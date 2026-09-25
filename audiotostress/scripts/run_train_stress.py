"""Train a classifier to pick the stressed vowel, and say how much the audio helps.

No single acoustic feature finds the stress: on the proof-of-concept slice
duration landed 51.1%, F0 median 56.5% and energy 42.0%, against 38.4% chance.
That is the shape a combination is meant to fix, and it is what RUAccent
trained rather than thresholding one number.

The comparison that decides whether any of this is worth doing is not
"classifier against chance" — it is **acoustic features against position
alone**. Ukrainian stress is far from uniform across syllables, so a model given
only "which vowel is this, of how many" already scores well without hearing
anything. If adding the recording does not beat that, the recording is not
contributing and the whole audio idea fails on its own evidence.

The split is by word form, not by row. The same form appearing in train and
test would let the model memorise the lexicon entry instead of learning to
listen, and every number would be inflated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

ACOUSTIC = [
    "duration_s", "log_duration", "duration_relative_word",
    "duration_relative_utterance", "rms_energy", "peak_energy",
    "relative_rms_energy", "f0_median_hz", "f0_range_hz", "f0_slope_hz_per_s",
]


def rows_from(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if 0 <= row["label"] < len(row["features"]):
                out.append(row)
    return out


def vectors(row: dict, *, acoustic: bool, positional: bool) -> np.ndarray:
    """One feature vector per vowel of this word."""
    count = len(row["features"])
    built = []
    for index, feature in enumerate(row["features"]):
        values: list[float] = []
        if acoustic:
            values += [float(feature.get(name) or 0.0) for name in ACOUSTIC]
            values.append(1.0 if feature.get("f0_valid") else 0.0)
            # Each feature against the word's own maximum: stress is relative,
            # and a loud speaker should not read as a stressed syllable.
            for name in ACOUSTIC:
                column = [float(f.get(name) or 0.0) for f in row["features"]]
                top = max(abs(v) for v in column) or 1.0
                values.append(column[index] / top)
        if positional:
            values += [float(index), float(count), index / max(count - 1, 1),
                       1.0 if index == 0 else 0.0,
                       1.0 if index == count - 1 else 0.0]
        built.append(values)
    return np.asarray(built, dtype=np.float64)


def evaluate(model, rows: list[dict], **kind) -> float:
    """Word-level accuracy: the argmax over a word's vowels must be the label."""
    hit = 0
    for row in rows:
        scores = model.predict_proba(vectors(row, **kind))[:, 1]
        hit += int(np.argmax(scores)) == row["label"]
    return hit / len(rows) if rows else float("nan")



def train_listwise(train: list[dict], test: list[dict], *, kind: dict,
                   epochs: int, seed: int) -> float:
    """Score a word's vowels against each other rather than one at a time.

    The binary arms decide each vowel on its own and take the argmax
    afterwards, which throws away the one thing known for certain: a word has
    exactly one stressed vowel. A softmax over the word's own vowels states
    that constraint in the loss, so the model spends its capacity on the
    comparison instead of rediscovering the rule.

    `VowelStressRanker` already implements this — a transformer over the vowel
    sequence with a scalar head — and nothing was calling it.
    """
    import torch

    from ukstress.ranker.model import VowelStressRanker, make_vowel_batch

    torch.manual_seed(seed)

    def batch_of(rows: list[dict]):
        prosody = [vectors(row, **kind).astype(np.float32) for row in rows]
        # The ranker concatenates an SSL block; there is none yet, so it is
        # empty and the prosodic vector carries everything.
        empty = [np.zeros((len(block), 0), dtype=np.float32) for block in prosody]
        return make_vowel_batch(empty, prosody,
                                targets=[row["label"] for row in rows])

    width = vectors(train[0], **kind).shape[1]
    ranker = VowelStressRanker(ssl_size=0, prosody_size=width, hidden_size=64,
                               transformer_layers=2, attention_heads=4, ffn_size=128)
    optimiser = torch.optim.AdamW(ranker.parameters(), lr=3e-4, weight_decay=0.01)

    # Standardise per column: raw hertz and seconds differ by orders of
    # magnitude, and a transformer fed unscaled inputs spends its first epochs
    # undoing that.
    stacked = np.vstack([vectors(row, **kind) for row in train])
    mean, deviation = stacked.mean(axis=0), stacked.std(axis=0) + 1e-6

    def normalise(rows: list[dict]):
        block = batch_of(rows)
        block.prosodic_features[:] = (block.prosodic_features - mean) / deviation
        return block

    train_batch = normalise(train)
    test_batch = normalise(test)
    targets = torch.as_tensor(train_batch.targets, dtype=torch.long)

    ranker.module.train()
    for _ in range(epochs):
        optimiser.zero_grad(set_to_none=True)
        logits = ranker.logits(train_batch)
        masked = logits.masked_fill(~torch.as_tensor(train_batch.valid_mask), float("-inf"))
        loss = torch.nn.functional.cross_entropy(masked, targets)
        loss.backward()
        optimiser.step()

    ranker.module.eval()
    with torch.no_grad():
        probabilities = ranker.probabilities(test_batch)
    picks = probabilities.argmax(axis=1)
    return float(np.mean(picks == np.asarray([row["label"] for row in test])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path,
                        default=Path("artifacts/audio_runs/mined/rows.jsonl"))
    parser.add_argument("--holdout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--listwise", action="store_true",
                        help="also train the transformer ranker over each "
                             "word's vowels with a softmax loss")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/audio_runs/mined/training_report.json"))
    args = parser.parse_args()

    rows = rows_from(args.rows)
    forms = sorted({row["form"] for row in rows})
    rng = np.random.default_rng(args.seed)
    rng.shuffle(forms)
    cut = int(len(forms) * (1 - args.holdout))
    train_forms = set(forms[:cut])
    train = [r for r in rows if r["form"] in train_forms]
    test = [r for r in rows if r["form"] not in train_forms]
    print(f"rows {len(rows):,} over {len(forms):,} forms | "
          f"train {len(train):,} / test {len(test):,}")
    if not test or not train:
        print("not enough data yet")
        return 1

    chance = sum(1 / len(r["features"]) for r in test) / len(test)
    longest = sum(1 for r in test
                  if int(np.argmax([f["duration_s"] for f in r["features"]])) == r["label"])
    print(f"\nchance level:          {100 * chance:.1f}%")
    print(f"longest vowel:         {100 * longest / len(test):.1f}%")

    report = {"rows": len(rows), "forms": len(forms), "test_rows": len(test),
              "chance": chance, "longest_vowel": longest / len(test), "arms": {}}

    arms = (("position only", {"acoustic": False, "positional": True}),
            ("acoustic only", {"acoustic": True, "positional": False}),
            ("acoustic + position", {"acoustic": True, "positional": True}))
    for name, kind in arms:
        features, labels = [], []
        for row in train:
            block = vectors(row, **kind)
            features.append(block)
            labels.append(np.eye(len(block), dtype=np.int64)[row["label"]])
        stacked = np.vstack(features)
        target = np.concatenate(labels)
        model = (GradientBoostingClassifier(random_state=args.seed)
                 if len(stacked) >= 200 else
                 LogisticRegression(max_iter=2000, random_state=args.seed))
        model.fit(stacked, target)
        accuracy = evaluate(model, test, **kind)
        report["arms"][name] = accuracy
        print(f"{name:<22}{100 * accuracy:>6.1f}%")

    if args.listwise:
        for name, kind in arms:
            accuracy = train_listwise(train, test, kind=kind,
                                      epochs=args.epochs, seed=args.seed)
            report["arms"][f"listwise: {name}"] = accuracy
            print(f"{'listwise: ' + name:<22}{100 * accuracy:>6.1f}%")

    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    gain = report["arms"]["acoustic + position"] - report["arms"]["position only"]
    print(f"\nwhat the audio adds over position alone: {100 * gain:+.1f} points")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
