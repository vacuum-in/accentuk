"""Train on the mined recording, test on the passage whose stress is known.

Every number so far has been measured inside one narrator reading one novel,
on words the lexicon already names unambiguously — which are exactly the words
that need no model. This scores the thing that matters instead: a different
voice, different material, and forms where the dictionary offers two readings
and only the recording says which was spoken.

The homograph subset is reported separately because it is the whole point. A
model can score well overall by learning that Ukrainian stress rarely lands on
the last syllable and still be useless on `за́мок` against `замки́`.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ACOUSTIC = [
    "duration_s", "log_duration", "duration_relative_word",
    "duration_relative_utterance", "rms_energy", "peak_energy",
    "relative_rms_energy", "f0_median_hz", "f0_range_hz", "f0_slope_hz_per_s",
]


def rows_from(path: Path) -> list[dict]:
    """Rows, with any SSL block attached from the matrix beside them.

    1024 numbers per vowel do not belong in JSON; the miner writes them to a
    `.ssl.npy` alongside and each row carries the slice it owns.
    """
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if 0 <= row["label"] < len(row["features"]):
                out.append(row)
    matrix_path = path.with_suffix(".ssl.npy")
    if matrix_path.is_file():
        matrix = np.load(matrix_path)
        for row in out:
            span = row.get("ssl_index")
            row["ssl"] = matrix[span[0]:span[1]] if span else None
    return out


def vectors(row: dict, *, acoustic: bool = True, positional: bool = True,
            ssl: bool = False) -> np.ndarray:
    count = len(row["features"])
    block = row.get("ssl") if ssl else None
    built = []
    for index, feature in enumerate(row["features"]):
        values: list[float] = []
        if acoustic:
            values += [float(feature.get(name) or 0.0) for name in ACOUSTIC]
            values.append(1.0 if feature.get("f0_valid") else 0.0)
            for name in ACOUSTIC:
                column = [float(f.get(name) or 0.0) for f in row["features"]]
                top = max(abs(v) for v in column) or 1.0
                values.append(column[index] / top)
        if positional:
            values += [float(index), float(count), index / max(count - 1, 1),
                       1.0 if index == 0 else 0.0,
                       1.0 if index == count - 1 else 0.0]
        if block is not None and index < len(block):
            frame = np.asarray(block[index], dtype=np.float64)
            # Against the word's own mean: stress is a contrast between this
            # vowel and its neighbours, and an absolute embedding carries the
            # speaker and the phoneme instead.
            values += list(frame - np.asarray(block, dtype=np.float64).mean(axis=0))
        built.append(values)
    return np.asarray(built, dtype=np.float64)


def score(model, rows: list[dict], **kind) -> tuple[int, int]:
    hit = 0
    for row in rows:
        pick = int(np.argmax(model.predict_proba(vectors(row, **kind))[:, 1]))
        hit += pick == row["label"]
    return hit, len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path,
                        default=Path("artifacts/audio_runs/mined/rows.jsonl"))
    parser.add_argument("--test", type=Path,
                        default=Path("artifacts/audio_runs/stressed_eval/rows.jsonl"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--linear", action="store_true",
                        help="use the regularised linear model for every arm, "
                             "so the comparison is between features rather "
                             "than between features and classifier at once")
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/audio_runs/crossset_report.json"))
    args = parser.parse_args()

    train = rows_from(args.train)
    test = rows_from(args.test)
    print(f"train {len(train):,} rows | test {len(test):,} rows "
          f"over {len({r['form'] for r in test})} forms")

    has_ssl = any(r.get("ssl") is not None for r in train)
    arms = [("position only", {"acoustic": False, "positional": True}),
            ("acoustic only", {"acoustic": True, "positional": False}),
            ("acoustic + position", {"acoustic": True, "positional": True})]
    if has_ssl:
        arms += [("ssl only", {"acoustic": False, "positional": False, "ssl": True}),
                 ("ssl + acoustic", {"acoustic": True, "positional": False, "ssl": True})]

    homographs = [r for r in test if r.get("ambiguous_in_text")]
    chance = sum(1 / len(r["features"]) for r in test) / len(test)
    longest = sum(1 for r in test
                  if int(np.argmax([f["duration_s"] for f in r["features"]])) == r["label"])
    print(f"\nchance level:        {100 * chance:.1f}%")
    print(f"longest vowel:       {100 * longest / len(test):.1f}%")
    print(f"homograph rows:      {len(homographs)}")

    report = {"train_rows": len(train), "test_rows": len(test),
              "homograph_rows": len(homographs), "chance": chance,
              "longest_vowel": longest / len(test), "arms": {}}

    print(f"\n{'arm':<22}{'all':>8}{'homographs':>13}")
    for name, kind in arms:
        features, labels = [], []
        for row in train:
            block = vectors(row, **kind)
            features.append(block)
            labels.append(np.eye(len(block), dtype=np.int64)[row["label"]])
        stacked = np.vstack(features)
        # Boosting cannot use a thousand columns from four thousand rows; a
        # regularised linear model can, and picking by width rather than by
        # taste keeps the arms comparable.
        model = (make_pipeline(StandardScaler(),
                               LogisticRegression(max_iter=3000, C=0.05,
                                                  random_state=args.seed))
                 if args.linear or stacked.shape[1] > 200
                 else GradientBoostingClassifier(random_state=args.seed))
        model.fit(stacked, np.concatenate(labels))

        hit, total = score(model, test, **kind)
        hom_hit, hom_total = score(model, homographs, **kind) if homographs else (0, 0)
        report["arms"][name] = {
            "all": hit / total,
            "homographs": hom_hit / hom_total if hom_total else None,
        }
        share = f"{100 * hom_hit / hom_total:.1f}%" if hom_total else "—"
        print(f"{name:<22}{100 * hit / total:>7.1f}%{share:>13}")

        if name == "acoustic + position":
            wrong = collections.Counter()
            for row in homographs:
                pick = int(np.argmax(model.predict_proba(vectors(row, **kind))[:, 1]))
                if pick != row["label"]:
                    wrong[row["form"]] += 1
            if wrong:
                print("\n  homographs it gets wrong:",
                      ", ".join(f"{f}×{n}" for f, n in wrong.most_common(12)))

    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
