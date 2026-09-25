"""Train the vowel ranker on Common Voice, held out by speaker.

Everything measured before this was one narrator. A model can reach 94% on one
voice by learning that voice — its habitual vowel length, its microphone, its
room — and tell you nothing about whether it hears stress. Common Voice ships
official train/dev/test splits built so that no speaker appears in two of
them, and this trains and scores across that boundary: the test voices are
ones the model has never heard.

The model is the listwise ranker rather than a per-vowel classifier, because
the question is which vowel of this word carries the stress, not whether each
vowel independently looks stressed. One softmax over a word's vowels states
that directly, and it cannot answer "two of them" or "none".

The report breaks the test set down by speaker as well as in aggregate. A
single mean hides the case that matters — a model that is excellent on most
voices and lost on a few is a different object from one that is uniformly
mediocre, and only the per-speaker spread tells them apart.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

PROSODY_MODE = "full"

ACOUSTIC = [
    "duration_s", "log_duration", "duration_relative_word",
    "duration_relative_utterance", "rms_energy", "peak_energy",
    "relative_rms_energy", "f0_median_hz", "f0_range_hz", "f0_slope_hz_per_s",
]


def prosody_matrix(row: dict, mode: str = "full") -> np.ndarray:
    """Per-vowel prosody, each feature also given relative to the word.

    The absolute numbers carry the speaker: a quiet voice has a low RMS on
    every vowel it utters. The within-word ratio is what survives a change of
    speaker, so both go in and the model may weigh them.
    """
    count = len(row["features"])
    columns = {name: [float(f.get(name) or 0.0) for f in row["features"]]
               for name in ACOUSTIC}
    built = []
    for index, feature in enumerate(row["features"]):
        values = [float(feature.get(name) or 0.0) for name in ACOUSTIC]
        values.append(1.0 if feature.get("f0_valid") else 0.0)
        for name in ACOUSTIC:
            top = max(abs(v) for v in columns[name]) or 1.0
            values.append(columns[name][index] / top)
        positional = [float(index), float(count), index / max(count - 1, 1),
                      1.0 if index == 0 else 0.0,
                      1.0 if index == count - 1 else 0.0]
        if mode == "position":
            values = positional
        elif mode == "none":
            values = [0.0]
        else:
            values += positional
        built.append(values)
    return np.asarray(built, dtype=np.float32)


def load(path: Path) -> tuple[list[dict], np.memmap | None, int]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if 0 <= row["label"] < len(row["features"]):
                rows.append(row)
    meta = path.with_suffix(".ssl.json")
    raw = path.with_suffix(".ssl.f16")
    if meta.is_file() and raw.is_file():
        dim = json.loads(meta.read_text(encoding="utf-8"))["dim"]
        # A memmap, not np.load: at a hundred thousand words this matrix runs
        # to gigabytes, and the training loop touches one word's slice at a
        # time.
        matrix = np.memmap(raw, dtype=np.float16, mode="r").reshape(-1, dim)
        return rows, matrix, dim
    return rows, None, 0


def ssl_block(row: dict, matrix: np.memmap | None, vowels: int) -> np.ndarray:
    if matrix is None or not row.get("ssl_index"):
        return np.zeros((vowels, 0), dtype=np.float32)
    start, end = row["ssl_index"]
    block = np.asarray(matrix[start:end], dtype=np.float32)
    if len(block) != vowels:
        return np.zeros((vowels, matrix.shape[1]), dtype=np.float32)
    # Against the word's own mean: stress is a contrast between this vowel and
    # its neighbours, and an absolute embedding carries the speaker.
    return block - block.mean(axis=0, keepdims=True)


def batches(rows: list[dict], matrix, dim: int, size: int, *, shuffle: bool,
            seed: int = 0):
    order = list(range(len(rows)))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), size):
        chunk = [rows[i] for i in order[start:start + size]]
        widest = max(len(r["features"]) for r in chunk)
        prosody_dim = len(prosody_matrix(chunk[0], PROSODY_MODE)[0])
        ssl = np.zeros((len(chunk), widest, dim), dtype=np.float32)
        prosody = np.zeros((len(chunk), widest, prosody_dim), dtype=np.float32)
        valid = np.zeros((len(chunk), widest), dtype=bool)
        targets = np.zeros(len(chunk), dtype=np.int64)
        for index, row in enumerate(chunk):
            vowels = len(row["features"])
            if dim:
                ssl[index, :vowels] = ssl_block(row, matrix, vowels)
            prosody[index, :vowels] = prosody_matrix(row, PROSODY_MODE)
            valid[index, :vowels] = True
            targets[index] = row["label"]
        yield chunk, ssl, prosody, valid, targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path,
                        default=Path("artifacts/audio_runs/commonvoice/rows.jsonl"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-ssl", action="store_true",
                        help="prosody and position only, to price what the "
                             "SSL embeddings are actually worth")
    parser.add_argument("--train-fraction", type=float, default=1.0,
                        help="train on this fraction of the training speakers, "
                             "to read the data-scaling curve. Held out by "
                             "speaker, so a smaller fraction is fewer voices "
                             "as well as fewer rows — which is how more data "
                             "would actually arrive")
    parser.add_argument("--prosody", choices=("full", "position", "none"),
                        default="full",
                        help="`position` keeps only the five free positional "
                             "features and drops the measured ones; `none` "
                             "drops those too. Ablating at inference put the "
                             "measured features' worth at 0.19 points, an "
                             "upper bound — a model trained without them can "
                             "compensate, and this says whether it does")
    parser.add_argument("--ssl-dims", default="all",
                        help="`all`, or `LOW:HIGH` to read only those columns "
                             "of the stored SSL block. With a miner run using "
                             "--ssl-parts, `0:1024` is exactly the mean-pooled "
                             "representation, so the two can be compared on "
                             "identical rows")
    parser.add_argument("--keep-prepositional", action="store_true",
                        help="keep clitic pronouns after a preposition. Their "
                             "trie label is the bare form's (мене́) while the "
                             "speaker says до ме́не, so they are mislabelled "
                             "rather than difficult; scripts/run_tag_context.py "
                             "marks them and they are dropped by default")
    parser.add_argument("--split", choices=("official", "speaker"),
                        default="official",
                        help="`official` uses Common Voice's own splits, which "
                             "are built for ASR and put 727 voices in test "
                             "against 70 in train; `speaker` re-cuts the same "
                             "corpus 75/10/15 by speaker, keeping the boundary "
                             "but spending the voices on training instead")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/audio_runs/speaker_model"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import torch

    from ukstress.ranker.model import (
        VowelStressRanker,
        load_ranker_checkpoint,
        masked_cross_entropy,
        save_ranker_checkpoint,
    )

    rows, matrix, dim = load(args.rows)
    if args.ssl_dims != "all" and matrix is not None:
        low, high = (int(v) for v in args.ssl_dims.split(":"))
        matrix, dim = matrix[:, low:high], high - low
        print(f"reading SSL columns {low}:{high}")
    if args.no_ssl:
        matrix, dim = None, 0
    if not args.keep_prepositional:
        before = len(rows)
        rows = [r for r in rows if not r.get("prepositional_clitic")]
        if before != len(rows):
            print(f"dropped {before - len(rows):,} mislabelled prepositional clitics")
    # Split by speaker, not by clip. Common Voice keeps its splits speaker-
    # disjoint, but validated.tsv holds clips that belong to no split at all,
    # and some of those are read by voices that do appear in test. Routing
    # them by clip put eight test voices into training.
    home: dict[str, str] = {}
    if args.split == "official":
        for row in rows:
            named = row.get("split")
            if named in ("train", "dev", "test"):
                home.setdefault(row["speaker"], named)
    else:
        # Common Voice sizes its splits for an ASR leaderboard, which leaves
        # 70 voices to learn from and 727 to be tested on. The boundary is
        # what matters here, not its position, so this re-cuts the corpus by
        # speaker and spends the voices where they teach.
        voices = sorted({row["speaker"] for row in rows})
        np.random.default_rng(args.seed).shuffle(voices)
        cut = int(0.75 * len(voices)), int(0.85 * len(voices))
        for index, speaker in enumerate(voices):
            home[speaker] = ("train" if index < cut[0]
                             else "dev" if index < cut[1] else "test")
    parts: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for row in rows:
        parts[home.get(row["speaker"], "train")].append(row)
    train, dev, test = parts["train"], parts["dev"], parts["test"]
    if args.train_fraction < 1.0:
        voices = sorted({r["speaker"] for r in train})
        np.random.default_rng(args.seed).shuffle(voices)
        keep = set(voices[:max(1, int(args.train_fraction * len(voices)))])
        train = [r for r in train if r["speaker"] in keep]

    voices = {name: {r["speaker"] for r in part}
              for name, part in (("train", train), ("dev", dev), ("test", test))}
    overlap = voices["train"] & voices["test"]
    print(f"train {len(train):,} rows / {len(voices['train']):,} speakers")
    print(f"dev   {len(dev):,} rows / {len(voices['dev']):,} speakers")
    print(f"test  {len(test):,} rows / {len(voices['test']):,} speakers")
    print(f"speaker overlap train/test: {len(overlap)}")
    if not train or not test:
        print("not enough mined rows yet")
        return 1

    global PROSODY_MODE
    PROSODY_MODE = args.prosody
    prosody_dim = len(prosody_matrix(train[0], PROSODY_MODE)[0])
    ranker = VowelStressRanker(ssl_size=dim, prosody_size=prosody_dim,
                               hidden_size=args.hidden,
                               transformer_layers=args.layers,
                               attention_heads=4, ffn_size=4 * args.hidden)
    ranker.module.to(args.device)
    optimiser = torch.optim.AdamW(ranker.parameters(), lr=args.lr,
                                  weight_decay=0.01)
    torch.manual_seed(args.seed)

    def accuracy(part: list[dict]) -> tuple[float, dict[str, list[int]]]:
        ranker.module.eval()
        hits: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        total = correct = 0
        with torch.no_grad():
            for chunk, ssl, prosody, valid, targets in batches(
                    part, matrix, dim, 256, shuffle=False):
                logits = ranker.module(
                    torch.as_tensor(ssl, device=args.device),
                    torch.as_tensor(prosody, device=args.device),
                    torch.as_tensor(valid, device=args.device))
                logits = logits.masked_fill(
                    ~torch.as_tensor(valid, device=args.device), float("-inf"))
                picks = logits.argmax(dim=-1).cpu().numpy()
                for row, pick, gold in zip(chunk, picks, targets, strict=True):
                    got = int(pick == gold)
                    hits[row["speaker"]][0] += got
                    hits[row["speaker"]][1] += 1
                    correct += got
                    total += 1
        return correct / max(total, 1), hits

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        ranker.module.train()
        loss_sum = steps = 0.0
        for _, ssl, prosody, valid, targets in batches(
                train, matrix, dim, args.batch, shuffle=True, seed=args.seed + epoch):
            valid_t = torch.as_tensor(valid, device=args.device)
            logits = ranker.module(
                torch.as_tensor(ssl, device=args.device),
                torch.as_tensor(prosody, device=args.device), valid_t)
            loss = masked_cross_entropy(
                logits, valid_t, torch.as_tensor(targets, device=args.device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            loss_sum += float(loss.detach())
            steps += 1
        dev_score, _ = accuracy(dev or test)
        print(f"epoch {epoch:>2}: loss {loss_sum / max(steps, 1):.4f}  "
              f"dev {100 * dev_score:.2f}%", flush=True)
        if dev_score >= best:
            best = dev_score
            save_ranker_checkpoint(
                args.out / "ranker.pt", ranker,
                metadata={"rows": str(len(train)), "dev": f"{dev_score:.4f}",
                          "ssl": str(bool(dim))})

    # Score the checkpoint, not whatever the last epoch happened to leave in
    # memory. Dev peaks a few epochs before the end and the saved weights are
    # the peak, so reporting the final state described a model nobody would
    # ever load — 95.62% against the checkpoint's real 95.92%.
    restored, _ = load_ranker_checkpoint(args.out / "ranker.pt", device=args.device)
    ranker.module.load_state_dict(restored.module.state_dict())
    ranker.module.to(args.device)
    test_score, per_speaker = accuracy(test)
    chance = sum(1 / len(r["features"]) for r in test) / len(test)
    longest = sum(1 for r in test
                  if int(np.argmax([f["duration_s"] for f in r["features"]])) == r["label"])
    scored = sorted(hit / seen for hit, seen in per_speaker.values() if seen >= 20)

    print(f"\nchance level:     {100 * chance:.1f}%")
    print(f"longest vowel:    {100 * longest / len(test):.1f}%")
    print(f"ranker (test):    {100 * test_score:.2f}%")
    if scored:
        print(f"\nper speaker over {len(scored)} voices with 20+ words:")
        print(f"  median          {100 * statistics.median(scored):.1f}%")
        print(f"  worst / best    {100 * scored[0]:.1f}% / {100 * scored[-1]:.1f}%")
        print(f"  10th pct        {100 * scored[max(0, len(scored) // 10)]:.1f}%")

    report = {
        "train_rows": len(train), "test_rows": len(test),
        "train_speakers": len(voices["train"]), "test_speakers": len(voices["test"]),
        "speaker_overlap": len(overlap), "ssl": bool(dim),
        "split": args.split,
        "chance": chance, "longest_vowel": longest / len(test),
        "test_accuracy": test_score, "dev_accuracy": best,
        "per_speaker": {"count": len(scored),
                        "median": statistics.median(scored) if scored else None,
                        "worst": scored[0] if scored else None,
                        "best": scored[-1] if scored else None},
    }
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport -> {args.out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
