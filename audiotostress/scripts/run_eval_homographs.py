"""Score a trained ranker on the words a dictionary cannot settle.

Every accuracy this project has reported was measured on forms the stress trie
names unambiguously — which are precisely the forms that need no model at all.
A ranker can post 95% there by learning where Ukrainian stress usually falls
and still be deaf to the distinction that matters: `за́мок` against `замки́`,
where only the recording says which was spoken.

This loads a checkpoint trained on Common Voice and runs it against the
stressed passage: different speakers, different recording chain, different
material, and a subset of rows whose form the dictionary offers two readings
for. The homograph number is the one worth quoting, and it is quoted with a
confidence interval because thirty-four rows over eighteen forms is a small
sample and pretending otherwise would be the easiest mistake here.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

import run_train_speakers as trainer  # noqa: E402
from run_train_speakers import ACOUSTIC, prosody_matrix  # noqa: E402


def wilson(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if not total:
        return (0.0, 0.0)
    p = hits / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def load_eval(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if 0 <= row["label"] < len(row["features"]):
                rows.append(row)
    matrix_path = path.with_suffix(".ssl.npy")
    if matrix_path.is_file():
        matrix = np.load(matrix_path)
        for row in rows:
            span = row.get("ssl_index")
            block = matrix[span[0]:span[1]] if span else None
            row["ssl"] = block
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/audio_runs/speaker_full/ranker.pt"))
    parser.add_argument("--rows", type=Path,
                        default=Path("artifacts/audio_runs/stressed_eval/rows.jsonl"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/audio_runs/homograph_report.json"))
    args = parser.parse_args()

    import torch

    from ukstress.ranker.model import load_ranker_checkpoint

    ranker, metadata = load_ranker_checkpoint(args.checkpoint, device=args.device)
    ranker.module.eval()
    ssl_size = ranker.architecture["ssl_size"]
    # The checkpoint's prosody width says which features it was trained on:
    # 26 is the full set, 5 the free positional ones, 1 none at all. Reading
    # it from the model means the two cannot fall out of step.
    trainer.PROSODY_MODE = {26: "full", 5: "position", 1: "none"}.get(
        ranker.architecture["prosody_size"], "full")
    print(f"prosody:    {trainer.PROSODY_MODE}")
    rows = load_eval(args.rows)
    print(f"checkpoint: {args.checkpoint}  (trained on {metadata.get('rows')} rows)")
    print(f"eval rows:  {len(rows):,}")

    picks = []
    with torch.no_grad():
        for row in rows:
            vowels = len(row["features"])
            prosody = prosody_matrix(row, trainer.PROSODY_MODE)[None, :, :]
            if ssl_size:
                block = row.get("ssl")
                if block is None or len(block) != vowels:
                    ssl = np.zeros((1, vowels, ssl_size), dtype=np.float32)
                else:
                    block = np.asarray(block, dtype=np.float32)
                    ssl = (block - block.mean(axis=0, keepdims=True))[None, :, :]
            else:
                ssl = np.zeros((1, vowels, 0), dtype=np.float32)
            valid = np.ones((1, vowels), dtype=bool)
            logits = ranker.module(
                torch.as_tensor(ssl, device=args.device),
                torch.as_tensor(prosody, device=args.device),
                torch.as_tensor(valid, device=args.device))
            picks.append(int(logits.argmax(dim=-1).item()))

    def report(name: str, subset: list[tuple[dict, int]]) -> dict:
        if not subset:
            return {}
        hits = sum(1 for row, pick in subset if pick == row["label"])
        low, high = wilson(hits, len(subset))
        chance = sum(1 / len(row["features"]) for row, _ in subset) / len(subset)
        longest = sum(1 for row, _ in subset
                      if int(np.argmax([f["duration_s"] for f in row["features"]]))
                      == row["label"])
        print(f"\n{name}  ({len(subset)} rows, "
              f"{len({r['form'] for r, _ in subset})} forms)")
        print(f"  chance          {100 * chance:.1f}%")
        print(f"  longest vowel   {100 * longest / len(subset):.1f}%")
        print(f"  ranker          {100 * hits / len(subset):.1f}%  "
              f"[{100 * low:.1f}, {100 * high:.1f}]")
        return {"rows": len(subset), "hits": hits,
                "accuracy": hits / len(subset), "ci": [low, high],
                "chance": chance, "longest_vowel": longest / len(subset)}

    paired = list(zip(rows, picks, strict=True))
    summary = {
        "checkpoint": str(args.checkpoint),
        "all": report("all words", paired),
        "homographs": report("homographs only",
                             [(r, p) for r, p in paired if r.get("ambiguous_in_text")]),
    }

    wrong = collections.Counter(
        r["form"] for r, p in paired if r.get("ambiguous_in_text") and p != r["label"])
    if wrong:
        print("\nmissed homograph forms: " +
              ", ".join(f"{form}×{count}" for form, count in wrong.most_common()))
        summary["missed"] = dict(wrong)

    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\nreport -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
