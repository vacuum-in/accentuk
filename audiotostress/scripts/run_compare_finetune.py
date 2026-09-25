"""Pair the fine-tuned model against the frozen one, row for row.

Both models are evaluated on the same speaker-held-out test set — the same
clips, the same words, in the same order — so the comparison is paired and
McNemar applies. Comparing two accuracy figures would not settle a difference
this size: the three-slice change was 0.45 points and needed the paired test
to show up at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from run_finetune_encoder import speaker_split  # noqa: E402
from run_train_speakers import batches, load  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-rows", type=Path,
                        default=Path("artifacts/audio_runs/cv_full_parts/rows.jsonl"))
    parser.add_argument("--frozen", type=Path,
                        default=Path("artifacts/audio_runs/production/ranker.pt"))
    parser.add_argument("--marks", type=Path,
                        default=Path("artifacts/audio_runs/finetuned/marks.jsonl"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/audio_runs/finetune_comparison.json"))
    args = parser.parse_args()

    import torch

    from ukstress.ranker.model import load_ranker_checkpoint

    rows, matrix, dim = load(args.frozen_rows)
    rows = [r for r in rows if not r.get("prepositional_clitic")]
    home = speaker_split(rows, args.seed)
    test = [r for r in rows if home[r["speaker"]] == "test"]

    ranker, _ = load_ranker_checkpoint(args.frozen, device=args.device)
    ranker.module.eval()
    frozen: dict[tuple[str, str], list[int]] = {}
    with torch.no_grad():
        for chunk, ssl, prosody, valid, targets in batches(test, matrix, dim, 256,
                                                           shuffle=False):
            mask = torch.as_tensor(valid, device=args.device)
            logits = ranker.module(torch.as_tensor(ssl, device=args.device),
                                   torch.as_tensor(prosody, device=args.device), mask)
            picks = logits.masked_fill(~mask, float("-inf")).argmax(dim=-1).cpu().numpy()
            for row, pick, gold in zip(chunk, picks, targets, strict=True):
                frozen.setdefault((row["clip"], row["form"]), []).append(int(pick == gold))

    tuned: dict[tuple[str, str], list[int]] = {}
    for line in args.marks.read_text(encoding="utf-8").splitlines():
        if line.strip():
            mark = json.loads(line)
            tuned.setdefault((mark["clip"], mark["form"]), []).append(mark["hit"])

    # Keyed on clip and form rather than position: the two runs filter rows
    # slightly differently, and a silent off-by-one would invent a difference.
    paired = []
    for key, hits in tuned.items():
        other = frozen.get(key)
        if other and len(other) == len(hits):
            paired.extend(zip(other, hits, strict=True))
    if not paired:
        print("no rows matched between the two runs")
        return 1

    a = [x for x, _ in paired]
    b = [y for _, y in paired]
    n01 = sum(1 for x, y in paired if not x and y)
    n10 = sum(1 for x, y in paired if x and not y)
    n = n01 + n10
    p = 1.0 if n == 0 else min(
        1.0, 2 * sum(comb(n, k) for k in range(min(n01, n10) + 1)) / 2 ** n)
    print(f"paired rows        {len(paired):,}")
    print(f"frozen encoder     {100 * sum(a) / len(a):.2f}%")
    print(f"fine-tuned encoder {100 * sum(b) / len(b):.2f}%")
    print(f"discordant: tuned-only-right {n01}, frozen-only-right {n10}")
    print(f"McNemar exact two-sided p = {p:.3g}")
    args.out.write_text(json.dumps({
        "paired_rows": len(paired), "frozen": sum(a) / len(a),
        "finetuned": sum(b) / len(b), "tuned_only": n01, "frozen_only": n10,
        "p_value": p}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
