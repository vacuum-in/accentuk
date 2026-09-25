"""K-fold cross-validation to certify the precision claim without new labels.

The balanced fine-tune reaches 0.9911 precision at margin 11, but on only 336
held-out multi-sense rows, so the 95% lower bound is 0.974 — the point estimate
clears 98%, the certified bound does not. Because 0.9911 sits comfortably above
0.98 (unlike the unbalanced model's 0.9808, which no sample size could certify),
the gap is a *sample size* problem, and roughly 2x the evaluation rows closes it.

Folding over forms supplies exactly that. Each form is held out in exactly one
fold, so pooling the held-out predictions evaluates every row in the corpus
instead of a single 25% slice — about 4x the multi-sense rows, for GPU time
rather than annotation spend.

What this certifies is the **procedure's** precision (train-on-75%-of-forms,
predict held-out forms), pooled across folds, not one checkpoint's. That is the
standard reading of cross-validated precision and is the honest label for it.

Folds split by form, never by sentence: sentences of one form on both sides
would measure memorisation.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ukstress_ml import crossencoder

sys.path.insert(0, str(Path(__file__).parent))
from run_eval_inflected import THRESHOLDS, load, score, wilson
from run_finetune_inflected import balance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--keep-fold", type=int, default=0,
                        help="keep this fold's weights; the rest are deleted (1.1 GB each)")
    parser.add_argument("--include-duplicate-gloss", action="store_true",
                        help="keep forms whose candidates share a gloss (unresolvable)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--balance-ratio", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--silver", type=Path, default=Path("output/ml/silver_inflected.json"))
    parser.add_argument(
        "--inventory", type=Path, default=Path("output/ml/ambiguous_forms_inflected.jsonl")
    )
    parser.add_argument("--base", type=Path, default=Path("models/v3-xenc/checkpoint"))
    parser.add_argument("--work", type=Path, default=Path("output/ml/models/cv"))
    parser.add_argument("--output", type=Path, default=Path("output/ml/cv_inflected.json"))
    args = parser.parse_args()

    rows, glosses = load(args.inventory, args.silver)
    if not args.include_duplicate_gloss:
        # Candidates carrying the same gloss give the cross-encoder identical
        # inputs, so the pick is a coin flip at high confidence. They were 32%
        # of high-margin errors; training on them teaches noise and evaluating
        # on them caps measured precision.
        by_form: dict[str, list[str]] = {}
        for row in rows:
            by_form.setdefault(row["form"], []).extend(
                glosses.get(c["sense_id"], {}).get("definition", "")
                for c in row["candidates"]
            )
        unresolvable = {
            form
            for form, defs in by_form.items()
            if len({" ".join(d.split()) for d in defs if d.strip()}) < len(
                {c for c in defs if c.strip()}
            )
            or (len(defs) > 1 and len({" ".join(d.split()) for d in defs}) == 1)
        }
        before = len(rows)
        rows = [r for r in rows if r["form"] not in unresolvable]
        print(f"excluded {len(unresolvable):,} duplicate-gloss forms "
              f"({before - len(rows):,} rows); {len(rows):,} remain", flush=True)

    forms = sorted({r["form"] for r in rows})
    random.Random(args.seed).shuffle(forms)
    folds = [set(forms[i :: args.folds]) for i in range(args.folds)]

    predictions: list[dict[str, Any]] = []
    pooled_margin: list[float] = []
    pooled_correct: list[bool] = []
    pooled_multi: list[bool] = []
    per_fold: list[dict[str, Any]] = []

    for index, held_forms in enumerate(folds):
        train = [r for r in rows if r["form"] not in held_forms]
        held = [r for r in rows if r["form"] in held_forms]
        if args.balance_ratio:
            train = balance(train, args.balance_ratio, args.seed + index)

        senses: dict[str, set[str]] = collections.defaultdict(set)
        for row in held:
            senses[row["form"]].add(row["gold_sense"])
        multi = {f for f, s in senses.items() if len(s) > 1}

        out = args.work / f"fold{index}"
        print(f"\n=== fold {index + 1}/{args.folds}: train={len(train):,} "
              f"held={len(held):,} ({len(held_forms)} forms, {len(multi)} multi-sense)",
              flush=True)
        run = crossencoder.train(
            train,
            held,
            glosses,
            out,
            config=crossencoder.CrossEncoderConfig(
                base_model=str(args.base),
                max_epochs=args.epochs,
                learning_rate=args.lr,
                seed=args.seed + index,
            ),
            corpus_version=f"inflected-cv-fold{index}",
        )
        margins, correct = score(out / "checkpoint", held, glosses)
        # Persist per-row out-of-fold predictions. Every row is predicted by a
        # model that never trained on its form, which is exactly the signal
        # needed to find likely label errors later — and it costs kilobytes
        # instead of the 1.1 GB a checkpoint would.
        predictions.extend(
            {
                "sentence_id": row["sentence_id"],
                "form": row["form"],
                "group_id": row["group_id"],
                "silver": row["gold_sense"],
                "margin": float(mg),
                "correct": bool(ok),
                "fold": index,
            }
            for row, mg, ok in zip(held, margins, correct, strict=True)
        )
        is_multi = np.array([r["form"] in multi for r in held])
        pooled_margin.extend(margins.tolist())
        pooled_correct.extend(correct.tolist())
        pooled_multi.extend(is_multi.tolist())
        entry = {
            "fold": index,
            "held_rows": len(held),
            "multi_sense_rows": int(is_multi.sum()),
            "accuracy_multi_sense": float(correct[is_multi].mean()) if is_multi.any() else 0.0,
            "best_dev_group_macro": run.best_dev_group_macro,
        }
        per_fold.append(entry)
        print(f"fold {index}: multi-sense acc={entry['accuracy_multi_sense']:.4f}", flush=True)
        # Each fold's checkpoint is 1.1 GB. Keep one so higher thresholds can
        # be explored later without retraining; delete the rest.
        if index != args.keep_fold:
            for weight in (out / "checkpoint").glob("*.safetensors"):
                weight.unlink()

    margin = np.array(pooled_margin)
    correct = np.array(pooled_correct)
    multi = np.array(pooled_multi)
    mm, mc = margin[multi], correct[multi]

    curve = []
    for t in THRESHOLDS:
        keep = mm >= t
        if not keep.sum():
            continue
        lo, hi = wilson(int(mc[keep].sum()), int(keep.sum()))
        curve.append(
            {
                "margin": float(t),
                "coverage": float(keep.mean()),
                "rows": int(keep.sum()),
                "precision": float(mc[keep].mean()),
                "ci_low": lo,
                "ci_high": hi,
                "certified_98": bool(lo >= 0.98),
            }
        )

    report = {
        "folds": args.folds,
        "balance_ratio": args.balance_ratio,
        "pooled_rows": int(correct.size),
        "pooled_multi_sense_rows": int(multi.sum()),
        "accuracy_multi_sense": float(mc.mean()),
        "per_fold": per_fold,
        "curve_multi_sense": curve,
        "note": "pooled over folds; certifies the training procedure, not one checkpoint",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    oof = args.output.with_name(args.output.stem + "_predictions.jsonl")
    with oof.open("w", encoding="utf-8") as handle:
        for record in predictions:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"out-of-fold predictions -> {oof} ({len(predictions):,} rows)")

    print(f"\npooled multi-sense rows: {int(multi.sum()):,}  accuracy={mc.mean():.4f}")
    print(f"{'margin':>8}{'cov':>8}{'rows':>7}{'precision':>11}{'95% CI':>20}")
    for c in curve:
        flag = "   CERTIFIED >=98%" if c["certified_98"] else ""
        print(f"{c['margin']:>8.2f}{c['coverage']:>8.3f}{c['rows']:>7}{c['precision']:>11.4f}"
              f"   [{c['ci_low']:.3f},{c['ci_high']:.3f}]{flag}")
    print(f"\nwritten to {args.output}")


if __name__ == "__main__":
    main()
