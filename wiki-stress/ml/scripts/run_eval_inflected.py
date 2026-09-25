"""Evaluate an inflected-forms model the way `RESULTS.md` §2 says to.

Aggregate accuracy on this corpus is not a result. 421 of 598 held-out forms
occur with only **one** sense, so they are answered by memorising a prior rather
than by reading context, and they inflate the model and its baseline together.
Measured on the held-out split, the majority baseline scores **0.8597** while
zero-shot v3-xenc scores 0.8631 — a 0.34pp gap that an aggregate number
presents as 86% "accuracy".

`RESULTS.md` §2 recorded this exact lesson for the lemma corpus ("35% of
evaluation groups appear with only one sense ... they inflate both the model and
its baseline"). Here it is 70%.

So this reports, for both the base and the fine-tuned checkpoint:

* the majority baseline, always, as the bar to clear;
* accuracy restricted to **multi-sense forms** — forms that genuinely occur with
  more than one sense in the held-out data, which is where context actually
  decides anything;
* **minority-sense recall** — rows whose gold sense is not the form's dominant
  one. The majority baseline scores exactly 0 here, so anything above 0 is
  context being read;
* the precision/coverage curve with Wilson intervals, on the multi-sense subset,
  because the certified claim is a lower bound and not a point estimate.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ukstress_ml import ambiguity, crossencoder

THRESHOLDS = (0.0, 1.2482, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0)


def wilson(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = hits / total
    den = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return ((centre - margin) / den, (centre + margin) / den)


def load(inventory: Path, silver: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    forms = {f.form: f for f in ambiguity.load(inventory)}
    glosses = {
        c.sense_id: {"definition": c.definition, "stressed": c.stressed}
        for f in forms.values()
        for c in f.candidates
    }
    rows = []
    for row in json.loads(silver.read_text(encoding="utf-8")):
        form = forms[row["form"]]
        rows.append(
            {
                **row,
                "sense_id": row["gold_sense"],
                "candidates": [
                    {"sense_id": c.sense_id, "signature": c.signature} for c in form.candidates
                ],
            }
        )
    return rows, glosses


def score(model_dir: Path, rows: list[dict[str, Any]], glosses: Any) -> tuple[np.ndarray, np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()
    loader = DataLoader(
        crossencoder.PairDataset(rows, glosses),
        batch_size=16,
        shuffle=False,
        collate_fn=crossencoder.make_collate(tokenizer, 192),
    )
    device = next(model.parameters()).device
    margins: list[float] = []
    correct: list[bool] = []
    with torch.no_grad():
        for batch in loader:
            encoded = {k: v.to(device) for k, v in batch["encoded"].items()}
            logits = model(**encoded).logits.squeeze(-1).float().cpu().numpy()
            offset = 0
            for width, gold in zip(batch["spans"], batch["golds"].tolist(), strict=True):
                scores = logits[offset : offset + width]
                order = np.argsort(-scores)
                margins.append(
                    float(scores[order[0]] - scores[order[1]]) if width > 1 else float("inf")
                )
                correct.append(int(order[0]) == gold)
                offset += width
    del model
    torch.cuda.empty_cache()
    return np.array(margins), np.array(correct)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--silver", type=Path, default=Path("output/ml/silver_inflected.json"))
    parser.add_argument(
        "--inventory", type=Path, default=Path("output/ml/ambiguous_forms_inflected.jsonl")
    )
    parser.add_argument("--base", type=Path, default=Path("models/v3-xenc/checkpoint"))
    parser.add_argument("--tuned", type=Path)
    parser.add_argument("--holdout", type=float, default=0.25)
    parser.add_argument("--holdout-by", choices=("form", "group"), default="form")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output", type=Path, default=Path("output/ml/eval_inflected.json"))
    args = parser.parse_args()

    rows, glosses = load(args.inventory, args.silver)
    key = "form" if args.holdout_by == "form" else "group_id"
    values = sorted({r[key] for r in rows}, key=str)
    random.Random(args.seed).shuffle(values)
    held_keys = set(values[: max(1, int(len(values) * args.holdout))])
    train = [r for r in rows if r[key] not in held_keys]
    held = [r for r in rows if r[key] in held_keys]

    # A form is "multi-sense" when the held-out data actually shows it with more
    # than one sense. Single-sense forms test recall of a prior, not context.
    senses = collections.defaultdict(set)
    for row in held:
        senses[row["form"]].add(row["gold_sense"])
    multi_forms = {f for f, s in senses.items() if len(s) > 1}
    dominant = {
        f: collections.Counter(r["gold_sense"] for r in held if r["form"] == f).most_common(1)[0][0]
        for f in multi_forms
    }

    prior = collections.defaultdict(collections.Counter)
    for row in train:
        prior[row["group_id"]][row["gold_sense"]] += 1
    baseline_hits = sum(
        1
        for r in held
        if prior.get(r["group_id"])
        and prior[r["group_id"]].most_common(1)[0][0] == r["gold_sense"]
    )

    is_multi = np.array([r["form"] in multi_forms for r in held])
    is_minority = np.array(
        [r["form"] in multi_forms and r["gold_sense"] != dominant.get(r["form"]) for r in held]
    )
    base_multi = sum(
        1
        for r in held
        if r["form"] in multi_forms and dominant[r["form"]] == r["gold_sense"]
    )

    report: dict[str, Any] = {
        "holdout_by": args.holdout_by,
        "held_out_rows": len(held),
        "held_out_forms": len(senses),
        "multi_sense_forms": len(multi_forms),
        "multi_sense_rows": int(is_multi.sum()),
        "minority_rows": int(is_minority.sum()),
        "majority_baseline_all": baseline_hits / len(held),
        "majority_baseline_multi_sense": base_multi / max(int(is_multi.sum()), 1),
        "models": {},
    }
    print(f"held-out {len(held):,} rows / {len(senses):,} forms  "
          f"(multi-sense forms {len(multi_forms):,}, rows {int(is_multi.sum()):,}, "
          f"minority rows {int(is_minority.sum()):,})")
    print(f"majority baseline: all={report['majority_baseline_all']:.4f}  "
          f"multi-sense={report['majority_baseline_multi_sense']:.4f}  minority=0.0000\n")

    targets = [("base", args.base)] + ([("tuned", args.tuned)] if args.tuned else [])
    for name, directory in targets:
        margins, correct = score(directory, held, glosses)
        entry: dict[str, Any] = {
            "path": str(directory),
            "accuracy_all": float(correct.mean()),
            "accuracy_multi_sense": float(correct[is_multi].mean()) if is_multi.any() else 0.0,
            "minority_recall": float(correct[is_minority].mean()) if is_minority.any() else 0.0,
            "curve_multi_sense": [],
        }
        mm, mc = margins[is_multi], correct[is_multi]
        for t in THRESHOLDS:
            keep = mm >= t
            if not keep.sum():
                continue
            lo, hi = wilson(int(mc[keep].sum()), int(keep.sum()))
            entry["curve_multi_sense"].append(
                {
                    "margin": float(t),
                    "coverage": float(keep.mean()),
                    "rows": int(keep.sum()),
                    "precision": float(mc[keep].mean()),
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
        report["models"][name] = entry
        print(f"[{name}] accuracy all={entry['accuracy_all']:.4f}  "
              f"multi-sense={entry['accuracy_multi_sense']:.4f}  "
              f"minority-recall={entry['minority_recall']:.4f}")
        print(f"{'margin':>8}{'cov':>8}{'rows':>7}{'precision':>11}{'95% CI':>20}")
        for c in entry["curve_multi_sense"]:
            flag = "  <= certified >=98%" if c["ci_low"] >= 0.98 else ""
            print(f"{c['margin']:>8.2f}{c['coverage']:>8.3f}{c['rows']:>7}"
                  f"{c['precision']:>11.4f}   [{c['ci_low']:.3f},{c['ci_high']:.3f}]{flag}")
        print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
