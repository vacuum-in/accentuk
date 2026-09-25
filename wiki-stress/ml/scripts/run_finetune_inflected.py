"""Fine-tune the cross-encoder on inflected forms and re-measure calibration.

Zero-shot on inflected surfaces the model never trained on measured 0.8928
accuracy — only 3.1pp below the lemma baseline, so the semantics transfer — but
its precision/coverage curve plateaus near 0.96, well short of 98%. High-margin
predictions on unseen surfaces are not trustworthy, which is a calibration
failure rather than a knowledge failure.

This asks the narrow question that decides whether the full 7M-token annotation
run is worth commissioning: **does fine-tuning on inflected forms restore the
calibration?**

The split is by *form*, never by sentence. Splitting sentences would put other
inflections of the same form on both sides and measure memorisation; held-out
forms measure what actually matters — generalisation to inflections the model
has not been trained on.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ukstress_ml import ambiguity, crossencoder

SILVER = Path("output/ml/silver_inflected.json")
INVENTORY = Path("output/ml/ambiguous_forms_inflected.jsonl")
BASE = Path("models/v3-xenc/checkpoint")
THRESHOLDS = (0.0, 1.2482, 3.0, 5.0, 6.2532, 7.0, 9.0, 11.0, 13.0)


def load_rows(silver: Path = SILVER,
              inventory: Path = INVENTORY) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    forms = {f.form: f for f in ambiguity.load(inventory)}
    glosses = {
        c.sense_id: {"definition": c.definition, "stressed": c.stressed}
        for f in forms.values()
        for c in f.candidates
    }
    rows, unknown = [], 0
    for row in json.loads(silver.read_text(encoding="utf-8")):
        # A corpus and an inventory are separate artifacts and do not have to
        # agree: `silver_mined_v10` carries 2,303 forms that
        # `ambiguous_forms_inflected` does not, and pairing them used to raise
        # KeyError on the first one. A row with no candidate list has nothing to
        # choose between, so skip it and report how many — silently dropping a
        # third of a corpus would be worse than the crash.
        form = forms.get(row["form"])
        if form is None:
            unknown += 1
            continue
        rows.append(
            {
                **row,
                "sense_id": row["gold_sense"],
                "candidates": [
                    {"sense_id": c.sense_id, "signature": c.signature} for c in form.candidates
                ],
            }
        )
    if unknown:
        print(f"skipped {unknown:,} rows whose form is absent from {inventory.name}",
              flush=True)
    return rows, glosses


def split_rows(
    rows: list[dict[str, Any]], seed: int, holdout: float, by: str
) -> tuple[list, list]:
    """Hold out whole forms or whole groups — never individual sentences.

    `form` measures generalisation to an unseen inflection of a group the model
    was trained on. `group` is the harder question the fine-tune left open:
    whether a group with no training sentence at all can still be resolved from
    its gloss. That distinction decides how much generated data the 601 starved
    groups actually need.
    """
    key = "form" if by == "form" else "group_id"
    values = sorted({r[key] for r in rows}, key=str)
    random.Random(seed).shuffle(values)
    held = set(values[: max(1, int(len(values) * holdout))])
    return [r for r in rows if r[key] not in held], [r for r in rows if r[key] in held]


def drop_one_sided(rows: list[dict[str, Any]], forms: dict[str, Any],
                   min_share: float = 0.0) -> tuple[list[dict[str, Any]], set[str]]:
    """Remove training rows for forms whose data is too one-sided to learn from.

    One-sided data is worse than no data. Measured on the gold set:

        обід      0 training rows            1.000
        плачу     0 training rows            1.000
        брати     0 training rows            1.000
        правило   260 rows, 0% minority      0.640
        поділ     35 rows, 9% minority       0.500

    With no rows the cross-encoder has nothing to lean on and scores the
    `(sentence, gloss)` pair on its merits, which is what it was built to do.
    With a few hundred rows that only ever show one sense it learns the prior
    instead and stops reading the context — the majority baseline dressed up as
    a model.

    So a form the corpus cannot yet balance is better left out of training than
    left in skewed. This is a stopgap: the real fix is generating the missing
    sense, and once a form has both it should train on both.

    `min_share` extends this past the all-or-nothing case. Dropping only forms
    with a *completely* missing sense fixed `правило` (0.640 -> 1.000) but left
    `поділ` at 0.500, because its minority sense is present at 9% — enough to
    survive the filter and still teach the prior. Measured on the gold set, a
    form scores ~1.00 once its weakest sense reaches roughly 20% and degrades in
    proportion below that, so the share is the honest cut, not the presence.
    """
    by_form: dict[str, Counter[str]] = {}
    for row in rows:
        by_form.setdefault(row["form"], Counter())[row["gold_sense"]] += 1
    dropped: set[str] = set()
    for form, counts in by_form.items():
        if form not in forms:
            continue
        senses = {c.sense_id for c in forms[form].candidates}
        total = sum(counts.values())
        weakest = min(counts.get(sense, 0) for sense in senses)
        if weakest == 0 or (min_share > 0 and total and weakest / total < min_share):
            dropped.add(form)
    return [row for row in rows if row["form"] not in dropped], dropped


def balance(rows: list[dict[str, Any]], ratio: float, seed: int) -> list[dict[str, Any]]:
    """Cap the dominant sense of each form at `ratio`x its rarest sibling.

    Trained on the natural skew, the model improves aggregate accuracy by
    leaning harder on the prior — measured here as minority-sense recall
    *falling* 0.6376 -> 0.6131 while overall accuracy rose. Minority recall is
    the load-bearing number: the majority baseline scores exactly 0 on it, so it
    is the only metric that shows context being read rather than a prior being
    recalled.

    Balancing is applied to **training rows only**. The held-out split keeps its
    natural distribution, because that skew is what production looks like and
    flattening it would flatter the evaluation (`RESULTS.md` §4).
    """
    if ratio <= 0:
        return rows
    rng = random.Random(seed)
    by_form: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        by_form.setdefault(row["form"], {}).setdefault(row["gold_sense"], []).append(row)
    kept: list[dict[str, Any]] = []
    for senses in by_form.values():
        floor = min(len(items) for items in senses.values())
        cap = max(1, int(floor * ratio))
        for items in senses.values():
            if len(items) > cap:
                items = rng.sample(items, cap)
            kept.extend(items)
    rng.shuffle(kept)
    return kept


def curve(model: Any, tokenizer: Any, rows: list[dict[str, Any]], glosses: Any) -> dict[str, Any]:
    loader = DataLoader(
        crossencoder.PairDataset(rows, glosses),
        batch_size=16,
        shuffle=False,
        collate_fn=crossencoder.make_collate(tokenizer, 192),
    )
    device = next(model.parameters()).device
    margins: list[float] = []
    correct: list[bool] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            encoded = {k: v.to(device) for k, v in batch["encoded"].items()}
            logits = model(**encoded).logits.squeeze(-1).float().cpu().numpy()
            offset = 0
            for width, gold in zip(batch["spans"], batch["golds"].tolist(), strict=True):
                if width > 1 and gold >= 0:
                    scores = logits[offset : offset + width]
                    order = np.argsort(-scores)
                    margins.append(float(scores[order[0]] - scores[order[1]]))
                    correct.append(int(order[0]) == gold)
                offset += width
    margin = np.array(margins)
    hit = np.array(correct)
    return {
        "rows": int(hit.size),
        "accuracy": float(hit.mean()) if hit.size else 0.0,
        "curve": [
            {
                "margin": float(t),
                "coverage": float((margin >= t).mean()),
                "precision": float(hit[margin >= t].mean()),
            }
            for t in THRESHOLDS
            if (margin >= t).sum()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--silver", type=Path, default=SILVER)
    parser.add_argument("--inventory", type=Path, default=INVENTORY,
                        help="candidate senses and glosses; the glossed inventory "
                             "carries 13,565 forms against the inflected one's 12,651")
    parser.add_argument("--base", default=str(BASE),
                        help="hub id or local checkpoint to start from")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-rows", type=int, default=16)
    parser.add_argument("--amp", choices=("bfloat16", "float16"), default=None,
                        help="autocast dtype; needed to fit a large encoder on 8 GB")
    parser.add_argument("--accumulate", type=int, default=1,
                        help="optimiser steps every N batches, to keep the "
                             "effective batch size when batch-rows is reduced")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--exemplars", type=int, default=0, metavar="K",
                        help="score against K labelled usages of each sense instead of "
                             "its dictionary definition; 0 keeps the definition. "
                             "Exemplars are drawn from the training split only.")
    parser.add_argument("--holdout", type=float, default=0.25)
    parser.add_argument("--holdout-by", choices=("form", "group"), default="form")
    parser.add_argument("--min-sense-share", type=float, default=0.0,
                        help="with --drop-one-sided, also drop forms whose weakest "
                             "sense is below this share of the form's rows")
    parser.add_argument("--drop-one-sided", action="store_true",
                        help="exclude forms whose training rows cover only one sense; "
                             "measured, they score worse than forms with no data at all")
    parser.add_argument("--balance-ratio", type=float, default=0.0,
                        help="cap dominant sense at N x rarest, training rows only")
    parser.add_argument("--output", type=Path, default=Path("output/ml/models/v4-inflected"))
    args = parser.parse_args()

    rows, glosses = load_rows(args.silver, args.inventory)
    train_rows, held_rows = split_rows(rows, args.seed, args.holdout, args.holdout_by)
    print(
        f"holdout_by={args.holdout_by}  rows={len(rows):,}  "
        f"train={len(train_rows):,} ({len({r['form'] for r in train_rows})} forms, "
        f"{len({r['group_id'] for r in train_rows})} groups)  "
        f"held-out={len(held_rows):,} ({len({r['form'] for r in held_rows})} forms, "
        f"{len({r['group_id'] for r in held_rows})} groups)",
        flush=True,
    )

    if args.drop_one_sided:
        forms_by_key = {f.form: f for f in ambiguity.load(args.inventory)}
        before = len(train_rows)
        train_rows, dropped = drop_one_sided(train_rows, forms_by_key, args.min_sense_share)
        print(f"dropped {before - len(train_rows):,} one-sided training rows "
              f"across {len(dropped):,} forms (they train the prior, not the context)",
              flush=True)
    if args.balance_ratio:
        before_n = len(train_rows)
        train_rows = balance(train_rows, args.balance_ratio, args.seed)
        print(f"balanced train rows {before_n:,} -> {len(train_rows):,} "
              f"(cap {args.balance_ratio}x rarest sense per form)", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    # A hub checkpoint has no trained classification head, so scoring it before
    # fine-tuning would measure a random projection, not a baseline.
    trained_head = Path(args.base).exists()
    if trained_head:
        before_model = AutoModelForSequenceClassification.from_pretrained(args.base).cuda()
        before = curve(before_model, tokenizer, held_rows, glosses)
        print(f"BEFORE fine-tune: accuracy={before['accuracy']:.4f}", flush=True)
        del before_model
        torch.cuda.empty_cache()
    else:
        before = {"rows": 0, "accuracy": 0.0, "curve": []}
        print(f"base {args.base} has no trained head; skipping before-measurement", flush=True)

    config = crossencoder.CrossEncoderConfig(
        base_model=args.base,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        batch_rows=args.batch_rows,
        amp_dtype=args.amp,
        accumulate=args.accumulate,
    )
    exemplars = None
    if args.exemplars:
        # Built from the training split alone. Drawing them from all rows would
        # put held-out sentences into the evidence for their own senses, and the
        # split is by form precisely to measure generalisation.
        exemplars = crossencoder.build_exemplars(
            train_rows, per_sense=args.exemplars, seed=args.seed
        )
        senses = sum(1 for v in exemplars.values() if v)
        print(f"exemplars: {senses:,} senses carry up to {args.exemplars} usages",
              flush=True)

    run = crossencoder.train(
        train_rows,
        held_rows,
        glosses,
        args.output,
        config=config,
        corpus_version="inflected-pilot" if not args.exemplars
                       else f"inflected-pilot+exemplars{args.exemplars}",
        exemplars=exemplars,
    )
    print(f"trained: best_epoch={run.best_epoch} dev_group_macro={run.best_dev_group_macro:.4f}",
          flush=True)

    after_model = AutoModelForSequenceClassification.from_pretrained(
        args.output / "checkpoint"
    ).cuda()
    after = curve(after_model, tokenizer, held_rows, glosses)

    report = {
        "held_out_forms": len({r["form"] for r in held_rows}),
        "held_out_rows": len(held_rows),
        "before": before,
        "after": after,
        "config": {"epochs": args.epochs, "lr": args.lr, "holdout": args.holdout,
                   "holdout_by": args.holdout_by,
                   "balance_ratio": args.balance_ratio},
        "held_out_groups": len({r["group_id"] for r in held_rows}),
    }
    (args.output / "calibration.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\n{'margin':>9}{'cov before':>12}{'prec before':>13}{'cov after':>11}{'prec after':>12}")
    after_by = {c["margin"]: c for c in after["curve"]}
    for entry in before["curve"]:
        other = after_by.get(entry["margin"])
        if other is None:
            continue
        print(
            f"{entry['margin']:>9.4f}{entry['coverage']:>12.4f}{entry['precision']:>13.4f}"
            f"{other['coverage']:>11.4f}{other['precision']:>12.4f}"
        )
    print(f"\naccuracy {before['accuracy']:.4f} -> {after['accuracy']:.4f}")
    print(f"written to {args.output / 'calibration.json'}")


if __name__ == "__main__":
    main()
