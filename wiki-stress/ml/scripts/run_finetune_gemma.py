"""Fine-tune Gemma 3 270M as the homograph cross-encoder.

The incumbent is XLM-R base (278M), an encoder that reads `(sentence, gloss)`
bidirectionally. Gemma 3 270M is a *decoder*: causal attention means the
sentence tokens cannot attend forward to the gloss, so the pair is only seen
jointly at the final position. Sequence classification on a decoder therefore
pools the last non-pad token rather than a [CLS] summary.

That is a real architectural handicap for pair scoring, and the model is also
slightly smaller. It may lose to 0.9717. The measurement is the point.

Two Gemma-specific details this handles that the XLM-R path did not need:

* **Vocabulary.** 262k tokens at hidden 640 is ~168M embedding parameters —
  62% of the model — for a task whose inputs are Ukrainian. Nothing here trims
  it, but it explains why a "270M" model trains no faster than the 278M encoder.
* **Padding.** Gemma ships no pad token, and last-token pooling on a
  right-padded batch would read padding instead of content. Pad is set to EOS
  and padding forced to the *left*, so the final position is always real text.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ukstress_ml import ambiguity
from ukstress_ml.corpus import CLOSE_MARK, OPEN_MARK

THRESHOLDS = (0.0, 1.25, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0)


def wilson(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = hits / total
    den = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return ((centre - margin) / den, (centre + margin) / den)


class PairData(Dataset):
    """One row -> one (marked sentence, gloss) pair per candidate sense."""

    def __init__(self, rows: list[dict[str, Any]], glosses: dict[str, str]) -> None:
        self.rows = rows
        self.glosses = glosses

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        sentence = (
            row["sentence"][: row["start"]]
            + OPEN_MARK
            + row["sentence"][row["start"] : row["end"]]
            + CLOSE_MARK
            + row["sentence"][row["end"] :]
        )
        pairs = [
            (c["sense_id"], self.glosses.get(c["sense_id"], ""))
            for c in sorted(row["candidates"], key=lambda c: c["sense_id"])
        ]
        gold = next((i for i, (s, _) in enumerate(pairs) if s == row["gold_sense"]), -1)
        return {"sentence": sentence, "pairs": pairs, "gold": gold, "form": row["form"]}


def make_collate(tokenizer: Any, max_length: int):
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        left: list[str] = []
        right: list[str] = []
        spans: list[int] = []
        golds: list[int] = []
        for item in batch:
            spans.append(len(item["pairs"]))
            golds.append(item["gold"])
            for _, gloss in item["pairs"]:
                left.append(item["sentence"])
                right.append(gloss)
        encoded = tokenizer(
            left, right, truncation=True, max_length=max_length, padding=True,
            return_tensors="pt",
        )
        return {
            "encoded": encoded,
            "spans": spans,
            "golds": torch.tensor(golds, dtype=torch.long),
            "forms": [item["form"] for item in batch],
        }

    return collate


def listwise_loss(logits: torch.Tensor, spans: list[int], golds: torch.Tensor) -> torch.Tensor:
    """Softmax over each row's candidates — the same objective as the encoder."""
    losses = []
    offset = 0
    for position, width in enumerate(spans):
        chunk = logits[offset : offset + width]
        gold = int(golds[position])
        if width > 1 and gold >= 0:
            losses.append(
                torch.nn.functional.cross_entropy(
                    chunk.unsqueeze(0), torch.tensor([gold], device=chunk.device)
                )
            )
        offset += width
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def score(model: Any, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    margins: list[float] = []
    correct: list[bool] = []
    with torch.no_grad():
        for batch in loader:
            encoded = {k: v.to(device) for k, v in batch["encoded"].items()}
            logits = model(**encoded).logits.squeeze(-1).float().cpu().numpy()
            offset = 0
            for width, gold in zip(batch["spans"], batch["golds"].tolist(), strict=True):
                if width > 1 and gold >= 0:
                    chunk = logits[offset : offset + width]
                    order = np.argsort(-chunk)
                    margins.append(float(chunk[order[0]] - chunk[order[1]]))
                    correct.append(int(order[0]) == gold)
                offset += width
    return np.array(margins), np.array(correct)


def balance(rows: list[dict[str, Any]], ratio: float, seed: int) -> list[dict[str, Any]]:
    if ratio <= 0:
        return rows
    rng = random.Random(seed)
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(row["form"], {}).setdefault(row["gold_sense"], []).append(row)
    kept: list[dict[str, Any]] = []
    for senses in grouped.values():
        cap = max(1, int(min(len(v) for v in senses.values()) * ratio))
        for items in senses.values():
            kept.extend(rng.sample(items, cap) if len(items) > cap else items)
    rng.shuffle(kept)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/gemma-3-270m")
    parser.add_argument("--silver", type=Path,
                        default=Path("output/ml/silver_inflected_clean.json"))
    parser.add_argument("--inventory", type=Path,
                        default=Path("output/ml/ambiguous_forms_inflected.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("output/ml/models/gemma3-270m"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-rows", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--holdout", type=float, default=0.25)
    parser.add_argument("--balance-ratio", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    forms = {f.form: f for f in ambiguity.load(args.inventory)}
    glosses = {c.sense_id: c.definition for f in forms.values() for c in f.candidates}
    rows = []
    for row in json.loads(args.silver.read_text(encoding="utf-8")):
        form = forms[row["form"]]
        rows.append({**row, "candidates": [
            {"sense_id": c.sense_id, "signature": c.signature} for c in form.candidates]})

    keys = sorted({r["form"] for r in rows})
    random.Random(args.seed).shuffle(keys)
    held_keys = set(keys[: max(1, int(len(keys) * args.holdout))])
    train_rows = [r for r in rows if r["form"] not in held_keys]
    held_rows = [r for r in rows if r["form"] in held_keys]
    train_rows = balance(train_rows, args.balance_ratio, args.seed)

    sense_by_form: dict[str, set[str]] = defaultdict(set)
    for row in held_rows:
        sense_by_form[row["form"]].add(row["gold_sense"])
    multi = {f for f, s in sense_by_form.items() if len(s) > 1}

    print(f"model={args.model_id}  train={len(train_rows):,}  held={len(held_rows):,} "
          f"({len(held_keys)} forms, {len(multi)} multi-sense)", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder sequence classification pools the final token; right padding would
    # make that a pad token for every short sequence in the batch.
    tokenizer.padding_side = "left"

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, num_labels=1, token=token, dtype=torch.bfloat16
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.gradient_checkpointing_enable()
    total = sum(p.numel() for p in model.parameters())
    print(f"parameters={total/1e6:.0f}M  device={device}  dtype={model.dtype}", flush=True)

    collate = make_collate(tokenizer, args.max_length)
    train_loader = DataLoader(PairData(train_rows, glosses), batch_size=args.batch_rows,
                              shuffle=True, collate_fn=collate)
    held_loader = DataLoader(PairData(held_rows, glosses), batch_size=args.batch_rows,
                             shuffle=False, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = max(1, len(train_loader) * args.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    history = []
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            encoded = {k: v.to(device) for k, v in batch["encoded"].items()}
            logits = model(**encoded).logits.squeeze(-1).float()
            loss = listwise_loss(logits, batch["spans"], batch["golds"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running += float(loss)
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {running/step:.4f} ({(time.time()-started)/60:.1f} min)", flush=True)
        margins, correct = score(model, held_loader, device)
        is_multi = np.array([r["form"] in multi for r in held_rows if True])[: correct.size]
        acc = float(correct.mean()) if correct.size else 0.0
        history.append({"epoch": epoch, "loss": running / max(len(train_loader), 1),
                        "held_accuracy": acc})
        print(f"epoch {epoch}  loss {running/len(train_loader):.4f}  held acc {acc:.4f}  "
              f"({(time.time()-started)/60:.1f} min)", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output / "checkpoint")
    tokenizer.save_pretrained(args.output / "checkpoint")

    # Final report restricted to multi-sense forms, the only rows where context
    # decides anything (single-sense forms measure recall of a prior).
    margins, correct = score(model, held_loader, device)
    flags: list[bool] = []
    for row in held_rows:
        width = len(forms[row["form"]].candidates)
        if width > 1:
            flags.append(row["form"] in multi)
    is_multi = np.array(flags[: correct.size])
    mm, mc = margins[is_multi], correct[is_multi]
    curve = []
    for t in THRESHOLDS:
        keep = mm >= t
        if not keep.sum():
            continue
        lo, hi = wilson(int(mc[keep].sum()), int(keep.sum()))
        curve.append({"margin": t, "coverage": float(keep.mean()), "rows": int(keep.sum()),
                      "precision": float(mc[keep].mean()), "ci_low": lo, "ci_high": hi})
    report = {"model_id": args.model_id, "parameters": total, "history": history,
              "multi_sense_rows": int(is_multi.sum()),
              "accuracy_multi_sense": float(mc.mean()) if mc.size else 0.0,
              "curve_multi_sense": curve,
              "minutes": (time.time() - started) / 60}
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\nmulti-sense rows {int(is_multi.sum()):,}  accuracy {report['accuracy_multi_sense']:.4f}")
    print(f"{'margin':>7}{'cov':>8}{'rows':>7}{'precision':>11}{'95% CI':>20}")
    for c in curve:
        print(f"{c['margin']:>7.2f}{c['coverage']:>8.3f}{c['rows']:>7}{c['precision']:>11.4f}"
              f"   [{c['ci_low']:.3f},{c['ci_high']:.3f}]")
    print(f"\nwritten to {args.output/'report.json'}")


if __name__ == "__main__":
    main()
