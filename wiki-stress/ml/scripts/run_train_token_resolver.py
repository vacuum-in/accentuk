"""Train a token classifier to pick the stress of an ambiguous word in context.

The cross-encoder this replaces scores (sentence, gloss) similarity, which
cannot separate a grammatical heteronym: `ко́леса` and `коле́са` are one lemma
with one definition. Measured on lang-uk, grammatical heteronyms are 289 of the
492 forms and carry 70% of the heteronym loss.

This model never sees a gloss. It reads the sentence, takes the hidden state of
the ambiguous span, and scores each signature the lexicon offers for that form
— a softmax over the candidates, so it cannot answer "both" or "neither", and
cannot invent a reading. Every ambiguous form is coverable, glossed or not.

Three test conditions are reported separately because they answer different
questions: a form seen in training but a new sentence, a form held out
entirely, and the lang-uk benchmark. The second is the one that says whether
this generalises rather than memorises.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
from pathlib import Path

import numpy as np


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def wilson(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if not total:
        return 0.0, 0.0
    p = hits / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("output/ml/corpus/tok-v1"))
    parser.add_argument("--base", default="ukr-models/xlm-roberta-base-uk")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, default=Path("models/tok-v1"))
    parser.add_argument("--balance-readings", type=float, default=0.0,
                        help="weight each ambiguous row by (rows of its form / "
                             "rows of its form with the same reading) ** this "
                             "power, so the minority reading of a form carries "
                             "more of the loss. 0 leaves the corpus's own "
                             "frequencies, which the classifier otherwise learns "
                             "as a prior instead of reading the context; 1 makes "
                             "every reading present carry equal weight; 0.5 is "
                             "the compromise to try first")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    import transformers
    from torch import nn

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base)
    encoder = transformers.AutoModel.from_pretrained(args.base).to(args.device)
    width = encoder.config.hidden_size
    # One vector per candidate signature, scored against the span's hidden
    # state. Signatures are small integers, so the table is tiny and shared
    # across forms — the model learns "the second vowel, in this context",
    # not a per-form lookup.
    head = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                         nn.Linear(width, args.max_candidates)).to(args.device)

    train = load(args.corpus / "train.jsonl")
    dev = load(args.corpus / "dev.jsonl")
    test = load(args.corpus / "test.jsonl")
    if args.balance_readings > 0:
        per_form: collections.Counter = collections.Counter()
        per_reading: collections.Counter = collections.Counter()
        for row in train:
            if len(row["candidates"]) > 1:
                per_form[row["form"].lower()] += 1
                per_reading[(row["form"].lower(), row["gold"])] += 1
        for row in train:
            if len(row["candidates"]) > 1:
                form = row["form"].lower()
                row["weight"] = (per_form[form] / per_reading[(form, row["gold"])]) ** args.balance_readings
        print(f"balance {args.balance_readings}: mean weight "
              f"{sum(r.get('weight', 1.0) for r in train) / len(train):.2f}")
    print(f"train {len(train):,} | dev {len(dev):,} | test {len(test):,}")

    def encode(batch: list[dict]):
        texts = [r["sentence"] for r in batch]
        got = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=args.max_length, return_offsets_mapping=True)
        offsets = got.pop("offset_mapping")
        spans = []
        for index, row in enumerate(batch):
            start, end = row["start"], row["end"]
            hit = [t for t, (a, b) in enumerate(offsets[index].tolist())
                   if a < end and b > start and b > a]
            spans.append(hit or [0])
        masks = torch.zeros(len(batch), offsets.shape[1], dtype=torch.bool)
        for index, hit in enumerate(spans):
            masks[index, hit] = True
        allowed = torch.zeros(len(batch), args.max_candidates, dtype=torch.bool)
        targets = torch.zeros(len(batch), dtype=torch.long)
        for index, row in enumerate(batch):
            for candidate in row["candidates"]:
                if not candidate.isdigit():
                    continue   # "0|1": two accents on one compound, no class for it
                value = int(candidate)
                if value < args.max_candidates:
                    allowed[index, value] = True
            if not allowed[index].any():
                allowed[index, 0] = True
            targets[index] = int(row["gold"]) if row["gold"].isdigit() else 0
        return ({k: v.to(args.device) for k, v in got.items()},
                masks.to(args.device), allowed.to(args.device), targets.to(args.device))

    def forward(inputs, masks, allowed):
        hidden = encoder(**inputs).last_hidden_state
        pooled = (hidden * masks.unsqueeze(-1)).sum(1) / masks.sum(1, keepdim=True).clamp(min=1)
        return head(pooled).masked_fill(~allowed, float("-inf"))

    parameters = list(encoder.parameters()) + list(head.parameters())
    optimiser = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    steps = max(1, len(train) // args.batch) * args.epochs
    schedule = transformers.get_linear_schedule_with_warmup(
        optimiser, int(0.06 * steps), steps)

    def evaluate(rows: list[dict], only_ambiguous: bool = True):
        encoder.eval()
        head.eval()
        marks = []
        with torch.no_grad():
            for start in range(0, len(rows), 64):
                chunk = [r for r in rows[start:start + 64]
                         if not only_ambiguous or len(r["candidates"]) > 1]
                if not chunk:
                    continue
                inputs, masks, allowed, targets = encode(chunk)
                picks = forward(inputs, masks, allowed).argmax(-1)
                marks.extend((r, int(p == t)) for r, p, t in
                             zip(chunk, picks.tolist(), targets.tolist(), strict=True))
        return marks

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        head.train()
        random.shuffle(train)
        total = seen = 0.0
        for start in range(0, len(train), args.batch):
            chunk = train[start:start + args.batch]
            inputs, masks, allowed, targets = encode(chunk)
            weights = torch.tensor([float(r.get("weight", 1.0)) for r in chunk], device=args.device)
            losses = nn.functional.cross_entropy(forward(inputs, masks, allowed), targets,
                                                 reduction="none")
            loss = (losses * weights).sum() / weights.sum()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimiser.step()
            schedule.step()
            total += float(loss.detach())
            seen += 1
            if seen % 200 == 0:
                print(f"  epoch {epoch} step {int(seen)}/{len(train)//args.batch} "
                      f"loss {total/seen:.4f}", flush=True)
        marks = evaluate(dev)
        score = sum(h for _, h in marks) / max(len(marks), 1)
        print(f"epoch {epoch}: loss {total/seen:.4f}  dev {100*score:.2f}% "
              f"({len(marks):,} ambiguous)", flush=True)
        if score >= best:
            best = score
            encoder.save_pretrained(args.out)
            tokenizer.save_pretrained(args.out)
            torch.save(head.state_dict(), args.out / "head.pt")
            # Not config.json: save_pretrained just wrote the encoder's own
            # there, and overwriting it leaves a model that cannot be loaded.
            (args.out / "resolver.json").write_text(
                json.dumps({"base": args.base, "max_candidates": args.max_candidates,
                            "dev": score}, ensure_ascii=False), encoding="utf-8")

    marks = evaluate(test)
    groups = collections.defaultdict(list)
    # The bar the model has to clear is not chance, it is answering
    # candidates[0] every time — tok-v1 scored 66.3% on unseen forms against
    # 66.4% for that rule, which is how it was found to have learned nothing.
    baseline = collections.defaultdict(list)
    for row, hit in marks:
        first = int(row["gold"] == row["candidates"][0])
        names = ["all", "unseen form" if row.get("unseen_form") else "seen form"]
        if "pipeline_agreed" in row:
            names.append("pipeline agreed" if row["pipeline_agreed"] else "pipeline disagreed")
        for name in names:
            groups[name].append(hit)
            baseline[name].append(first)
    print(f"\n{'condition':<22}{'rows':>8}{'accuracy':>10}{'95% CI':>18}{'candidates[0]':>16}")
    for name, hits in groups.items():
        low, high = wilson(sum(hits), len(hits))
        print(f"  {name:<20}{len(hits):>8,}{100*sum(hits)/len(hits):>9.1f}%"
              f"{f'[{100*low:.1f}, {100*high:.1f}]':>18}"
              f"{100*sum(baseline[name])/len(hits):>15.1f}%")
    (args.out / "report.json").write_text(json.dumps(
        {name: {"rows": len(h), "accuracy": sum(h)/len(h),
                "first_candidate": sum(baseline[name])/len(h)} for name, h in groups.items()},
        ensure_ascii=False, indent=2), encoding="utf-8")
    # The serving-side coverage: how often each ambiguous form was seen in
    # training decides whether the resolver may answer for it (min_seen).
    seen = collections.Counter(r["form"].lower() for r in train if len(r["candidates"]) > 1)
    (args.out / "coverage.json").write_text(json.dumps(
        {"model_version": args.out.name, "min_seen": 100, "forms": dict(sorted(seen.items()))},
        ensure_ascii=False), encoding="utf-8")
    print(f"coverage: {len(seen):,} forms, {sum(1 for n in seen.values() if n >= 100):,} at 100+")
    print(f"\nmodel -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
