"""LoRA on a causal LLM for the fixed-choice stress decision.

Same prompt as the zero-shot picker, same decision (softmax over the option
letters' logits at the answer position), but the adapter learns it from the
classifier corpus: cross-entropy over the letters only, the option order
drawn at random per row so no letter is favoured. Rows with one candidate
(the negatives) are skipped — there is nothing to choose.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ukstress_ml.llm_picker import LETTERS, build_prompt, letter_ids, load_glosses  # noqa: E402


def rows_of(path: Path, limit: int, seed: int) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if len(r["candidates"]) >= 2 and str(r["gold"]) in r["candidates"]:
                rows.append(r)
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--corpus", type=Path, default=Path("output/ml/corpus/tok-v5"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=0, help="train rows to use (0 = all)")
    parser.add_argument("--dev-rows", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--glosses", type=Path)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpointing", action="store_true",
                        help="gradient checkpointing: a third more compute for a third "
                             "of the activation memory; a 1.7B model at batch 32 fits "
                             "a 32 GB card without it")
    args = parser.parse_args()

    import torch
    import transformers
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).to(args.device)
    if args.checkpointing:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    ids = torch.tensor(letter_ids(tokenizer), device=args.device)
    glosses = load_glosses(args.glosses) if args.glosses else {}

    train = rows_of(args.corpus / "train.jsonl", args.rows, args.seed)
    dev = rows_of(args.corpus / "dev.jsonl", args.dev_rows, args.seed)
    print(f"{len(train):,} train rows, {len(dev):,} dev rows, base {args.base}", flush=True)
    rng = random.Random(args.seed)

    def batch_tensors(rows: list[dict], shuffle_order: bool):
        prompts, targets, widths = [], [], []
        for r in rows:
            cands = list(r["candidates"][:len(LETTERS)])
            if shuffle_order:
                rng.shuffle(cands)
            prompts.append(build_prompt(tokenizer, glosses, r["sentence"], r["start"], r["end"], r["form"], cands))
            targets.append(cands.index(str(r["gold"])))
            widths.append(len(cands))
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=args.max_length).to(args.device)
        return enc, torch.tensor(targets, device=args.device), widths

    def restricted(logits, widths):
        # letters beyond a row's candidate count are masked out of its softmax
        picked = logits[:, ids]
        mask = torch.arange(len(LETTERS), device=args.device)[None, :] >= torch.tensor(widths, device=args.device)[:, None]
        return picked.masked_fill(mask, float("-inf"))

    def evaluate() -> float:
        model.eval()
        hits = 0
        with torch.inference_mode():
            for at in range(0, len(dev), args.batch * 2):
                enc, target, widths = batch_tensors(dev[at:at + args.batch * 2], shuffle_order=False)
                out = restricted(model(**enc, logits_to_keep=1).logits[:, -1, :].float(), widths)
                hits += int((out.argmax(-1) == target).sum())
        model.train()
        return hits / max(len(dev), 1)

    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps_total = args.epochs * (len(train) // (args.batch * args.accum))
    # 200 steps of warmup, not 50: at 2e-4 a three-epoch run collapsed to
    # chance (loss ln 2) for its first thirteen thousand steps before escaping
    warmup = args.warmup
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda s: min(1.0, (s + 1) / warmup) * max(0.05, 1 - s / max(steps_total, 1)))
    print(f"dev before training: {evaluate():.1%}", flush=True)
    model.train()
    step, started, best = 0, time.time(), 0.0
    args.out.mkdir(parents=True, exist_ok=True)
    def length_bucketed(rows: list[dict], block: int = 4096) -> list[dict]:
        # shuffle, then sort within blocks by sentence length: batches of like
        # length carry little padding, which is most of the speed at batch 32+
        rng.shuffle(rows)
        out = []
        for at in range(0, len(rows), block):
            out += sorted(rows[at:at + block], key=lambda r: len(r["sentence"]))
        return out

    for epoch in range(args.epochs):
        train = length_bucketed(train)
        running = 0.0
        for i in range(0, len(train), args.batch):
            enc, target, widths = batch_tensors(train[i:i + args.batch], shuffle_order=True)
            # the LM head only at the answer position: the full [batch, seq, vocab]
            # tensor is 150k wide and was the whole memory bill
            out = restricted(model(**enc, logits_to_keep=1).logits[:, -1, :].float(), widths)
            loss = torch.nn.functional.cross_entropy(out, target) / args.accum
            loss.backward()
            running += float(loss) * args.accum
            if (i // args.batch + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step(); schedule.step(); optimiser.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    seen = i + args.batch
                    print(f"  epoch {epoch + 1} step {step}/{steps_total} loss {running / (50 * args.accum):.4f} "
                          f"{seen / (time.time() - started):.1f} rows/s", flush=True)
                    running = 0.0
                if step % args.eval_every == 0:
                    acc = evaluate()
                    print(f"  dev {acc:.1%}", flush=True)
                    if acc > best:
                        best = acc
                        model.save_pretrained(args.out)
        acc = evaluate()
        print(f"epoch {epoch + 1}: dev {acc:.1%}", flush=True)
        if acc >= best:
            best = acc
            model.save_pretrained(args.out)
    (args.out / "train.json").write_text(json.dumps({"base": args.base, "rows": len(train), "dev": best,
                                                     "rank": args.rank, "lr": args.lr}, indent=2), encoding="utf-8")
    print(f"best dev {best:.1%} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
