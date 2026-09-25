"""A stress placer that reads the word's letters, not a tokenizer's pieces.

Stress in a word the lexicon lacks is decided mostly by its ending and its
shape — the suffix table is right 72% of the time knowing nothing else — and
a subword tokenizer hides exactly that. This model sees the characters: a
bidirectional GRU over the word (with the previous and next word as context,
separated by a marker), and a softmax over the positions of its vowels. It
trains in minutes on the placer corpus and is scored on the same two sets as
the transformer placer, so the comparison is direct.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

VOWELS = "аеєиіїоуюяй"
SEP, PAD, UNK = "|", "\x00", "\x01"


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def neighbours(sentence: str, start: int, end: int) -> tuple[str, str]:
    left = sentence[:start].split()
    right = sentence[end:].split()
    return (left[-1].lower() if left else ""), (right[0].lower() if right else "")


def encode(sentence: str, start: int, end: int, form: str, vocab: dict, max_len: int = 48):
    before, after = neighbours(sentence, start, end)
    text = f"{before[-12:]}{SEP}{form}{SEP}{after[:12]}"
    offset = len(before[-12:]) + 1        # where the form starts in `text`
    ids = [vocab.get(c, vocab[UNK]) for c in text[:max_len]]
    # Positions the stress may take: real vowels only, never й.
    vowel_at = [offset + i for i, c in enumerate(form) if c in VOWELS and c != "й" and offset + i < max_len]
    return ids, vowel_at


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("output/ml/corpus/placer-v1"))
    parser.add_argument("--out", type=Path, default=Path("models/placer-char"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--review", type=Path, default=Path("output/ml/live_review20.json"))
    parser.add_argument("--sweeps", type=Path,
                        default=Path("/home/devops/audiotostress/artifacts/audio_runs"))
    parser.add_argument("--suffix-table", type=Path, default=Path("output/ml/suffix_table.json"),
                        help="the served suffix fallback, for the ensemble report")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--eval-only", action="store_true",
                        help="load --out/placer.pt and only run the reports")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    from torch import nn

    torch.manual_seed(17)
    random.seed(17)
    train, dev, test = (load(args.corpus / f"{n}.jsonl") for n in ("train", "dev", "test"))
    chars = collections.Counter(c for r in train for c in r["form"] + r["sentence"][:0])
    for r in train:
        b, a = neighbours(r["sentence"], r["start"], r["end"])
        chars.update(b + a)
    vocab = {PAD: 0, UNK: 1, SEP: 2}
    for c, n in chars.most_common():
        if n >= 3 and c not in vocab:
            vocab[c] = len(vocab)
    print(f"train {len(train):,} | dev {len(dev):,} | test {len(test):,} | {len(vocab)} chars")

    class Placer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(len(vocab), 64, padding_idx=0)
            self.rnn = nn.GRU(64, args.hidden, num_layers=args.layers, batch_first=True,
                              bidirectional=True, dropout=0.2)
            self.score = nn.Sequential(nn.Linear(2 * args.hidden, args.hidden), nn.GELU(),
                                       nn.Linear(args.hidden, 1))

        def forward(self, ids, vowel_pos, vowel_mask):
            h, _ = self.rnn(self.embed(ids))                         # B×T×2H
            gathered = torch.gather(h, 1, vowel_pos.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
            logits = self.score(gathered).squeeze(-1)                # B×V
            return logits.masked_fill(~vowel_mask, float("-inf"))

    if args.eval_only:
        saved = torch.load(args.out / "placer.pt")
        vocab = saved["vocab"]
        args.hidden, args.layers = saved["hidden"], saved.get("layers", 2)
    model = Placer().to(args.device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)

    def batchify(rows):
        encoded = [encode(r["sentence"], r["start"], r["end"], r["form"].lower(), vocab) for r in rows]
        T = max(len(ids) for ids, _ in encoded)
        V = max(len(v) for _, v in encoded)
        ids = torch.zeros(len(rows), T, dtype=torch.long)
        pos = torch.zeros(len(rows), V, dtype=torch.long)
        mask = torch.zeros(len(rows), V, dtype=torch.bool)
        gold = torch.zeros(len(rows), dtype=torch.long)
        for i, ((seq, vowels), r) in enumerate(zip(encoded, rows)):
            ids[i, :len(seq)] = torch.tensor(seq)
            pos[i, :len(vowels)] = torch.tensor(vowels)
            mask[i, :len(vowels)] = True
            gold[i] = r["candidates"].index(r["gold"]) if r["gold"] in r["candidates"] else 0
        return ids.to(args.device), pos.to(args.device), mask.to(args.device), gold.to(args.device)

    def accuracy(rows):
        model.eval()
        hits = 0
        with torch.no_grad():
            for s in range(0, len(rows), 512):
                ids, pos, mask, gold = batchify(rows[s:s + 512])
                hits += int((model(ids, pos, mask).argmax(-1) == gold).sum())
        return hits / max(len(rows), 1)

    best, started = 0.0, time.time()
    for epoch in range(1, 0 if args.eval_only else args.epochs + 1):
        model.train()
        random.shuffle(train)
        total = 0.0
        for s in range(0, len(train), args.batch):
            ids, pos, mask, gold = batchify(train[s:s + args.batch])
            loss = nn.functional.cross_entropy(model(ids, pos, mask), gold)
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += float(loss)
        score = accuracy(dev)
        print(f"epoch {epoch}: loss {total / max(1, len(train) // args.batch):.4f}  "
              f"dev {score:.2%}  {(time.time() - started) / 60:.1f} min", flush=True)
        if score >= best:
            best = score
            torch.save({"state": model.state_dict(), "vocab": vocab, "hidden": args.hidden,
                        "layers": args.layers}, args.out / "placer.pt")
    model.load_state_dict(torch.load(args.out / "placer.pt")["state"])
    print(f"test (held-out forms): {accuracy(test):.2%}")

    # The same two external sets as run_eval_placer.py, plus the ensemble:
    # the suffix table answers when its longest matching ending is at least
    # `--table-min` characters (the table is a memory of 1.9M forms and is
    # right when it has really seen the ending), the placer otherwise.
    import unicodedata
    table = json.loads(args.suffix_table.read_text(encoding="utf-8")) if args.suffix_table.exists() else {}

    def table_answer(form: str):
        decomposed = unicodedata.normalize("NFD", form)
        vowels = sum(1 for c in form if c in VOWELS)
        for length in range(min(13, len(decomposed)), 0, -1):
            position = table.get(decomposed[-length:])
            if position is not None and position < vowels:
                return str(vowels - 1 - position), length
        return None, 0

    from run_eval_placer import external_sets
    for name, rows, golds, pipeline in external_sets(args.review, args.sweeps):
        model.eval()
        picks_all, probs_all = [], []
        with torch.no_grad():
            for s in range(0, len(rows), 512):
                chunk = rows[s:s + 512]
                ids, pos, mask, _ = batchify([{**r, "gold": "0"} for r in chunk])
                logits = model(ids, pos, mask)
                probs = torch.softmax(logits, -1)
                picks_all += logits.argmax(-1).tolist()
                probs_all += probs.max(-1).values.tolist()
        placer = [c["candidates"][p] if p < len(c["candidates"]) else None for p, c in zip(picks_all, rows)]
        hits = sum(a == g for a, g in zip(placer, golds))
        print(f"{name}: {len(rows):,} tokens  pipeline {pipeline:.1%}  char placer {hits / len(rows):.1%}")
        for min_len in (4, 5, 6, 7):
            for min_prob in (0.0, 0.6, 0.8):
                ens = 0
                for a, pr, c, g in zip(placer, probs_all, rows, golds):
                    t, length = table_answer(c["form"])
                    use_table = t is not None and length >= min_len and pr < max(min_prob, 1e-9) if min_prob else (t is not None and length >= min_len)
                    ens += (t if use_table else a) == g
                print(f"    table if ending ≥{min_len} chars" + (f" and placer <{min_prob}" if min_prob else "") +
                      f": {ens / len(rows):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
