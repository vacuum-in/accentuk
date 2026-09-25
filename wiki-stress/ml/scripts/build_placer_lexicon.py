"""Add the lexicon's own single-reading forms to a placer corpus.

The book corpus teaches how narrators stress words the lexicon lacks —
derivations, dialect, names from novels. The lexicon teaches the shape of
Ukrainian stress at scale: 1.9M forms, no context. A placer trained on both
sees fifteen times the words, and the proper nouns and loanwords the
benchmark's fallback set is made of are mostly lexicon shapes.

Lexicon rows carry the word alone as the sentence. They go to train only: the
dev and test sets stay the book's held-out forms, and the external sets stay
what they are.
"""

from __future__ import annotations

import argparse
import json
import random
import unicodedata
from pathlib import Path

VOWELS = "аеєиіїоуюяй"


def candidates_of(form: str) -> list[str]:
    out, seen = [], -1
    for ch in form:
        if ch in VOWELS:
            seen += 1
            if ch != "й":
                out.append(str(seen))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="a placer corpus to extend")
    parser.add_argument("--lexicon", type=Path, required=True, help="TSV: form_normalized, signature")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    held = set()
    for name in ("dev", "test"):
        for line in (args.corpus / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                held.add(json.loads(line)["form"].lower())

    rows = []
    for line in args.lexicon.read_text(encoding="utf-8").splitlines():
        if "|" not in line:
            continue
        form, signature = line.split("|", 1)
        form = unicodedata.normalize("NFC", form.strip().lower())
        signature = signature.strip()
        if not signature.isdigit() or form in held or not form.isalpha():
            continue
        cands = candidates_of(form)
        if len(cands) < 2 or signature not in cands:
            continue
        rows.append({"sentence": form, "start": 0, "end": len(form), "form": form,
                     "candidates": cands, "gold": signature, "source": "lexicon", "weight": 1.0})
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.sample]

    train = (args.corpus / "train.jsonl").read_text(encoding="utf-8")
    (args.out / "train.jsonl").write_text(
        train + "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    for name in ("dev", "test"):
        (args.out / f"{name}.jsonl").write_text(
            (args.corpus / f"{name}.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    print(f"{len(rows):,} lexicon rows added to {train.count(chr(10)):,} book rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
