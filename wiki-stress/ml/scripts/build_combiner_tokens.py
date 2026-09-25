"""The labelled ambiguous tokens a combiner learns from and is judged on.

Two kinds of gold, kept apart because they are wrong in different ways:

  cv       Common Voice verify sweeps: what a speaker said, read by the audio
           ranker at confidence >= 0.99. Modern text, frequent forms; the
           ranker is ~99% on majority readings and weaker on rare ones.
  languk   lang-uk's benchmark: a person's marks on sentences built to force
           the rare reading. Adversarial; the independent check.

Candidates are taken from the live API now, not from sweep time, so every
token carries the served candidate list (in served order) and the live
pipeline's own answer and tier — the baseline a combiner has to beat.
Splits are by sentence hash: cv 60/20/20 train/dev/test, languk 50/50
dev/test (lang-uk is never trained on unless asked).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import unicodedata
import urllib.request
from pathlib import Path

VOWELS = "аеєиіїоуюя"
ACUTE = "́"
SWEEPS = ("verify_60k.jsonl", "verify_30k.jsonl", "verify_after.jsonl")


def signature_of(text: str) -> str | None:
    ordinal = -1
    for ch in unicodedata.normalize("NFD", text):
        if ch.lower() in VOWELS:
            ordinal += 1
        elif ch == ACUTE:
            return str(ordinal)
    return None


def bucket(sentence: str) -> float:
    return int(hashlib.md5(sentence.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def languk_gold(plain: str, gold: str) -> dict[int, int]:
    """Character offset in `plain` of every vowel the '+' notation stresses."""
    stressed, position = {}, 0
    for ch in gold:
        if ch == "+":
            stressed[position - 1] = 1
            continue
        position += 1
    return stressed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweeps", type=Path, default=Path("/home/devops/audiotostress/artifacts/audio_runs"))
    parser.add_argument("--languk", type=Path, default=Path("output/ml/live_review29.json"),
                        help="a saved live bench: it carries lang-uk's sentences and gold")
    parser.add_argument("--api", default="http://127.0.0.1:8080")
    parser.add_argument("--min-gold", type=float, default=0.99)
    parser.add_argument("--out", type=Path, default=Path("output/ml/combiner/tokens.jsonl"))
    args = parser.parse_args()

    # Gold per (sentence, start): CV from the sweeps, lang-uk from the '+' marks.
    gold: dict[tuple[str, int], dict] = {}
    for name in SWEEPS:
        path = args.sweeps / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if '"form"' not in line:
                continue
            r = json.loads(line)
            if r.get("start") is None or not r.get("sentence") or r.get("confidence", 0) < args.min_gold:
                continue
            gold[(r["sentence"], r["start"])] = {"source": "cv", "gold": str(r["audio"]),
                                                 "clip": r.get("clip")}
    cv_sentences = {s for s, _ in gold}
    languk_rows = json.loads(args.languk.read_text(encoding="utf-8"))
    languk_marks = {r["plain"]: languk_gold(r["plain"], r["gold"]) for r in languk_rows}
    print(f"{len(gold):,} cv tokens over {len(cv_sentences):,} sentences; "
          f"{len(languk_marks):,} lang-uk sentences", flush=True)

    def ask(text: str):
        request = urllib.request.Request(f"{args.api}/v1/stress", data=json.dumps({"text": text}).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            return text, json.load(response)["tokens"]

    sentences = sorted(cv_sentences | set(languk_marks))
    answers: dict[str, list] = {}
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for text, tokens in pool.map(ask, sentences):
            answers[text] = tokens
    print(f"{len(answers):,} sentences answered by the live API", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with args.out.open("w", encoding="utf-8") as handle:
        for sentence, tokens in answers.items():
            for token in tokens:
                candidates = [c for c in token.get("candidates") or [] if c.isdigit()]
                if len(set(candidates)) < 2:
                    continue
                label = None
                if (sentence, token["start"]) in gold:
                    label = gold[(sentence, token["start"])]
                elif sentence in languk_marks:
                    marks = languk_marks[sentence]
                    word = sentence[token["start"]:token["end"]]
                    ordinal, found = -1, None
                    for offset, ch in enumerate(word):
                        if ch.lower() in VOWELS:
                            ordinal += 1
                            if marks.get(token["start"] + offset):
                                found = str(ordinal)
                                break
                    if found is not None:
                        label = {"source": "languk", "gold": found}
                if label is None or label["gold"] not in candidates:
                    continue
                b = bucket(sentence)
                if label["source"] == "cv":
                    split = "train" if b < 0.6 else "dev" if b < 0.8 else "test"
                else:
                    split = "dev" if b < 0.5 else "test"
                row = {"sentence": sentence, "start": token["start"], "end": token["end"],
                       "text": token["text"], "form": unicodedata.normalize("NFC", token["text"].lower()),
                       "candidates": candidates, "pipeline": signature_of(token.get("output_text", "")),
                       "status": token.get("status"), "split": split, **label}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                key = f"{label['source']}:{split}"
                counts[key] = counts.get(key, 0) + 1
    print(json.dumps(counts, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
