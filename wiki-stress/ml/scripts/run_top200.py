"""RUAccent's yardstick on Ukrainian: the 200 most frequent ambiguous forms.

RUAccent reports 0.9637 on the top-200 homographs of its own audio-labelled
test (confidence > 0.99). The same construction here: the 200 most frequent
ambiguous forms of the Common Voice verify sweeps, gold from the audio ranker
at >= 0.99, answers from the live API. Lived in /tmp until 2026-09-23; now
it also saves every token's answer, so two configurations can be compared
form by form (`--save`).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import unicodedata
import urllib.request
from pathlib import Path

VOWELS = "аеєиіїоуюяй"
ACUTE = "́"
SWEEPS = ("verify_60k.jsonl", "verify_30k.jsonl", "verify_after.jsonl")


def signature(text: str) -> str | None:
    ordinal = -1
    for ch in unicodedata.normalize("NFD", text):
        if ch.lower() in VOWELS:
            ordinal += 1
        if ch == ACUTE:
            return str(ordinal)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweeps", type=Path, default=Path("/home/devops/audiotostress/artifacts/audio_runs"))
    parser.add_argument("--api", default="http://127.0.0.1:8080")
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--min-gold", type=float, default=0.99)
    parser.add_argument("--save", type=Path, help="write every scored token here as JSON lines")
    parser.add_argument("--combiner", choices=("on", "off"),
                        help="ask for the learned combiner per request; unset, the service default")
    args = parser.parse_args()

    seen = {}
    for name in SWEEPS:
        for line in (args.sweeps / name).read_text(encoding="utf-8").splitlines():
            if '"form"' not in line:
                continue
            r = json.loads(line)
            if r.get("start") is None or not r.get("sentence"):
                continue
            candidates = [c for c in r.get("candidates") or [] if c.isdigit()]
            if len(set(candidates)) < 2 or r.get("confidence", 0) < args.min_gold:
                continue
            seen[(r.get("clip"), r["form"], r["start"])] = r
    rows = list(seen.values())
    freq = collections.Counter(r["form"].lower() for r in rows)
    top = {f for f, _ in freq.most_common(args.top)}
    subset = [r for r in rows if r["form"].lower() in top]
    sentences = sorted({r["sentence"] for r in subset})

    def ask(text: str):
        request = urllib.request.Request(f"{args.api}/v1/stress", data=json.dumps({"text": text, **({"combiner": args.combiner == "on"} if args.combiner else {})}).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return text, json.load(response)["tokens"]

    answers = {}
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for text, tokens in pool.map(ask, sentences):
            answers[text] = tokens

    hits = total = 0
    by_tier: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    saved = []
    for r in subset:
        token = next((t for t in answers.get(r["sentence"], []) if t["start"] == r["start"]), None)
        if not token:
            continue
        got = signature(token.get("output_text", ""))
        ok = got == str(r["audio"])
        total += 1
        hits += ok
        by_tier[token["status"]][0] += 1
        by_tier[token["status"]][1] += ok
        saved.append({"clip": r.get("clip"), "form": r["form"].lower(), "start": r["start"],
                      "gold": str(r["audio"]), "got": got, "status": token["status"], "ok": ok})
    print(f"top-{args.top} ambiguous forms: {total:,} tokens ({len(subset):,} in set), live accuracy {hits / total:.2%}")
    for status, (n, right) in sorted(by_tier.items(), key=lambda kv: -kv[1][0]):
        print(f"  {status:<20}{n:>6,}  {right / n:.1%}")
    if args.save:
        args.save.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in saved) + "\n", encoding="utf-8")
        print(f"-> {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
