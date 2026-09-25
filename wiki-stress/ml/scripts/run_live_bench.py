"""Score the running API against lang-uk's benchmark, through HTTP.

The harness in `run_langukbench.py` reimplements the tier order in Python, which
makes it the right place to try a change but the wrong place to believe one: it
can agree with itself while the product does something else. This posts the
benchmark's own sentences to the live service and hands the answers to the
maintainers' evaluator, so what is measured is the deployment.

Callers run concurrently because the service is meant to be used that way, and
because a tier that quietly degrades under load — as the morphology pool once
did — will not show up in a serial run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ACUTE = "́"


def to_plus(text: str) -> str:
    """Rewrite our combining acute as the benchmark's '+'.

    Both mark the vowel they follow, so this is a substitution, not a move. It
    goes through NFD to reach the accent and back through NFC because NFD also
    splits `й` into `и` plus a breve, which must be put back or every word
    carrying one is scored wrong.
    """
    decomposed = unicodedata.normalize("NFD", text).replace(ACUTE, "+")
    return unicodedata.normalize("NFC", decomposed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1/stress")
    parser.add_argument("--callers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--label", default="live API")
    parser.add_argument("--save", type=Path,
                        help="write gold and produced sentences side by side")
    args = parser.parse_args()

    client = httpx.Client(timeout=args.timeout)

    tokens: dict[str, list] = {}

    def stressify(text: str) -> str:
        response = client.post(args.url, json={"text": text})
        response.raise_for_status()
        body = response.json()
        # Per-token status and margin, kept so precision can be measured per
        # tier: a mode that only marks what it is sure of needs to know which
        # tiers are sure.
        tokens[text] = body.get("tokens", [])
        return to_plus(body["text"])

    # The evaluator's import expects to run from its own directory, so the
    # chdir below invalidates every relative path; resolve ours first.
    if args.save:
        args.save = args.save.resolve()

    # The evaluator calls the callable one sentence at a time, so concurrency
    # has to come from prefetching every sentence before handing it the cache.
    sys.path.insert(0, str(args.benchmark))
    os.chdir(args.benchmark / "lexical_stress_benchmark" / "examples")
    import pandas as pd
    from lexical_stress_benchmark import evaluate_stressification

    frame = pd.read_csv(args.benchmark / "lexical_stress_benchmark" / "data"
                        / "lexical_stress_dataset.csv")
    plain = [str(s).replace("+", "") for s in frame["StressedSentence"]]
    print(f"sentences: {len(plain):,}  callers: {args.callers}", flush=True)

    with ThreadPoolExecutor(max_workers=args.callers) as pool:
        answers = list(pool.map(stressify, plain))
    cache = dict(zip(plain, answers, strict=True))
    print("all sentences answered", flush=True)

    if args.save:
        gold = [str(s) for s in frame["StressedSentence"]]
        args.save.write_text(json.dumps(
            [{"plain": p_, "gold": g, "got": cache[p_], "tokens": tokens.get(p_, [])}
             for p_, g in zip(plain, gold, strict=True)],
            ensure_ascii=False), encoding="utf-8")
        print(f"side by side -> {args.save}", flush=True)

    scores = evaluate_stressification(lambda text: cache[text])
    names = ["sentence_accuracy", "word_accuracy", "heteronym_accuracy",
             "unambiguous_accuracy", "macro_f1_heteronyms"]
    print(f"\n{args.label}")
    for name, value in zip(names, scores.values(), strict=True):
        print(f"  {name:<24}{value * 100:>8.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
