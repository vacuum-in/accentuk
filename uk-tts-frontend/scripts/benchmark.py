"""Score the stress stage on lang-uk's lexical stress benchmark.

The metrics are not reimplemented here. The benchmark's own evaluator is
imported from a checkout of `lang-uk/ukrainian-tts-preprocessing` and handed a
stressify function, exactly as its bundled examples do, so a number produced
here is comparable with any other number produced by that harness.

    python scripts/benchmark.py --benchmark ../ukrainian-tts-preprocessing
    python scripts/benchmark.py --benchmark ... --system baseline

`--system pipeline` calls the running stress API over HTTP, so what is measured
is the deployed service rather than a harness-local reconstruction of it.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import sys
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

ACUTE = "́"


def to_plus_notation(text: str) -> str:
    """Rewrite `нови́й` as `нови+й`.

    The benchmark marks stress with a `+` following the stressed vowel, which is
    where the combining acute sits too, so this is a substitution and not a
    move. Decompose first: the acute is a separate code point only in NFD, and
    `й` and `ї` carry a combining mark of their own that must survive.
    """

    decomposed = unicodedata.normalize("NFD", unicodedata.normalize("NFC", text))
    return unicodedata.normalize("NFC", decomposed.replace(ACUTE, "+"))


def pipeline_stressify(api: str, timeout: float, on_ambiguity: str) -> Callable[[str], str]:
    import httpx

    client = httpx.Client(base_url=api.rstrip("/"), timeout=timeout)

    def stressify(sentence: str) -> str:
        response = client.post(
            "/v1/stress", json={"text": sentence, "on_ambiguity": on_ambiguity}
        )
        response.raise_for_status()
        return to_plus_notation(response.json()["text"])

    return stressify


def baseline_stressify() -> Callable[[str], str]:
    """`ukrainian-word-stress`, configured as the benchmark's own example does."""

    from ukrainian_word_stress import OnAmbiguity, Stressifier

    return Stressifier(stress_symbol="+", on_ambiguity=OnAmbiguity.First)


class _Sentence:
    """A restored per-sentence result, shaped like the evaluator's own.

    `DatasetAccuracy.update_with_sentence` reads these attributes and nothing
    else, so a baseline saved to disk can be re-aggregated — and therefore
    bootstrapped — without re-running the system that produced it.
    """

    __slots__ = ("total_words", "correctly_stressified_words", "total_heteronyms",
                 "correctly_stressified_heteronyms", "total_unambiguous_words",
                 "correctly_stressified_unambiguous", "heteronyms_dictionary")

    def __init__(self, counts: dict) -> None:
        self.total_words = counts["w"]
        self.correctly_stressified_words = counts["wc"]
        self.total_heteronyms = counts["h"]
        self.correctly_stressified_heteronyms = counts["hc"]
        self.total_unambiguous_words = counts["u"]
        self.correctly_stressified_unambiguous = counts["uc"]
        self.heteronyms_dictionary = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        for word, variants in counts["hd"].items():
            for variant, predictions in variants.items():
                self.heteronyms_dictionary[word][variant].extend(predictions)

    def is_sentence_correct(self) -> bool:
        return self.total_words == self.correctly_stressified_words


def _sentence_counts(accuracy: Any) -> dict:
    return {
        "w": accuracy.total_words,
        "wc": accuracy.correctly_stressified_words,
        "h": accuracy.total_heteronyms,
        "hc": accuracy.correctly_stressified_heteronyms,
        "u": accuracy.total_unambiguous_words,
        "uc": accuracy.correctly_stressified_unambiguous,
        "hd": {w: dict(v) for w, v in accuracy.heteronyms_dictionary.items()},
    }


def _restore_sentence(counts: dict) -> _Sentence:
    return _Sentence(counts)


def _load_evaluator(benchmark: Path):
    """Import the benchmark's own evaluator and chdir to where its data lives."""
    root = benchmark.resolve()
    if not (root / "lexical_stress_benchmark" / "data" / "lexical_stress_dataset.csv").is_file():
        raise SystemExit(f"not a ukrainian-tts-preprocessing checkout: {root}")
    sys.path.insert(0, str(root))
    os.chdir(root / "lexical_stress_benchmark" / "examples")
    from lexical_stress_benchmark.benchmark.accuracy import DatasetAccuracy
    from lexical_stress_benchmark.benchmark.sentence_stress_evaluator import (
        evaluate_stress_sentence_level,
        evaluate_stress_word_level,
        is_word_heteronym,
    )

    return (DatasetAccuracy, evaluate_stress_sentence_level,
            evaluate_stress_word_level, is_word_heteronym)


def _aggregate(DatasetAccuracy, per_sentence: list[Any]) -> dict[str, float]:
    """Roll per-sentence results up exactly as the upstream harness does."""
    metrics = DatasetAccuracy()
    for accuracy in per_sentence:
        metrics.update_with_sentence(accuracy)
    return metrics.compute_averages(len(per_sentence))


def run(benchmark: Path, stressify: Callable[[str], str]) -> dict[str, Any]:
    """Score a system, keeping per-sentence and per-token detail for pairing.

    The metrics come from the benchmark's own `evaluate_stress_sentence_level`
    and `DatasetAccuracy`, which is what its aggregate entry point calls, so a
    number here is the number that entry point would print. What is added is
    keeping the intermediate results rather than discarding them.
    """
    import pandas as pd

    previous = Path.cwd()
    try:
        (DatasetAccuracy, evaluate_sentence, evaluate_word,
         is_heteronym) = _load_evaluator(benchmark)
        dataset = pd.read_csv(
            Path("..") / "data" / "lexical_stress_dataset.csv"
        )["StressedSentence"]

        started = time.perf_counter()
        per_sentence, tokens, predictions, skipped = [], [], [], 0
        for index, gold in enumerate(dataset):
            predicted = stressify(gold.replace("+", ""))
            predictions.append({"index": index, "gold": gold, "predicted": predicted})
            accuracy = evaluate_sentence(
                gold, predicted, raise_on_mismatch=False, ignore_mismatch=True
            )
            if accuracy is None:
                skipped += 1
                continue
            per_sentence.append(accuracy)
            # Token-level correctness, by the same rule the sentence evaluator
            # applies, so a paired test compares like with like.
            gold_words = gold.strip().lower().split()
            got_words = predicted.strip().lower().split()
            if len(gold_words) != len(got_words):
                continue
            for position, (g, o) in enumerate(zip(gold_words, got_words, strict=True)):
                plain = g.replace("+", "")
                if not is_heteronym(plain):
                    continue
                tokens.append({
                    "key": f"{index}:{position}",
                    "form": plain,
                    "correct": bool(evaluate_word(g, o)),
                })

        metrics = _aggregate(DatasetAccuracy, per_sentence)
        metrics["seconds"] = round(time.perf_counter() - started, 1)
        metrics["sentences_scored"] = len(per_sentence)
        metrics["sentences_skipped"] = skipped
        metrics["heteronym_tokens"] = len(tokens)
        return {"metrics": metrics, "per_sentence": per_sentence,
                "tokens": tokens, "predictions": predictions,
                "_aggregate": lambda rows: _aggregate(DatasetAccuracy, rows)}
    finally:
        os.chdir(previous)


def mcnemar(baseline: list[dict], candidate: list[dict]) -> dict[str, Any]:
    """Exact paired test over the tokens both systems scored.

    Only discordant tokens carry information: b, fixed by the candidate, and c,
    broken by it. Everything else cancels, which is why a paired test resolves a
    difference an unpaired interval cannot.
    """
    left = {row["key"]: row["correct"] for row in baseline}
    right = {row["key"]: row["correct"] for row in candidate}
    shared = left.keys() & right.keys()
    fixed = sorted(k for k in shared if right[k] and not left[k])
    broken = sorted(k for k in shared if left[k] and not right[k])
    b, c = len(fixed), len(broken)

    # Two-sided exact binomial on the discordant pairs, p = 0.5 under the null.
    if b + c == 0:
        p_value = 1.0
    else:
        n = b + c
        tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n)
        p_value = min(1.0, 2 * tail)
    forms = collections.Counter(
        row["form"] for row in candidate if row["key"] in set(fixed) | set(broken)
    )
    return {
        "paired_tokens": len(shared), "fixed": b, "broken": c,
        "net_tokens": b - c,
        "net_points": round(100 * (b - c) / max(len(shared), 1), 3),
        "p_value": round(p_value, 5),
        "changed_forms": forms.most_common(12),
    }


def bootstrap(baseline: dict, candidate: dict, metrics: tuple[str, ...],
              draws: int = 2000, seed: int = 20260831) -> dict[str, dict[str, float]]:
    """Percentile intervals for every metric at once, resampling *sentences*.

    Sentences are the independent unit: tokens inside one are correlated, so
    resampling tokens would report an interval narrower than the truth. All
    metrics share each resample, both because it is five times cheaper and
    because it keeps the intervals mutually consistent.
    """
    rng = random.Random(seed)
    left, right = baseline["per_sentence"], candidate["per_sentence"]
    size = min(len(left), len(right))
    if size == 0:
        return {name: {"low": 0.0, "high": 0.0, "draws": 0} for name in metrics}

    differences: dict[str, list[float]] = {name: [] for name in metrics}
    aggregate = candidate["_aggregate"]
    for _ in range(draws):
        sample = [rng.randrange(size) for _ in range(size)]
        a = aggregate([left[i] for i in sample])
        b = aggregate([right[i] for i in sample])
        for name in metrics:
            differences[name].append(100 * (b[name] - a[name]))

    out = {}
    for name, values in differences.items():
        values.sort()
        out[name] = {
            "low": round(values[int(0.025 * draws)], 3),
            "high": round(values[int(0.975 * draws)], 3),
            "draws": draws,
        }
    return out


METRICS = (
    "sentence_accuracy",
    "word_accuracy",
    "heteronym_accuracy",
    "unambiguous_accuracy",
    "macro_average_f1_across_heteronyms",
)


def _stressify_for(args) -> Callable[[str], str]:
    if args.system == "baseline":
        return baseline_stressify()
    return pipeline_stressify(args.api, args.timeout, args.on_ambiguity)


def _report(result: dict) -> None:
    metrics = result["metrics"]
    for name in METRICS:
        print(f"{name:38} {metrics[name] * 100:6.2f}%")
    print(f"{'heteronym tokens':38} {metrics['heteronym_tokens']:6}")
    print(f"{'sentences scored':38} {metrics['sentences_scored']:6}"
          f"  (skipped {metrics['sentences_skipped']})")
    print(f"{'seconds':38} {metrics['seconds']:6.1f}")


def _report_comparison(base: dict, cand: dict, draws: int) -> None:
    print("\n" + "=" * 74)
    print("PAIRED COMPARISON   baseline -> candidate")
    print("=" * 74)
    print(f"{'metric':38} {'baseline':>9} {'candidate':>10} {'95% CI of change':>22}")
    intervals = bootstrap(base, cand, METRICS, draws=draws)
    for name in METRICS:
        a, b = base["metrics"][name] * 100, cand["metrics"][name] * 100
        ci = intervals[name]
        verdict = "  noise" if ci["low"] <= 0 <= ci["high"] else ""
        print(f"{name:38} {a:8.2f}% {b:9.2f}%   "
              f"[{ci['low']:+6.2f}, {ci['high']:+6.2f}]{verdict}")

    test = mcnemar(base["tokens"], cand["tokens"])
    print(f"\nMcNemar over {test['paired_tokens']} paired heteronym tokens")
    print(f"  fixed by candidate : {test['fixed']}")
    print(f"  broken by candidate: {test['broken']}")
    print(f"  net                : {test['net_tokens']:+d} tokens "
          f"({test['net_points']:+.2f} points)")
    print(f"  exact p-value      : {test['p_value']}")
    if test["p_value"] > 0.05:
        print("  -> not distinguishable from no change at the 5% level")
    if test["changed_forms"]:
        print("  forms that moved   : "
              + ", ".join(f"{w} x{n}" for w, n in test["changed_forms"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True,
                        help="checkout of lang-uk/ukrainian-tts-preprocessing")
    parser.add_argument("--system", choices=["pipeline", "baseline"], default="pipeline")
    parser.add_argument("--api", default="http://127.0.0.1:8080", help="stress API base URL")
    parser.add_argument("--on-ambiguity", choices=["default", "preserve"], default="default")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--out", type=Path, help="write the metrics as JSON")
    parser.add_argument("--save", type=Path,
                        help="write per-token results here, for use as a later --baseline")
    parser.add_argument("--baseline", type=Path,
                        help="a file written by --save; report the paired change against it")
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="resamples for the confidence interval")
    args = parser.parse_args(argv)

    result = run(args.benchmark, _stressify_for(args))
    print()
    _report(result)

    if args.save:
        args.save.write_text(json.dumps({
            "system": args.system,
            "metrics": result["metrics"],
            "tokens": result["tokens"],
            "per_sentence": [_sentence_counts(s) for s in result["per_sentence"]],
        }, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nper-token results -> {args.save}")

    if args.baseline:
        saved = json.loads(args.baseline.read_text(encoding="utf-8"))
        base = {
            "metrics": saved["metrics"],
            "tokens": saved["tokens"],
            "per_sentence": [_restore_sentence(c) for c in saved["per_sentence"]],
            "_aggregate": result["_aggregate"],
        }
        _report_comparison(base, result, args.bootstrap)

    if args.out:
        args.out.write_text(
            json.dumps({"system": args.system, **result["metrics"]},
                       indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
