"""The heteronym error budget, split by ambiguity class and by deciding tier.

Pairing note: API tokens are matched to whitespace words by character offset,
never by position. The API skips standalone punctuation, so one dash
desynchronises a positional zip for the rest of the sentence — which silently
scrambled the tier attribution in an earlier version of this analysis.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--api", default="http://127.0.0.1:8080")
    parser.add_argument("--spacy",
                        default="/home/devops/wiki-stress/models/uk_core_news_sm")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, "/home/devops/wiki-stress/ml/src")
    import httpx
    from benchmark import to_plus_notation
    from ukstress_ml.morphology import SpacyMorphologyTier
    from ukstress_ml.triage import classify_readings

    data = args.benchmark / "lexical_stress_benchmark" / "data"
    rows = [r["StressedSentence"] for r in
            csv.DictReader((data / "lexical_stress_dataset.csv").open(encoding="utf-8"))]
    heteronyms = {r["Heteronym"] for r in
                  csv.DictReader((data / "heteronyms_list.csv").open(encoding="utf-8"))}

    tier = SpacyMorphologyTier(model_path=args.spacy)
    cache: dict[str, str] = {}

    def klass(form: str) -> str:
        if form not in cache:
            readings = tier.readings(form)
            cache[form] = classify_readings(readings)[0] if readings else "not_in_trie"
        return cache[form]

    api = httpx.Client(base_url=args.api, timeout=180)
    totals: collections.Counter[str] = collections.Counter()
    errors: collections.Counter[str] = collections.Counter()
    by_status: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter)
    status_total: collections.Counter[str] = collections.Counter()
    status_wrong: collections.Counter[str] = collections.Counter()

    for gold_sentence in rows:
        plain = gold_sentence.replace("+", "")
        served = api.post("/v1/stress",
                          json={"text": plain, "include_tokens": True}).json()
        gold_words = gold_sentence.lower().split()
        got_words = to_plus_notation(served["text"]).lower().split()
        if len(gold_words) != len(got_words):
            continue
        word_at: dict[int, int] = {}
        cursor = 0
        for position, word in enumerate(plain.split()):
            cursor = plain.index(word, cursor)
            for offset in range(cursor, cursor + len(word)):
                word_at[offset] = position
            cursor += len(word)
        for token in served["tokens"]:
            position = word_at.get(token["start"])
            if position is None or position >= len(gold_words):
                continue
            gold, got = gold_words[position], got_words[position]
            form = gold.replace("+", "")
            if form not in heteronyms:
                continue
            group = klass(form)
            totals[group] += 1
            status_total[token["status"]] += 1
            if gold != got:
                errors[group] += 1
                by_status[group][token["status"]] += 1
                status_wrong[token["status"]] += 1

    total, wrong = sum(totals.values()), sum(errors.values())
    report = {
        "tokens": total, "errors": wrong,
        "accuracy": round(100 * (1 - wrong / total), 2),
        "by_class": {c: {"tokens": totals[c], "errors": errors[c],
                         "accuracy": round(100 * (1 - errors[c] / totals[c]), 1),
                         "deciding_tier": by_status[c].most_common(4)}
                     for c, _ in totals.most_common()},
        "by_tier": {s: {"tokens": n, "errors": status_wrong[s],
                        "accuracy": round(100 * (1 - status_wrong[s] / n), 1)}
                    for s, n in status_total.most_common()},
    }
    print(f"{total} heteronym tokens, {wrong} errors, {report['accuracy']}%\n")
    print(f"{'class':16} {'tok':>5} {'err':>5} {'acc':>8}  decided by")
    for c, row in report["by_class"].items():
        decided = ", ".join(f"{k} {v}" for k, v in row["deciding_tier"])
        print(f"{c:16} {row['tokens']:5} {row['errors']:5} {row['accuracy']:7.1f}%  {decided}")
    print(f"\n{'tier':22} {'tok':>5} {'err':>5} {'acc':>8}")
    for s, row in report["by_tier"].items():
        print(f"{s:22} {row['tokens']:5} {row['errors']:5} {row['accuracy']:7.1f}%")
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
