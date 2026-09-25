"""Score collected gold decisions under a serving policy.

`run_gold_eval.py` records what each tier *said*; this decides what gets served
and what that scores. Splitting the two makes the expensive part (a GPU pass
over the set) reusable and the cheap part (a threshold, a fallback rule) a
parameter, so a policy question is answered by re-reading a file.

Two policy axes:

`--threshold`
    The margin the cross-encoder must clear to be believed. The shipped
    manifest value was chosen on silver precision at low coverage; the gold set
    is the first evidence about it that is not self-referential.

`--fallback`
    What to serve for an ambiguous form nothing decided. `none` reproduces the
    deployed API, which emits the *unstressed* token — it scores zero by
    construction, and 200 of 1000 gold rows landed there. `dictionary` serves
    the lexicon's highest-confidence reading, `manifest` the inventory's
    dominant sense, `morphology` prefers a morphological parse where one
    exists and falls back to the dictionary.

The fallback choice is not a detail. On a sense-balanced set any fixed default
approaches 0.5; on running text a frequency-weighted default is much better
than that. Reporting one number for both is how a fallback that scores 0.4044
went unnoticed — so `--baseline` also prints what always-serve-the-default
would score, which is the only number a tier's accuracy is meaningful against.
"""

from __future__ import annotations

import argparse
import collections
import json
import unicodedata
from pathlib import Path
from typing import Any

FALLBACKS = ("none", "dictionary", "manifest", "morphology")


def same_stress(first: str | None, second: str | None) -> bool:
    if not first or not second:
        return False
    return unicodedata.normalize("NFD", first).lower() == \
        unicodedata.normalize("NFD", second).lower()


def by_signature(record: dict[str, Any]) -> dict[str, str]:
    return {c["signature"]: c["stressed"] for c in record["candidates"]}


def fallback_answer(record: dict[str, Any], mode: str) -> tuple[str | None, str]:
    """Return (stressed form, tier name) for a row nothing decided."""
    candidates = record["candidates"]
    if not candidates:
        return None, "not_in_lexicon"
    if mode == "none":
        return None, "unresolved"
    if mode == "morphology" and record["morphology"]:
        return record["morphology"], "morphology"
    if mode == "manifest":
        stressed = by_signature(record).get(record["manifest_default"] or "")
        if stressed:
            return stressed, "manifest_default"
    return candidates[0]["stressed"], "dictionary_default"


def decide(record: dict[str, Any], threshold: float, fallback: str) -> dict[str, Any]:
    if not record["span_found"]:
        return {"got": None, "tier": "span_not_found"}
    if record["monosyllabic"]:
        # Ukrainian marks no stress on monosyllables, so the target is already
        # its own answer; if the gold file marks one, score it as given.
        return {"got": record["target"], "tier": "monosyllabic"}
    if not record["candidates"]:
        return {"got": None, "tier": "not_in_lexicon"}
    if len(record["candidates"]) == 1:
        return {"got": record["candidates"][0]["stressed"], "tier": "dictionary"}

    model = record["model"]
    if model is not None and model["margin"] >= threshold:
        stressed = by_signature(record).get(model["signature"])
        if stressed:
            return {"got": stressed, "tier": "model", "margin": model["margin"]}

    got, tier = fallback_answer(record, fallback)
    if model is not None:
        tier = f"abstained_{tier}"
    elif record["morphology"] and fallback != "none":
        # Outside the model's coverage but inside morphology's: report it as the
        # morphology tier so its accuracy stays separable from the dictionary's.
        return {"got": record["morphology"], "tier": "morphology"}
    else:
        tier = f"uncovered_{tier}"
    return {"got": got, "tier": tier, "margin": (model or {}).get("margin")}


def score(decisions: list[dict[str, Any]], threshold: float,
          fallback: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored = []
    for record in decisions:
        outcome = decide(record, threshold, fallback)
        scored.append({**{k: v for k, v in record.items() if k != "candidates"},
                       **outcome, "correct": same_stress(outcome["got"], record["gold"])})
    by_tier: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in scored:
        by_tier[row["tier"]].append(row)
    total = len(scored)
    correct = sum(1 for r in scored if r["correct"])
    summary = {
        "total": total, "correct": correct, "accuracy": correct / total if total else 0.0,
        "threshold": threshold, "fallback": fallback,
        "by_tier": {tier: {"rows": len(rows), "correct": sum(1 for r in rows if r["correct"])}
                    for tier, rows in sorted(by_tier.items())},
    }
    return scored, summary


def baseline(decisions: list[dict[str, Any]]) -> float:
    """Accuracy of always serving the lexicon's best reading, no model at all."""
    hit = 0
    for record in decisions:
        if record["monosyllabic"]:
            hit += same_stress(record["target"], record["gold"])
        elif record["candidates"]:
            hit += same_stress(record["candidates"][0]["stressed"], record["gold"])
    return hit / len(decisions) if decisions else 0.0


def report(summary: dict[str, Any]) -> None:
    total = summary["total"]
    print(f"\nGOLD ACCURACY: {summary['correct']}/{total} = {summary['accuracy']:.4f}"
          f"   (threshold={summary['threshold']}, fallback={summary['fallback']})\n")
    print(f"{'tier':<28}{'rows':>7}{'correct':>9}{'accuracy':>10}{'share':>8}")
    for tier, stats in sorted(summary["by_tier"].items(), key=lambda kv: -kv[1]["rows"]):
        rows, hit = stats["rows"], stats["correct"]
        print(f"{tier:<28}{rows:>7}{hit:>9}{hit/rows:>10.4f}{rows/total:>8.1%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path,
                        default=Path("output/ml/gold_eval/decisions.json"))
    parser.add_argument("--out", type=Path, default=Path("output/ml/gold_eval"))
    parser.add_argument("--threshold", type=float, default=13.0)
    parser.add_argument("--fallback", choices=FALLBACKS, default="none")
    parser.add_argument("--sweep", action="store_true",
                        help="print accuracy across thresholds and fallbacks and exit")
    args = parser.parse_args()

    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))

    if args.sweep:
        thresholds = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 11.0, 13.0, 15.0, 20.0]
        print(f"always-serve-the-lexicon-default baseline: {baseline(decisions):.4f}\n")
        header = "  ".join(f"{f:>12}" for f in FALLBACKS)
        print(f"{'threshold':>9}  {header}")
        for threshold in thresholds:
            cells = []
            for fallback in FALLBACKS:
                _, summary = score(decisions, threshold, fallback)
                cells.append(f"{summary['accuracy']:>12.4f}")
            print(f"{threshold:>9.1f}  " + "  ".join(cells))
        return

    scored, summary = score(decisions, args.threshold, args.fallback)
    summary["baseline_dictionary_default"] = baseline(decisions)
    report(summary)
    print(f"\nalways-serve-the-lexicon-default baseline: "
          f"{summary['baseline_dictionary_default']:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rows.json").write_text(
        json.dumps(scored, ensure_ascii=False, indent=1), encoding="utf-8")
    wrong = [r for r in scored if not r["correct"]]
    (args.out / "errors.json").write_text(
        json.dumps(wrong, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"errors: {len(wrong)}  -> {args.out}/errors.json")


if __name__ == "__main__":
    main()
