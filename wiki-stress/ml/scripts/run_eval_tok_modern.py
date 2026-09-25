"""Score a token classifier on modern text, against what speakers said.

The Common Voice verify sweeps hold every ambiguous token the pipeline met,
with the audio ranker's reading at its confidence, the pipeline's own answer
at sweep time, and the tier that gave it. A classifier trained on books has
seen none of these sentences, so this is a held-out test on the material the
user actually cares about — with one caveat the numbers must be read under:
the gold here is the same ranker that labelled the classifier's training
books, so a systematic ranker error on a form counts as agreement. lang-uk
(`run_eval_tok_langukbench.py`) is the independent check; a rule is changed
only when both move the right way.

Three totals per gate: the pipeline as swept, the classifier everywhere, and
the deployment rule — the classifier where the pipeline would take the
dictionary default or the cross-encoder's pick and the form clears the gate,
the pipeline elsewhere.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ukstress_ml.token_resolver import TokenResolver  # noqa: E402

SWEEPS = ("verify_60k.jsonl", "verify_30k.jsonl", "verify_after.jsonl")
TAKEN = {"dictionary_default", "ambiguous", "model_ineligible", "stressed"}


def load_rows(sweeps: Path, min_gold: float) -> list[dict]:
    """Later sweeps win, as build_token_corpus does: verify_after re-ran the
    first clips against a corrected lexicon."""
    seen: dict[tuple, dict] = {}
    for name in SWEEPS:
        path = sweeps / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if '"form"' not in line:
                continue
            row = json.loads(line)
            if row.get("start") is None or not row.get("sentence"):
                continue
            candidates = [c for c in row.get("candidates") or [] if c.isdigit()]
            if len(set(candidates)) < 2 or row.get("confidence", 0) < min_gold:
                continue
            row["candidates"] = candidates
            row["end"] = row.get("end") or row["start"] + len(row["form"])
            seen[(row.get("clip"), row["form"], row["start"])] = row
    return list(seen.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/tok-v5"))
    parser.add_argument("--coverage", type=Path,
                        help="training rows that define what the classifier has "
                             "seen; defaults to the corpus named like the model")
    parser.add_argument("--sweeps", type=Path,
                        default=Path("/home/devops/audiotostress/artifacts/audio_runs"))
    parser.add_argument("--min-gold", type=float, default=0.99,
                        help="ranker confidence a token needs to count as gold")
    parser.add_argument("--min-seen", type=int, nargs="+", default=[10, 30, 100])
    parser.add_argument("--min-confidence", type=float, nargs="+", default=[0.0, 0.9, 0.95])
    parser.add_argument("--override", default="dictionary_default,stressed",
                        help="pipeline tiers the rule lets the classifier replace")
    parser.add_argument("--tiers", action="store_true",
                        help="also print the per-tier table at the first gate")
    parser.add_argument("--allowance", type=Path,
                        help="learn a per-form allowance instead of a global gate: the "
                             "clips split in two by hash; on the dev half the classifier "
                             "is allowed on a form only where it beats the pipeline by "
                             "--margin tokens; the rule is then scored on the test half "
                             "and the allowed forms written here as JSON")
    parser.add_argument("--margin", type=int, default=2,
                        help="tokens by which the classifier must beat the pipeline on "
                             "the dev half of a form's tokens to be allowed on it")
    parser.add_argument("--llm", help="score with a causal LLM zero-shot (restricted softmax "
                                      "over the option letters) instead of the classifier")
    parser.add_argument("--llm-adapter", type=Path, help="a LoRA adapter for --llm")
    parser.add_argument("--glosses", type=Path, help="sense lexicon whose definitions ride "
                                                    "along as option descriptions (--llm)")
    parser.add_argument("--limit", type=int, default=0, help="score only this many tokens (pilot)")
    parser.add_argument("--allow", type=Path,
                        help="score the rule with this allowance (a JSON list of forms) "
                             "in place of min-seen")
    args = parser.parse_args()

    rows = load_rows(args.sweeps, args.min_gold)
    if args.limit:
        import random
        random.Random(17).shuffle(rows)
        rows = rows[:args.limit]
    print(f"{len(rows):,} ambiguous Common Voice tokens at audio confidence ≥{args.min_gold}")

    if args.llm:
        from ukstress_ml.llm_picker import LLMPicker
        resolver = LLMPicker(args.llm, glosses=args.glosses, adapter=args.llm_adapter)
    else:
        resolver = TokenResolver(args.model, min_seen=1)
    coverage = args.coverage or Path(str(args.model).replace("models/", "output/ml/corpus/")) / "train.jsonl"
    seen: collections.Counter[str] = collections.Counter()
    for line in coverage.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if len(r["candidates"]) > 1:
                seen[r["form"].lower()] += 1

    targets = [{"index": i, "sentence": r["sentence"], "start": r["start"], "end": r["end"],
                "form": r["form"].lower(), "candidates": r["candidates"]}
               for i, r in enumerate(rows)]
    answers: dict[int, dict] = {}
    for start in range(0, len(targets), 64):
        for d in resolver.resolve(targets[start:start + 64]):
            answers[d["index"]] = d

    override = set(args.override.split(","))
    pipeline_hits = sum(str(r["text"]) == str(r["audio"]) for r in rows)
    print(f"pipeline as swept: {pipeline_hits / len(rows):.1%}\n")

    if args.allowance:
        return learn_allowance(rows, answers, seen, override, args)
    allow = set(json.loads(args.allow.read_text(encoding="utf-8"))) if args.allow else None
    print(f"{'min seen':>9}{'min conf':>10}{'everywhere':>12}{'rule':>8}{'tokens taken':>14}{'taken right':>13}")
    first = True
    for min_seen in args.min_seen:
        for min_conf in args.min_confidence:
            every = rule = taken = taken_right = 0
            tiers: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0])
            for i, r in enumerate(rows):
                gold = str(r["audio"])
                pipe = str(r["text"]) == gold
                d = answers.get(i)
                eligible = (r["form"].lower() in allow if allow is not None
                            else seen.get(r["form"].lower(), 0) >= min_seen)
                covered = d is not None and eligible and d["confidence"] >= min_conf
                pick = d["signature"] == gold if d else False
                every += pick if d else pipe
                use = covered and r.get("status") in override
                rule += pick if use else pipe
                taken += use
                taken_right += pick if use else 0
                if first:
                    key = f"{r.get('status', '?')} " + ("seen" if seen.get(r["form"].lower(), 0) >= min_seen else "unseen")
                    t = tiers[key]
                    t[0] += 1
                    t[1] += pipe
                    t[2] += pick if d else pipe
            print(f"{min_seen:>9}{min_conf:>10.2f}{100 * every / len(rows):>11.1f}%{100 * rule / len(rows):>7.1f}%"
                  f"{taken:>14,}{100 * taken_right / max(taken, 1):>12.1f}%")
            if first and args.tiers:
                print(f"\n  {'tier, coverage':<32}{'n':>7}{'pipeline':>10}{'classifier':>12}")
                for key, (n, p, c) in sorted(tiers.items(), key=lambda kv: -kv[1][0]):
                    if n >= 30:
                        print(f"  {key:<32}{n:>7,}{100 * p / n:>9.1f}%{100 * c / n:>11.1f}%")
                print()
            first = False
    return 0


def learn_allowance(rows, answers, seen, override, args) -> int:
    """Per form, on half the clips: does the classifier beat the pipeline on
    the tokens the rule would hand it? Allow it there; score on the other
    half, against the global gate."""
    import hashlib
    half = {i: int(hashlib.md5(str(r.get("clip")).encode()).hexdigest(), 16) % 2 for i, r in enumerate(rows)}
    per_form: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0])  # n, pipeline, classifier
    for i, r in enumerate(rows):
        d = answers.get(i)
        if half[i] != 0 or d is None or r.get("status") not in override:
            continue
        gold = str(r["audio"])
        t = per_form[r["form"].lower()]
        t[0] += 1
        t[1] += str(r["text"]) == gold
        t[2] += d["signature"] == gold
    allowed = sorted(f for f, (n, p, c) in per_form.items() if c - p >= args.margin)
    print(f"dev half: {sum(v[0] for v in per_form.values()):,} rule tokens over {len(per_form):,} forms; "
          f"{len(allowed):,} forms allowed at margin {args.margin} "
          f"({sum(1 for f in allowed if seen.get(f, 0) >= 100)} of them seen 100+)")
    args.allowance.write_text(json.dumps(allowed, ensure_ascii=False, indent=0), encoding="utf-8")

    print(f"\n{'test half':<28}{'n':>7}{'pipeline':>10}{'rule':>8}{'taken':>8}{'taken right':>13}")
    for name, eligible in (("global gate, seen 100+", lambda f: seen.get(f, 0) >= 100),
                           ("global gate, seen 80+", lambda f: seen.get(f, 0) >= 80),
                           (f"allowance, margin {args.margin}", lambda f: f in set(allowed))):
        n = pipe = rule = taken = right = 0
        for i, r in enumerate(rows):
            if half[i] != 1:
                continue
            gold = str(r["audio"]); d = answers.get(i)
            p = str(r["text"]) == gold
            use = d is not None and eligible(r["form"].lower()) and r.get("status") in override
            pick = d["signature"] == gold if d else False
            n += 1; pipe += p; rule += pick if use else p; taken += use; right += pick if use else 0
        print(f"{name:<28}{n:>7,}{100 * pipe / n:>9.1f}%{100 * rule / n:>7.1f}%{taken:>8,}{100 * right / max(taken, 1):>12.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
