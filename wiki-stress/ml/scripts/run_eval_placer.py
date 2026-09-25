"""Score the stress placer where the pipeline currently guesses.

Two sets, both of words the lexicon does not have:

* **lang-uk**: every benchmark token the live pipeline sent to the suffix,
  compound or positional fallback (from a `run_live_bench.py --save` review),
  with the benchmark's human gold. This is the acceptance set: 72% today.
* **Common Voice**: every verify-sweep token with no lexicon candidates, with
  the audio ranker's reading at ≥0.99. Larger, and closer to modern text:
  70% today.

The placer's candidates are all the word's vowels; a hyphenated word is
scored on its last segment like the positional default, since that is what
the surface signature addresses.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ukstress_ml.token_resolver import TokenResolver  # noqa: E402

VOWELS = "аеєиіїоуюяй"
TOKEN = re.compile(r"[А-Яа-яЇїІіЄєҐґ'’]+")
FALLBACK = {"suffix", "compound", "default_position", "not_found"}

def vowel_candidates(form: str) -> list[str]:
    """Ordinals a stress may land on. й has an ordinal in this project's
    signatures but is never stressed, and offering it produced «й́ому»."""
    out, seen = [], -1
    for ch in form:
        if ch in VOWELS:
            seen += 1
            if ch != "й":
                out.append(str(seen))
    return out



def ordinal_of(marked: str, mark: str) -> str | None:
    seen = -1
    for ch in unicodedata.normalize("NFD", marked):
        if ch.lower() in VOWELS:
            seen += 1
        if ch == mark:
            return str(seen)
    return None


def external_sets(review: Path, sweeps: Path, min_gold: float = 0.99):
    """(name, rows, golds, pipeline accuracy) for lang-uk's fallback tokens
    and Common Voice's no-lexicon tokens. Rows carry sentence/start/end/form/
    candidates in the resolver's shape."""
    out = []
    targets, golds, pipe = [], [], 0
    for row in json.loads(review.read_text(encoding="utf-8")):
        gold_words, cursor = row["gold"].split(), 0
        for t in [t for t in row.get("tokens", []) if t.get("status") in FALLBACK]:
            bare = unicodedata.normalize("NFC", t["text"].lower())
            while cursor < len(gold_words) and unicodedata.normalize(
                    "NFC", "".join(TOKEN.findall(gold_words[cursor].replace("+", ""))).lower()) != bare:
                cursor += 1
            if cursor >= len(gold_words):
                break
            gold = ordinal_of(gold_words[cursor], "+")
            cursor += 1
            vowels = sum(1 for c in bare if c in VOWELS)
            if gold is None or vowels < 2:
                continue
            targets.append({"index": len(targets), "sentence": row["plain"], "start": t["start"],
                            "end": t["end"], "form": bare, "candidates": vowel_candidates(bare)})
            golds.append(gold)
            pipe += ordinal_of(t.get("output_text", ""), "́") == gold
    out.append(("lang-uk fallback tokens", targets, golds, pipe / max(len(golds), 1)))

    seen: dict[tuple, dict] = {}
    for name in ("verify_60k.jsonl", "verify_30k.jsonl", "verify_after.jsonl"):
        path = sweeps / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if '"form"' not in line:
                continue
            r = json.loads(line)
            if r.get("candidates") or r.get("start") is None or r.get("confidence", 0) < min_gold:
                continue
            form = r["form"].lower()
            if sum(1 for c in form if c in VOWELS) < 2:
                continue
            seen[(r.get("clip"), form, r["start"])] = r
    rows = list(seen.values())
    cv = [{"index": i, "sentence": r["sentence"], "start": r["start"],
           "end": r.get("end") or r["start"] + len(r["form"]), "form": r["form"].lower(),
           "candidates": vowel_candidates(r["form"].lower())}
          for i, r in enumerate(rows)]
    cv_golds = [str(r["audio"]) for r in rows]
    cv_pipe = sum(str(r.get("text")) == str(r["audio"]) for r in rows) / max(len(rows), 1)
    out.append(("Common Voice no-lexicon tokens", cv, cv_golds, cv_pipe))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--review", type=Path, default=Path("output/ml/live_review20.json"))
    parser.add_argument("--sweeps", type=Path,
                        default=Path("/home/devops/audiotostress/artifacts/audio_runs"))
    parser.add_argument("--min-gold", type=float, default=0.99)
    args = parser.parse_args()

    resolver = TokenResolver(args.model, min_seen=0)
    resolver.forms = collections.defaultdict(lambda: 10 ** 9)   # every form is "covered"

    def ask(rows: list[dict]) -> dict[int, dict]:
        out = {}
        for start in range(0, len(rows), 64):
            for d in resolver.resolve(rows[start:start + 64]):
                out[d["index"]] = d
        return out

    # lang-uk: the tokens the live pipeline could not look up.
    targets, golds, tiers = [], [], []
    for row in json.loads(args.review.read_text(encoding="utf-8")):
        gold_words, cursor = row["gold"].split(), 0
        tokens = [t for t in row.get("tokens", []) if t.get("status") in FALLBACK]
        for t in tokens:
            bare = unicodedata.normalize("NFC", t["text"].lower())
            while cursor < len(gold_words) and unicodedata.normalize(
                    "NFC", "".join(TOKEN.findall(gold_words[cursor].replace("+", ""))).lower()) != bare:
                cursor += 1
            if cursor >= len(gold_words):
                break
            gold = ordinal_of(gold_words[cursor], "+")
            cursor += 1
            vowels = sum(1 for c in bare if c in VOWELS)
            if gold is None or vowels < 2:
                continue
            targets.append({"index": len(targets), "sentence": row["plain"], "start": t["start"],
                            "end": t["end"], "form": bare, "candidates": vowel_candidates(bare)})
            golds.append(gold)
            tiers.append((t["status"], unicodedata.normalize("NFD", t.get("output_text", "")).count("́") > 0
                          and ordinal_of(t["output_text"], "́") == gold))
    answers = ask(targets)
    hits = sum(answers[i]["signature"] == g for i, g in enumerate(golds) if i in answers)
    pipe = sum(ok for _, ok in tiers)
    print(f"lang-uk fallback tokens: {len(golds):,}  pipeline {pipe / len(golds):.1%}  "
          f"placer {hits / len(golds):.1%}")
    by_tier = collections.defaultdict(lambda: [0, 0, 0])
    for i, (tier, ok) in enumerate(tiers):
        by_tier[tier][0] += 1
        by_tier[tier][1] += ok
        by_tier[tier][2] += answers.get(i, {}).get("signature") == golds[i]
    for tier, (n, p, h) in sorted(by_tier.items(), key=lambda kv: -kv[1][0]):
        print(f"  {tier:<18}{n:>6,}  pipeline {p / n:.1%}  placer {h / n:.1%}")

    # Common Voice: tokens with no lexicon candidates, audio gold.
    seen: dict[tuple, dict] = {}
    for name in ("verify_60k.jsonl", "verify_30k.jsonl", "verify_after.jsonl"):
        path = args.sweeps / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if '"form"' not in line:
                continue
            r = json.loads(line)
            if r.get("candidates") or r.get("start") is None or r.get("confidence", 0) < args.min_gold:
                continue
            form = r["form"].lower()
            if sum(1 for c in form if c in VOWELS) < 2:
                continue
            seen[(r.get("clip"), form, r["start"])] = r
    rows = list(seen.values())
    cv = [{"index": i, "sentence": r["sentence"], "start": r["start"],
           "end": r.get("end") or r["start"] + len(r["form"]), "form": r["form"].lower(),
           "candidates": vowel_candidates(r["form"].lower())}
          for i, r in enumerate(rows)]
    answers = ask(cv)
    hits = sum(answers[i]["signature"] == str(r["audio"]) for i, r in enumerate(rows) if i in answers)
    pipe = sum(r.get("text") is not None and str(r["text"]) == str(r["audio"]) for r in rows)
    print(f"\nCommon Voice no-lexicon tokens: {len(rows):,}  pipeline {pipe / len(rows):.1%}  "
          f"placer {hits / len(rows):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
