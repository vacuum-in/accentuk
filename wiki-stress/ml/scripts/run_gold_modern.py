"""Score the live API against the modern-text gold (PLAN_QUALITY step 4.4).

Each gold line carries U+0301 after every stressed vowel. The marks are
stripped, the plain line goes to /v1/stress, and every word with two or more
vowels is compared with the gold: right, wrong, or left unstressed (which
counts as wrong, as on lang-uk). Reported overall, per deciding tier, and
split by whether the reviewer corrected the line — the corrected lines are
the ones a person certainly read; the others still carry the draft's marks
(the pipeline's at drafting time, the narrators' on the form under test).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import re
import unicodedata
import urllib.request
from pathlib import Path

ACUTE = "́"
VOWELS = "аеєиіїоуюя"
WORD = re.compile(r"[\w'’ʼ́-]+")


def stress_of(word: str) -> str | None:
    ordinal = -1
    for ch in unicodedata.normalize("NFD", word):
        if ch.lower() in VOWELS:
            ordinal += 1
        elif ch == ACUTE:
            return str(ordinal)
    return None


def vowels(word: str) -> int:
    return sum(1 for ch in unicodedata.normalize("NFD", word) if ch.lower() in VOWELS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=Path("ml/data/gold_modern_reviewed_1.txt"))
    parser.add_argument("--draft", type=Path, default=Path("ml/data/gold_modern_draft_1.txt"),
                        help="to tell the corrected lines from the drafted ones")
    parser.add_argument("--api", default="http://127.0.0.1:8080")
    parser.add_argument("--out", type=Path, default=Path("output/ml/gold_modern_report.json"))
    parser.add_argument("--show", type=int, default=40, help="errors to print")
    args = parser.parse_args()

    def lines_of(path: Path) -> list[str]:
        return [unicodedata.normalize("NFC", l.strip()) for l in path.read_text(encoding="utf-8").splitlines()
                if l.strip() and not l.lstrip().startswith("#")]

    gold = lines_of(args.gold)
    drafted = set(lines_of(args.draft))

    def ask(line: str):
        plain = unicodedata.normalize("NFC", unicodedata.normalize("NFD", line).replace(ACUTE, ""))
        request = urllib.request.Request(f"{args.api}/v1/stress", data=json.dumps({"text": plain}).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            return line, json.load(response)

    answers = {}
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for line, answer in pool.map(ask, gold):
            answers[line] = answer

    totals: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    by_tier: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    errors = []
    for line in gold:
        answer = answers[line]
        corrected = line not in drafted
        # the API keeps the input text and only inserts marks, so the words align
        gold_words = WORD.findall(line)
        got_words = WORD.findall(unicodedata.normalize("NFC", answer["text"]))
        tokens = {t["text"].lower(): t.get("status", "") for t in answer.get("tokens", [])}
        if len(gold_words) != len(got_words):
            continue
        for g, o in zip(gold_words, got_words):
            if vowels(g) < 2:
                continue
            want, got = stress_of(g), stress_of(o)
            if want is None:
                continue
            ok = got == want
            plain = unicodedata.normalize("NFC", unicodedata.normalize("NFD", g).replace(ACUTE, "")).lower()
            tier = tokens.get(plain, "?")
            for key in ("all", "corrected lines" if corrected else "drafted lines"):
                totals[key][0] += 1
                totals[key][1] += ok
            by_tier[tier][0] += 1
            by_tier[tier][1] += ok
            if not ok:
                errors.append({"word": plain, "gold": g, "got": o, "tier": tier, "corrected_line": corrected,
                               "line": line})

    print(f"{len(gold)} lines, {sum(1 for l in gold if l not in drafted)} corrected by the reviewer\n")
    for key in ("all", "corrected lines", "drafted lines"):
        n, ok = totals[key]
        if n:
            print(f"  {key:<18}{n:>6,} words  {100 * ok / n:6.2f}%")
    print("\n  by the tier that decided:")
    for tier, (n, ok) in sorted(by_tier.items(), key=lambda kv: -kv[1][0]):
        print(f"    {tier:<20}{n:>6,}  {100 * ok / n:6.2f}%  ({n - ok} wrong)")
    print(f"\n  errors ({len(errors)}), corrected lines first:")
    for e in sorted(errors, key=lambda e: not e["corrected_line"])[:args.show]:
        print(f"    {'*' if e['corrected_line'] else ' '} {e['gold']:<18} got {e['got']:<18} [{e['tier']}]")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"totals": totals, "by_tier": by_tier, "errors": errors},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
