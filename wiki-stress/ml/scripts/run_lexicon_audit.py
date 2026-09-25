"""Audit the stress of high-frequency single-reading lexicon entries.

The `lexicon` tier answers 3,422 of the benchmark's unambiguous tokens and gets
1.93% of them wrong -- `укра́їни` for `Украї́ни`, `йо́го` for `його́`,
`ді́тей` for `діте́й`. These are not ambiguity failures and no amount of context
modelling touches them: the stored stress is simply wrong, and it is wrong on
words that occur constantly.

Source rank does not separate the bad entries (rank 5 scores 97.46%, rank 10
scores 98.35%), so the entries have to be checked rather than filtered. This
asks two models independently, on the bare word with no sentence and no
candidate list, and reports only where both agree with each other and disagree
with the lexicon -- the same two-model rule the labelling pipeline uses.
"""

from __future__ import annotations

import argparse
import collections
import json
import unicodedata
from pathlib import Path
from typing import Any

import psycopg

from ukstress_ml import annotate

VOWELS = frozenset("аеєиіїоуюя")
BREVE = "̆"
ACUTE = "́"

AUDIT_SYSTEM = (
    "Ти — фахівець з української орфоепії. Визначаєш наголос у слові за "
    "нормами сучасної української літературної мови. Відповідаєш лише "
    "валідним JSON."
)


def syllable_count(word: str) -> int:
    """Vowels in the word. `й` is a consonant even though NFD spells it и+breve."""
    decomposed = unicodedata.normalize("NFD", word).lower()
    total = 0
    for index, char in enumerate(decomposed):
        if char in VOWELS:
            if char == "и" and index + 1 < len(decomposed) and decomposed[index + 1] == BREVE:
                continue
            total += 1
    return total


def stress_ordinal(stressed: str) -> int | None:
    """Which vowel carries the acute, counted the same way."""
    decomposed = unicodedata.normalize("NFD", stressed).lower()
    ordinal = -1
    for index, char in enumerate(decomposed):
        if char == ACUTE:
            return ordinal
        if char in VOWELS:
            if char == "и" and index + 1 < len(decomposed) and decomposed[index + 1] == BREVE:
                continue
            ordinal += 1
    return None


def audit_prompt(_form: Any, rows: list[dict[str, Any]]) -> str:
    listed = "\n".join(f"{i}. {row['word']}" for i, row in enumerate(rows))
    return f"""Для кожного слова визнач, на який склад падає наголос.

Слова:
{listed}

"vowel" — порядковий номер наголошеного голосного, рахуючи з нуля
(й не голосний). Наприклад, у слові "Украї́на" наголошений голосний — "ї",
це голосний номер 2 (у=0, а=1, ї=2).
Якщо слово має варіантні наголоси, назви основний нормативний.
Якщо не впевнений — "unclear": true.

Формат відповіді (лише JSON, без пояснень):
{{"items": [{{"i": 0, "vowel": 0, "unclear": false}}]}}"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("output/ml/silver_mined_v10.json"),
                        help="frequency source; audit the words that actually occur")
    parser.add_argument("--database-url",
                        default="postgresql://ukstress_owner:ukstress_owner@localhost:5432/ukstress")
    parser.add_argument("--top", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--raw", type=Path, default=Path("output/ml/raw"))
    parser.add_argument("--out", type=Path, default=Path("output/ml/lexicon_audit.json"))
    args = parser.parse_args()

    frequency: collections.Counter[str] = collections.Counter()
    for row in json.loads(args.corpus.read_text(encoding="utf-8")):
        for token in str(row["sentence"]).lower().split():
            token = token.strip(".,!?;:()«»\"'—–-")
            if token:
                frequency[token] += 1

    with psycopg.connect(args.database_url) as conn:
        stored = conn.execute(
            """SELECT form_normalized, min(stressed_form), count(DISTINCT stress_signature)
               FROM stress_lookup
               WHERE dataset_id IN (2,3,4,5,6,7) AND form_normalized = ANY(%s)
               GROUP BY form_normalized""",
            [[word for word, _ in frequency.most_common(args.top * 4)]],
        ).fetchall()

    # Single-reading entries only: a form with two readings is the model's
    # problem, not the lexicon's, and its "correct" stress depends on context.
    words: list[dict[str, Any]] = []
    for form, stressed, readings in stored:
        if readings != 1 or syllable_count(form) < 2:
            continue
        ordinal = stress_ordinal(str(stressed))
        if ordinal is None or ordinal < 0:
            continue
        # `run_job` groups rows by "form" to build one prompt per form; here a
        # batch is just a list of unrelated words, so they share one bucket.
        words.append({"sentence_id": form, "form": "audit", "word": form,
                      "stored": str(stressed), "stored_vowel": ordinal,
                      "count": frequency[form]})
    words.sort(key=lambda w: -w["count"])
    words = words[: args.top]
    print(f"auditing {len(words):,} single-reading forms by corpus frequency", flush=True)

    config = annotate.load_config()
    answers: dict[str, dict[str, int]] = {}
    for name, deployment in (("a", annotate.LABEL_DEPLOYMENT), ("b", annotate.VERIFY_DEPLOYMENT)):
        raw_path = args.raw / f"lexaudit_{name}.jsonl"
        annotate.run_job(
            job="label", rows=words, forms={"audit": None}, deployment=deployment, config=config,
            raw_path=raw_path, batch_size=args.batch_size, workers=args.workers,
            max_tokens=6000, system=AUDIT_SYSTEM, prompt_builder=audit_prompt,
            progress_every=10,
        )
        with raw_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                content = (record.get("content") or "").strip()
                content = content.removeprefix("```json").removeprefix("```").removesuffix("```")
                try:
                    items = json.loads(content).get("items", [])
                except json.JSONDecodeError:
                    continue
                ids = record.get("sentence_ids") or []
                for item in items:
                    index = item.get("i")
                    if not isinstance(index, int) or index >= len(ids) or item.get("unclear"):
                        continue
                    vowel = item.get("vowel")
                    if isinstance(vowel, int):
                        answers.setdefault(str(ids[index]), {})[name] = vowel

    disputed = []
    agreed = both = 0
    for word in words:
        pair = answers.get(word["word"], {})
        if len(pair) < 2 or pair["a"] != pair["b"]:
            continue
        both += 1
        if pair["a"] == word["stored_vowel"]:
            agreed += 1
        else:
            disputed.append({**word, "proposed_vowel": pair["a"]})
    print(f"\nboth models answered and agreed : {both:,}")
    print(f"  agree with the lexicon         : {agreed:,}")
    print(f"  DISPUTE the lexicon            : {len(disputed):,}")
    args.out.write_text(json.dumps(disputed, ensure_ascii=False, indent=1), encoding="utf-8")
    for item in disputed[:20]:
        print(f"    {item['word']:16} lexicon {item['stored']:16} "
              f"vowel {item['stored_vowel']} -> {item['proposed_vowel']}  (x{item['count']})")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
