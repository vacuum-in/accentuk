"""Close the lexicon gap: find unstressed words in real text, ask the models,
write the confident ones back to PostgreSQL, and keep the rest as training data.

Measured on 3M words of OpenSubtitles, 6.4% of polysyllabic tokens have no
lexicon entry at all. That is the largest remaining gap and no amount of model
work touches it — the pipeline never gets a candidate to choose between.

The stages:

1. **Scan** the corpus, skipping monosyllables (Ukrainian marks no stress on
   them, so their absence is correct, not a gap) and anything the lexicon
   already answers. Rank the misses by frequency: the top 100 forms carry a
   quarter of all misses, so frequency ordering is most of the value.

2. **Ask both vendors independently** — DeepSeek-V4-Pro and gpt-5.4 — for the
   stressed form, given real usage sentences from the corpus for context.

3. **Agreement decides.** Identical answers are written to PostgreSQL as a new
   `llm_gap_fill` dataset with `source_kind='llm_gap_fill'` and a confidence
   below the Wiktionary parser's, so ordering never lets a machine-written
   stress outrank a curated one. Disagreement is *not* resolved by a third
   opinion or a tiebreak: two competent models disagreeing is evidence the word
   is genuinely ambiguous.

4. **Disagreements become dataset candidates.** A word where the models differ
   is a homograph candidate, and the usage sentences already collected are
   exactly what the contextual model needs. Those are written out for the
   annotation pipeline instead of being forced into a single answer.

Nothing here overwrites an existing entry. The new rows live in their own
dataset, so they can be published, inspected, or rolled back independently.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import psycopg
from openai import OpenAI
from ukstress.normalizer import ACUTE, canonical_stressed_form, lookup_key, validate_stress

from ukstress_ml.annotate import LABEL_DEPLOYMENT, Usage, load_config

WORD = re.compile(r"[^\W\d_]+(?:['’ʼ-][^\W\d_]+)*", re.UNICODE)
CYRILLIC = re.compile(r"[а-щьюяєіїґА-ЩЬЮЯЄІЇҐ]")
VOWELS = frozenset("аеєиіїоуюя")

SYSTEM = (
    "Ти — український фонетист. Ставиш наголос у слові. "
    "Відповідаєш лише валідним JSON."
)


def syllables(word: str) -> int:
    return sum(1 for c in word.lower() if c in VOWELS)


def build_prompt(form: str, examples: list[str]) -> str:
    usage = "\n".join(f"  - {s}" for s in examples[:4]) or "  (немає прикладів)"
    return f"""Слово: {form}

Приклади вживання з реального тексту:
{usage}

Постав наголос у слові «{form}»: познач наголошений голосний,
повернувши слово зі знаком наголосу (U+0301) після наголошеної голосної.

Якщо слово має РІЗНІ наголоси в різних значеннях (омограф) — постав
"ambiguous": true і перелічи варіанти.

Формат (лише JSON):
{{"stressed": "сло́во", "ambiguous": false, "variants": []}}"""


def parse_answer(content: str, form: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError:
        return {}
    out: dict[str, Any] = {"ambiguous": bool(payload.get("ambiguous"))}
    variants = []
    for value in [payload.get("stressed"), *payload.get("variants", [])]:
        if not isinstance(value, str) or not value.strip():
            continue
        canonical = canonical_stressed_form(value.strip())
        # The answer must be the word we asked about, differing only by the
        # acute: a model that "corrects" the spelling has answered a different
        # question, and writing that to the lexicon would corrupt it.
        if lookup_key(canonical) != form:
            continue
        if ACUTE not in unicodedata.normalize("NFD", canonical):
            continue
        if validate_stress(canonical).classification != "valid":
            continue
        variants.append(canonical)
    out["variants"] = list(dict.fromkeys(variants))
    out["stressed"] = out["variants"][0] if out["variants"] else None
    return out


def _same_stress(first: str, second: str | None) -> bool:
    """True when two answers mark the same vowel, ignoring letter case."""
    if not second:
        return False
    return unicodedata.normalize("NFD", first).lower() == \
        unicodedata.normalize("NFD", second).lower()


def ask(client: OpenAI, model: str, prompt: str, usage: Usage, max_tokens: int) -> str:
    for attempt in range(1, 4):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": prompt}],
            }
            if model.startswith("gpt-5"):
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens
                kwargs["temperature"] = 0
            response = client.chat.completions.create(**kwargs)
            if response.usage:
                usage.add(response.usage.prompt_tokens, response.usage.completion_tokens)
            return response.choices[0].message.content or ""
        except Exception:  # noqa: BLE001
            if attempt == 3:
                usage.fail()
                return ""
            time.sleep(5 * attempt)
    return ""


def scan(corpus: Path, dsn: str, dataset_id: int, limit_words: int,
         examples_per_form: int) -> tuple[collections.Counter, dict[str, list[str]]]:
    """Return frequency-ranked misses plus usage sentences for each."""
    counts: collections.Counter[str] = collections.Counter()
    examples: dict[str, list[str]] = collections.defaultdict(list)
    seen = 0
    opener = gzip.open if corpus.suffix == ".gz" else open
    pending: list[tuple[str, list[str]]] = []

    def drain(conn: Any) -> None:
        nonlocal pending
        if not pending:
            return
        keys = sorted({k for _, ks in pending for k in ks})
        known = {
            str(r[0]) for r in conn.execute(
                "SELECT DISTINCT form_normalized FROM stress_lookup "
                "WHERE dataset_id=%s AND form_normalized = ANY(%s)", (dataset_id, keys)).fetchall()
        }
        for sentence, ks in pending:
            for key in ks:
                if key in known:
                    continue
                counts[key] += 1
                if len(examples[key]) < examples_per_form and 4 <= len(sentence.split()) <= 30:
                    examples[key].append(sentence)
        pending = []

    with psycopg.connect(dsn) as conn, \
         opener(corpus, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            sentence = line.strip()
            if not sentence or not CYRILLIC.search(sentence):
                continue
            keys = []
            for match in WORD.finditer(sentence):
                word = match.group()
                if not CYRILLIC.search(word) or syllables(word) <= 1:
                    continue
                seen += 1
                keys.append(lookup_key(word))
            if keys:
                pending.append((sentence, keys))
            if len(pending) >= 500:
                drain(conn)
            if seen >= limit_words:
                break
        drain(conn)
    return counts, examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path,
                        default=Path("data/corpus/opensubtitles-uk.txt.gz"))
    parser.add_argument("--words", type=int, default=3_000_000)
    parser.add_argument("--top", type=int, default=1500, help="how many missing forms to resolve")
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--database-url",
                        default="postgresql://ukstress_owner:ukstress_owner@localhost:5432/ukstress")
    parser.add_argument("--dataset-key", default="llm-gap-fill-v1")
    parser.add_argument("--raw", type=Path, default=Path("output/ml/raw/gap_fill.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("output/ml/gap_fill"))
    parser.add_argument("--write-db", action="store_true", help="apply agreed stresses to PG")
    args = parser.parse_args()

    with psycopg.connect(args.database_url) as conn:
        row = conn.execute("SELECT dataset_id FROM active_dataset WHERE singleton").fetchone()
        active = int(row[0]) if row else 0

    print(f"scanning {args.corpus.name} for up to {args.words:,} polysyllabic words "
          f"against dataset {active}", flush=True)
    counts, examples = scan(args.corpus, args.database_url, active, args.words, args.examples)
    ranked = [(f, n) for f, n in counts.most_common(args.top) if n >= args.min_count]
    print(f"missing forms: {len(counts):,} distinct, {sum(counts.values()):,} tokens\n"
          f"resolving top {len(ranked):,} (>= {args.min_count} occurrences)", flush=True)

    config = load_config()
    primary = LABEL_DEPLOYMENT.client(config)
    primary_model = LABEL_DEPLOYMENT.resolved_model(config)
    second = OpenAI(base_url=str(config["AZURE_OPENAI_ENDPOINT_INITAITESTING"]).rstrip("/"),
                    api_key=str(config["AZURE_OPENAI_API_KEY_INITAITESTING"]),
                    timeout=240.0, max_retries=2)
    second_model = str(config.get("AZURE_OPENAI_DEPLOYMENT_INITAITESTING") or "gpt-5.4")

    done: set[str] = set()
    if args.raw.exists():
        for line in args.raw.open(encoding="utf-8"):
            record = json.loads(line)
            if record.get("a_raw") or record.get("b_raw"):
                done.add(record["form"])
    pending = [(f, n) for f, n in ranked if f not in done]

    usage = Usage()
    tally: collections.Counter[str] = collections.Counter()
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    with args.raw.open("a", encoding="utf-8") as handle:
        for position, (form, count) in enumerate(pending, 1):
            prompt = build_prompt(form, examples.get(form, []))
            a_raw = ask(primary, primary_model, prompt, usage, 400)
            b_raw = ask(second, second_model, prompt, usage, 3000)
            a, b = parse_answer(a_raw, form), parse_answer(b_raw, form)
            if a.get("ambiguous") or b.get("ambiguous") or len(a.get("variants", [])) > 1 \
                    or len(b.get("variants", [])) > 1:
                verdict = "ambiguous"
            elif a.get("stressed") and _same_stress(a["stressed"], b.get("stressed")):
                # Compare the stress, not the casing. The models routinely
                # differ on whether a subtitle name is capitalised (Ска́рлетт vs
                # ска́рлетт) while agreeing exactly on the stressed vowel, and
                # counting that as a dispute floods the dataset candidates with
                # proper nouns that have no stress ambiguity at all.
                verdict = "agreed"
            elif a.get("stressed") and b.get("stressed"):
                verdict = "disagreed"
            else:
                verdict = "unresolved"
            tally[verdict] += 1
            handle.write(json.dumps({
                "form": form, "count": count, "verdict": verdict,
                "a": a.get("stressed"), "b": b.get("stressed"),
                "a_variants": a.get("variants", []), "b_variants": b.get("variants", []),
                "examples": examples.get(form, []),
                "a_raw": a_raw, "b_raw": b_raw,
            }, ensure_ascii=False) + "\n")
            handle.flush()
            if position % 25 == 0 or position == len(pending):
                print(f"  {position}/{len(pending)}  {dict(tally)}  {usage.snapshot()}", flush=True)

    records = [json.loads(line) for line in args.raw.open(encoding="utf-8")]
    agreed = [r for r in records if r["verdict"] == "agreed"]
    contested = [r for r in records if r["verdict"] in ("ambiguous", "disagreed")]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "agreed.tsv").write_text(
        "count\tform\tstressed\n" + "".join(
            f"{r['count']}\t{r['form']}\t{r['a']}\n" for r in
            sorted(agreed, key=lambda r: -r["count"])), encoding="utf-8")
    (args.out / "dataset_candidates.json").write_text(
        json.dumps([{
            "form": r["form"], "count": r["count"], "reason": r["verdict"],
            "variants": sorted({*r["a_variants"], *r["b_variants"]}),
            "examples": r["examples"],
        } for r in sorted(contested, key=lambda r: -r["count"])],
            ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nagreed (both models)      : {len(agreed):,}  -> {args.out}/agreed.tsv")
    print(f"contested -> dataset       : {len(contested):,}  -> {args.out}/dataset_candidates.json")
    print(f"tokens recovered if applied: {sum(r['count'] for r in agreed):,}")
    print(json.dumps({**dict(tally), **usage.snapshot()}, indent=2))

    if args.write_db and agreed:
        written = write_dataset(args.database_url, args.dataset_key, agreed)
        print(f"\nwrote {written:,} rows to dataset '{args.dataset_key}' (unpublished)")
    elif agreed:
        print("\n--write-db not set; nothing written to PostgreSQL")


def write_dataset(dsn: str, dataset_key: str, agreed: list[dict[str, Any]]) -> int:
    """Add agreed stresses as their own unpublished dataset.

    A separate dataset keeps machine-written stresses inspectable and
    reversible, and `confidence` sits below the Wiktionary parser's 0.95 so a
    generated entry can never outrank a curated one in `stress_lookup`'s
    ordering.
    """
    from ukstress.normalizer import stress_signature

    with psycopg.connect(dsn) as conn, conn.transaction():
        row = conn.execute("SELECT id FROM import_run WHERE dataset_key=%s",
                           (dataset_key,)).fetchone()
        if row:
            dataset_id = int(row[0])
            conn.execute("DELETE FROM lexeme WHERE dataset_id=%s", (dataset_id,))
        else:
            dataset_id = int(conn.execute(
                """INSERT INTO import_run (dataset_key, status, dump_url, dump_sha256,
                       parser_version, normalization_version, schema_version, statistics)
                   VALUES (%s,'building','llm://gap-fill',%s,'gap-fill-1','1','007','{}')
                   RETURNING id""", (dataset_key, "0" * 64)).fetchone()[0])
        written = 0
        for record in agreed:
            stressed = record["a"]
            lexeme_id = int(conn.execute(
                """INSERT INTO lexeme (dataset_id, lemma, lemma_normalized, stressed_lemma,
                       sense_key, source_title, source_section, confidence, natural_key)
                   VALUES (%s,%s,%s,%s,'0',%s,'LLMGapFill',0.70,%s) RETURNING id""",
                (dataset_id, record["form"], record["form"], stressed, record["form"],
                 f"gapfill:{dataset_key}:{record['form']}")).fetchone()[0])
            word_form_id = int(conn.execute(
                """INSERT INTO word_form (dataset_id, lexeme_id, form, form_normalized,
                       morphology_key, is_lemma, is_variant, confidence, source_rank, natural_key)
                   VALUES (%s,%s,%s,%s,'',true,false,0.70,30,%s) RETURNING id""",
                (dataset_id, lexeme_id, stressed, record["form"],
                 f"gapfill:wf:{record['form']}")).fetchone()[0])
            conn.execute(
                """INSERT INTO stress_variant (dataset_id, word_form_id, stressed_form,
                       stress_signature, variant_type, confidence, natural_key)
                   VALUES (%s,%s,%s,%s,'primary',0.70,%s)""",
                (dataset_id, word_form_id, stressed, stress_signature(stressed),
                 f"gapfill:sv:{record['form']}"))
            conn.execute(
                """INSERT INTO source_ref (dataset_id, lexeme_id, word_form_id, page_title,
                       source_kind, source_fragment, natural_key)
                   VALUES (%s,%s,%s,%s,'llm_gap_fill',%s,%s)""",
                (dataset_id, lexeme_id, word_form_id, record["form"], stressed,
                 f"gapfill:sr:{record['form']}"))
            written += 1
    return written


if __name__ == "__main__":
    main()
