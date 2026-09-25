"""Find homographs the lexicon records as having only one stress.

`run_recover_readings.py` recovers a second reading wherever the source trie
records one, which fixed 5 of the 7 words the gold set caught being served at
chance. The other two are the residue this handles: `мати` has one reading in
the trie and `радій` is absent from it entirely, so no amount of reading the
trie will ever produce `мати́` or `раді́й`. The evidence has to come from
somewhere else.

The method is the gap filler's, applied to a different question. Two
deployments on different resources are asked independently whether a form has
more than one stress, and only an exact agreement on the same set of stressed
forms is written. A disagreement is *not* resolved by a third opinion or a
tiebreak — two competent models disagreeing about whether a word is a homograph
is evidence that it is a marginal case, and marginal cases belong in a review
file, not in the lexicon.

Candidates are drawn frequency-first from a corpus, restricted to forms the
lexicon answers with exactly one stress. Asking about a form nobody writes buys
nothing, and asking about a form the lexicon already reports as ambiguous
answers a question that is not open.

Confidence 0.55 sits below the trie recovery's 0.60 and far below the
Wiktionary parser's 0.95: this is the weakest evidence in the lexicon, and the
ordering has to say so.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql
from ukrainian_word_stress.stressify_ import (
    _load_dictionary,
    _parse_dictionary_value,
    _trie_value,
)
from ukstress.database import build_lookup_projection
from ukstress.normalizer import (
    NormalizationError,
    canonical_stressed_form,
    lookup_key,
    stress_signature,
    validate_stress,
)

from ukstress_ml.annotate import (
    FLASH_DEPLOYMENT,
    LABEL_DEPLOYMENT,
    VERIFY_DEPLOYMENT,
    Usage,
    load_config,
)

WORD = re.compile(r"[^\W\d_]+(?:['’ʼ-][^\W\d_]+)*", re.UNICODE)
CYRILLIC = re.compile(r"[а-щьюяєіїґА-ЩЬЮЯЄІЇҐ]")
VOWELS = frozenset("аеєиіїоуюя")

DEPLOYMENTS = {"label": LABEL_DEPLOYMENT, "verify": VERIFY_DEPLOYMENT, "flash": FLASH_DEPLOYMENT}

MAX_OUTPUT_TOKENS = 700

SYSTEM = ("Ти — український фонетист і лексикограф. Визначаєш, чи слово є "
          "омографом. Відповідаєш лише валідним JSON.")


def build_prompt(form: str, known: str, examples: list[str]) -> str:
    usage = "\n".join(f"  - {s}" for s in examples[:4]) or "  (немає прикладів)"
    return f"""Слово: {form}
Наголос, який знає наш словник: {known}

Приклади вживання з реального тексту:
{usage}

Питання: чи має слово «{form}» БІЛЬШЕ НІЖ ОДИН наголос —
тобто чи це омограф, у якого від наголосу змінюється значення або
граматична форма?

Відповідай "homograph": true лише тоді, коли обидва наголоси справді
вживаються в сучасній українській мові. Варіативний наголос в одному й
тому самому значенні (коли обидва варіанти означають те саме і є просто
допустимими вимовами) — це НЕ омограф, для нього постав false.

Якщо так, перелічи ВСІ наголоси: слово зі знаком наголосу (U+0301 після
наголошеної голосної) і стисле тлумачення цього значення.

Формат (лише JSON):
{{"homograph": true, "readings": [{{"stressed": "сло́во", "meaning": "..."}}]}}"""


def parse_response(form: str, content: str) -> dict[str, str] | None:
    """Return {signature: stressed} for a claimed homograph, else None."""
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(content[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not payload.get("homograph"):
        return {}
    readings: dict[str, str] = {}
    for item in payload.get("readings", []):
        stressed = canonical_stressed_form(str(item.get("stressed", "")))
        if lookup_key(stressed) != form:
            # A reading of some other word cannot be a reading of this one; the
            # models occasionally answer about the lemma they thought of.
            continue
        if validate_stress(stressed).classification == "rejected":
            continue
        try:
            signature = stress_signature(stressed)
        except NormalizationError:
            continue
        if "|" in signature:
            continue
        readings.setdefault(signature, stressed)
    return readings if len(readings) > 1 else {}


def build_candidates(args: argparse.Namespace) -> None:
    """Rank corpus forms the lexicon answers with exactly one stress."""
    counts: collections.Counter[str] = collections.Counter()
    examples: dict[str, list[str]] = collections.defaultdict(list)
    opener = gzip.open if args.corpus.suffix == ".gz" else open
    scanned = 0
    with opener(args.corpus, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if scanned >= args.words:
                break
            for match in WORD.finditer(line):
                token = match.group()
                if not CYRILLIC.search(token):
                    continue
                scanned += 1
                if sum(1 for c in token.lower() if c in VOWELS) <= 1:
                    continue
                key = lookup_key(token)
                counts[key] += 1
                if len(examples[key]) < 4 and 20 < len(line) < 200:
                    examples[key].append(line.strip())

    with psycopg.connect(args.database_url) as conn:
        active = int(conn.execute(
            "SELECT dataset_id FROM active_dataset WHERE singleton").fetchone()[0])
        datasets = args.dataset or [active, 3, 4]
        keys = [form for form, _ in counts.most_common(args.scan_top)]
        rows = conn.execute(
            "SELECT form_normalized, count(DISTINCT stress_signature), min(stressed_form) "
            "FROM stress_lookup WHERE dataset_id = ANY(%s) AND form_normalized = ANY(%s) "
            "GROUP BY form_normalized", (datasets, keys)).fetchall()
    lexicon = {str(form): (int(n), str(stressed)) for form, n, stressed in rows}

    trie = _load_dictionary()
    candidates: list[dict[str, Any]] = []
    tally: collections.Counter[str] = collections.Counter()
    for form, count in counts.most_common(args.scan_top):
        entry = lexicon.get(form)
        if entry is None:
            tally["not_in_lexicon"] += 1
            continue
        signatures, stressed = entry
        if signatures > 1:
            tally["already_ambiguous"] += 1
            continue
        values = _trie_value(trie, form)
        if values is not None and len({tuple(a) for _, a in _parse_dictionary_value(values[0])}) > 1:
            # The trie already knows; `run_recover_readings.py` owns these and
            # needs no model opinion.
            tally["trie_recoverable"] += 1
            continue
        tally["candidate"] += 1
        candidates.append({"form": form, "count": count, "known": stressed,
                           "examples": examples.get(form, [])})

    args.candidates.parent.mkdir(parents=True, exist_ok=True)
    args.candidates.write_text(json.dumps(candidates[:args.limit or None],
                                          ensure_ascii=False, indent=1), encoding="utf-8")
    for key, value in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"{key:<20}{value:>8,}")
    print(f"\nwords scanned: {scanned:,}   candidates: {len(candidates):,}"
          f"  -> {args.candidates}")


def ask_all(candidates: list[dict[str, Any]], name: str, workers: int,
            raw_path: Path) -> None:
    """Ask one deployment about every candidate, appending raw responses."""
    done: set[str] = set()
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("content"):
                done.add(str(record["form"]))
    todo = [c for c in candidates if c["form"] not in done]
    if not todo:
        return

    config = load_config(Path(".env"))
    deployment = DEPLOYMENTS[name]
    client = deployment.client(config)
    model = deployment.resolved_model(config)
    limits: dict[str, Any] = {"temperature": 0.0, "max_tokens": MAX_OUTPUT_TOKENS}
    limits_lock = threading.Lock()
    usage = Usage()
    lock = threading.Lock()
    print(f"{model}: asking about {len(todo):,} forms", flush=True)

    def ask(candidate: dict[str, Any]) -> dict[str, Any]:
        prompt = build_prompt(candidate["form"], candidate["known"], candidate["examples"])
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt}]
        for attempt in range(6):
            try:
                response = client.chat.completions.create(
                    model=model, messages=messages, **limits)
            except Exception as error:
                text = str(error)
                # `limits` is shared across the pool, so the retry must not
                # depend on *this* thread being the one that removed the
                # offending key: every sibling call already in flight would
                # otherwise raise instead of retrying.
                if "nsupported parameter" in text or "unsupported_parameter" in text:
                    with limits_lock:
                        if "max_tokens" in text and "max_tokens" in limits:
                            limits.pop("max_tokens")
                            limits["max_completion_tokens"] = MAX_OUTPUT_TOKENS
                        elif "temperature" in text and "temperature" in limits:
                            limits.pop("temperature")
                    if attempt < 5:
                        continue
                    raise
                if "429" not in text or attempt == 5:
                    raise
                time.sleep(2 ** attempt + random.random())
                continue
            if response.usage is not None:
                usage.add(response.usage.prompt_tokens, response.usage.completion_tokens)
            return {"form": candidate["form"], "content": response.choices[0].message.content}
        raise RuntimeError("unreachable")

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    finished = 0
    with raw_path.open("a", encoding="utf-8") as raw, \
            ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(ask, c): c for c in todo}
        for future in as_completed(futures):
            candidate = futures[future]
            try:
                record = future.result()
            except Exception as error:  # noqa: BLE001
                usage.fail()
                record = {"form": candidate["form"], "content": None, "error": str(error)}
            with lock:
                raw.write(json.dumps(record, ensure_ascii=False) + "\n")
                raw.flush()
                finished += 1
                if finished % 100 == 0 or finished == len(todo):
                    print(f"  {model}: {finished:,}/{len(todo):,}", flush=True)
    print(json.dumps({model: usage.snapshot()}, indent=2))


def read_verdicts(form_keys: set[str], raw_path: Path) -> dict[str, dict[str, str]]:
    verdicts: dict[str, dict[str, str]] = {}
    if not raw_path.exists():
        return verdicts
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        form = str(record.get("form", ""))
        if form not in form_keys or not record.get("content"):
            continue
        parsed = parse_response(form, str(record["content"]))
        if parsed is not None:
            verdicts[form] = parsed
    return verdicts


def write_dataset(dsn: str, dataset_key: str, agreed: list[dict[str, Any]]) -> int:
    written = 0
    with psycopg.connect(dsn) as conn, conn.transaction():
        row = conn.execute("SELECT id FROM import_run WHERE dataset_key=%s",
                           (dataset_key,)).fetchone()
        if row:
            dataset_id = int(row[0])
            # Delete children first, each by `dataset_id`. Deleting the lexemes
            # and letting the foreign keys cascade looks equivalent and is not:
            # every child table indexes `(dataset_id, ...)`, so a cascade
            # lookup by `lexeme_id` alone cannot use any of them and falls back
            # to a sequential scan of a 2.9M-row table *per lexeme*. Rewriting
            # six thousand rows that way ran for over half an hour; this is
            # four index scans.
            for table in ("source_ref", "stress_variant", "word_form", "lexeme"):
                conn.execute(
                    sql.SQL("DELETE FROM {} WHERE dataset_id = %s").format(
                        sql.Identifier(table)), (dataset_id,))
        else:
            dataset_id = int(conn.execute(
                """INSERT INTO import_run (dataset_key, status, dump_url, dump_sha256,
                       parser_version, normalization_version, schema_version, statistics)
                   VALUES (%s,'building','llm://homograph-audit',%s,
                           'homograph-audit-1','1','007','{}')
                   RETURNING id""", (dataset_key, "0" * 64)).fetchone()[0])
        for record in agreed:
            form = record["form"]
            for signature, stressed in sorted(record["missing"].items()):
                key = f"{form}:{signature}"
                lexeme_id = int(conn.execute(
                    """INSERT INTO lexeme (dataset_id, lemma, lemma_normalized, stressed_lemma,
                           sense_key, source_title, source_section, confidence, natural_key)
                       VALUES (%s,%s,%s,%s,%s,%s,'HomographAudit',0.55,%s) RETURNING id""",
                    (dataset_id, form, form, stressed, signature, form,
                     f"audit:{dataset_key}:{key}")).fetchone()[0])
                word_form_id = int(conn.execute(
                    """INSERT INTO word_form (dataset_id, lexeme_id, form, form_normalized,
                           morphology_key, is_lemma, is_variant, confidence, source_rank,
                           natural_key)
                       VALUES (%s,%s,%s,%s,'',true,true,0.55,45,%s) RETURNING id""",
                    (dataset_id, lexeme_id, stressed, form,
                     f"audit:wf:{key}")).fetchone()[0])
                conn.execute(
                    """INSERT INTO stress_variant (dataset_id, word_form_id, stressed_form,
                           stress_signature, variant_type, confidence, natural_key)
                       VALUES (%s,%s,%s,%s,'primary',0.55,%s)""",
                    (dataset_id, word_form_id, stressed, signature, f"audit:sv:{key}"))
                conn.execute(
                    """INSERT INTO source_ref (dataset_id, lexeme_id, word_form_id, page_title,
                           source_kind, source_fragment, natural_key)
                       VALUES (%s,%s,%s,%s,'llm_homograph_audit',%s,%s)""",
                    (dataset_id, lexeme_id, word_form_id, form, stressed, f"audit:sr:{key}"))
                written += 1
        build_lookup_projection(conn, dataset_id)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-candidates", action="store_true")
    parser.add_argument("--corpus", type=Path,
                        default=Path("data/corpus/opensubtitles-uk.txt.gz"))
    parser.add_argument("--words", type=int, default=3_000_000)
    parser.add_argument("--scan-top", type=int, default=20_000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--candidates", type=Path,
                        default=Path("output/ml/audit_candidates.json"))
    parser.add_argument("--database-url",
                        default="postgresql://ukstress_owner:ukstress_owner@localhost:5432/ukstress")
    parser.add_argument("--dataset", type=int, action="append", default=None)
    parser.add_argument("--deployments", default="label,verify")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("output/ml/homograph_audit"))
    parser.add_argument("--dataset-key", default="homograph-audit-v1")
    parser.add_argument("--write-db", action="store_true")
    args = parser.parse_args()

    if args.build_candidates:
        build_candidates(args)
        return

    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    if args.limit:
        candidates = candidates[:args.limit]
    names = [n.strip() for n in args.deployments.split(",") if n.strip()]
    if len(names) < 2:
        raise SystemExit("two deployments are required; one model cannot confirm itself")
    args.out.mkdir(parents=True, exist_ok=True)

    raw_paths = {name: args.out / f"raw.{name}.jsonl" for name in names}
    for name in names:
        ask_all(candidates, name, args.workers, raw_paths[name])

    keys = {c["form"] for c in candidates}
    verdicts = {name: read_verdicts(keys, raw_paths[name]) for name in names}
    known = {c["form"]: c for c in candidates}

    agreed: list[dict[str, Any]] = []
    contested: list[dict[str, Any]] = []
    tally: collections.Counter[str] = collections.Counter()
    for form in sorted(keys):
        answers = [verdicts[name].get(form) for name in names]
        if any(a is None for a in answers):
            tally["unanswered"] += 1
            continue
        if all(not a for a in answers):
            tally["agreed_not_homograph"] += 1
            continue
        if len({tuple(sorted(a)) for a in answers}) > 1:
            tally["contested"] += 1
            contested.append({"form": form, "count": known[form]["count"],
                              "answers": {n: verdicts[n][form] for n in names}})
            continue
        combined = answers[0]
        existing = stress_signature(canonical_stressed_form(known[form]["known"]))
        missing = {s: v for s, v in combined.items() if s != existing}
        if not missing:
            tally["agreed_no_new_reading"] += 1
            continue
        tally["agreed_homograph"] += 1
        agreed.append({"form": form, "count": known[form]["count"],
                       "known": known[form]["known"], "missing": missing})

    agreed.sort(key=lambda r: -r["count"])
    contested.sort(key=lambda r: -r["count"])
    (args.out / "agreed.json").write_text(
        json.dumps(agreed, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out / "contested.json").write_text(
        json.dumps(contested, ensure_ascii=False, indent=1), encoding="utf-8")

    for key, value in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"{key:<24}{value:>8,}")
    print(f"\nagreed homographs: {len(agreed):,}  -> {args.out}/agreed.json")
    print(f"contested        : {len(contested):,}  -> {args.out}/contested.json")

    if args.write_db and agreed:
        written = write_dataset(args.database_url, args.dataset_key, agreed)
        print(f"wrote {written:,} readings to dataset '{args.dataset_key}' (unpublished)")


if __name__ == "__main__":
    main()
