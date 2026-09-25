"""Run real text through the full pipeline and log what it could not stress.

Everything measured so far has been on the project's own corpus, which is drawn
from the forms the pipeline already knows about. That answers "is the model
right when asked", not "how much of ordinary Ukrainian gets stressed at all".
Running an outside corpus answers the second question, and the interesting
output is the failures.

Each token lands in exactly one bucket:

    dictionary          one stress in the lexicon, served directly
    model               ambiguous, the model answered above threshold
    model_abstained     ambiguous, below threshold -> dictionary default
    outside_coverage    ambiguous but absent from the model manifest
    not_in_lexicon      no entry at all -- the real coverage gap
    skipped             non-Cyrillic or single-letter token

`not_in_lexicon` is written out with frequencies, because the highest-frequency
misses are worth more than the long tail.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import time
from pathlib import Path
from typing import Any

import psycopg
from ukstress.normalizer import lookup_key

from ukstress_ml.morphology import MorphologyTier
from ukstress_ml.morphology import resolve as morph_resolve
from ukstress_ml.serving import ContextualStressModel, Target

WORD = re.compile(r"[^\W\d_]+(?:['’ʼ-][^\W\d_]+)*", re.UNICODE)
CYRILLIC = re.compile(r"[а-щьюяєіїґА-ЩЬЮЯЄІЇҐ]")
VOWELS = frozenset("аеєиіїоуюя")


def is_monosyllabic(word: str) -> bool:
    """Ukrainian does not mark stress on single-syllable words, so their
    absence from a stress lexicon is correct rather than a coverage gap.
    Counting them as misses put the failure rate at 30.2% when the real
    figure is 6.3%."""
    return sum(1 for c in word.lower() if c in VOWELS) <= 1


def read_lines(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        yield from handle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path,
                        default=Path("data/corpus/opensubtitles-uk.txt.gz"))
    parser.add_argument("--words", type=int, default=200_000)
    parser.add_argument("--database-url",
                        default="postgresql://ukstress_owner:ukstress_owner@localhost:5432/ukstress")
    parser.add_argument("--model", type=Path, default=Path("output/ml/models/v8-generated2"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("output/ml/serving_manifest_expanded.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-sentences", type=int, default=64)
    parser.add_argument("--extra-dataset", type=int, action="append", default=[],
                        help="also consult these dataset ids (e.g. an unpublished gap fill)")
    parser.add_argument("--no-morphology", action="store_true",
                        help="skip tier 2 to measure its contribution")
    parser.add_argument("--out", type=Path, default=Path("output/ml/corpus_pilot"))
    args = parser.parse_args()

    model = ContextualStressModel(args.model, args.manifest, backend="torch", device=args.device)
    morphology = None if args.no_morphology else MorphologyTier()
    reading_cache: dict[str, list] = {}
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["forms"]
    with psycopg.connect(args.database_url) as conn:
        row = conn.execute("SELECT dataset_id FROM active_dataset WHERE singleton").fetchone()
        dataset_id = int(row[0]) if row else 0

    tally: collections.Counter[str] = collections.Counter()
    missing: collections.Counter[str] = collections.Counter()
    abstained: collections.Counter[str] = collections.Counter()
    uncovered: collections.Counter[str] = collections.Counter()
    seen_words = 0
    started = time.time()

    batch: list[tuple[str, list[tuple[str, int, int]]]] = []

    def flush(conn: Any) -> None:
        nonlocal batch
        if not batch:
            return
        keys = sorted({lookup_key(w) for _, spans in batch for w, _, _ in spans})
        rows = conn.execute(
            """
            SELECT form_normalized, stress_signature, min(stressed_form)
            FROM stress_lookup WHERE dataset_id = ANY(%s) AND form_normalized = ANY(%s)
            GROUP BY form_normalized, stress_signature
            """,
            ([dataset_id, *args.extra_dataset], keys),
        ).fetchall()
        found: dict[str, dict[str, str]] = collections.defaultdict(dict)
        for form, signature, stressed in rows:
            found[str(form)][str(signature)] = str(stressed)

        # Only a token that is ambiguous *and* outside the model manifest can
        # reach tier 2 (~6% of tokens), and Stanza's per-call overhead dominates
        # at sentence granularity, so collect the sentences that need it and
        # parse them in one batched call.
        parses: dict[int, dict] = {}
        if morphology is not None:
            need: list[str] = []
            for sentence, spans in batch:
                for word, _, _ in spans:
                    key = lookup_key(word)
                    variants = found.get(key)
                    if not variants or len(variants) == 1:
                        continue
                    entry = manifest.get(key)
                    if entry is None or sorted(entry["signatures"]) != sorted(variants):
                        need.append(sentence)
                        break
            if need:
                try:
                    for sentence, spans_map in zip(need, morphology.parse_batch(need), strict=True):
                        parses[id(sentence)] = spans_map
                except Exception as error:  # noqa: BLE001 - never stop the run
                    print(f"  parse batch failed ({error}); tier 2 skipped for "
                          f"{len(need)} sentences", flush=True)

        targets: list[Target] = []
        target_words: list[str] = []
        for sentence, spans in batch:
            for word, start, end in spans:
                key = lookup_key(word)
                variants = found.get(key)
                if not variants:
                    tally["not_in_lexicon"] += 1
                    missing[key] += 1
                    continue
                if len(variants) == 1:
                    tally["dictionary"] += 1
                    continue
                entry = manifest.get(key)
                if entry is None or sorted(entry["signatures"]) != sorted(variants):
                    # Tier 2: the readings may be separated by their tags, in
                    # which case the parse decides and no model is needed.
                    if morphology is not None:
                        if key not in reading_cache:
                            reading_cache[key] = morphology.readings(key)
                        parse = parses.get(id(sentence), {}).get((start, end))
                        if parse and reading_cache[key]:
                            got = morph_resolve(reading_cache[key], parse[0], parse[1])
                            if got is not None:
                                tally["morphology"] += 1
                                continue
                    tally["outside_coverage"] += 1
                    uncovered[key] += 1
                    continue
                targets.append(Target(sentence, start, end, key, tuple(sorted(variants))))
                target_words.append(key)
        if targets:
            for decision, key in zip(model.resolve(targets), target_words, strict=True):
                if decision["status"] == "selected":
                    tally["model"] += 1
                else:
                    tally["model_abstained"] += 1
                    abstained[key] += 1
        batch = []

    with psycopg.connect(args.database_url) as conn:
        for line in read_lines(args.corpus):
            sentence = line.strip()
            if not sentence or not CYRILLIC.search(sentence):
                continue
            spans = []
            for match in WORD.finditer(sentence):
                word = match.group()
                if len(word) < 2 or not CYRILLIC.search(word):
                    tally["skipped"] += 1
                    continue
                if is_monosyllabic(word):
                    tally["monosyllabic"] += 1
                    continue
                spans.append((word, match.start(), match.end()))
            if not spans:
                continue
            seen_words += len(spans)
            batch.append((sentence, spans))
            if len(batch) >= args.batch_sentences:
                flush(conn)
                if seen_words % 20000 < args.batch_sentences * 12:
                    print(f"  {seen_words:,} words  {time.time()-started:.0f}s  {dict(tally)}",
                          flush=True)
            if seen_words >= args.words:
                break
        flush(conn)

    args.out.mkdir(parents=True, exist_ok=True)
    total = sum(v for k, v in tally.items() if k not in {"skipped", "monosyllabic"})
    report = {
        "corpus": str(args.corpus),
        "words_scanned": seen_words,
        "tokens_classified": total,
        "seconds": round(time.time() - started, 1),
        "tally": dict(tally),
        "share": {k: round(v / total, 4) for k, v in tally.items()
                  if k not in {"skipped", "monosyllabic"}},
        "distinct_missing": len(missing),
        "distinct_uncovered": len(uncovered),
    }
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, counter in (("not_in_lexicon", missing), ("outside_coverage", uncovered),
                          ("model_abstained", abstained)):
        with (args.out / f"{name}.tsv").open("w", encoding="utf-8") as handle:
            handle.write("count\tform\n")
            for form, count in counter.most_common():
                handle.write(f"{count}\t{form}\n")

    print(f"\nwords scanned      : {seen_words:,}   ({report['seconds']}s)")
    print(f"{'bucket':<20}{'tokens':>10}{'share':>9}")
    for key, value in tally.most_common():
        if key in {"skipped", "monosyllabic"}:
            continue
        print(f"{key:<20}{value:>10,}{value/total:>8.1%}")
    print(f"\ndistinct forms not in lexicon : {len(missing):,}")
    print(f"distinct forms outside model  : {len(uncovered):,}")
    print(f"written to {args.out}/")


if __name__ == "__main__":
    main()
