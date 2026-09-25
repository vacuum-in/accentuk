"""Mine training sentences for under-represented senses from a plain-text corpus.

Wikipedia is an encyclopedia and starves exactly the senses this model needs.
78% of mined groups came back single-sense, and every remaining model error sits
on a group whose minority sense never appeared in training: `правило` has 96
rows and scores 0.708 because 95 of them are one sense. Generating the missing
sentences works but is bounded by an API quota that admits roughly one request
in flight, which makes it a multi-day job.

Mining a *different* corpus is bounded by CPU instead. Malyuk (UberText 2.0 +
OSCAR + Ukrainian News, 38.9M documents) carries fiction, speech, news and court
text, which is where imperatives, colloquial senses and the non-encyclopedic
half of a homograph pair actually occur.

This mines candidates only — a mined sentence has no sense label. Labelling is
a separate, cheaper stage: a morphological parse settles the `grammatical`
triage class for free, and one batched model call can label many sentences,
against roughly seven kept sentences per generation call.

Targets are the deficient (form, sense) pairs, so a group that is already
balanced costs nothing here.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ukstress_ml import ambiguity, mine


def read_parquet(paths: list[Path], text_column: str = "text") -> Iterator[tuple[str, str]]:
    """Stream a text column out of parquet shards, skipping unreadable ones.

    A truncated download is a readable *file* and an unreadable *parquet*: the
    footer carries the schema, so a cut-short shard raises only when opened.
    Skipping it and saying so is better than aborting a multi-hour mine over one
    bad shard out of thirty.
    """
    import pyarrow.parquet as pq

    for path in sorted(paths):
        try:
            parquet = pq.ParquetFile(path)
        except Exception as error:  # noqa: BLE001
            print(f"  skipping unreadable shard {path.name}: {str(error)[:60]}", flush=True)
            continue
        for batch in parquet.iter_batches(batch_size=2048, columns=[text_column]):
            column = batch.column(0)
            for index in range(len(column)):
                value = column[index].as_py()
                if value:
                    yield f"{path.stem}:{index}", value


def read_text(paths: list[Path]) -> Iterator[tuple[str, str]]:
    """Each line is a document; enough for subtitle and one-sentence-per-line dumps."""
    for path in sorted(paths):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle):
                line = line.strip()
                if line:
                    yield f"{path.stem}:{number}", line


def deficient_forms(inventory: Path, manifest: Path, corpora: list[Path],
                    floor: int) -> list[ambiguity.AmbiguousForm]:
    """Forms whose group has a sense the training corpus under-represents."""
    servable = {entry["group_id"]
                for entry in json.loads(manifest.read_text(encoding="utf-8"))["forms"].values()}
    have: dict[int, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for path in corpora:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("rows", [])
        for row in rows:
            sense = row.get("gold_sense") or row.get("sense_id")
            if sense is not None:
                have[int(row["group_id"])][str(sense)] += 1

    wanted: list[ambiguity.AmbiguousForm] = []
    for form in ambiguity.load(inventory):
        if form.group_id not in servable:
            continue
        if len({c.signature for c in form.candidates}) < 2:
            continue
        counts = have.get(form.group_id, collections.Counter())
        if any(counts.get(c.sense_id, 0) < floor for c in form.candidates):
            wanted.append(form)
    return wanted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/malyuk"))
    parser.add_argument("--glob", default="*.parquet")
    parser.add_argument("--format", choices=("parquet", "text"), default="parquet")
    parser.add_argument("--inventory", type=Path,
                        default=Path("output/ml/ambiguous_forms_glossed.jsonl"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("output/ml/serving_manifest_expanded.json"))
    # See merge_corpus.py: an append action with a default list cannot be
    # overridden, only added to.
    parser.add_argument("--corpus-rows", type=Path, action="append", default=None)
    parser.add_argument("--floor", type=int, default=12)
    parser.add_argument("--cap-per-form", type=int, default=60)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--corpus-name", default=None,
                        help="provenance tag for the rows; defaults to the corpus "
                             "directory name")
    parser.add_argument("--licence", default=None)
    parser.add_argument("--out", type=Path, default=Path("output/ml/mined_malyuk"))
    args = parser.parse_args()

    corpus_rows = args.corpus_rows or [Path("output/ml/silver_plus_generated_v2.json")]
    forms = deficient_forms(args.inventory, args.manifest, corpus_rows, args.floor)
    paths = sorted(args.corpus_dir.glob(args.glob))
    paths = [p for p in paths if p.stat().st_size > 1_000_000]
    print(f"deficient forms: {len(forms):,}   shards: {len(paths)}", flush=True)
    if not paths:
        raise SystemExit("no corpus shards found")

    documents = read_parquet(paths) if args.format == "parquet" else read_text(paths)
    reservoirs, stats = mine.mine_documents(
        documents, forms, cap_per_form=args.cap_per_form,
        max_documents=args.max_documents)

    corpus_name = args.corpus_name or args.corpus_dir.name
    licence = args.licence or f"see source corpus: {corpus_name}"
    args.out.mkdir(parents=True, exist_ok=True)
    rows = 0
    with (args.out / "candidates.jsonl").open("w", encoding="utf-8") as handle:
        for key in sorted(reservoirs):
            for item in reservoirs[key].items:
                # Hardcoding the corpus name tagged 30,462 subtitle-mined
                # sentences as Malyuk — a wrong licence on real text. The tag
                # follows the source directory unless the caller names it.
                handle.write(json.dumps({**item, "corpus": corpus_name,
                                         "licence": licence,
                                         "source_tier": "mined_corpus"},
                                        ensure_ascii=False) + "\n")
                rows += 1
    report: dict[str, Any] = {
        "documents": stats.pages, "prefiltered": stats.pages_prefiltered,
        "sentences_examined": stats.sentences_examined, "hits": stats.hits,
        "kept": stats.kept, "rows_written": rows,
        "forms_with_any": len(reservoirs), "forms_targeted": len(forms),
        "rejected": {"multi_form": stats.rejected_multi_form,
                     "stress_marks": stats.rejected_stress_marks,
                     "length": stats.rejected_length,
                     "script_ratio": stats.rejected_script_ratio},
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\n{rows:,} candidate sentences for {len(reservoirs):,} of {len(forms):,} forms"
          f"  -> {args.out}/candidates.jsonl")


if __name__ == "__main__":
    main()
