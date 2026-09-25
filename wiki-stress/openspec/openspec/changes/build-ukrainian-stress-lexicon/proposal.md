# Proposal: Build Ukrainian Stress Lexicon

## Intent

Create a production-ready pipeline and lookup service that transform the
Ukrainian Wiktionary article dump into a high-performance PostgreSQL lexicon of
Ukrainian lemmas, stressed lemmas, inflected forms, stressed forms,
grammatical metadata, ambiguity, confidence, and source provenance.

The implementation must use Python for the complete offline data pipeline and
Go only for the read-only HTTP lookup API.

## Problem

Ukrainian Wikipedia prose normally omits lexical stress, while Ukrainian
Wiktionary contains stressed headwords and inflection structures in varied and
sometimes inconsistent MediaWiki markup. A useful TTS or NLP dictionary must
therefore:

- isolate Ukrainian entries from multilingual pages;
- understand supported headword, inflection-template, and table conventions;
- preserve combining acute accents and valid alternative stresses;
- associate forms with lexemes and grammatical features;
- reject or quarantine malformed extraction results;
- load data efficiently without per-row inserts;
- expose low-latency exact and batch lookup;
- remain reproducible across dump and parser versions.

## Scope

### In scope

- Official Ukrainian Wiktionary `pages-articles` XML dump as the initial source.
- Streaming `.bz2` processing with bounded memory.
- Ukrainian lemma and inflected-form extraction.
- Explicit support for multiple lexemes, homonyms, and stress variants.
- Unicode-safe canonicalization and validation.
- PostgreSQL normalized source tables and a read-optimized lookup projection.
- Versioned import runs and atomic dataset publication.
- Python CLI for download, parse, validate, import, publish, export, statistics,
  and ETL benchmarks.
- Go HTTP API for exact, best-candidate, lemma, and batch lookup.
- Docker Compose, tests, generated reports, API schema, and measured benchmarks.

### Out of scope

- A web UI.
- Editing Wiktionary or sending data back to Wikimedia.
- Generic MediaWiki rendering or Lua/template execution.
- Machine-learning stress prediction for words absent from the lexicon.
- Contextual disambiguation of homographs inside sentences.
- Morphological generation of forms that are not explicitly present in or
  deterministically encoded by supported Wiktionary structures.
- Fuzzy lookup in the initial release.
- Any Go-based ETL, parsing, normalization-authoring, migrations, or import.
- Any Python-based public HTTP lookup service.

## Proposed capabilities

1. **Dump ingestion** — Python streams and parses the dump safely.
2. **Stress lexicon** — Python produces validated normalized lexemes, forms,
   variants, grammatical tags, source references, and quality reports.
3. **Dataset publication** — Python bulk-loads a versioned dataset and switches
   the active dataset atomically.
4. **Lookup API** — Go performs read-only PostgreSQL exact and batch lookups.
5. **Operations** — Both components expose reproducible commands, tests,
   observability, and measured performance reports.

## Key decisions

- PostgreSQL is the authoritative online store.
- Normalized relational tables preserve linguistic structure and provenance.
- A flattened `stress_lookup` projection serves latency-sensitive API queries.
- Python uses PostgreSQL `COPY` through `psycopg` rather than row-by-row inserts.
- Go uses `pgx/v5` with a bounded `pgxpool`.
- The API returns all valid candidates by default and never hides ambiguity.
- The API performs only conservative request normalization required to match the
  stored canonical key; canonical linguistic rules are defined and tested in
  Python and exported as database metadata/test vectors.
- Dataset publication is versioned and reversible.

## Assumptions

- The initial deployment has one PostgreSQL primary and one API deployment,
  although the API must be horizontally scalable.
- A typical build host has at least 8 CPU cores, 16 GB RAM, and SSD storage.
- The dump format and Wiktionary templates can evolve; unsupported structures
  must be measurable and extensible through handlers.
- Exact throughput and latency targets are acceptance targets, not guaranteed
  results. The implementation must report actual measurements.

## Impact

### New components

- `etl/`: Python package and CLI.
- `api/`: Go HTTP lookup service.
- `db/`: SQL migrations owned and executed by Python tooling.
- `reports/`: generated quality and benchmark artifacts.
- `openspec/`: project and capability specifications.

### Data impact

The first import creates new PostgreSQL schemas and tables. Later imports create
a new immutable dataset version, validate it, publish it atomically, and retain
a configurable number of previous versions for rollback.

### API impact

The change introduces a versioned `/v1` HTTP contract. No existing public API is
modified.

## Risks

- Wiktionary markup inconsistency can reduce extraction coverage.
- A broad fallback regex can create false positives.
- Unicode mishandling can corrupt lookup keys or stress positions.
- Multiple lexical analyses may produce large ambiguous result sets.
- Keeping many dataset versions may increase storage use.
- A flattened lookup table can duplicate data and increase build time.
- Request-side normalization implemented independently in Go can drift from the
  Python canonicalizer.

## Mitigations

- Prefer structured handlers and assign lower confidence to fallbacks.
- Maintain real-entry fixtures and golden extraction tests.
- Store parser version, dump checksum, and source revision metadata.
- Generate unhandled-template and rejected-form reports.
- Export normalization conformance vectors from Python and run them in Go tests.
- Cap API result count while returning an explicit truncation indicator.
- Benchmark normalized joins against the flattened projection before retaining
  unnecessary indexes.
- Retain and activate prior dataset versions for rollback.

## Rollback

- Application rollback: deploy the previous Go API image.
- Data rollback: atomically set `active_dataset.dataset_id` to a previously
  published dataset.
- Schema rollback: migrations are forward-only in production; destructive
  schema rollback is replaced by application/data rollback and a corrective
  migration.
