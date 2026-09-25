# Design: Ukrainian Stress Lexicon

## 1. System overview

```text
Official Ukrainian Wiktionary dump
                 |
                 v
+----------------------------------------------+
| Python ETL                                   |
| download -> stream -> parse -> normalize     |
| -> validate -> deduplicate -> stage -> COPY  |
| -> build lookup projection -> publish        |
+----------------------------------------------+
                 |
                 v
          PostgreSQL
  normalized tables + stress_lookup
                 |
                 v
+----------------------------------------------+
| Go read-only HTTP API                        |
| validate request -> normalize lookup key     |
| -> prepared PostgreSQL query -> JSON         |
+----------------------------------------------+
```

There is no runtime call from Go to Python. There is no HTTP service in the
Python component. PostgreSQL is the integration boundary.

## 2. Repository layout

```text
.
├── api/
│   ├── cmd/stress-api/main.go
│   ├── internal/config/
│   ├── internal/httpapi/
│   ├── internal/lookup/
│   ├── internal/postgres/
│   ├── internal/observability/
│   ├── migrations_test/
│   ├── go.mod
│   └── Dockerfile
├── etl/
│   ├── pyproject.toml
│   ├── src/ukstress/
│   │   ├── cli.py
│   │   ├── download.py
│   │   ├── dump_reader.py
│   │   ├── sections.py
│   │   ├── wikicode.py
│   │   ├── handlers/
│   │   ├── normalize.py
│   │   ├── stress.py
│   │   ├── grammar.py
│   │   ├── validate.py
│   │   ├── deduplicate.py
│   │   ├── staging.py
│   │   ├── database.py
│   │   ├── publish.py
│   │   ├── reports.py
│   │   └── benchmark.py
│   ├── tests/
│   └── Dockerfile
├── db/
│   ├── migrations/
│   └── queries/
├── deploy/
│   └── docker-compose.yml
├── reports/
├── Makefile
└── openspec/
```

## 3. Technology choices

### Python ETL

- Python 3.12 or newer.
- `bz2` plus `lxml.etree.iterparse` for bounded-memory XML processing.
- `mwparserfromhell` for MediaWiki templates and links.
- Dedicated table parser for supported wiki-table conventions.
- `unicodedata` and explicit Ukrainian character tables for normalization.
- `zstandard` for compressed staging artifacts.
- `psycopg` 3 for PostgreSQL `COPY`, transactions, and migrations.
- `pydantic` or dataclasses for typed intermediate records.
- `pytest`, `hypothesis`, `ruff`, and `mypy`.

Python owns all schema migrations because schema changes are part of the data
pipeline lifecycle, not the lookup service lifecycle.

### Go lookup API

- Go 1.24 or newer.
- Standard `net/http` routing or a minimal router; do not introduce a full web
  framework without a measured need.
- `github.com/jackc/pgx/v5/pgxpool`.
- JSON encoding with strict request-size and batch-size limits.
- Prometheus-compatible metrics endpoint.
- OpenAPI document stored in the repository and contract-tested.

## 4. Python pipeline

### 4.1 Download

`ukstress download`:

- resolves a configured dump URL;
- downloads to a temporary file;
- verifies expected size when available;
- calculates SHA-256;
- atomically renames the completed file;
- records URL, filename, checksum, and acquisition time.

The pipeline may also accept a local dump and explicit checksum.

### 4.2 Streaming XML reader

The XML reader:

- opens `.bz2` directly;
- iterates page elements;
- reads main namespace pages;
- extracts page ID, title, redirect target, revision ID, revision timestamp,
  and wikitext;
- clears processed XML elements and previous siblings;
- rejects pages above a configurable maximum uncompressed size;
- emits bounded records through a bounded queue.

One process streams XML. Parsing workers may process independent pages. Output
records carry deterministic page/revision identifiers so retries can be
deduplicated.

### 4.3 Ukrainian section isolation

The parser recognizes Ukrainian language sections using heading structure and
known language markers. It does not rely on one substring.

It must:

- isolate only the Ukrainian section;
- preserve its heading hierarchy;
- handle multiple Ukrainian lexical entries on one page;
- avoid extracting Russian, Polish, Belarusian, Bulgarian, or other language
  forms;
- report unrecognized language-marker patterns.

### 4.4 Structured extraction

Extraction priority:

1. explicit Ukrainian headword/stress templates;
2. recognized inflection/conjugation templates;
3. recognized morphology tables;
4. bold headwords in a confirmed Ukrainian section;
5. controlled fallback extraction.

Each handler returns typed candidates with:

- plain and stressed text;
- lexeme context;
- part of speech;
- grammatical tags;
- source kind;
- source fragment;
- source coordinates;
- confidence;
- warnings.

Unknown templates are counted and sampled. No template is executed.

### 4.5 Unicode normalization

Canonical rules:

- Decompose with NFD.
- Stress mark is `U+0301 COMBINING ACUTE ACCENT`.
- Remove only configured stress marks when generating unstressed keys.
- Canonical apostrophe is `U+02BC MODIFIER LETTER APOSTROPHE`.
- Normalize common internal hyphens/dashes to `-`.
- Lowercase with explicit Ukrainian conformance tests.
- Permit Ukrainian letters, apostrophe, hyphen, whitespace for multiword
  records, and combining acute accent.
- Reject isolated accents, accents after consonants, control characters,
  residual HTML/Wikicode, and empty normalized forms.
- Calculate stress positions by Unicode vowel ordinal, never byte index.

Canonical lookup normalization must be represented by:

- Python implementation;
- generated JSON conformance vectors;
- database metadata identifying the normalization version;
- a minimal Go matcher implementation tested against every vector.

Go may apply only the already-specified matching normalization to user input.
It may not introduce new linguistic transformations.

### 4.6 Stress representation

Examples:

```text
мо́ва          -> plain: мова, stress signature: 0
вода́          -> plain: вода, stress signature: 1
переклада́ти   -> plain: перекладати, stress signature: 3
```

For multi-token or hyphenated forms, use:

```text
token-index:vowel-index|token-index:vowel-index
```

Multiple valid stressed forms remain separate variants.

### 4.7 Validation and confidence

Validation distinguishes:

- valid;
- plausible with warnings;
- rejected.

Suggested source confidence:

- 1.00: explicit supported structured template;
- 0.95: supported morphology table with unambiguous headers;
- 0.90: bold headword in a confirmed Ukrainian entry;
- 0.75: recognized manually formatted table;
- 0.50: controlled fallback;
- below 0.50: quarantined unless explicitly enabled.

Confidence is provenance quality, not a linguistic probability.

### 4.8 Staging

Write deterministic Zstandard-compressed JSON Lines:

```text
output/<dataset-id>/lexemes.jsonl.zst
output/<dataset-id>/word_forms.jsonl.zst
output/<dataset-id>/stress_variants.jsonl.zst
output/<dataset-id>/source_refs.jsonl.zst
output/<dataset-id>/parse_errors.jsonl.zst
output/<dataset-id>/manifest.json
```

Every record has a stable natural key so repeated parsing or import is
idempotent.

### 4.9 Bulk import and publication

Import stages:

1. Create an `import_run` in `building` state.
2. Create or truncate dataset-scoped staging tables.
3. Stream staging files into PostgreSQL using `COPY`.
4. Run SQL validation and uniqueness checks.
5. Merge into normalized dataset-scoped tables.
6. Build the `stress_lookup` projection.
7. Create/validate indexes.
8. Run `ANALYZE`.
9. Compare quality gates with the previous active dataset.
10. In one transaction:
    - mark the new run `published`;
    - update the singleton `active_dataset`;
    - mark the previous run `superseded`.
11. Generate final reports.

A failed build never changes the active dataset.

## 5. PostgreSQL model

### 5.1 Dataset lifecycle

```sql
CREATE TABLE import_run (
    id                    BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_key           TEXT NOT NULL UNIQUE,
    status                TEXT NOT NULL,
    dump_url              TEXT NOT NULL,
    dump_sha256           TEXT NOT NULL,
    dump_timestamp        TIMESTAMPTZ,
    parser_version        TEXT NOT NULL,
    normalization_version TEXT NOT NULL,
    schema_version        TEXT NOT NULL,
    git_commit            TEXT,
    started_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at           TIMESTAMPTZ,
    statistics            JSONB NOT NULL DEFAULT '{}'::jsonb,
    CHECK (status IN ('building', 'validated', 'published', 'superseded', 'failed'))
);

CREATE TABLE active_dataset (
    singleton             BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    dataset_id            BIGINT NOT NULL REFERENCES import_run(id),
    activated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### 5.2 Normalized tables

Core tables:

- `part_of_speech`
- `lexeme`
- `word_form`
- `stress_variant`
- `source_ref`
- `parse_error`
- `unhandled_template`
- `rejected_form`

All lexicon records contain `dataset_id`. Natural uniqueness includes dataset,
lexeme identity, normalized form, stress signature, morphology key, and source
context as appropriate.

Do not merge homonyms solely by spelling.

### 5.3 Read projection

```sql
CREATE TABLE stress_lookup (
    dataset_id          BIGINT NOT NULL,
    form_normalized     TEXT NOT NULL,
    stressed_form       TEXT NOT NULL,
    stress_signature    TEXT NOT NULL,
    lemma_normalized    TEXT NOT NULL,
    stressed_lemma      TEXT,
    part_of_speech      TEXT,
    grammatical_tags    TEXT[] NOT NULL DEFAULT '{}',
    lexeme_id           BIGINT NOT NULL,
    word_form_id        BIGINT NOT NULL,
    is_lemma            BOOLEAN NOT NULL,
    is_variant          BOOLEAN NOT NULL,
    is_obsolete         BOOLEAN NOT NULL,
    confidence          REAL NOT NULL,
    source_rank         SMALLINT NOT NULL,
    PRIMARY KEY (
        dataset_id,
        form_normalized,
        stressed_form,
        lexeme_id,
        word_form_id
    )
);
```

Primary exact-lookup index:

```sql
CREATE INDEX stress_lookup_exact_idx
ON stress_lookup (dataset_id, form_normalized, confidence DESC, source_rank)
INCLUDE (
    stressed_form,
    stress_signature,
    lemma_normalized,
    stressed_lemma,
    part_of_speech,
    grammatical_tags,
    lexeme_id,
    word_form_id,
    is_lemma,
    is_variant,
    is_obsolete
);
```

Lemma index:

```sql
CREATE INDEX stress_lookup_lemma_idx
ON stress_lookup (dataset_id, lemma_normalized)
INCLUDE (
    form_normalized,
    stressed_form,
    grammatical_tags,
    confidence
);
```

Do not add trigram or full-text indexes until a specified fuzzy/prefix feature
requires them and a benchmark proves value.

## 6. Go HTTP API

### 6.1 Responsibility

The Go process:

- starts an HTTP server;
- validates requests;
- applies only conformance-tested lookup-key normalization;
- queries the active PostgreSQL dataset;
- returns JSON;
- emits metrics and health status.

It has a read-only database role and contains no migration command.

### 6.2 Active dataset handling

At startup and periodically, Go reads `active_dataset.dataset_id` and
normalization metadata. The current ID is stored atomically in memory.

A lookup query always includes the dataset ID as a parameter. A publication can
therefore become visible without restarting the API. Existing in-flight
requests may finish against the previous immutable dataset.

### 6.3 Endpoints

#### `GET /v1/lookup?word=<word>&mode=all|best`

Returns exact matches.

Default `mode=all`.

Response:

```json
{
  "input": "замок",
  "normalized": "замок",
  "dataset": "ukwiktionary-2026-07-01-parser-1.0.0",
  "status": "ambiguous",
  "truncated": false,
  "candidates": [
    {
      "stressed_form": "за́мок",
      "stress_signature": "0",
      "lemma": "замок",
      "stressed_lemma": "за́мок",
      "part_of_speech": "noun",
      "grammatical_tags": ["nominative", "singular"],
      "is_lemma": true,
      "confidence": 1.0
    },
    {
      "stressed_form": "замо́к",
      "stress_signature": "1",
      "lemma": "замок",
      "stressed_lemma": "замо́к",
      "part_of_speech": "noun",
      "grammatical_tags": ["nominative", "singular"],
      "is_lemma": true,
      "confidence": 1.0
    }
  ]
}
```

`mode=best` returns the highest-ranked candidate plus `ambiguous=true` when
other valid candidates exist. It never implies contextual correctness.

#### `POST /v1/lookup:batch`

Request:

```json
{
  "words": ["мова", "замок", "невідомеслово"],
  "mode": "all"
}
```

Maximum default batch size: 10,000 items. The server preserves input order and
duplicates.

The SQL uses an array with ordinality:

```sql
WITH input AS (
    SELECT ordinality, word
    FROM unnest($1::text[]) WITH ORDINALITY AS t(word, ordinality)
)
SELECT ...
FROM input
LEFT JOIN stress_lookup l
  ON l.dataset_id = $2
 AND l.form_normalized = input.word
ORDER BY input.ordinality, l.confidence DESC, l.source_rank;
```

#### `GET /v1/lemmas/{lemma}/forms`

Returns all stored forms for the exact normalized lemma, with pagination and a
configurable maximum page size.

#### Operational endpoints

- `GET /health/live`
- `GET /health/ready`
- `GET /metrics`
- `GET /version`

Readiness fails when PostgreSQL is unavailable or no dataset is active.

### 6.4 Error model

```json
{
  "error": {
    "code": "INVALID_WORD",
    "message": "word contains unsupported characters",
    "request_id": "..."
  }
}
```

Required status codes:

- `400` invalid query or JSON;
- `413` request body or batch too large;
- `422` syntactically valid but unsupported lookup token;
- `429` only when an explicitly configured limiter rejects a request;
- `500` internal failure;
- `503` database or active-dataset unavailable.

No raw SQL or internal stack trace is returned.

### 6.5 Query behavior

- Exact normalized key only.
- All candidates ordered deterministically by:
  1. non-obsolete before obsolete;
  2. confidence descending;
  3. source rank ascending;
  4. lemma records before non-lemma records when otherwise equal;
  5. stressed form code-point order for stable output.
- Result count is capped; response declares truncation.
- SQL is parameterized.
- Statement timeout is set per connection or transaction.
- Pool limits are explicit.
- API requests never write to lexicon tables.

## 7. Performance

### ETL measurements

Generate:

- pages/second;
- Ukrainian sections/second;
- extracted candidates/second;
- peak resident memory;
- staging size;
- COPY rate;
- projection build time;
- table/index sizes;
- total build time.

### API measurements

Generate cold and warm measurements for:

- exact single lookup;
- ambiguous lookup;
- unknown word;
- batch sizes 100, 1,000, and 10,000;
- concurrent clients;
- connection-pool saturation;
- active-dataset switch during traffic.

Target acceptance goals on the reference host:

- warm exact lookup p50 below 5 ms;
- warm exact lookup p95 below 20 ms;
- 10,000-token batch below 1 second where practical;
- no correctness errors during dataset switch.

Actual reports must state hardware, PostgreSQL settings, dataset size, command,
sample size, and percentiles. Targets are not substituted for results.

## 8. Security

- Treat dump content as untrusted data.
- Do not execute templates, Lua, HTML, shell, Python, or SQL from the dump.
- Enforce page, token, field, log, request-body, and batch limits.
- Use safe temporary paths and atomic file moves.
- Use parameterized queries.
- Run Go with a read-only database user.
- Redact credentials from logs and reports.
- Return bounded source fragments in parse-error reports.
- Pin dependencies and scan images/dependencies in CI.
- API containers run as non-root with a read-only filesystem where practical.

## 9. Observability

Python emits structured logs with:

- import run;
- page/revision;
- parser stage;
- handler;
- counts;
- elapsed time;
- warnings/errors.

Go emits structured logs with:

- request ID;
- route;
- status;
- duration;
- result count;
- dataset ID;
- database error class without sensitive detail.

Metrics include:

- HTTP request count and duration;
- PostgreSQL query duration;
- pool acquired/idle/max and acquire wait;
- lookup hit/miss/ambiguous/truncated counts;
- active dataset;
- ETL page/form/error counters.

Do not use the input word as an unbounded metric label.

## 10. Testing strategy

### Python

- unit tests for Unicode, stress, apostrophe, hyphen, validation, signatures;
- golden tests for real Wiktionary entries;
- property tests for normalization idempotence;
- multilingual-section isolation tests;
- handler tests for noun, adjective, verb, pronoun, numeral, participle forms;
- integration tests for staging, COPY, publication, rollback;
- deterministic-output test across worker counts;
- malformed and adversarial dump-content tests.

### Go

- normalization conformance tests generated by Python;
- handler contract tests;
- PostgreSQL repository integration tests;
- batch-order and duplicate-preservation tests;
- ambiguity and truncation tests;
- read-only-role tests;
- dataset-switch tests;
- race tests;
- load tests.

### Cross-language contract

Python writes:

```text
artifacts/normalization-conformance.json
artifacts/api-fixtures.json
```

Go tests consume them. Python remains the normative implementation.

## 11. Alternatives considered

### All Python

Rejected for the requested production architecture. Python remains fully
capable of serving the API, but the selected design uses Go for a small,
statically deployable, high-concurrency read service.

### All Go

Rejected because the extraction problem benefits from Python's MediaWiki and
linguistic tooling, rapid handler iteration, and data-analysis ecosystem. It
would also violate the required responsibility split.

### SQLite/RocksDB only

Rejected as the authoritative online database because PostgreSQL provides
versioned imports, robust bulk loading, operational tooling, SQL analysis, and
concurrent network access. Compact offline exports may be added later.

### API querying normalized tables only

Retained as a fallback and correctness reference, but not the preferred hot
path. The flattened projection minimizes joins and makes query plans stable.

### In-memory API dictionary

Rejected for the initial release because it duplicates the full dataset in each
replica, complicates atomic reloads, and can exceed memory budgets. It may be
reconsidered only after measured PostgreSQL limits.
