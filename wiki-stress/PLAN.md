# Build a High-Performance Ukrainian Stress Dictionary from the Ukrainian Wiktionary Dump

## Objective

Implement a production-ready pipeline that downloads, parses, normalizes, validates, and imports the Ukrainian Wiktionary dump into a high-performance PostgreSQL database.

The resulting database must contain:

* Ukrainian lemmas
* stressed lemmas
* all extractable inflected word forms
* stressed inflected forms
* grammatical metadata
* part of speech
* source page and revision metadata
* alternative stress variants
* normalized forms for fast lookup
* extraction confidence and provenance

The primary use case is high-speed Ukrainian word stress lookup for:

* TTS text normalization
* speech synthesis
* forced alignment
* NLP preprocessing
* stress restoration
* dictionary APIs

Do not implement a UI.

Implement the complete working system, including source code, database migrations, ingestion pipeline, tests, documentation, Docker Compose, and benchmark scripts.

---

# 1. Technology Stack

Use:

* Python 3.12+
* PostgreSQL 16+
* `lxml` or another streaming XML parser
* `mwparserfromhell` for MediaWiki markup parsing
* SQLAlchemy 2.x only where useful
* PostgreSQL native bulk loading with `COPY`
* Docker Compose
* `pytest`
* `ruff`
* `mypy`

Do not load the entire XML dump into memory.

Use streaming parsing.

The implementation must work with:

```text
ukwiktionary-latest-pages-articles.xml.bz2
```

Official source:

```text
https://dumps.wikimedia.org/ukwiktionary/latest/ukwiktionary-latest-pages-articles.xml.bz2
```

---

# 2. Repository Structure

Create this structure:

```text
ukrainian-stress-dictionary/
├── README.md
├── pyproject.toml
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── .env.example
├── migrations/
│   ├── 001_extensions.sql
│   ├── 002_schema.sql
│   ├── 003_indexes.sql
│   └── 004_functions.sql
├── src/
│   └── ukstress/
│       ├── __init__.py
│       ├── config.py
│       ├── cli.py
│       ├── downloader.py
│       ├── dump_reader.py
│       ├── language_sections.py
│       ├── wikicode_parser.py
│       ├── entry_parser.py
│       ├── template_parser.py
│       ├── table_parser.py
│       ├── stress_parser.py
│       ├── grammar_parser.py
│       ├── normalizer.py
│       ├── validator.py
│       ├── models.py
│       ├── records.py
│       ├── deduplicator.py
│       ├── staging_writer.py
│       ├── database.py
│       ├── importer.py
│       ├── exporter.py
│       ├── benchmark.py
│       └── logging_config.py
├── scripts/
│   ├── download_dump.sh
│   ├── build_database.sh
│   ├── benchmark.sh
│   └── export_dictionary.sh
├── tests/
│   ├── fixtures/
│   ├── test_normalizer.py
│   ├── test_stress_parser.py
│   ├── test_language_sections.py
│   ├── test_entry_parser.py
│   ├── test_table_parser.py
│   ├── test_deduplicator.py
│   ├── test_importer.py
│   └── test_lookup.py
└── benchmarks/
    └── lookup_words.txt
```

---

# 3. Core Data Model

Use PostgreSQL with a schema optimized for lookup performance and compact storage.

Create the following tables.

## 3.1 `lexeme`

Represents a dictionary lexeme or lemma.

```sql
CREATE TABLE lexeme (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    language_code       TEXT NOT NULL DEFAULT 'uk',
    lemma               TEXT NOT NULL,
    lemma_normalized    TEXT NOT NULL,
    stressed_lemma      TEXT,
    stress_signature    TEXT,
    part_of_speech_id   SMALLINT,
    wiktionary_page_id  BIGINT,
    revision_id         BIGINT,
    source_title        TEXT NOT NULL,
    source_section      TEXT,
    is_multiword        BOOLEAN NOT NULL DEFAULT FALSE,
    is_proper_name      BOOLEAN NOT NULL DEFAULT FALSE,
    is_abbreviation     BOOLEAN NOT NULL DEFAULT FALSE,
    is_obsolete         BOOLEAN NOT NULL DEFAULT FALSE,
    confidence          REAL NOT NULL DEFAULT 1.0,
    parser_version      TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

A Wiktionary page may describe multiple lexemes with the same written lemma. Do not assume one page equals one lexeme.

## 3.2 `word_form`

Represents an inflected form.

```sql
CREATE TABLE word_form (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    lexeme_id           BIGINT NOT NULL REFERENCES lexeme(id) ON DELETE CASCADE,
    form                TEXT NOT NULL,
    form_normalized     TEXT NOT NULL,
    stressed_form       TEXT,
    stress_signature    TEXT,
    grammatical_tags    TEXT[] NOT NULL DEFAULT '{}',
    morphology_key      TEXT,
    is_lemma            BOOLEAN NOT NULL DEFAULT FALSE,
    is_variant          BOOLEAN NOT NULL DEFAULT FALSE,
    confidence          REAL NOT NULL DEFAULT 1.0,
    source_kind         SMALLINT NOT NULL,
    source_fragment     TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

## 3.3 `stress_variant`

Store multiple possible stress patterns for the same form.

```sql
CREATE TABLE stress_variant (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    word_form_id        BIGINT NOT NULL REFERENCES word_form(id) ON DELETE CASCADE,
    stressed_form       TEXT NOT NULL,
    stress_signature    TEXT NOT NULL,
    variant_type        SMALLINT NOT NULL,
    region              TEXT,
    usage_label         TEXT,
    confidence          REAL NOT NULL DEFAULT 1.0,
    source_fragment     TEXT
);
```

## 3.4 `part_of_speech`

Use a compact lookup table.

At minimum support:

* noun
* verb
* adjective
* adverb
* pronoun
* numeral
* participle
* converb
* preposition
* conjunction
* particle
* interjection
* abbreviation
* proper noun
* phrase
* unknown

## 3.5 `grammatical_feature`

Optional dictionary table for canonical grammatical features.

Examples:

* nominative
* genitive
* dative
* accusative
* instrumental
* locative
* vocative
* singular
* plural
* masculine
* feminine
* neuter
* animate
* inanimate
* present
* past
* future
* imperative
* infinitive
* first_person
* second_person
* third_person
* perfective
* imperfective
* comparative
* superlative
* short_form
* active
* passive

## 3.6 `parse_error`

Store parser failures without terminating the complete import.

```sql
CREATE TABLE parse_error (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    page_id             BIGINT,
    revision_id         BIGINT,
    page_title          TEXT,
    error_type          TEXT NOT NULL,
    error_message       TEXT NOT NULL,
    source_fragment     TEXT,
    parser_version      TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

## 3.7 `import_run`

Track reproducibility.

Include:

* dump URL
* dump filename
* dump checksum
* dump timestamp
* parser version
* Git commit
* start time
* finish time
* status
* total pages
* Ukrainian pages
* extracted lexemes
* extracted forms
* rejected forms
* error count

---

# 4. Unicode and Stress Representation

Ukrainian stress is normally represented using Unicode combining acute accent:

```text
U+0301
```

Example:

```text
мо́ва
```

Internally this may be:

```text
м + о + U+0301 + в + а
```

Implement the following canonical rules.

## 4.1 Storage normalization

Store stressed forms in Unicode NFD form.

Store display/export forms in NFC-compatible output where appropriate, but do not rely on NFC to compose Ukrainian accented vowels because most accented Ukrainian vowels have no single precomposed Unicode character.

Use NFD internally to ensure predictable handling of `U+0301`.

## 4.2 Unstressed form

Generate the plain form by removing only stress-related combining marks.

Remove:

* `U+0301 COMBINING ACUTE ACCENT`
* optionally any explicitly configured alternative stress mark

Do not remove unrelated combining marks blindly.

## 4.3 Apostrophes

Normalize these characters to one canonical apostrophe:

```text
'
’
ʼ
‘
`
```

Canonical internal apostrophe:

```text
ʼ
```

Unicode:

```text
U+02BC MODIFIER LETTER APOSTROPHE
```

Examples:

```text
п'ять
п’ять
пʼять
```

must normalize to:

```text
пʼять
```

Preserve the original value in source metadata when useful.

## 4.4 Hyphens

Normalize common dash and hyphen variants inside words to ASCII hyphen-minus:

```text
-
```

Do not merge punctuation-separated unrelated tokens.

## 4.5 Letter case

Store:

* original form
* lowercase normalized form

Use Ukrainian-aware lowercasing.

Do not incorrectly map Ukrainian `І`, `Ї`, `Є`, or `Ґ`.

## 4.6 Ukrainian alphabet validation

Valid Ukrainian letters:

```text
а б в г ґ д е є ж з и і ї й к л м н о п р с т у ф х ц ч ш щ ь ю я
```

Also allow:

* apostrophe
* hyphen
* combining acute accent

Handle uppercase equivalents.

Reject or separately classify tokens containing unrelated scripts unless the page explicitly represents a foreign form.

## 4.7 Stress signature

Create a compact canonical stress signature.

For a single word, store the zero-based index of the stressed vowel among vowels.

Examples:

```text
мо́ва      -> 0
вода́      -> 1
переклада́ти -> 3
```

For multiple stresses or multiple tokens, use a deterministic representation.

Example:

```text
украї́нсько-по́льський
```

Possible signature:

```text
0:2|1:0
```

where:

* first number is token index
* second number is stressed-vowel index within the token

Document the exact format.

Do not calculate stress position using byte offsets.

Use Unicode code points and vowel positions.

Ukrainian vowels:

```text
а е є и і ї о у ю я
```

---

# 5. Dump Processing

Process the `.bz2` XML file directly without first decompressing it to disk.

Use:

* `bz2.BZ2File`
* streaming XML parsing
* element cleanup after each page

Memory consumption must stay approximately constant as dump size grows.

For each page, extract:

* page ID
* namespace
* title
* revision ID
* timestamp
* wikitext

Only process main namespace pages unless configurable otherwise.

Skip:

* discussion pages
* categories
* templates as standalone pages
* files
* help pages
* project pages
* redirects, unless redirects are used to build aliases

Support redirects as aliases when they point to Ukrainian lexical entries.

---

# 6. Ukrainian Language Section Detection

A page may contain entries in several languages.

Extract only Ukrainian-language sections.

Support multiple Ukrainian Wiktionary section formats, including variations such as:

```text
== Українська ==
```

and language templates or markers used by Ukrainian Wiktionary.

Do not use only one brittle substring check.

Implement a section parser that identifies heading hierarchy and templates.

Add fixtures copied from real representative Ukrainian Wiktionary entry structures.

The parser must not extract stressed Polish, Russian, Belarusian, Bulgarian, or other language forms from non-Ukrainian sections.

---

# 7. Extracting Lemmas and Forms

Implement several extraction strategies.

Each extracted record must include `source_kind`.

Suggested values:

```text
1 = page_title
2 = bold_headword
3 = headword_template
4 = inflection_template
5 = morphology_table
6 = explicit_stress_template
7 = synonym_or_variant
8 = redirect
9 = fallback_regex
```

Prefer structured extraction over regular expressions.

## 7.1 Lemma extraction priority

Extract stressed lemma from, in priority order:

1. explicit Ukrainian headword template
2. bold stressed headword
3. inflection template lemma
4. page title combined with explicit stress metadata
5. manually validated fallback extraction

The page title usually has no stress mark.

Do not assume every stressed token in the article is the lemma.

## 7.2 Inflected forms

Extract all forms present in:

* noun declension tables
* adjective declension tables
* pronoun tables
* numeral tables
* verb conjugation tables
* participle tables
* comparative and superlative sections
* explicit alternative-form templates
* stress-specific templates
* manually formatted wiki tables where recognizable

Extract grammatical context from:

* row headers
* column headers
* template parameters
* table captions
* surrounding headings

Store canonical grammatical tags.

## 7.3 Multiple forms in one cell

A table cell may contain:

* several variants
* slash-separated forms
* comma-separated forms
* regional forms
* optional segments
* notes
* footnote markers
* wikilinks
* templates
* `<br>` separators

Split only when structurally justified.

Do not split hyphenated words.

Remove:

* references
* footnote markers
* HTML markup
* wiki formatting
* nonlexical explanatory text

Preserve legitimate apostrophes and hyphens.

## 7.4 Optional word segments

Do not generate combinatorial forms from arbitrary parenthesized strings unless there is a clearly defined Wiktionary convention.

For uncertain forms:

* store the literal source form
* lower confidence
* preserve `source_fragment`

## 7.5 Reflexive verbs

Preserve Ukrainian reflexive suffixes:

```text
-ся
-сь
```

Treat them as part of the word.

## 7.6 Multiword expressions

Store multiword entries, but classify them separately with:

```text
is_multiword = true
```

For the primary word-stress lookup table, do not mix multiword expressions into single-token exact lookup unless requested by the query.

---

# 8. Parsing MediaWiki Markup

Use `mwparserfromhell` for templates, links, and basic Wikicode traversal.

Also support raw table parsing where library abstractions are insufficient.

Create a template-dispatch architecture.

Example:

```python
TEMPLATE_HANDLERS = {
    "specific-template-name": parse_specific_template,
}
```

Template handlers must be independently testable.

Because Wiktionary templates may evolve, create:

* known-template registry
* unknown-template counter
* optional report of most frequent unhandled templates

Do not silently ignore every unknown template.

At the end of an import, generate a report such as:

```text
reports/unhandled_templates.csv
```

with:

* template name
* occurrence count
* sample page titles
* sample invocations

---

# 9. Record Validation

Validate every extracted form.

A valid stressed Ukrainian lexical form should normally satisfy:

* contains at least one Ukrainian vowel
* contains at least one `U+0301`
* stress mark follows a Ukrainian vowel
* no stress mark follows a consonant
* no isolated combining accent
* no HTML or Wikicode remains
* no control characters
* normalized plain form is nonempty
* form length is within configured limits

Flag rather than automatically discard unusual but plausible forms.

Confidence levels:

```text
1.00 = explicit structured template or clear morphology table
0.90 = bold headword in Ukrainian section
0.80 = clearly labeled manual wiki table
0.60 = controlled fallback extraction
0.30 = ambiguous extraction
```

Do not assign high confidence to regex-only extraction.

---

# 10. Deduplication

Deduplicate on canonical normalized values.

A lexeme identity should account for:

* normalized lemma
* part of speech
* homonym or sense marker when available
* source entry context

Do not merge unrelated homonyms solely because their spelling is identical.

A word form identity should account for:

* lexeme ID
* normalized form
* stress signature
* morphology key
* variant classification

If duplicate records differ only by source quality:

* retain one canonical record
* keep the highest confidence
* preserve source provenance where practical

If the same unstressed form has different legitimate stress variants, preserve all variants.

Examples:

```text
за́мок
замо́к
```

These must not overwrite each other.

---

# 11. Staging and Bulk Import

Do not insert rows one by one.

Implement a two-stage pipeline:

1. Parse dump into compressed intermediate files
2. Bulk load into PostgreSQL staging tables with `COPY`

Recommended intermediate format:

```text
JSON Lines compressed with Zstandard
```

or:

```text
TSV with robust escaping
```

Preferred output files:

```text
output/lexemes.jsonl.zst
output/forms.jsonl.zst
output/stress_variants.jsonl.zst
output/errors.jsonl.zst
output/import_manifest.json
```

Then:

* load staging tables
* validate staging data
* merge into final tables
* create indexes after initial bulk import
* run `ANALYZE`
* optionally run `VACUUM`

The pipeline must support resume after failure.

Use checkpoints by page count or page ID.

---

# 12. Indexing Strategy

Optimize for these queries:

1. Exact unstressed word lookup
2. Exact stressed word lookup
3. Return every stress variant
4. Prefix autocomplete
5. Lookup by lemma
6. Retrieve all forms of a lemma
7. Retrieve grammatical metadata
8. Batch lookup of thousands of words
9. Prefer the highest-confidence result

Create at least these indexes:

```sql
CREATE INDEX idx_word_form_normalized
ON word_form (form_normalized);

CREATE INDEX idx_word_form_stressed
ON word_form (stressed_form);

CREATE INDEX idx_word_form_lexeme
ON word_form (lexeme_id);

CREATE INDEX idx_lexeme_normalized
ON lexeme (lemma_normalized);

CREATE INDEX idx_lexeme_stressed
ON lexeme (stressed_lemma);

CREATE INDEX idx_word_form_lookup_covering
ON word_form (form_normalized, confidence DESC)
INCLUDE (
    stressed_form,
    lexeme_id,
    grammatical_tags,
    morphology_key,
    is_lemma,
    source_kind
);
```

Enable PostgreSQL extensions where useful:

```sql
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

For prefix lookup, evaluate:

* `text_pattern_ops`
* trigram index
* dedicated prefix table
* `varchar_pattern_ops`

Select the best strategy based on benchmarks.

Do not add expensive indexes that do not improve measured queries.

---

# 13. Fast Lookup SQL API

Implement SQL functions or prepared queries.

## 13.1 Exact lookup

Input:

```text
замок
```

Output all valid stress variants:

```json
[
  {
    "form": "замок",
    "stressed_form": "за́мок",
    "lemma": "замок",
    "stressed_lemma": "за́мок",
    "part_of_speech": "noun",
    "grammatical_tags": ["nominative", "singular"],
    "confidence": 1.0
  },
  {
    "form": "замок",
    "stressed_form": "замо́к",
    "lemma": "замок",
    "stressed_lemma": "замо́к",
    "part_of_speech": "noun",
    "grammatical_tags": ["nominative", "singular"],
    "confidence": 1.0
  }
]
```

## 13.2 Batch lookup

Support efficient lookup for at least 10,000 input tokens per call using:

* temporary tables
* array input
* `unnest`
* indexed join

Preserve input order and duplicates.

Example input:

```text
["український", "мова", "замок", "невідомеслово"]
```

Return:

* input token
* normalized token
* all matching stress variants
* match status
* confidence

## 13.3 Best candidate

Implement an optional best-candidate query.

It may rank by:

1. exact normalized match
2. lemma match
3. inflected form match
4. confidence
5. structured source kind
6. non-obsolete usage

Do not silently discard ambiguity.

Return ambiguity metadata.

---

# 14. Optional Materialized Lookup Table

Evaluate creating a flattened lookup table:

```sql
CREATE TABLE stress_lookup (
    form_normalized     TEXT NOT NULL,
    stressed_form       TEXT NOT NULL,
    lemma_normalized    TEXT,
    stressed_lemma      TEXT,
    part_of_speech_id   SMALLINT,
    grammatical_tags    TEXT[],
    confidence          REAL NOT NULL,
    lexeme_id           BIGINT NOT NULL,
    word_form_id        BIGINT NOT NULL,
    PRIMARY KEY (
        form_normalized,
        stressed_form,
        lexeme_id,
        word_form_id
    )
);
```

Use it only if benchmarks show that it meaningfully improves lookup latency.

Keep normalized source tables as the authoritative data model.

Provide a refresh command.

---

# 15. CLI

Implement a CLI named:

```text
ukstress
```

Commands:

```bash
ukstress download
ukstress parse
ukstress validate
ukstress import
ukstress build
ukstress lookup "український"
ukstress batch-lookup input.txt
ukstress export
ukstress benchmark
ukstress stats
ukstress report-unhandled
```

Examples:

```bash
ukstress build \
  --dump data/ukwiktionary-latest-pages-articles.xml.bz2 \
  --database-url postgresql://ukstress:ukstress@localhost:5432/ukstress
```

Support:

```bash
ukstress parse --workers 8
```

However, do not parallelize XML reading in a way that requires loading the entire dump or corrupts page boundaries.

A valid approach is:

* one streaming producer
* bounded work queue
* multiple page-parsing workers
* one ordered or batched output writer

Use bounded queues to control memory.

---

# 16. Parallelism

Implement configurable parsing workers.

Requirements:

* bounded queue
* deterministic record normalization
* no duplicate output caused by retries
* graceful shutdown
* worker exception propagation
* progress reporting
* checkpointing

Avoid Python multiprocessing overhead for tiny records where it makes performance worse.

Benchmark:

* single process
* thread pool
* process pool

Choose the default based on measured throughput.

---

# 17. Performance Targets

Provide benchmark scripts and report actual measured results.

Target hardware assumption:

* 8 CPU cores
* 16 GB RAM
* SSD
* PostgreSQL running locally

Targets:

* streaming memory use below 2 GB where practical
* exact lookup p50 below 5 ms after cache warm-up
* exact lookup p95 below 20 ms after cache warm-up
* batch lookup of 10,000 tokens below 1 second where practical
* sustained parsing throughput reported in pages per second
* database size reported
* index size reported

Do not fabricate benchmark results.

The benchmark command must generate the results on the current machine.

Output:

```text
reports/benchmark.json
reports/benchmark.md
```

---

# 18. Statistics and Quality Reports

Generate these statistics:

* total Wiktionary pages
* total Ukrainian sections
* total lexemes
* total stressed lemmas
* total forms
* total stressed forms
* forms without stress
* forms with multiple stress variants
* forms grouped by part of speech
* forms grouped by source kind
* confidence distribution
* malformed stress count
* rejected form count
* top unhandled templates
* top parse errors
* duplicate count
* database and index sizes

Generate:

```text
reports/statistics.json
reports/statistics.md
reports/parse_errors.csv
reports/rejected_forms.csv
reports/unhandled_templates.csv
reports/ambiguous_forms.csv
```

---

# 19. Export Formats

Support exports to:

## 19.1 TSV

```text
form<TAB>stressed_form<TAB>lemma<TAB>stressed_lemma<TAB>pos<TAB>tags<TAB>confidence
```

## 19.2 JSON Lines

Example:

```json
{
  "form": "мови",
  "stressed_form": "мо́ви",
  "lemma": "мова",
  "stressed_lemma": "мо́ва",
  "part_of_speech": "noun",
  "grammatical_tags": [
    "genitive",
    "singular"
  ],
  "confidence": 1.0
}
```

## 19.3 Compact TTS dictionary

```text
мова|мо́ва
мови|мо́ви
український|украї́нський
```

When one form has multiple stress variants:

```text
замок|за́мок
замок|замо́к
```

Do not arbitrarily pick one variant unless explicitly using `--best-only`.

---

# 20. Testing

Write comprehensive unit and integration tests.

Cover:

* combining acute accent handling
* NFD normalization
* apostrophe normalization
* hyphen normalization
* Ukrainian lowercase conversion
* stress removal
* stress signature generation
* multiple stress marks
* malformed accents
* homographs
* multiple Ukrainian entries on one page
* multilingual pages
* noun tables
* adjective tables
* verb conjugation tables
* `<br>`-separated forms
* wikilinks
* templates inside table cells
* footnotes
* alternative forms
* regional forms
* reflexive verbs
* multiword expressions
* redirects
* duplicate forms
* import retry
* batch lookup order preservation

Include representative fixtures.

Do not make tests depend on the live Wikimedia website.

---

# 21. Sample Validation Cases

At minimum validate these expected transformations:

```text
мо́ва -> мова
украї́нський -> український
пʼя́ть -> пʼять
обʼє́кт -> обʼєкт
будь-яки́й -> будь-який
```

Validate homographic ambiguity:

```text
за́мок
замо́к
```

Validate that both map to:

```text
замок
```

but remain separate stress variants.

Validate that:

```text
о́бласть
областе́й
```

can belong to the same lexeme with different grammatical forms.

---

# 22. Error Handling

The import must not terminate because of one malformed page.

For each failure:

* log page ID
* log revision ID
* log title
* log parser stage
* log exception type
* preserve a limited source fragment
* continue processing

Add:

```text
--fail-fast
```

for development and testing.

Default production behavior must be continue-on-error.

Do not log full dump pages by default.

---

# 23. Security and Robustness

Treat dump content as untrusted input.

Requirements:

* no arbitrary template execution
* no shell execution from dump content
* no HTML rendering
* no dynamic Python evaluation
* no unsafe deserialization
* parameterized SQL
* bounded record sizes
* bounded logs
* configurable maximum page size
* safe temporary file handling
* checksum verification for downloaded dumps

Do not execute Lua modules, Wiktionary templates, or remote code.

Parse template syntax only as data.

---

# 24. Reproducibility

Record:

* source dump URL
* source dump timestamp
* source SHA-256
* parser version
* Git commit
* Python version
* dependency versions
* database schema version
* command-line options
* import start and finish time

Generate an import manifest:

```json
{
  "dump_file": "ukwiktionary-latest-pages-articles.xml.bz2",
  "dump_sha256": "...",
  "parser_version": "...",
  "schema_version": "...",
  "started_at": "...",
  "finished_at": "...",
  "statistics": {}
}
```

---

# 25. Docker Compose

Provide a working `docker-compose.yml` with:

* PostgreSQL 16
* parser/import application
* persistent PostgreSQL volume
* dump/output volume
* health checks

Example commands:

```bash
docker compose up -d postgres
docker compose run --rm app ukstress build --dump /data/ukwiktionary.xml.bz2
docker compose run --rm app ukstress stats
```

Tune PostgreSQL conservatively for bulk import and lookup.

Do not hardcode unsafe production settings.

Document optional settings for:

* `shared_buffers`
* `maintenance_work_mem`
* `work_mem`
* `effective_cache_size`
* WAL optimization during initial build

---

# 26. Makefile

Implement:

```bash
make setup
make lint
make typecheck
make test
make download
make parse
make import
make build
make benchmark
make export
make clean
```

---

# 27. README

The README must explain:

1. project purpose
2. architecture
3. database schema
4. Unicode stress representation
5. dump download
6. local setup
7. Docker setup
8. full build
9. incremental rebuild
10. exact lookup
11. batch lookup
12. exports
13. benchmarks
14. known Wiktionary limitations
15. parser extension strategy
16. licensing and attribution

Clearly state that Ukrainian Wiktionary is incomplete and inconsistent, so the generated database is not guaranteed to contain every Ukrainian word form or always provide a correct stress pattern.

Do not claim complete linguistic coverage.

---

# 28. Licensing and Attribution

Preserve source attribution metadata required for Wikimedia-derived content.

Document the applicable Wikimedia and Wiktionary licensing considerations.

Include:

* source URL
* dump date
* page title
* page ID
* revision ID where available

Do not state a definitive legal conclusion beyond clearly documented source-license requirements.

---

# 29. Implementation Priorities

Implement in this order:

1. Unicode normalization
2. streaming dump parser
3. Ukrainian section detection
4. headword extraction
5. structured template extraction
6. morphology table extraction
7. validation
8. deduplication
9. intermediate files
10. PostgreSQL schema
11. bulk import
12. exact lookup
13. batch lookup
14. indexes
15. statistics
16. benchmarks
17. documentation

---

# 30. Acceptance Criteria

The project is complete only when:

* the dump is parsed without loading it entirely into RAM
* Ukrainian sections are isolated correctly
* stressed lemmas are extracted
* inflected forms are extracted from supported tables and templates
* multiple stress variants are preserved
* Unicode normalization is deterministic
* malformed records are reported
* PostgreSQL is populated through bulk loading
* exact lookup is indexed
* batch lookup is implemented
* tests pass
* Docker Compose works
* benchmark scripts run
* reports are generated
* README contains reproducible commands
* no benchmark numbers are invented
* no claim of complete Wiktionary coverage is made

---

# 31. Required Final Agent Output

At completion, provide:

1. concise architecture summary
2. list of created files
3. database schema overview
4. extraction strategies implemented
5. unsupported templates or structures
6. exact commands to build the database
7. exact lookup examples
8. actual test results
9. actual benchmark results
10. known limitations

Do not return only an architecture proposal.

Create the actual implementation.

