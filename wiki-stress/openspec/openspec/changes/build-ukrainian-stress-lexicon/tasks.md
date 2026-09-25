# Tasks: Build Ukrainian Stress Lexicon

## 1. Project foundation

- [x] 1.1 Create repository layout for `etl/`, `api/`, `db/`, `deploy/`,
      `reports/`, and `openspec/`.
- [x] 1.2 Add Python packaging, lockfile, linting, typing, and test configuration.
- [x] 1.3 Add Go module, linting, tests, race-test target, and reproducible build.
- [x] 1.4 Add Dockerfiles and Docker Compose for PostgreSQL, ETL, and API.
- [x] 1.5 Add Make targets for setup, test, build, parse, import, publish,
      benchmark, export, and clean.
- [x] 1.6 Add CI that runs Python tests, Go tests, migration checks, API contract
      tests, dependency scans, and image builds.

## 2. PostgreSQL schema

- [x] 2.1 Implement forward-only migrations for import runs and active dataset.
- [x] 2.2 Implement reference tables for parts of speech and canonical
      grammatical features.
- [x] 2.3 Implement dataset-scoped `lexeme`, `word_form`, `stress_variant`, and
      `source_ref` tables.
- [x] 2.4 Implement `parse_error`, `rejected_form`, and `unhandled_template`
      reporting tables.
- [x] 2.5 Implement staging tables and validation constraints.
- [x] 2.6 Implement `stress_lookup` projection and exact/lemma indexes.
- [x] 2.7 Create separate migration-time owner, ETL writer, and Go read-only
      database roles.
- [x] 2.8 Add migration integration tests and verify the Go role cannot mutate
      lexicon tables.

## 3. Python normalization core

- [x] 3.1 Implement NFD normalization and explicit `U+0301` stress handling.
- [x] 3.2 Implement canonical `U+02BC` apostrophe normalization.
- [x] 3.3 Implement canonical internal hyphen normalization.
- [x] 3.4 Implement Ukrainian character and vowel validation.
- [x] 3.5 Implement unstressed-key generation without stripping unrelated marks.
- [x] 3.6 Implement deterministic stress signatures using vowel ordinals.
- [x] 3.7 Implement multi-token and hyphenated-form signature rules.
- [x] 3.8 Add examples, golden tests, property tests, and idempotence tests.
- [x] 3.9 Generate normalization conformance vectors for Go.

## 4. Python dump acquisition and streaming

- [x] 4.1 Implement resumable/safe dump download with temporary files.
- [x] 4.2 Implement SHA-256 calculation and manifest recording.
- [x] 4.3 Implement direct `.bz2` streaming with `lxml.etree.iterparse`.
- [x] 4.4 Extract page, redirect, revision, timestamp, and wikitext metadata.
- [x] 4.5 Enforce page-size and field-size limits.
- [x] 4.6 Clear processed XML elements to keep memory bounded.
- [x] 4.7 Add bounded producer/worker queues and graceful shutdown.
- [x] 4.8 Add deterministic output tests for one worker versus multiple workers.
- [x] 4.9 Add malformed XML and oversized-page tests.

## 5. Ukrainian section parser

- [x] 5.1 Implement heading-tree parsing.
- [x] 5.2 Implement known Ukrainian language markers and aliases.
- [x] 5.3 Support multiple lexical entries in one Ukrainian section.
- [x] 5.4 Prevent extraction from non-Ukrainian sections.
- [x] 5.5 Report unknown language-marker structures.
- [x] 5.6 Add multilingual real-entry fixtures and golden tests.

## 6. Wikicode and extraction handlers

- [x] 6.1 Build a non-executing `mwparserfromhell` traversal layer.
- [x] 6.2 Build template registry and normalized template-name matching.
- [x] 6.3 Implement explicit stressed-headword handlers.
- [x] 6.4 Implement noun declension handlers.
- [x] 6.5 Implement adjective declension handlers.
- [x] 6.6 Implement verb conjugation and reflexive-form handlers.
- [x] 6.7 Implement pronoun, numeral, participle, adverb, comparative, and
      superlative handlers where represented.
- [x] 6.8 Implement supported manual wiki-table parsing with row/column context.
- [x] 6.9 Handle wikilinks, `<br>`, variants, notes, footnotes, and references.
- [x] 6.10 Implement redirects as aliases without treating redirect text as a
      stressed form.
- [x] 6.11 Implement controlled fallback extraction with reduced confidence.
- [x] 6.12 Record unknown templates with counts and representative samples.
- [x] 6.13 Add handler fixtures copied from representative dump pages.

## 7. Candidate validation and deduplication

- [x] 7.1 Implement valid/plausible/rejected classification.
- [x] 7.2 Reject isolated stress marks and stress after non-vowels.
- [x] 7.3 Reject residual HTML, Wikicode, controls, and unsupported scripts.
- [x] 7.4 Preserve multiword expressions while classifying them separately.
- [x] 7.5 Preserve homonyms and all legitimate stress variants.
- [x] 7.6 Define stable natural keys for lexemes, forms, variants, and sources.
- [x] 7.7 Merge duplicate provenance without merging unrelated homonyms.
- [x] 7.8 Add ambiguity, duplicate, and malformed-form tests.

## 8. Staging and reports

- [x] 8.1 Implement deterministic JSONL.zst writers.
- [x] 8.2 Implement parser checkpoints and resumable output.
- [x] 8.3 Write import manifest with dump, parser, schema, Git, and environment
      versions.
- [x] 8.4 Generate parse-error, rejected-form, unhandled-template, duplicate,
      and ambiguous-form reports.
- [x] 8.5 Generate counts by part of speech, source kind, confidence, and
      grammatical feature.
- [x] 8.6 Ensure logs and source fragments are bounded.
- [x] 8.7 Add staging replay and idempotence tests.

## 9. Python PostgreSQL import and publication

- [x] 9.1 Implement Python-owned migration command.
- [x] 9.2 Implement streaming PostgreSQL `COPY` into staging tables.
- [x] 9.3 Implement staging validation and natural-key uniqueness checks.
- [x] 9.4 Merge normalized records into the dataset-scoped final tables.
- [x] 9.5 Build and validate the `stress_lookup` projection.
- [x] 9.6 Create indexes after bulk load and run `ANALYZE`.
- [x] 9.7 Implement configurable quality gates against the active dataset.
- [x] 9.8 Implement atomic active-dataset publication.
- [x] 9.9 Implement data rollback to a previous published dataset.
- [x] 9.10 Implement retention cleanup that never deletes the active dataset.
- [x] 9.11 Add failure-injection tests proving failed imports do not become active.
- [x] 9.12 Add publication tests with concurrent read transactions.

## 10. Python CLI and exports

- [x] 10.1 Implement `ukstress download`.
- [x] 10.2 Implement `ukstress parse`.
- [x] 10.3 Implement `ukstress validate`.
- [x] 10.4 Implement `ukstress migrate`.
- [x] 10.5 Implement `ukstress import`.
- [x] 10.6 Implement `ukstress publish` and `ukstress rollback`.
- [x] 10.7 Implement `ukstress build` orchestration.
- [x] 10.8 Implement `ukstress stats` and `ukstress report-unhandled`.
- [x] 10.9 Implement TSV, JSONL, and compact TTS exports.
- [x] 10.10 Implement `--best-only` without changing the authoritative data.
- [x] 10.11 Add CLI help, exit-code, interruption, and retry tests.

## 11. Go PostgreSQL repository

- [x] 11.1 Implement configuration through environment variables with validation.
- [x] 11.2 Implement bounded `pgxpool` with connect and statement timeouts.
- [x] 11.3 Implement active-dataset loading and atomic periodic refresh.
- [x] 11.4 Implement prepared exact lookup query.
- [x] 11.5 Implement deterministic best-candidate query while retaining
      ambiguity metadata.
- [x] 11.6 Implement batch lookup with `unnest(... WITH ORDINALITY)`.
- [x] 11.7 Implement lemma-forms query with pagination.
- [x] 11.8 Enforce read-only transactions and database role.
- [x] 11.9 Add repository integration tests against PostgreSQL.
- [x] 11.10 Test an active-dataset switch during concurrent queries.

## 12. Go HTTP API

- [x] 12.1 Implement `GET /v1/lookup`.
- [x] 12.2 Implement `POST /v1/lookup:batch`.
- [x] 12.3 Implement `GET /v1/lemmas/{lemma}/forms`.
- [x] 12.4 Implement `/health/live`, `/health/ready`, `/metrics`, and `/version`.
- [x] 12.5 Implement strict body, token, batch, result, and pagination limits.
- [x] 12.6 Implement stable JSON response and error contracts.
- [x] 12.7 Preserve batch input order and duplicates.
- [x] 12.8 Return explicit hit, miss, ambiguous, and truncated statuses.
- [x] 12.9 Add request IDs, timeouts, graceful shutdown, and panic recovery.
- [x] 12.10 Do not log raw lookup words by default.
- [x] 12.11 Run Python-generated normalization conformance vectors in Go tests.
- [x] 12.12 Add OpenAPI schema and contract tests.
- [x] 12.13 Run `go test -race ./...`.

## 13. Observability and security

- [x] 13.1 Add structured Python import logs.
- [x] 13.2 Add structured Go request and database logs.
- [x] 13.3 Add bounded-cardinality API and pool metrics.
- [x] 13.4 Add ETL throughput, error, and quality metrics.
- [x] 13.5 Verify dump content cannot trigger code, shell, SQL, Lua, or template
      execution.
- [x] 13.6 Run containers as non-root and remove unnecessary capabilities.
- [x] 13.7 Scan dependencies and container images.
- [x] 13.8 Verify credentials and raw SQL errors are absent from logs/responses.
- [x] 13.9 Document backup, restore, dataset rollback, and retention procedures.

## 14. Benchmarks and acceptance

- [x] 14.1 Implement ETL benchmark command and machine-readable report.
- [x] 14.2 Record pages/second, forms/second, memory, staging size, COPY speed,
      projection time, and table/index sizes.
- [x] 14.3 Implement Go single-lookup and batch load tests.
- [x] 14.4 Measure cold/warm p50, p95, p99 and throughput.
- [x] 14.5 Test batches of 100, 1,000, and 10,000 tokens.
- [x] 14.6 Test pool saturation and PostgreSQL unavailability.
- [x] 14.7 Test publication while API traffic is active.
- [x] 14.8 Generate `reports/benchmark.json` and `reports/benchmark.md`.
- [x] 14.9 State hardware, dataset, commands, samples, and PostgreSQL settings.
- [x] 14.10 Do not mark targets achieved unless measured output proves them.

## 15. Documentation and release

- [x] 15.1 Document architecture and the Python/Go boundary.
- [x] 15.2 Document Unicode representation and normalization versioning.
- [x] 15.3 Document local, Docker, full-build, publish, rollback, API, export, and
      benchmark commands.
- [x] 15.4 Document known template coverage and linguistic limitations.
- [x] 15.5 Document source attribution, dump metadata, and applicable licensing
      considerations without presenting unsupported legal conclusions.
- [x] 15.6 Add representative API examples and database query plans.
- [x] 15.7 Validate OpenSpec change with strict validation.
- [x] 15.8 Complete an end-to-end build from a real dump.
- [x] 15.9 Attach actual tests, quality reports, and benchmarks to the release.
