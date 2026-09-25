# Ukrainian Stress Lexicon

## Purpose

Build a reproducible dictionary database containing Ukrainian lemmas and every
stressed inflected form that can be reliably extracted from Ukrainian Wiktionary.
Expose exact and batch lookup through a small read-only Go HTTP API.

## Architecture boundary

### Python

Python SHALL own:

- dump download and checksum verification;
- streaming XML and MediaWiki parsing;
- Ukrainian-language section detection;
- template and table extraction;
- Unicode and apostrophe normalization;
- stress validation and stress-signature generation;
- grammatical-tag canonicalization;
- deduplication and confidence assignment;
- staging-file generation;
- PostgreSQL schema migrations and bulk loading;
- dataset publication and rollback;
- quality reports, exports, and ETL benchmarks.

### Go

Go SHALL own only:

- the HTTP server;
- request parsing and validation;
- read-only PostgreSQL lookup queries;
- response serialization;
- connection pooling;
- health, readiness, and metrics endpoints;
- API-level latency and load benchmarks.

Go SHALL NOT parse dumps, infer morphology, normalize source records, run
migrations, publish datasets, or mutate lexicon data.

## Quality principle

“All forms” means all forms extractable from supported dump structures. The
project SHALL report unsupported templates, malformed entries, rejected forms,
and coverage gaps instead of claiming a complete Ukrainian dictionary.
