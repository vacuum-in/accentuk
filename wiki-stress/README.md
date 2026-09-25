# Ukrainian Stress Lexicon

`ukstress` streams the Ukrainian Wiktionary XML dump into a normalized,
versioned PostgreSQL lexicon for Ukrainian word-stress lookup. Python
(`etl/`) owns download, extraction, normalization, validation, and the
full dataset lifecycle (import → publish → rollback → retention). A
read-only Go HTTP service (`api/`) is the only public interface.

See [`docs/architecture.md`](docs/architecture.md) for the full
Python/Go boundary and normalization details,
[`docs/operations.md`](docs/operations.md) for every command including
backup/restore/rollback, [`docs/api.md`](docs/api.md) for real request/
response examples and query plans, and
[`docs/coverage-and-limitations.md`](docs/coverage-and-limitations.md)
for what is and isn't extracted, plus source attribution.
[`docs/state.md`](docs/state.md) is what is deployed right now and what is
waiting to be; [`docs/lessons.md`](docs/lessons.md) is how the audiobook
dataset and the token classifier were built, told by the mistakes, with the
rules that came out of them.

## Quick start

```bash
cp .env.example .env
make setup                          # uv --directory etl sync --all-groups

cd deploy
docker compose up -d postgres
docker exec -e PGPASSWORD=ukstress_owner deploy-postgres-1 \
  psql -U ukstress_owner -d ukstress -c \
  "ALTER ROLE ukstress_api WITH PASSWORD 'ukstress_api';
   ALTER ROLE ukstress_etl WITH PASSWORD 'ukstress_etl';"

cd ../etl
uv run ukstress migrate \
  --database-url postgresql://ukstress_owner:ukstress_owner@localhost:5432/ukstress \
  --migrations ../db/migrations

cd ../deploy
docker compose up -d --build api
curl -G http://localhost:8080/v1/lookup --data-urlencode 'word=мова'
```

That gets you an empty, ready API (`{"status":"not_found", ...}` for
every word) — there's no published dataset yet. To build and publish one
from a real dump, see "Full build from a real dump" in
[`docs/operations.md`](docs/operations.md).

## Lookup API

```bash
curl -G http://localhost:8080/v1/lookup --data-urlencode 'word=мова'
curl -X POST http://localhost:8080/v1/lookup:batch \
  -H 'Content-Type: application/json' \
  -d '{"words": ["український", "мова", "замок", "невідомеслово"]}'
curl http://localhost:8080/v1/lemmas/замок/forms
curl http://localhost:8080/health/live
curl http://localhost:8080/health/ready
```

Batch lookup preserves input order and duplicates, accepts up to
`MAX_BATCH_SIZE` (default 10,000) words in one set-oriented PostgreSQL
query, and reports `ambiguous` results with every candidate rather than
silently picking one. Full contract: [`api/openapi.yaml`](api/openapi.yaml);
real captured examples: [`docs/api.md`](docs/api.md).

## Schema

`lexeme` and `word_form` are dataset-scoped normalized records;
`stress_variant` holds every recorded stress realization of a form;
`source_ref` retains page/revision/section provenance for attribution.
`stress_lookup` is a denormalized, indexed read projection built after
import specifically for the Go API's two hot queries (exact match by
`form_normalized`, forms by `lemma_normalized`) — see the real `EXPLAIN
ANALYZE` output in [`docs/api.md`](docs/api.md#query-plans). `import_run`
and `active_dataset` implement the publish/rollback/retention lifecycle
documented in [`docs/architecture.md`](docs/architecture.md#dataset-lifecycle).

Unrecognized templates are never executed and never silently dropped —
they're counted in `unhandled_template`
(`ukstress report-unhandled`), which is how coverage gaps get prioritized.
See [`docs/coverage-and-limitations.md`](docs/coverage-and-limitations.md)
for exactly what is and isn't currently extracted.

## Commands

```bash
make lint typecheck test                  # etl: ruff + mypy + pytest
cd api && make fmt vet test test-race     # api: gofmt + vet + go test (+ -race)
cd api && make test-integration           # requires UKSTRESS_TEST_DATABASE_URL

uv run ukstress download
uv run ukstress parse --dump data/ukwiktionary-latest-pages-articles.xml.bz2 --workers 4
uv run ukstress import --database-url "$DATABASE_URL" --input output/<dataset-key>
uv run ukstress publish --database-url "$DATABASE_URL" <dataset_id>
uv run ukstress rollback --database-url "$DATABASE_URL" <dataset_id>
uv run ukstress cleanup --database-url "$DATABASE_URL" --retain-count 3
uv run ukstress export --database-url "$DATABASE_URL" --format jsonl --output export.jsonl
uv run ukstress stats --database-url "$DATABASE_URL"
uv run ukstress report-unhandled --database-url "$DATABASE_URL"

cd api && go build -o bin/bench-lookup ./cmd/bench-lookup
./bin/bench-lookup --base-url http://localhost:8080 --words ../benchmarks/lookup_words.txt
```

Full command reference, including Docker Compose and the full-build
pipeline: [`docs/operations.md`](docs/operations.md). Benchmark results —
real measurements against a live 364,130-row dataset, not pre-populated
numbers, and three real bugs the process found: [`reports/benchmark.md`](reports/benchmark.md).

## CI

`.github/workflows/ci.yml` runs Python lint/typecheck/test, Go
vet/test/race/integration, a migration + role-permission check, dependency
scans (`pip-audit`, `govulncheck`), and container image builds + Trivy
scans on every push/PR.
