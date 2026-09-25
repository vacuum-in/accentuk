# Operations

Every command below is real and was run against a live instance while
writing this document — none are aspirational.

## Local development (no Docker)

```bash
make setup                 # uv --directory etl sync --all-groups
cd etl && uv run ukstress migrate \
  --database-url postgresql://ukstress_owner:ukstress_owner@localhost:5432/ukstress \
  --migrations ../db/migrations

cd ../api && go build -o bin/stress-api ./cmd/stress-api
DATABASE_URL=postgresql://ukstress_api:ukstress_api@localhost:5432/ukstress \
  ./bin/stress-api
```

## Docker Compose

`deploy/docker-compose.yml` defines `postgres`, `stress-model`, `api`, and the
one-off `etl` tool service (profile `tools`, so it doesn't start with a bare `up`).
Both `etl` and `api` run `cap_drop: [ALL]` and `no-new-privileges`; `api`
additionally runs with a **read-only root filesystem** — it's a static Go
binary with no runtime writes, verified live (see the container-hardening
note below).

```bash
cd deploy
docker compose up -d postgres        # persistent named volume: deploy_postgres-data
docker compose up -d --build api     # rebuilds and (re)starts the API

# ETL runs as a one-off, not a long-lived service:
docker compose run --rm etl parse --dump /data/ukwiktionary-latest-pages-articles.xml.bz2
```

After migrations and setting the role passwords shown below, load the frozen
contextual candidate inventory into its
database table. This supplements model-covered forms missing from the published
Wiktionary projection while leaving the 363,272-form dataset unchanged:

```bash
docker compose --profile tools run --rm \
  -v "$PWD/../models/v3-xenc/serving_manifest.json:/data/serving_manifest.json:ro" \
  etl import-serving-inventory \
  --database-url postgresql://ukstress_etl:ukstress_etl@postgres:5432/ukstress \
  --manifest /data/serving_manifest.json
```

The compose file sets `ukstress_api`'s and `ukstress_etl`'s **passwords**
nowhere — `db/migrations/005_roles.sql` creates the roles with `LOGIN` but
no password ("Application passwords are injected at deployment"). Set them
once per fresh volume:

```bash
docker exec -e PGPASSWORD=ukstress_owner deploy-postgres-1 \
  psql -U ukstress_owner -d ukstress -c \
  "ALTER ROLE ukstress_api WITH PASSWORD 'ukstress_api';
   ALTER ROLE ukstress_etl WITH PASSWORD 'ukstress_etl';"
```

## Full build from a real dump

```bash
cd etl
uv run ukstress build \
  --dump ../data/ukwiktionary-latest-pages-articles.xml.bz2 \
  --output ../output \
  --database-url postgresql://ukstress_owner:ukstress_owner@localhost:5432/ukstress \
  --migrations ../db/migrations \
  --workers 4 \
  --publish
```

`build` runs migrate → parse → import → (optionally) publish in one
command. Without `--publish`, the result sits `validated` until you
explicitly `ukstress publish <dataset_id>`. A real run against the full
ukwiktionary dump (see `output/9dacc408065a68ee-forms-v4/manifest.json`)
produced 29,579 lexemes and 363,272 word forms from 65,346 pages, with
6,750 parse errors recorded (not silently dropped) in
`reports/parse_errors.json`.

## Dataset lifecycle commands

```bash
# Publish a validated dataset (supersedes whatever was previously active)
uv run ukstress publish --database-url "$DATABASE_URL" <dataset_id>

# Roll back to a retained published/superseded dataset
uv run ukstress rollback --database-url "$DATABASE_URL" <dataset_id>

# Delete superseded/failed datasets beyond the N most recent
# (never deletes the currently active dataset — enforced in the query)
uv run ukstress cleanup --database-url "$DATABASE_URL" --retain-count 3
```

Both `publish` and `rollback` are single-transaction, atomic operations
(see `publish_dataset`/`rollback_dataset` in `etl/src/ukstress/database.py`).
Proven live under continuous API traffic: 435 requests spanning two
dataset switches, zero errors, zero responses mixing two dataset
versions — see "Publication while API traffic is active" in
[`reports/benchmark.md`](../reports/benchmark.md).

## The volume follows the project name

`postgres-data` is declared as an ordinary Compose volume, so its real name is
`<project>_postgres-data`. Bring the same stack up under a different project
name and PostgreSQL starts on an empty cluster while the published lexicon sits
untouched in the old volume — the server is healthy, the schema is missing, and
the API dies on `load active dataset`.

This happened: the stack was first published under project `deploy`, later run
as `uk-tts`, and a `docker compose -p uk-tts up -d --build api` recreated
PostgreSQL through `depends_on` onto a fresh `uk-tts_postgres-data`. Nothing was
lost — `deploy_postgres-data` still held all 21.7 GB — but the lexicon was
offline for six minutes and a long-running sweep against the API died with it.

The deployment now pins the volume:

```yaml
volumes:
  postgres-data:
    external: true
    name: ${POSTGRES_VOLUME:-deploy_postgres-data}
```

Two habits follow. Check `depends_on` before running `up` on any service —
naming one service does not restrict Compose to that service. And when the API
reports `load active dataset` failures or an empty schema, check which volume is
mounted before assuming data loss:

```bash
docker inspect <postgres-container> --format '{{range .Mounts}}{{.Name}}{{end}}'
docker volume ls | grep postgres-data
```

## Backup and restore

There is no dedicated backup command — PostgreSQL's own tooling is the
supported path, because `import_run`/`active_dataset`/`stress_lookup` are
ordinary tables with no external state to reconcile:

```bash
# Backup (any published or retained dataset, or the whole database)
docker exec deploy-postgres-1 pg_dump -U ukstress_owner -Fc ukstress > backup.dump

# Restore into a fresh database
docker exec -i deploy-postgres-1 pg_restore -U ukstress_owner -d ukstress --clean < backup.dump
```

Because publication is dataset-scoped (`dataset_id` on every lexicon row,
`active_dataset` a single pointer row), a restore from an older backup
naturally restores the `active_dataset` pointer that was active at
backup time — there is nothing dataset-lifecycle-specific to reconcile
after a restore beyond the normal `pg_restore` invocation above.

## Rollback and retention procedure

1. **Something wrong with the active dataset** (bad build, bad publish):
   `ukstress rollback <previous_dataset_id>`. This is the fast path — no
   database restore needed, since the previous dataset's rows are still
   present (retention only deletes `superseded`/`failed` datasets, and
   only beyond `--retain-count`).
2. **Reclaiming disk from old datasets**: `ukstress cleanup --retain-count N`
   periodically (not currently scheduled automatically — run it from cron
   or a CI job if disk growth matters for your deployment). Pick `N` large
   enough to always have a rollback target; `N=1` keeps the single most
   recent superseded dataset in addition to the active one.
3. **A dataset was deleted that you needed**: restore from a `pg_dump`
   backup taken before the `cleanup` run. `cleanup` is a hard `DELETE`,
   not a soft-delete — there is no in-database undo.

## Container hardening

Verified live (not just declared in the compose file): built the `api`
image, ran it with `--cap-drop ALL --security-opt no-new-privileges:true
--read-only` exactly as `deploy/docker-compose.yml` specifies, and
confirmed `/health/live`, `/health/ready`, and a real `/v1/lookup` all
served correctly — a read-only root filesystem does not break it because
the binary performs no runtime writes.

## Benchmarking

**API (Go)** — latency/throughput against a running `stress-api`:

```bash
cd api
go build -o bin/bench-lookup ./cmd/bench-lookup
./bin/bench-lookup --base-url http://localhost:8080 \
  --words ../benchmarks/lookup_words.txt \
  --output ../reports/benchmark.json
```

`benchmarks/lookup_words.txt` is a real 10,000-word sample drawn from a
published dataset's `stress_lookup.form_normalized` column, not synthetic
data — the benchmark's job is to catch real behavior a curated fixture
would hide, and it has: see "Bugs found during this benchmark run" in
[`reports/benchmark.md`](../reports/benchmark.md).

**ETL (Python)** — parse throughput, import COPY/projection timing, and
table/index sizes from a real parse + import run:

```bash
cd etl
uv run ukstress benchmark \
  --dump ../data/ukwiktionary-latest-pages-articles.xml.bz2 \
  --output /tmp/etl-benchmark-staging \
  --database-url "$DATABASE_URL" \
  --migrations ../db/migrations \
  --workers 4 \
  --report ../reports/etl_benchmark.json \
  --report-markdown ../reports/etl_benchmark.md
```

Point `--database-url` at a **throwaway** database, not a deployment with
existing datasets: the benchmark imports the dump under its
content-derived `dataset_key`, which collides if that exact dump was
already imported there. A real run against the full ukwiktionary dump
(65,346 pages) measured 330 pages/s, 1,845 word forms/s, 165 MB peak RSS,
a 10.4s COPY phase, and a 17.5s projection-build phase — see
[`reports/etl_benchmark.md`](../reports/etl_benchmark.md) for the full
report including per-table/index byte sizes.
