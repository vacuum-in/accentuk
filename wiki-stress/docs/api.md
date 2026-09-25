# Lookup API

Full contract: [`api/openapi.yaml`](../api/openapi.yaml). Every example
below is a real response captured from the live deployment against dataset
`9dacc408065a68ee-forms-v4` (dataset_id 3), not hand-written.

## Full-text stress

`POST /v1/stress` tokenizes complete Ukrainian orthographic words, performs
one set-oriented lookup, and sends all eligible ambiguous occurrences in one
internal XLM-R call. The model can select only a signature returned by the
active database and only for a form in its frozen training manifest.

```bash
curl -X POST http://localhost:8080/v1/stress \
  -H 'Content-Type: application/json' \
  -d '{"text":"Мова, стіл і замок."}'
```

Every token reports one of `stressed`, `not_required`, `already_stressed`,
`not_found`, `ambiguous`, `model_ineligible`, or `invalid_candidate`.
Offsets are Unicode code-point offsets. Model outage preserves ambiguous
words while database-unambiguous words are still stressed. `homographs.db`
is not part of either runtime container.

## Exact lookup

```bash
curl -G http://localhost:8080/v1/lookup --data-urlencode 'word=мова'
```

```json
{
  "input": "мова",
  "status": "ambiguous",
  "candidates": [
    {
      "stressed_form": "мо́ва",
      "stress_signature": "0",
      "lemma": "мова",
      "stressed_lemma": "мо́ва",
      "part_of_speech": "noun",
      "grammatical_tags": ["nominative", "singular", "source_field:nom-sg"],
      "is_lemma": true,
      "is_variant": false,
      "is_obsolete": false,
      "confidence": 0.95
    },
    {
      "stressed_form": "мо́ва",
      "stress_signature": "0",
      "lemma": "мова",
      "stressed_lemma": "мо́ва",
      "part_of_speech": "unknown",
      "grammatical_tags": [],
      "is_lemma": true,
      "is_variant": false,
      "is_obsolete": false,
      "confidence": 0.9
    }
  ],
  "ambiguous": true,
  "truncated": false,
  "dataset_id": 3
}
```

Two entries with the *same* stressed form and signature, different
metadata — a real duplicate-provenance artifact from two source parses of
the same word, not contrived. The API reports it rather than silently
picking one; `mode=best` picks the top-ranked candidate but still marks
`ambiguous: true` so a client can't mistake it for a confident answer.

```bash
curl -G http://localhost:8080/v1/lookup --data-urlencode 'word=мова' --data-urlencode 'mode=best'
```

## Multi-token entries

Tab is a legitimate structural separator inside a lookup key, not an
error — see [architecture.md](architecture.md#unicode-representation-and-normalization-versioning):

```bash
curl -G http://localhost:8080/v1/lookup --data-urlencode $'word=бруковиця\tбруковиці'
```

```json
{
  "input": "бруковиця\tбруковиці",
  "status": "found",
  "candidates": [{
    "stressed_form": "брукови́ця\tбрукови́ці",
    "stress_signature": "0:2|1:2",
    "lemma": "бруковиця",
    "part_of_speech": "unknown",
    "grammatical_tags": [],
    "is_lemma": false,
    "is_variant": false,
    "is_obsolete": false,
    "confidence": 0.9
  }],
  "ambiguous": false,
  "truncated": false,
  "dataset_id": 3
}
```

## Not found

```bash
curl -G http://localhost:8080/v1/lookup --data-urlencode 'word=xyzabc'
```

```json
{"input": "xyzabc", "status": "not_found", "candidates": [], "ambiguous": false, "truncated": false, "dataset_id": 3}
```

No stress form is invented for an unknown word — `not_found` is a
first-class status, not an empty-candidates edge case a client has to
infer.

## Batch lookup

Preserves input order and duplicates; one PostgreSQL round trip regardless
of batch size (up to `MAX_BATCH_SIZE`, default 10,000):

```bash
curl -X POST http://localhost:8080/v1/lookup:batch \
  -H 'Content-Type: application/json' \
  -d '{"words": ["мова", "мова", "замок", "xyzabc"]}'
```

```json
{
  "results": [
    {"input": "мова", "status": "ambiguous", "candidates": [ /* … */ ], "ambiguous": true, "truncated": false, "dataset_id": 3},
    {"input": "мова", "status": "ambiguous", "candidates": [ /* … */ ], "ambiguous": true, "truncated": false, "dataset_id": 3},
    {"input": "замок", "status": "not_found", "candidates": null, "ambiguous": false, "truncated": false, "dataset_id": 3},
    {"input": "xyzabc", "status": "not_found", "candidates": null, "ambiguous": false, "truncated": false, "dataset_id": 3}
  ],
  "dataset_id": 3
}
```

## Lemma forms (paginated)

```bash
curl 'http://localhost:8080/v1/lemmas/замок/forms?limit=1'
```

```json
{"lemma": "замок", "forms": [{"form": "замок", "stressed_form": "за́мок", "grammatical_tags": [], "confidence": 0.9}], "offset": 0, "limit": 1}
```

## Errors

Stable code/message/request_id contract — never SQL, credentials, or a
stack trace (enforced by `TestContractErrorShape` in
`api/internal/httpapi/contract_test.go`):

```bash
curl -G http://localhost:8080/v1/lookup --data-urlencode 'word=мова' --data-urlencode 'mode=bogus'
```

```json
{"code": "invalid_mode", "message": "mode must be 'exact' or 'best'", "request_id": "df834a4a923d7bab"}
```

```bash
# With PostgreSQL unreachable:
curl -G http://localhost:8080/v1/lookup --data-urlencode 'word=мова'
```

```json
{"code": "database_unavailable", "message": "the lookup service is temporarily unavailable", "request_id": "41f8363670ef2de8"}
```
— `503`, and `/health/ready` reports the same condition (`not_ready`,
"database is unreachable") within one refresh tick; `/health/live` stays
`200` throughout, since the process itself is fine. Proven live by
stopping PostgreSQL under a running deployment — see
[`reports/benchmark.md`](../reports/benchmark.md#postgresql-unavailability).

## Query plans

The two hot-path queries both hit their target index (`stress_lookup_exact_idx`,
`stress_lookup_lemma_idx`) rather than scanning — real `EXPLAIN (ANALYZE,
BUFFERS)` output against the live dataset (364,130 rows):

**Exact lookup** — index-only scan, 0.3ms:

```text
Limit (actual time=0.227..0.228 rows=2 loops=1)
  -> Incremental Sort (actual time=0.226..0.226 rows=2 loops=1)
       Presorted Key: is_obsolete, confidence, source_rank
       -> Index Only Scan using stress_lookup_exact_idx on stress_lookup
            (actual time=0.096..0.097 rows=2 loops=1)
            Index Cond: ((dataset_id = 3) AND (form_normalized = 'мова'::text))
            Heap Fetches: 0
Execution Time: 0.303 ms
```

**Lemma forms** — index scan (not index-only, since it also projects
`grammatical_tags`/`confidence` beyond the index's `INCLUDE` columns for
some rows), 3.0ms:

```text
Limit (actual time=2.993..2.995 rows=19 loops=1)
  -> Sort (actual time=2.991..2.992 rows=19 loops=1)
       -> Index Scan using stress_lookup_lemma_idx on stress_lookup
            (actual time=2.255..2.914 rows=19 loops=1)
            Index Cond: ((dataset_id = 3) AND (lemma_normalized = 'мова'::text))
Execution Time: 3.052 ms
```
