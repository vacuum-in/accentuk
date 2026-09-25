# Architecture

## Component boundary

Python (`etl/`) owns every offline concern: dump download, streaming
extraction, normalization, validation, deduplication, PostgreSQL schema
migrations, bulk import, dataset publication/rollback/retention, reports,
and exports. Go (`api/`) owns exactly one thing: a read-only HTTP lookup
service over PostgreSQL. This boundary is enforced by convention and by
`openspec/config.yaml`'s `operations.apply.guidance`
("do not silently broaden Go responsibilities"), not by a build-time
check — a Go PR that adds dump parsing, morphology extraction, or a
database migration is a boundary violation regardless of whether it
compiles.

```text
bz2 XML dump
  → Ukrainian-section isolation (etl/src/ukstress/language_sections.py)
  → template/table/headword extraction (wikicode.py, table_parser.py, entry_parser.py)
  → normalization + validation (normalizer.py, validator.py)
  → deduplication (deduplicator.py)
  → zstd JSONL staging (staging.py)
  → PostgreSQL COPY + set-based merge (database.py)
  → stress_lookup projection (indexed for exact/lemma retrieval)
  → Go HTTP API (api/internal/repository, api/internal/httpapi)
```

The Go side never touches staging tables, migrations, or the ETL roles —
see `db/migrations/005_roles.sql`: `ukstress_api` has `SELECT`-only grants,
enforced by PostgreSQL itself, not just application code (proved in
`api/internal/repository/repository_test.go` and by `db/tests/verify_migrations.sh`
in CI).

`build-homograph-stress-disambiguation` (a separate, later openspec
change, not yet implemented) amends this boundary narrowly: it adds an
**internal-only** Python inference service that Go may call for spans it
has already identified as ambiguous. Go still owns no model lifecycle
code and still exposes the only public HTTP interface. See that change's
`design.md` and its `specs/disambiguation-api/spec.md` MODIFIED
requirement for the full amendment.

## Which reading wins

A form can be named by six sources, and until recently their precedence was
not written down anywhere. It is worth writing down, because one of the layers
was silently overruling a layer above it for a year.

The order, from the answer that ships to the answer of last resort:

1. **A contextual tier** — the morphology parse, or the XLM-R model. If either
   resolves a form, its answer stands and the token is reported `stressed` or
   with a `rule`. Nothing below is consulted.
2. **The candidate order from the lexicon**, whose head becomes
   `dictionary_default` when no contextual tier spoke.
3. **The suffix table**, for forms the lexicon has never seen.
4. **The compound fallback**, for forms it can split.

Layer 2 is where the interesting failure lived. The candidate list comes from
`Repository.batchSignatures`, ordered by `is_obsolete, confidence DESC,
source_rank`, across the active dataset plus `SUPPLEMENTARY_DATASETS`. Then,
after the query, `preferTrieDefault` moved the reading the source trie names to
the front of that list.

That reordering is a good prior. Where the trie and the lexicon order disagree
on lang-uk's benchmark, the trie is right 26 times against 12. But it applied
unconditionally, including over a reading a person had reviewed and written at
confidence 1.00 — so a correction could be applied to the lexicon, verified in
the database, and still not appear in the response. `ма́ю` and `того́` were
corrected and served as `маю́` and `то́го` for as long as anyone looked.

`REVIEWED_DATASETS` is the exception. Datasets named there hold decisions a
person made; the batch query marks their rows, and a form carrying one leads
with that reading instead of the trie's. With the variable unset the behaviour
is exactly as before. On lang-uk's benchmark through the live service the fix
moved heteronym accuracy from 83.10% to 84.62% and sentence accuracy from
69.30% to 69.98%.

### Ordering was not enough, and why

`REVIEWED_DATASETS` put the reviewed reading first, and `маю` was still served
as `маю́`. The reason is layer 1, working exactly as designed: `маю` carried
two candidates, and **two candidates are what makes a form model-eligible**
(`handleStress`: `case 1` serves a single candidate outright, `default` sends
the token to the contextual model). So the model was handed a form a person had
already decided, and chose the reading the review had rejected. No amount of
reordering fixes that — the tier is *supposed* to override the default.

So the corrections are now split by what the review actually said.

**Forms whose other reading is simply wrong** go to
`manual-corrections-exclusive-v1`, named in `REVIEWED_EXCLUSIVE_DATASETS`. The
batch query marks their rows and the candidate list collapses to that one
signature, so the form takes the `case 1` path and **no tier is offered it at
all**. Sixteen forms: `його`, `Київ`, `народу`, `народові`, `маю`, `дівчата`,
`дівчат`, `беру`, `народом`, `сором`, `розумів`, `гроші`, `користуватися`,
`богдане`, `одержав`, `болярин`.

**Forms with two valid readings**, where only the default was wrong, stay in
`manual-corrections-v1` with both candidates and the reviewed one leading —
there the tier *should* decide, and if it decides badly that is a training-data
problem, not a precedence one. Seven forms: `того`, `була`, `кого`, `років`,
`всього`, `залишилися`, `ніяк`.

`run_manual_corrections.py` writes both datasets. Given `--manifest` it also
drops the corrected forms from the model's coverage — a flag that existed for
precisely this reason and had never been passed, so the model kept being asked
about forms already settled. Pass it.

On lang-uk through the live service the split moved heteronym accuracy 84.62% →
84.73%, sentence 69.98% → 70.27%. Small, because it touches sixteen forms; the
mechanism is the point, since every future correction met the same wall.

### One thing this still does not fix

**Supplementary datasets fill gaps; they do not override.** A reading written
to a supplementary dataset reaches the response only where the form is absent
from the active dataset — which is why correcting an absent form like
`болярин` worked immediately and correcting `маю` needed the exclusive path. To
change a form the active dataset already holds, edit the active dataset
(`ml/scripts/apply_stress_overrides.py`), not a supplementary one.

### Where each layer lives

| layer | source | configured by |
| --- | --- | --- |
| morphology | spaCy parse, internal Python service | `MORPHOLOGY_TIMEOUT` |
| model | XLM-R cross-encoder, frozen manifest | `MODEL_URL`, `MODEL_MANIFEST` |
| reviewed readings (leading) | `stress_lookup` rows a person decided | `REVIEWED_DATASETS` |
| reviewed readings (exclusive) | the same, collapsed to one candidate | `REVIEWED_EXCLUSIVE_DATASETS` |
| trie default | JSON map from `run_trie_defaults.py` | `TRIE_DEFAULTS` |
| lexicon order | `stress_lookup`, confidence then source rank | `SUPPLEMENTARY_DATASETS` |
| suffix table | JSON map | `SUFFIX_TABLE` |

## Dataset lifecycle

`import_run` rows move through `building → validated → published →
superseded` (or `failed`). `active_dataset` is a singleton table holding
exactly one `dataset_id`; the Go API reads it every
`DATASET_REFRESH_INTERVAL` (default 5s) and serves whichever dataset was
active at the start of each request — a publish never causes one response
to mix rows from two dataset versions (proved under live traffic in
`reports/benchmark.md`, "Publication while API traffic is active").

- `ukstress import` loads a staging directory, runs quality gates
  (`enforce_quality_gates` in `database.py`), and leaves the dataset in
  `validated` state — never directly active.
- `ukstress publish <id>` atomically activates a `validated` dataset and
  marks the previous active one `superseded`.
- `ukstress rollback <id>` atomically reactivates a retained `published`
  or `superseded` dataset.
- `ukstress cleanup --retain-count N` deletes `superseded`/`failed`
  datasets beyond the N most recent, **never** the currently active one
  (enforced in the query itself, not just by convention — see
  `cleanup_retained_datasets` in `database.py`).

See [operations.md](operations.md) for exact commands and
[the benchmark report](../reports/benchmark.md) for a live PostgreSQL-outage
and publish-under-traffic test.

## Unicode representation and normalization versioning

All stressed text is stored as **NFD** (canonical decomposition), never
NFC — a stressed vowel is the base letter followed by a combining acute
accent (`U+0301`), not a precomposed character. This is deliberate: NFD
lets the normalizer treat the acute as an independent, strippable code
point (`lookup_key` = NFD text, lowercased, with only `U+0301` removed)
without needing a table of every precomposed Cyrillic+acute combination.

Canonicalization rules, implemented once in
`etl/src/ukstress/normalizer.py` and re-implemented — deliberately, not
duplicated by accident — in Go at `api/internal/normalize/normalize.go`:

| Rule | Detail |
| --- | --- |
| Apostrophe | `' \` U+2018 U+2019 U+02BB U+FF07` → canonical `U+02BC` |
| Internal hyphen | `U+2010–U+2014, U+2212` between two non-space characters → plain `-` |
| Stress mark | Only `U+0301` (combining acute) is ever added or stripped |
| Structural separators | Space and **tab** are valid within a lookup key (tab marks a multi-token entry, e.g. `"бруковиця\tбруковиці"` — see the "Bugs found" section of `reports/benchmark.md` for how this was actually discovered) |
| Case | `lookup_key` lowercases after NFD/canonicalization, never before |

**Conformance is enforced, not just documented.** Python generates
`artifacts/normalization-conformance.json` (`etl/src/ukstress/conformance.py`);
Go's `TestConformanceVectors` in `api/internal/normalize/normalize_test.go`
fails the build if its output ever diverges from that file. There is
exactly one normalization implementation per language, and they are
proven to agree — not merely asserted to.

Stress signatures (`normalizer.stress_signature`) encode stress position
as a **zero-based stressed-vowel ordinal**, never a byte or code-point
offset, because those shift under re-normalization and a vowel ordinal
does not: `мо́ва` → `0` (stress on the 0th vowel), `вода́` → `1`. Multi-token
values use `segment:ordinal` pairs joined by `|`, e.g. `0:2|1:2`.

The normalization rule set itself is versioned (`normalization_version`
in `ImportManifest` and `import_run.normalization_version`, currently
`"1"`); a future rule change bumps that version rather than silently
reinterpreting already-published data.
