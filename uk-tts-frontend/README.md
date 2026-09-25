# uk-tts-frontend

**A text frontend for Ukrainian speech synthesis: it spells out everything a
voice cannot read, then marks which vowel of each word is stressed.**

```console
$ uktts prepare "О 7:45 ранку 3 квітня 2025 р. потяг № 12 прибув до Львова."
О сьо́мій со́рок п'ять ра́нку тре́тього кві́тня дві ти́сячі два́дцять п'я́того
ро́ку по́тяг но́мер двана́дцять прибу́в до Льво́ва.

$ uktts prepare "Замок на горі. Замок у дверях."
За́мок на горі́. Замо́к у́ две́рях.
```

The second line is the harder half. `замок` is a heteronym — the castle on the
hill and the lock in the door are spelled identically and stressed differently —
and Ukrainian has thousands of them. Getting one wrong is immediately audible.

## What it does

```
source ──▶ NFC ──▶ chunk ──▶ verbalize ──▶ stress ──▶ TTS-ready text
```

| stage | what it decides | how |
| --- | --- | --- |
| **verbalization** | how `14:30`, `1 500 грн`, `№ 12`, `iPhone 15 Pro` are said | a 52.7M-parameter Marian sequence-to-sequence model |
| **stress** | which vowel of each word carries the stress | a four-tier service: lexicon → morphology → contextual cross-encoder → suffix and compound fallbacks |

This repository is the orchestration between them: chunking, stage ordering,
batching, offset bookkeeping, a CLI, an HTTP service, and a Gradio page. It
implements no verbalization rules and no stress rules of its own — see
[docs/architecture.md](docs/architecture.md) for why that boundary is strict.

Phonemization and acoustic synthesis are downstream and out of scope.

## Why verbalization runs first

A stress lexicon has no entry for `14:30` or `1 500`. Stressing first leaves
every word the verbalizer later produces unstressed, and the voice guesses at
`чотирнадцятій` — a word that was not in the input when the stress tier saw it.
Verbalizing first hands the stress tier ordinary Ukrainian words it can look up.
The order is not configurable for that reason.

## What you need to supply

Neither model is in this repository, and neither is small.

| artifact | what it is | where it comes from |
| --- | --- | --- |
| Marian verbalizer checkpoint | ~200 MB, `config.json` + weights + SentencePiece | the [marian-uk-verbalizer](#related-projects) project |
| stress service | PostgreSQL lexicon, cross-encoder, spaCy tagger | the [ukstress](#related-projects) project |

Point `UKTTS_VERBALIZER_CHECKPOINT` at the first and `UKTTS_STRESS_URL` at the
second. Either stage can be switched off — `--no-verbalize` or `--no-stress` —
so you can run and evaluate one of them alone.

## Install

```bash
uv venv && uv pip install -e ".[service,ui,dev]"
```

Requires Python 3.10+, and `torch`, `transformers`, `sentencepiece`, `httpx`.
The service extra adds `fastapi` and `uvicorn`; the ui extra adds `gradio`.

## Run

```bash
# both stages
uktts prepare "У 1991 році Україна проголосила незалежність."

# verbalization only — no PostgreSQL needed
uktts prepare --no-stress "Компанія Apple випустила iPhone 15 Pro за \$999."

# a corpus, one input per line, JSON out
uktts prepare --file corpus.txt --lines --json > prepared.jsonl

# component readiness
uktts ready

# HTTP service on :8000
uktts serve --host 0.0.0.0 --port 8000
```

### HTTP API

```console
$ curl -s localhost:8000/v1/prepare -H 'content-type: application/json' \
    -d '{"text": "Ціна 3,5 млн грн.", "include_tokens": true}'
```

`POST /v1/prepare` takes `text` or `texts`, and optionally `on_ambiguity`
(`default` | `preserve`) and `include_tokens`. It returns one result per input:

| field | meaning |
| --- | --- |
| `source` | what you sent |
| `verbalized` | after stage 1, before stage 2 |
| `text` | the TTS-ready result |
| `chunks` | how many pieces stage 1 saw |
| `tokens` | per-word decisions, when `include_tokens` is set |
| `warnings` | anything the pipeline wants you to know |
| `timings_ms` | per-stage latency |

`GET /health/ready` returns 503 while the stress service is unreachable, and
names which component is down. `GET /health/live` stays up regardless.

### A page to poke at it

```bash
make ui     # python -m uktts.gradio_app --api http://127.0.0.1:8000
```

A Gradio page on `:7861` that calls `/v1/prepare` over HTTP like any other
client. It shows the text after verbalization and after stress side by side,
plus a per-word table naming **which tier decided each stress**. That table is
the point: in the finished line a lucky dictionary default and a confident model
decision look identical.

```
Замок  За́мок  dictionary_default  several readings, none chosen on evidence
горі   горі́    morphology          the grammatical tier decided it
Замок  Замо́к   stressed            a tier decided it
```

## Deployment

`deploy/docker-compose.yml` builds all four services — PostgreSQL, the stress
model service, the stress API, and this gateway — from two sibling checkouts.

```bash
cp deploy/.env.example deploy/.env    # then edit the paths
make up
curl -s localhost:8000/health/ready
```

See [docs/deployment.md](docs/deployment.md) for the full setup, for reusing an
existing lexicon volume rather than re-importing one, and for a troubleshooting
table of the failures this stack actually produces.

## Configuration

Every setting is an environment variable and every one has a CLI flag.

| variable | default | meaning |
| --- | --- | --- |
| `UKTTS_VERBALIZER_CHECKPOINT` | *(required)* | Marian checkpoint directory |
| `UKTTS_VERBALIZER_DEVICE` | `auto` | `auto`, `cpu`, or `cuda` |
| `UKTTS_BATCH_SIZE` | `8` | chunks per model call |
| `UKTTS_NUM_BEAMS` | `1` | beam width for verbalization |
| `UKTTS_MAX_SOURCE_TOKENS` | `384` | the checkpoint's source window |
| `UKTTS_STRESS_URL` | `http://127.0.0.1:8080` | stress API base URL |
| `UKTTS_ON_AMBIGUITY` | `default` | `default` serves the lexicon's best reading for an undecided word, `preserve` emits it unstressed |
| `UKTTS_STRESS_TIMEOUT` | `30` | seconds |
| `UKTTS_ROUTE` | `1` | send a chunk to the verbalizer only when it has something to spell out |
| `UKTTS_VERBALIZE` / `UKTTS_STRESS` | `1` | switch a stage off |

## Accuracy

Measured on lang-uk's 1,026-sentence lexical stress benchmark, **through the
running HTTP stack**, using the benchmark's own evaluator:

| metric | this pipeline | `ukrainian-word-stress` | Δ |
| --- | ---: | ---: | ---: |
| heteronym | **82.22%** | 64.34% | **+17.88** |
| macro-F1 (heteronyms) | **62.47%** | 47.26% | **+15.21** |
| sentence | **65.01%** | 41.52% | **+23.49** |
| word | **94.94%** | 88.67% | **+6.27** |
| unambiguous | 98.53% | **98.60%** | −0.07 |

Reproduce both columns with `scripts/benchmark.py` — see
[docs/benchmark.md](docs/benchmark.md).

This scores stress only; verbalization is measured separately by its own
project, at 82.90% exact match with numeric and digit fidelity reported as 100%.
There is no end-to-end figure because no public set scores both stages. Read
[docs/limitations.md](docs/limitations.md) before trusting any of them.

## Related projects

Two upstream projects supply the models. Both are separate codebases with their
own licences and their own data provenance; this repository vendors neither.

- **marian-uk-verbalizer** — trains the Marian verbalizer. Its runtime policy is
  that no hardcoded verbalization rule may exist: quality is changed by data and
  training, never by patching the model's output. This pipeline honours that.
- **ukstress** — streams the Ukrainian Wiktionary dump into a versioned
  PostgreSQL stress lexicon and serves it, with a contextual cross-encoder for
  heteronyms and a spaCy morphology tier for grammatical alternations.

## Licence and attribution

The code in this repository is [Apache-2.0](LICENSE).

**The models and data are not, and are not distributed here.** Before you
publish anything built with this pipeline, check each of these yourself:

| component | what to check |
| --- | --- |
| stress lexicon | derived from the Ukrainian Wiktionary dump — Wiktionary content is CC BY-SA 4.0, which carries attribution and share-alike obligations on the lexicon |
| verbalizer training data | traces to `skypro1111/ubertext-2-news-verbalized`; confirm that dataset's terms for your use |
| verbalizer checkpoint | a Marian model trained from scratch — no Gemma weights are in the shipped lineage, though a separate distillation branch exists in that project |
| spaCy `uk_core_news_sm` | MIT, from Ukr-Synth (MIT) |
| lang-uk benchmark | used for evaluation only; check its terms before redistributing results |

Neither upstream project carries a licence file at the time of writing. Add one
to each before publishing this stack, or the models have no grant attached.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The short version: the round-trip
invariant on chunking and the "no rules of our own" boundary are the two things
a change must not break, and both have tests.
