# Deployment

The compose stack runs four services:

| service | what it is | port |
| --- | --- | --- |
| `postgres` | the stress lexicon | — |
| `stress-model` | cross-encoder + spaCy morphology | — |
| `api` | the stress API (Go) | 8080 |
| `frontend` | this pipeline | 8000 |

## Setup

Two checkouts must exist alongside this one — the `ukstress` project and the
verbalizer project — and you need a populated lexicon database and two model
directories.

```bash
cp deploy/.env.example deploy/.env
$EDITOR deploy/.env          # every path, and ACTIVE_INVENTORY_HASH
make up
curl -s localhost:8000/health/ready
```

`make up` is `docker compose --env-file deploy/.env -f deploy/docker-compose.yml
up -d --build`, plus `deploy/docker-compose.local.yml` when that file exists.

### What each path must point at

| variable | expects |
| --- | --- |
| `UKSTRESS_DIR` | the ukstress checkout — the build context for both stress images |
| `STRESS_MODEL_DIR` | a directory containing `checkpoint/` **and** `serving_manifest.json` |
| `SERVING_MANIFEST` | that same manifest file |
| `SPACY_MODEL_DIR` | a spaCy Ukrainian model directory, e.g. `uk_core_news_sm` |
| `SUFFIX_TABLE`, `TRIE_DEFAULTS` | the two JSON lookup tables the stress API reads |
| `ACTIVE_INVENTORY_HASH` | must equal the manifest's own `inventory_hash`, or the API refuses the model and quietly serves without it |
| `VERBALIZER_CHECKPOINT` | a Marian checkpoint directory |

The manifest has to live **inside** `STRESS_MODEL_DIR` on the host. runc cannot
create a mountpoint inside a read-only bind, so mounting it separately into the
model directory fails to start the container.

## Reusing an existing lexicon

Importing the Wiktionary dump takes hours and the resulting volume is tens of
gigabytes. To reuse one you already have, declare it external in a local overlay:

```yaml
# deploy/docker-compose.local.yml   (gitignored)
services: {}
volumes:
  postgres-data:
    external: true
    name: <the existing volume name>
```

`make up` picks the file up automatically. Declaring it external also means
`docker compose down -v` cannot destroy it.

**Never run two PostgreSQL servers against one data directory.** If the ukstress
project has its own compose stack, keep only one of them up. This stack uses the
project name `uk-tts` so the two do not otherwise collide.

## Security posture

Every container drops all capabilities, sets `no-new-privileges`, and runs with
a read-only root filesystem plus a small tmpfs where a writable path is needed.
Model directories are mounted read-only.

`stress-model` runs as `HOST_UID:HOST_GID` rather than as the image's own user.
Safetensors files are written mode 600, so a mounted checkpoint is unreadable to
any other uid — and `transformers` reports that as a *missing file*, not as a
permission error.

## Verifying

```console
$ curl -s localhost:8000/health/ready | jq
{"ready": true, "verbalizer": "marian", "stress": {"backend": "http", "ready": true, ...}}

$ curl -s localhost:8000/v1/prepare -H 'content-type: application/json' \
    -d '{"text": "Замок на горі. Замок у дверях."}'
```

The stress API is published on 8080 as well, so a suspect reading can be checked
against that tier alone:

```bash
curl -s localhost:8080/v1/stress -H 'content-type: application/json' \
  -d '{"text": "Не вистачає руки."}'
```

The API logs one line per optional tier at startup. All three should appear;
their absence means that tier is silently off:

```
supplementary datasets active  ids=[3 5 7 9 10]
trie defaults enabled          forms=53995
suffix fallback enabled
```

## Troubleshooting

Failures this stack actually produces, and what each one means.

| symptom | cause |
| --- | --- |
| build sends tens of GB to the daemon | the ukstress repo root is the build context; it needs a `.dockerignore` |
| `ResolutionImpossible` on `optimum[onnxruntime]` | pip cannot satisfy an extra that a *separate* distribution provides; resolve that step with `uv` instead |
| `read-only file system` creating a mountpoint | a bind nested inside another read-only bind — see the manifest note above |
| `FileNotFoundError: ... model.safetensors` although the file is there | wrong uid; the file is mode 600. Set `HOST_UID`/`HOST_GID` |
| `requires torch >= 2.6 ... CVE-2025-32434` | the checkpoint is `pytorch_model.bin`, not safetensors; `transformers` refuses `torch.load` below 2.6 |
| heteronym accuracy collapses with no error | a spaCy model that loads but tags nothing resolves 0 tokens on the morphology tier and costs ~17 points silently |
| `/health/ready` 503 with `unreachable` | the stress API is down; the pipeline refuses rather than returning unstressed text |
