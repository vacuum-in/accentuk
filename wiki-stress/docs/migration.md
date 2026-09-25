# Moving the stack to another machine

Written 2026-09-15. Three repositories, one database volume, a set of model
and data directories, and one `.env`. None of the repositories has a remote:
**the `.git` directories are the only copy of the history**, and they travel
as git bundles.

`scripts/pack_for_move.sh` produces everything under one directory with a
manifest and checksums; the "Restore" section is the other end.

## What there is, and what to do with each

| item | size | action | why |
| --- | ---: | --- | --- |
| `wiki-stress` (repo, no remote) | 1 MB bundle | **bundle** | API, ML, docs; all 64 commits. (`.git` is 670 MB of dangling loose objects from files once staged and never committed; the bundle does not carry them and nothing needs them) |
| `audiotostress` (repo, no remote) | small | **bundle** | audio ranker, miners, book scripts |
| `uk-tts-frontend` (repo, no remote) | small | **bundle** | compose, frontend, Gradio UI |
| `uk-tts-frontend/deploy/.env` | 1 KB | **copy, then edit paths** | gitignored; every host path in it changes |
| Postgres volume `deploy_postgres-data` | 18 GB database, 23.4 GB volume | **pg_dump** (custom format, compressed) | the lexicon: 21.7 GB of published data, months of imports and corrections |
| `wiki-stress/models/` | 4.2 GB | **copy** | `tok-v2` (deployed), `v3-xenc` (baked into the image), spaCy `uk_core_news_trf_380` (deployed), `tok-v1` |
| `wiki-stress/output/ml/models/` | 4.3 GB | **copy** | `v19-v10` is the deployed cross-encoder with its serving manifest |
| `wiki-stress/output/ml/` (rest) | 5.1 GB | **copy** | ONNX graphs, corpora (`corpus/tok-v2`), suffix table, trie defaults, counted forms, silver corpora, every benchmark review |
| `wiki-stress/output/` (non-ml) | 1 GB | copy | merged wordlist, form exports |
| `audiotostress/artifacts/` | 7.8 GB | **copy** | `audio_runs/all_model/ranker.pt` (the labeller), verify sweeps, `books/*.rows2*.jsonl` + labels (1.22M rows, three days of GPU), anchors, book texts |
| `verbolizer/…/e39-people-counts/final` | 202 MB | **copy** | the verbalizer checkpoint `.env` points at |
| `/home/devops/books` | 14 GB | copy if re-mining is ever wanted | commercial audio + ebooks, **not redistributable**; the derived rows are already in `artifacts/books` |
| `audiotostress/data/cv-corpus-26.0-2026-06-12` | 3 GB | copy or re-download | Common Voice 26.0 uk; needed only to re-mine or re-verify |
| `wiki-stress/data/malyuk` | 33 GB | re-download | text corpus for the placer/context experiments; not needed to serve |
| `wiki-stress/data/ukwiki-…xml.bz2` | 2.5 GB | re-download | source of the lexicon; the lexicon itself is in Postgres |
| HF cache (needed subset) | ≈8 GB | re-download offline-first, or copy the listed dirs | see below |
| `wiki-stress/backups/` | 665 MB | optional | August dumps, superseded by the fresh dump |
| `.venv`s, `~/.cache/uv` (28 GB), mypy caches, `api/bin` | 36 GB | **skip** | `uv sync` rebuilds them; docker rebuilds the images |
| docker images | ≈4.5 GB | skip | `docker compose build` |

**HF cache, the subset actually used** (`~/.cache/huggingface/hub/`):
`models--ukr-models--xlm-roberta-base-uk` (tok-v2's tokenizer/base),
`models--Yehor--wav2vec2-xls-r-300m-uk-with-small-lm` (SSL encoder for the
ranker, 1.2 GB), `models--mobiuslabsgmbh--faster-whisper-large-v3-turbo`
(book mining, 1.6 GB), `models--Systran--faster-whisper-large-v3` (anchoring,
2.9 GB), `models--skypro1111--m2m100-ukr-verbalization-ct2` (1.9 GB, if the
frontend uses it), the WhisperX alignment model for `uk` and pyannote VAD
(inside the whisperx assets). Everything else in the 27 GB cache is an
experiment (gemma, wavlm, mmBERT, xlm-roberta-large, styletts2) and can be
re-fetched if wanted.

## Pack

```
bash scripts/pack_for_move.sh /path/to/staging      # a disk with ≥ 35 GB free
```

It writes, under the staging directory: three `*.bundle` files, `lexicon.dump`
(custom-format pg_dump from the running container), `env.txt` (the `.env`,
secrets included — treat the staging directory accordingly), an `rsync` of the
copy-marked directories above (books and Common Voice only with `--with-books`
/ `--with-cv`), `MANIFEST.txt` with sizes, and `SHA256SUMS`. It does not stop
any service.

## Restore

1. `git clone wiki-stress.bundle wiki-stress` (and the other two). Bundles
   clone like remotes; `git bundle verify` first if in doubt.
2. `rsync` the copied directories back to the same relative places
   (`wiki-stress/models`, `wiki-stress/output`, `audiotostress/artifacts`, …).
   The absolute prefix may change; nothing in the repos hard-codes
   `/home/devops` except `.env` and a handful of scripts under
   `audiotostress/scripts` (`grep -rn /home/devops`).
3. `uv sync` in `wiki-stress/ml`, `wiki-stress/etl`, `audiotostress`.
   `audiotostress` also needs `ukrainian_word_stress` from the uv archive; it
   is a dependency in its `pyproject.toml`, so `uv sync` brings it, but the
   scripts reach into `~/.cache/uv/archive-v0/<hash>` for the trie — fix that
   path or symlink it (`grep -rn archive-v0 audiotostress/scripts`).
4. Postgres: bring up only the database, then
   `docker exec -i <postgres> pg_restore -U ukstress_owner -d ukstress --no-owner < lexicon.dump`.
   The compose pins the volume name `deploy_postgres-data` as external: create
   it first (`docker volume create deploy_postgres-data`) or set
   `POSTGRES_VOLUME` in `.env`. **Never start the server on an empty volume
   with the data elsewhere** — that is how the lexicon went offline once.
5. `.env`: rewrite every path (`UKSTRESS_DIR`, `STRESS_MODEL_DIR`,
   `SPACY_MODEL_DIR`, `SERVING_MANIFEST`, `SUFFIX_TABLE`, `TRIE_DEFAULTS`,
   `VERBALIZER_CHECKPOINT`, `COUNTED_FORM_LIST`, `TOKEN_MODEL_DIR`) and
   `HOST_UID`/`HOST_GID` to the new user. `ACTIVE_INVENTORY_HASH` stays.
6. `docker compose build && docker compose up -d`, then the checks in
   `docs/state.md`: `/health/ready` on `stress-model` must report
   `token_model` and a passing morphology canary; `run_live_bench.py` must
   reproduce 96.03% / 84.84%.
7. GPU work (mining, training) needs CUDA and the `nvidia` libraries the
   scripts export on `LD_LIBRARY_PATH`; the serving stack does not.

## What is not on this machine at all

The lang-uk benchmark clone lives in a session scratchpad
(`/tmp/claude-…/ukrainian-tts-preprocessing`); re-clone
`github.com/lang-uk/ukrainian-tts-preprocessing` on the new host.
