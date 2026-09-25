# Continuing training on a CUDA machine

Everything up to and including the corpus is finished and machine-independent.
What remains is model training, which on this Mac runs at roughly **36 minutes
per epoch** on MPS. The same run on a single modern CUDA GPU should take a small
fraction of that, which makes the experiments listed in §6 practical rather than
overnight.

No code changes are required to run on CUDA. `select_device()` in both
`ml/src/ukstress_ml/train.py` and `ml/src/ukstress_ml/crossencoder.py` already
prefers `cuda` when it is available, then `mps`, then CPU.

## 1. What to copy

```bash
bash ml/scripts/make_gpu_bundle.sh
```

That writes `ukstress-gpu-bundle.tar.gz` (~20 MB) and prints its SHA-256.

| Included | Why |
| --- | --- |
| `ml/`, `etl/` sources and `pyproject.toml` | The pipeline; `ml` depends on `etl` for canonicalization |
| `output/ml/corpus/v3/` | Train 11 165 · dev 427 · test_natural 759, plus the manifest |
| `output/ml/inventory_v1.jsonl` + manifest | Frozen senses and the content hash every artifact records |
| `output/ml/ambiguous_forms.jsonl` | Candidate glosses — the cross-encoder needs one per sense |
| Coverage and generation reports | Provenance for the numbers in `RESULTS.md` |

| Excluded | Why |
| --- | --- |
| `data/ukwiki-*.xml.bz2` (2.7 GB) | Mining is done; re-download only to re-mine |
| `output/ml/models/*` (1.6 GB) | Shipped separately — see §1a |
| `output/ml/raw/`, `mined/ukwiki/candidates.jsonl` | Only needed to re-assemble the corpus |
| `.env` | **Never copy.** Azure credentials are not needed for training |

## 1a. Moving the trained model

The code-and-corpus bundle is ~3 MB and travels anywhere. The trained weights
are 1.1 GB and are packaged separately, so moving the pipeline never means moving
a gigabyte:

```bash
bash ml/scripts/make_model_bundle.sh output/ml/models/v3-xenc
```

That produces `v3-xenc-model.tar.gz` with the checkpoint **and its provenance**
(`training_run.json`, `evaluation.json`, `release_report.json`,
`benchmark.json`). The provenance travels with the weights deliberately: without
it a checkpoint cannot be tied back to the corpus version and inventory hash it
was trained against, and an untraceable checkpoint is not a releasable one.

Restore from the repository root with `tar -xzf v3-xenc-model.tar.gz`.

### Git LFS (configured)

The released model is versioned with Git LFS. Two directories, two purposes:

| Path | Contents | Git |
| --- | --- | --- |
| `output/ml/models/` | Every training run | Ignored |
| `models/` | Released artifacts only | Tracked, LFS |

Promotion is deliberate, so experiments never enter history by accident:

```bash
bash ml/scripts/promote_model.sh output/ml/models/<run> <name>
```

It moves the checkpoint into `models/<name>/`, copies the provenance JSON
alongside, and stages the result. `.gitattributes` routes `*.safetensors`,
`*.bin`, `*.onnx`, `*.spm` and `models/**/tokenizer.json` through LFS while
keeping the small provenance JSON as readable, diffable text.

Verified: the 1.1 GB `model.safetensors` is a **135-byte pointer** in the index;
the blob lives in `.git/lfs/objects`.

**On another machine:**

```bash
git lfs install --local
git clone <remote> && cd wiki-stress     # or: git lfs pull
./ml/.venv/bin/python ml/scripts/run_final_report.py models/v3-xenc output/ml/corpus/v3
```

Without `git lfs install`, a clone yields pointer files instead of weights and
`from_pretrained` fails on an unreadable safetensors file.

**Quota warning.** GitHub's free tier gives **1 GB LFS storage and 1 GB/month
bandwidth**. This single checkpoint is 1.1 GB, so pushing it to a free GitHub
account will fail or require a paid data pack. Each promoted model version adds
another ~1.1 GB, and LFS history is not pruned by deleting the file. Before
pushing, confirm the remote's LFS quota, or keep the weights on a self-hosted
remote, an artifact store, or a private Hugging Face repo and version only the
provenance JSON in git.

`*.tar.gz` and `output/` remain in `.gitignore`, so neither bundle nor any
training run can be committed by accident.

### Continuing training from the checkpoint

`base_model` accepts a local directory as well as a Hub id, and loading a
checkpoint preserves the trained classifier head rather than re-initialising it
(verified by comparing head weights with and without `num_labels=1`).

```bash
# continue from the trained weights into a NEW run directory
./ml/.venv/bin/python ml/scripts/run_crossencoder.py v3 output/ml/models/v3-xenc/checkpoint xenc2
```

The third argument is the run tag. It is required when resuming: without it the
run would write back into the directory holding the checkpoint it started from,
and the script now refuses that rather than overwriting its own starting point.

Note that resuming restarts the learning-rate schedule and the optimiser state —
`AdamW` moments are not saved. For a short continuation use a lower
`learning_rate` (1e-5 or below); for a clean comparison, retrain from the
pretrained base instead.

## 2. Setup on the GPU machine

```bash
tar -xzf ukstress-gpu-bundle.tar.gz && cd wiki-stress
uv venv --python 3.12 ml/.venv
VIRTUAL_ENV=ml/.venv uv pip install torch --index-url https://download.pytorch.org/whl/cu124
VIRTUAL_ENV=ml/.venv uv pip install -e etl -e ml
./ml/.venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0), torch.__version__)"
```

Install the CUDA build of torch **before** `-e ml`, otherwise the CPU wheel is
picked up as a dependency. Match the `cu` suffix to the host driver.

## 3. Reproduce the current result first

Before changing anything, confirm the pipeline behaves identically:

```bash
./ml/.venv/bin/python -m pytest ml/tests -q
./ml/.venv/bin/python ml/scripts/run_crossencoder.py v3
```

Expect `test_natural` group-macro ≈ 0.90 against a 0.7678 majority baseline. A
materially different number means the environment differs, not the method — stop
and compare `training_run.json` (it records seed, library versions and hardware)
before drawing conclusions.

## 4. Settings worth changing on CUDA

In `ml/src/ukstress_ml/crossencoder.py`, `CrossEncoderConfig`:

| Field | Mac (MPS) | CUDA suggestion | Note |
| --- | --- | --- | --- |
| `batch_rows` | 16 | 64–128 | Each row expands to 2–3 pairs, so effective batch is 2–3× |
| `learning_rate` | 2e-5 | 2e-5 → 3e-5 | Scale with batch size, not blindly |
| `max_epochs` | 14 | 10–20 | Cheap enough to let early stopping decide |
| `max_length` | 192 | 192 | Sentence plus two glosses fits comfortably |
| `base_model` | `xlm-roberta-base` | try `xlm-roberta-large` | §6 |

Mixed precision is not wired in. On CUDA it is worth adding around the forward
and backward pass in `crossencoder.train`:

```python
scaler = torch.amp.GradScaler("cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    logits = model(**encoded).logits.squeeze(-1)
    loss = _listwise_loss(logits, batch["spans"], batch["golds"])
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
scaler.step(optimizer)
scaler.update()
```

Keep `_listwise_loss` in fp32 if bf16 makes the softmax across candidates
unstable — it operates on 2–3 logits per row, so the cost is negligible.

## 5. Where the numbers stand

Measured on this machine, `test_natural` (759 held-out natural Wikipedia
sentences, 202 known groups, generated text excluded):

| Metric | Marian 30M from scratch | Cross-encoder 278M | Majority baseline |
| --- | --- | --- | --- |
| Group-macro | 0.7161 | 0.8961 | 0.7678 |
| Multi-sense groups | 0.5899 | 0.8362 | 0.5930 |
| Minority-sense recall | 0.4200 | 0.7867 | 0.0 |

The 6-epoch cross-encoder had not converged; a 14-epoch run reached dev-macro
0.9092 by epoch 4. See `RESULTS.md` for the full history and the reasoning.

## 6. Experiments the GPU makes practical, in priority order

1. **`xlm-roberta-large` (560M).** The single most likely accuracy gain. Out of
   reach on MPS at any reasonable epoch time. Expect to lower `batch_rows` and
   `learning_rate` (1e-5 is a sane start).
2. **Longer schedules with real early stopping.** The base model was still
   improving when the budget ran out; `patience=3` over 20 epochs settles it.
3. **A Ukrainian-specific encoder** — e.g. `youscan/ukr-roberta-base` — against
   the multilingual one, same corpus and seed.
4. **More data per sense.** The corpus averages 22 rows per group. Raising the
   generation floor from 12 to 40 (`ml/scripts/run_generate.py`, `FLOOR`) is the
   direct lever on minority-sense recall; it needs Azure credentials and re-runs
   `run_generate.py` → `run_assemble.py` → training.
5. **Seed variance.** Every figure so far is a single run. Three seeds give an
   honest error bar before anything is called an improvement.

## 7. What is still incomplete, regardless of hardware

These are data problems and no GPU fixes them:

- **Coverage is 570 of 2 250 groups.** Extending needs definitions for the
  1 900 senses that lack one — the cross-encoder requires a gloss per sense.
- **Lemma forms only.** The model sees `замок` but never `замка́ми`. This needs a
  vendored accented morphological dictionary (VESUM / `dict_uk`); transferring
  stress across a mobile-stress paradigm without one would be guessing. It also
  caused 36% of generated sentences to be rejected for using an inflected form.
- **524 of 2 035 ambiguous forms have zero natural examples** in Ukrainian
  Wikipedia, so they depend entirely on generation.
- **Evaluation covers 202 groups** — those occurring with more than one sense in
  natural text. The rest are untested for disambiguation.

## 8. Serving

Training hardware and serving hardware are separate decisions. The serving
target is CPU behind the existing Go API, which calls the model only for spans
`stress_lookup` reports as ambiguous. `ml/scripts/run_benchmark.py` measures
fp32 against int8-dynamic at 1/2/4/8 spans per request; run it on an idle
machine, because a latency figure taken under training load is not a serving
figure. If int8 CPU latency proves unacceptable, the fallback is the abstention
path already specified: below the margin threshold the API returns the
dictionary's default sense rather than a model choice.
