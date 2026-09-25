# Plan — Marian model for meaning-based homograph stress (Ukrainian)

## 1. Goal and scope

The PostgreSQL lexicon already resolves stress for every word whose spelling maps to a single
stressed realization. It cannot resolve **homographs**: one unstressed spelling, several stressed
forms, choice determined by meaning or grammar (`за́мок` / `замо́к`, `Абра́мович` / `Абрамо́вич`).

This plan covers a Marian (seq2seq) model that takes a sentence containing an ambiguous token and
returns the correct stressed form of that token, chosen from context.

Out of scope: full-sentence accentuation of non-ambiguous words (dictionary already does that),
TTS integration, dictionary coverage expansion.

## 2. Current state (measured from `output/homographs_precess_pilot_enriched2.csv`)

| Fact | Value |
| --- | --- |
| Sense rows | 4 829 |
| Homograph groups (distinct spellings) | 2 250 |
| Groups with 2 senses / 3 / 4 / 5+ | 1 982 / 221 / 38 / 9 |
| Groups where all senses share a POS (**semantic**, hard) | 1 714 |
| Groups where senses differ by POS (**syntactic**, easier) | 536 |
| Senses with a filled `definition` | 2 811 (2 018 missing) |
| Groups with ≥1 sense missing a definition | 1 200 |
| Definitions stored as a JSON-list *string* (`["…","…"]`) | 2 419 |
| Senses with `gram_style` | 4 364 (465 missing) |
| Senses with a `forms` paradigm attached | 394 |
| Groups where two rows carry the **same** stressed form | 201 |
| Senses with `priority` set (default-sense hint) | 332 |

Sources already merged: `huggingface:patriotyk/homographs-storage` + `ukwiktionary word_forms`,
plus a Wiktionary enrichment pilot (4 rows carry a Wiktionary URL).

Two data facts drive the plan:

- **1 714 semantic groups** mean POS tagging alone is not a solution. A contextual model is required.
- **201 duplicate-stressed groups** are not stress-ambiguous at all — two senses, one pronunciation.
  They must be collapsed to a single label, not trained as a distinction the model cannot learn.

## 3. Design decision — model input/output format

Three viable framings. Recommendation is **B**.

**A. Full-sentence accentor.** `Він відчинив замок ключем.` → `Він відчи́нив за́мок ключе́м.`
Rejected as the primary target: the decoder must copy every character, which invites insertions and
deletions on text the dictionary already handles perfectly, and it wastes the whole model budget on
a solved problem.

**B. Marked-span, word-output (recommended).**
Source: sentence with the ambiguous token delimited, e.g.
`Він відчинив ⟦замок⟧ старим ключем .`
Target: the stressed token only — `за́мок`.
Short target ⇒ near-zero hallucination risk, fast decoding, and it composes cleanly with the
existing dictionary: PG resolves everything unambiguous, the model is called only for spans where
`stress_lookup` returns more than one distinct signature. This is exactly the behaviour the Go API
already reports as "ambiguous".

**C. Encoder classifier.** XLM-R/`ukr-roberta` + per-group softmax over senses. Faster and probably
stronger per-parameter, but it needs a per-group head and does not generalize to unseen homographs.
Build it anyway as the **baseline to beat** — it is ~1 day of work and makes the Marian numbers
meaningful.

Inference safeguard for B: **constrained scoring**. At serve time the candidate set for a spelling
is known from the dictionary (2–9 stressed variants). Force-decode each candidate, compare sequence
log-probabilities, return the argmax. Generation becomes classification with zero possibility of
emitting a non-existent form, and it yields a calibrated confidence for free.

## 4. Phase 0 — finish the homograph inventory

The list is "in progress"; this is the blocking dependency for generation quality. The generator
prompt is only as good as the sense description it is given.

1. **Normalize `definition`.** 2 419 rows hold a Python/JSON list rendered as a string. Parse to a
   real list, keep the first 1–2 glosses, strip SUM-style register labels
   (`Народно-поетичне слово, розмовне слово чи вираз …`) into a separate `register` column.
2. **Fill 2 018 missing definitions.** Order of preference: Ukrainian Wiktionary page for the
   spelling → SUM-11/SUM-20 → LLM proposal flagged `needs_review`. Never ship an LLM-authored gloss
   unreviewed as ground truth for a *sense boundary*; it is the label definition.
3. **Fill 465 missing `gram_style`** from the PG `part_of_speech` / `grammatical_feature` tables by
   joining on the stressed lemma.
4. **Collapse the 201 duplicate-stressed groups.** Merge the sense rows into one label; keep both
   definitions in a `merged_senses` field so generated sentences still cover both meanings.
5. **Add a `contrast` field per sense** — one clause stating how it differs from its siblings. The
   pilot already does this (`відрізняється від прізвища Абрамо́вич місцем наголосу`). This field is
   the single most important prompt input: it is what forces the generator to produce
   *discriminating* context rather than generic filler.
6. **Freeze as `homographs_v1.jsonl`** with a content hash. Every downstream artifact records it.

Sense record schema:

```json
{
  "group_id": 1, "spelling": "абрамович", "sense_id": "1.a",
  "stressed_lemma": "Абра́мович", "pos": "noun", "gram": "masc, animate, patronymic",
  "definition": "чоловіче по батькові, утворене від особового імені Абрам",
  "register": null, "contrast": "на відміну від прізвища Абрамо́вич",
  "priority": 0, "forms": [{"form": "Абра́мовича", "feats": "gen.sg"}],
  "review_status": "wiktionary|sum|llm_proposed|human_confirmed"
}
```

## 5. Phase 1 — ambiguity surface and paradigm expansion

Training on lemmas only teaches the model to disambiguate 4 829 word types while real text contains
inflected forms. Two senses may be ambiguous in some cells of the paradigm and not in others.

1. Expand each sense to its full paradigm from PG `word_form` / `stress_variant` (only 394/4 829
   rows currently carry forms — most of the paradigm must come from the database join, not the CSV).
2. For each group, compute the **ambiguous form set**: unstressed form strings that collide across
   two or more senses of the group. Only these need training data. Forms that disambiguate
   orthographically are already solved by the dictionary.
3. Emit `ambiguous_forms.jsonl`: `(group_id, unstressed_form, [{sense_id, stressed_form}])`.
4. Report per group: `n_ambiguous_forms`, whether the ambiguity is nominative-only or paradigm-wide.
   Budget generation proportional to this count, not uniformly per group.

## 6. Phase 2 — dataset generation with Azure AI Foundry

### 6.1 Two complementary sources

- **Mined + labelled (preferred where available).** Pull natural sentences containing the ambiguous
  form from UberText 2.0 / Ukrainian Wikipedia / the existing dump. The LLM only *labels* which
  sense is present, plus `unclear` when the context genuinely does not decide. Cheaper, real
  distribution, real syntax.
- **Generated (required for rare senses).** For senses with too few mined hits (dialectal, obsolete,
  toponymic adjectives such as `су́хівський`), the LLM writes sentences from the sense description.

Target mix ≈ 60 % mined / 40 % generated. Generated-only groups get flagged in eval so the
distribution-shift effect is measurable.

### 6.2 Generation call design

One call per **group**, not per sense — the model must see all sibling senses at once to write
contexts that separate them.

Deployment: a `gpt-4.1` / `gpt-5`-class model on Azure AI Foundry for generation quality; a
`mini`-class deployment for the labelling pass. Use the **Batch API** (24 h window, ~50 % discount)
for the bulk run and the sync endpoint only for iteration.

Prompt skeleton (system + user, structured output enforced):

```
System:
Ти — укладач корпусу для навчання моделі наголошування. Пишеш природні українські речення.
Правила:
- Кожне речення містить цільову словоформу РІВНО один раз, без знаків наголосу.
- Контекст має однозначно вказувати на задане значення, а не на інші значення того самого написання.
- Не використовуй метамовних пояснень («слово X означає…»), не називай інші значення.
- Різні речення — різні синтаксичні конструкції, регістри й теми.
- Довжина 8–25 слів.

User:
Написання: {spelling}
Цільова словоформа: {unstressed_form}  (граматика: {feats})
Значення A ({stressed_A}): {definition_A} — {contrast_A}
Значення B ({stressed_B}): {definition_B} — {contrast_B}
Згенеруй {n} речень для значення {target_sense_id}.
```

Structured-output schema:

```json
{"type":"object","properties":{"sentences":{"type":"array","items":{"type":"object",
 "properties":{"text":{"type":"string"},
   "cue":{"type":"string","description":"слова в реченні, що визначають значення"},
   "register":{"enum":["neutral","colloquial","literary","technical"]}},
 "required":["text","cue","register"],"additionalProperties":false}}},
 "required":["sentences"],"additionalProperties":false}
```

The `cue` field is not decoration: it is a cheap self-check, and a sentence whose `cue` is empty or
not present in `text` is discarded automatically.

Sampling: `temperature` 0.9, `top_p` 0.95, `n` sentences per call = 10, several calls per sense with
a rotating topic hint (`побут`, `історія`, `техніка`, `новини`, `художня література`) to prevent the
model from writing ten variations of one scene.

### 6.3 Volume and cost

| Tier | Senses | Sentences/sense | Total |
| --- | --- | --- | --- |
| Ambiguous in many forms, common | ~1 500 | 80 | 120 k |
| Standard | ~2 400 | 50 | 120 k |
| Rare / obsolete / toponymic | ~900 | 25 | 22 k |
| **Total** | | | **≈ 260 k** |

Roughly 26 k batch calls, ~18 M input and ~13 M output tokens. At mini/4.1-class batch rates this is
in the tens of dollars — confirm against current Foundry pricing before committing the full run.
Run a **200-group pilot first** and gate the full run on its QA pass rate.

### 6.4 Engineering

New package `ml/` alongside `etl/`, same tooling (uv, ruff, mypy, pytest):

```
ml/
  src/ukstress_ml/
    inventory.py      # phase 0/1: normalize, expand paradigms, ambiguous form sets
    generate.py       # Azure Foundry batch client, resumable, checkpointed by group_id
    mine.py           # corpus mining of natural sentences
    validate.py       # phase 3 filters
    dataset.py        # splits, balancing, Marian source/target rendering
    train.py          # HF Seq2SeqTrainer
    evaluate.py       # metrics + confusion reports
    serve.py          # CTranslate2 inference sidecar
  tests/
```

Requirements: resumable by `group_id` checkpoint, all raw responses persisted to
`output/ml/raw/*.jsonl.zst` before any parsing, cost/token counters logged per batch, model
deployment name + API version recorded in the run manifest. Secrets via `AZURE_OPENAI_ENDPOINT`,
`AZURE_OPENAI_API_KEY` (or Entra ID token) in `.env`, never committed.

## 7. Phase 3 — validation (automatic, then human)

Every generated or mined sentence passes all of:

1. **Form check.** After NFD normalization and U+0301 removal, the target form occurs exactly once.
2. **Paradigm check.** The assigned stressed form is a real form of the assigned sense.
3. **Leakage check.** The sentence contains no stress marks, no other sense's stressed spelling, no
   metalinguistic phrasing (`наголос`, `означає`, `у значенні`).
4. **Cue check.** `cue` is non-empty and its tokens appear in the sentence.
5. **Dedup.** Exact + near-duplicate (MinHash, Jaccard ≥ 0.8 on 5-grams) within and across senses.
6. **Blind re-label.** A *different* deployment reads the sentence with the target masked and the
   sense list, and picks the sense. Disagreement with the intended label ⇒ quarantine. This is the
   strongest single filter: it verifies the context actually disambiguates.
7. **Length/register spread.** Reject a sense's batch if >40 % of sentences share the first three
   tokens.

Then a human reviews a stratified sample: 300 sentences across the hardest groups (same-POS,
low-frequency). Ship gate: **≥ 95 % correct** in the sample. Below that, revise the prompt and
regenerate rather than filtering harder.

Expected yield after filters: 70–80 % of generated volume.

## 8. Phase 4 — splits and encoding

**Splits.** Sentence-level, stratified by `(group_id, sense_id)`, so every group appears in train.
Ratio 80/10/10. Additionally hold out:

- `test_unseen_groups`: 150 whole groups never seen in training — measures generalization to
  homographs outside the inventory.
- `test_natural`: human-labelled sentences mined from real text only — the honest number.
- `test_hard`: same-POS groups only.

**Encoding (format B).**

```
src: Він відчинив ⟦ замок ⟧ старим ключем .
tgt: за́мок
```

Context window: ±12 tokens around the target, truncated to 128 source tokens. Target NFD-normalized,
consistent with `etl/src/ukstress/normalizer.py` — reuse that module, do not reimplement the
normalization rules (apostrophe `ʼ`, dash unification, NFD, U+0301 handling).

**Tokenizer.** SentencePiece unigram, shared vocab 8 k, trained on the generated corpus plus a
Ukrainian Wikipedia sample; `character_coverage=1.0`, `⟦`/`⟧` as user-defined symbols. Target side
uses a character-level SentencePiece (~120 symbols) since outputs are single words — this keeps the
combining acute a first-class token. Build `source.spm`, `target.spm`, `vocab.json` and load with
`MarianTokenizer`.

**Balancing.** Cap the sense-frequency ratio inside a group at 4:1. Do not balance `test_natural` —
its skew is the point.

## 9. Phase 5 — training

Base: `transformers.MarianMTModel` initialized **from scratch** from a `MarianConfig` (there is no
useful pretrained uk→uk-stress checkpoint; a translation checkpoint's embeddings do not transfer to
this label space).

Starting config — asymmetric, because the target is one word:

```python
MarianConfig(
    vocab_size=8000, d_model=512,
    encoder_layers=6, decoder_layers=2,
    encoder_attention_heads=8, decoder_attention_heads=8,
    encoder_ffn_dim=2048, decoder_ffn_dim=1024,
    max_position_embeddings=256, dropout=0.1, activation_function="swish",
    share_encoder_decoder_embeddings=False,
)
```

≈ 35 M parameters. Training: `Seq2SeqTrainer`, effective batch 128 sentences, lr 5e-4, inverse-sqrt
schedule with 4 000 warmup steps, label smoothing 0.1, bf16, 20–40 epochs over ~200 k examples with
early stopping on dev **macro-accuracy over groups** (not loss, not BLEU). Single A10/4090:
roughly 2–4 hours per run.

Ablations worth running (they decide the final recipe, and each is a config flip):

- encoder depth 4 vs 6 vs 8;
- context window ±8 / ±12 / full sentence;
- span markers `⟦ ⟧` vs a prefix tag `<hom> замок </hom>` vs positional index token;
- with/without the mined half of the corpus.

If throughput becomes the constraint, switch to native `marian-nmt` for training and keep the same
data format; the HF path is chosen for iteration speed, not final performance.

## 10. Phase 6 — evaluation

Primary metric: **macro accuracy over homograph groups** (each group weighted equally, so 2 000
rare groups are not drowned by a few frequent ones).

Report also:

| Metric | Why |
| --- | --- |
| Micro accuracy on `test_natural` | Real-world expectation |
| Accuracy on `test_hard` (same-POS) | The actual difficulty |
| Accuracy on `test_unseen_groups` | Does it learn semantics or memorize labels |
| Accuracy vs. **priority baseline** (always pick the default sense) | The bar to clear |
| Accuracy vs. **classifier baseline** (C) | Architecture sanity check |
| Confidence calibration (ECE) on constrained scoring | Needed for the abstain threshold |
| Per-group confusion table, worst 50 groups | Drives the next data round |

Ship gates: macro accuracy ≥ 90 %, `test_natural` ≥ 92 %, no group below 60 % without being listed
in a known-limitations file, and calibrated abstention such that at the chosen threshold the model
either answers correctly or defers to `priority` ≥ 97 % of the time.

## 11. Phase 7 — serving

1. Convert to CTranslate2 (`ct2-transformers-converter`), int8 quantized. Expect a few ms per token
   on CPU — the model runs only on ambiguous spans, so per-sentence cost stays low.
2. Python FastAPI sidecar exposing `POST /v1/disambiguate` with
   `{"sentence": …, "spans": [{"start": …, "end": …, "candidates": [ … ]}]}`, returning the argmax
   candidate plus log-prob margin. Constrained scoring over `candidates` — never free generation.
3. Go API flow: tokenize → `stress_lookup` per token → single-variant tokens resolved from PG →
   remaining ambiguous tokens batched into one sidecar call → margin below threshold falls back to
   the `priority` sense → response marks which tokens were model-resolved.
4. Both containers into `deploy/`, sidecar health-checked, model version pinned in the manifest
   alongside the `homographs_v1` hash.

## 12. Risks

| Risk | Mitigation |
| --- | --- |
| LLM-written sentences are stylistically uniform; model learns the generator, not the language | Mined half of corpus; `test_natural` is the reported number |
| Sense definitions are wrong or overlap ⇒ unlearnable labels | Phase 0 human review; blind re-label agreement rate per group as a data-quality signal |
| 1 200 groups still lack definitions | Phase 0 is a hard gate; generation for a group is blocked until its senses are complete |
| Generated contexts leak the answer (metalinguistic phrasing) | Filter 3, plus blind re-label with the target masked |
| Rare senses stay rare after balancing | Per-group floor of 25 sentences; report per-group accuracy, never only the average |
| Marian emits a non-existent form | Constrained candidate scoring at inference — structurally impossible |
| Azure cost overrun | 200-group pilot gate, Batch API, token counters per batch, hard spend cap |

## 13. Milestones

| # | Deliverable | Gate |
| --- | --- | --- |
| M0 | `homographs_v1.jsonl` — 4 829 senses complete, duplicates collapsed, contrasts written | Human review of 200 senses |
| M1 | `ambiguous_forms.jsonl` + per-group generation budget | Paradigm coverage report |
| M2 | Pilot corpus, 200 groups | QA pass ≥ 95 % on human sample |
| M3 | Full corpus ≈ 260 k sentences, post-filter | Filter yield ≥ 70 %, dedup verified |
| M4 | Classifier baseline (C) numbers | Baseline recorded |
| M5 | Marian v1 trained | Beats priority baseline and classifier baseline |
| M6 | Eval report + known-limitations list | Ship gates in §10 met |
| M7 | CT2 sidecar + Go integration | End-to-end latency measured on this machine |

## 14. Open questions

1. Full-sentence accentuation (option A) as a later product, or does the dictionary + span model
   remain the shipping surface? Affects whether the corpus keeps full stressed sentences.
2. Licensing of the mined corpus (UberText 2.0 terms, Wikipedia CC BY-SA attribution) if the dataset
   is published.
3. Is the generated dataset itself a deliverable (HF dataset card, provenance per row) or internal
   training input only? Provenance columns are cheap now and impossible to reconstruct later.
