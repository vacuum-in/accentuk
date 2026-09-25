# Design: Ukrainian Audio-Derived Lexical Stress Mining

## 1. Goals

Build a reproducible pipeline that learns to recognize Ukrainian lexical stress from audio and uses it as a high-precision labeling oracle for context-dependent stress variants.

The final production consumer is expected to be a text-only contextual stress resolver, so audio is used offline to create training data rather than at TTS inference time.

Primary design principle:

> Optimize the mining system for label precision and auditability, not maximum corpus coverage.

## 2. Non-Goals

- Replace general Ukrainian ASR.
- Predict arbitrary pronunciation beyond lexical stress.
- Train the final context-only homograph resolver in this change.
- Treat text-derived labels for ambiguous forms as gold.

## 3. High-Level Architecture

```text
Corpus manifests
      |
      v
Audio ingest + normalization
      |
      +--> trusted transcript ------------------+
      |                                         |
      +--> ASR transcript (when needed)         |
                                                v
                                      Transcript normalization
                                                |
                                                v
                                        Word alignment
                                                |
                                      target-word discovery
                                                |
                                                v
                                    Fine phone/vowel alignment
                                                |
                         +----------------------+----------------------+
                         |                                             |
                         v                                             v
                SSL speech embeddings                         Prosodic features
                         |                           duration/energy/F0/etc.
                         +----------------------+----------------------+
                                                |
                                                v
                                      per-vowel feature sequence
                                                |
                                                v
                                  Acoustic Vowel Stress Ranker
                                                |
                                    candidate-aware probabilities
                                                |
                           +--------------------+--------------------+
                           |                    |                    |
                           v                    v                    v
                     ranker score      optional candidate      optional stress-
                                           scorer               aware CTC vote
                           |                    |                    |
                           +--------------------+--------------------+
                                                |
                                                v
                                    Calibration + acceptance policy
                                                |
                              +-----------------+-----------------+
                              |                                   |
                              v                                   v
                    accepted high precision                 rejected/review
                              |
                              v
                    versioned Parquet corpus
                              |
                              v
                 text-resolver training export
```

## 4. Repository Layout

Reference implementation layout:

```text
src/ukstress/
  cli/
    ingest.py
    align.py
    build_bootstrap.py
    train_ranker.py
    evaluate.py
    mine.py
    export_text_resolver.py
  config/
  audio/
    io.py
    normalize.py
    vad.py
  text/
    normalize.py
    tokenize.py
  lexicon/
    interface.py
    loader.py
    stress.py
  asr/
    interface.py
    whisper_backend.py
  alignment/
    schema.py
    interface.py
    whisperx_backend.py
    mfa_backend.py
    validation.py
  features/
    ssl.py
    pooling.py
    prosody.py
    schema.py
  models/
    vowel_ranker.py
    candidate_scorer.py
    stress_ctc_adapter.py
    calibration.py
  datasets/
    schemas.py
    ids.py
    bootstrap.py
    mining.py
    parquet.py
    splits.py
  pipeline/
    ingest.py
    align.py
    mine.py
    export.py
  evaluation/
    metrics.py
    leakage.py
    report.py
  provenance/
    manifest.py
    fingerprints.py
  tests/

configs/
  ingest/
  alignment/
  features/
  models/
  mining/
  experiments/

schemas/
  corpus-manifest.schema.json
  mining-record.schema.json

data/
  README.md

pyproject.toml
```

Exact package names may change during implementation, but module boundaries SHOULD remain equivalent.

## 5. Canonical Data Models

Use typed Python models/dataclasses internally and Arrow/Parquet externally.

### 5.1 CorpusManifestRecord

```python
class CorpusManifestRecord:
    source_id: str
    utterance_id: str
    audio_uri: str
    transcript: str | None
    speaker_id: str | None
    segment_start_s: float | None
    segment_end_s: float | None
    language: str
    license_id: str | None
    commercial_use: bool | None
    redistribution: bool | None
    provenance: dict[str, str]
```

### 5.2 WordAlignment

```python
class WordAlignment:
    token: str
    normalized_token: str
    start_s: float
    end_s: float
    confidence: float | None
    char_start: int | None
    char_end: int | None
```

### 5.3 VowelInterval

```python
class VowelInterval:
    vowel_index: int
    grapheme: str
    phone: str | None
    start_s: float
    end_s: float
    confidence: float | None
```

`vowel_index` is zero-based over vowels in the normalized target word, not over Unicode characters.

### 5.4 StressCandidate

```python
class StressCandidate:
    stressed_form: str
    vowel_index: int
    source: str
```

### 5.5 MiningRecord

```python
class MiningRecord:
    schema_version: str
    record_id: str
    source_id: str
    utterance_id: str
    speaker_id: str | None
    audio_uri: str

    sentence_original: str
    sentence_normalized: str
    target_word: str
    target_char_start: int | None
    target_char_end: int | None

    word_start_s: float
    word_end_s: float
    vowels: list[VowelInterval]
    candidates: list[StressCandidate]

    ranker_probs: list[float] | None
    candidate_scorer_probs: list[float] | None
    stress_ctc_candidate: int | None

    predicted_candidate: int | None
    calibrated_confidence: float | None
    candidate_margin: float | None
    alignment_score: float | None
    identity_score: float | None

    accepted: bool
    rejection_reasons: list[str]

    lexicon_fingerprint: str
    model_versions: dict[str, str]
    config_fingerprint: str
    license_id: str | None
    provenance: dict[str, str]
```

## 6. Stable IDs and Fingerprints

Stable record IDs are necessary for idempotency and dataset lineage.

Recommended record ID input:

```text
source_id
utterance_id
target normalized surface
word start/end quantized to 1 ms
pipeline schema version
```

Hash with SHA-256 and encode a short stable prefix for display while retaining the full hash in metadata when desired.

Dataset fingerprints MUST include ordered manifest IDs, lexicon fingerprint, normalization version, and split configuration.

## 7. Lexicon Layer

The lexicon adapter MUST normalize all stress notation into one internal representation.

Recommended internal convention:
- store plain surface separately
- store stressed form with Unicode combining acute accent or another single configured canonical representation
- compute zero-based vowel index explicitly

Never derive candidate identity during training by comparing visually normalized strings only; compare normalized vowel indices plus canonical stressed form.

### Candidate lookup

```python
lookup("замок") -> [
    StressCandidate("за́мок", vowel_index=0),
    StressCandidate("замо́к", vowel_index=1),
]
```

The bootstrap set accepts only `len(candidates) == 1`.

## 8. Transcript Strategy

Priority:

1. trusted supplied transcript
2. supplied transcript cross-checked by ASR where configured
3. ASR-generated transcript

Text normalization MUST be deterministic and conservative. It may normalize punctuation/case/Unicode for matching, but MUST NOT resolve ambiguous stress.

Target identity quality is separate from stress confidence.

## 9. Alignment Strategy

### 9.1 Word alignment

Selected initial backend: WhisperX 3.8.6 with the Ukrainian
`Yehor/wav2vec2-xls-r-300m-uk-with-small-lm` Wav2Vec2 CTC checkpoint (currently resolved by
Hugging Face to `Yehor/w2v-xls-r-uk`, revision
`e3ced4def0d70be3aab0f2db598a59961fe9ab3b`). Canary auxiliary CTC forced alignment through
CrispASR remains an optional benchmark candidate, not the production default.

Reasons:
- scalable
- forced against the trusted/reference transcript rather than accepting ASR text as identity
- a Ukrainian-specific CTC vocabulary without a separate romanization step
- direct integration with the existing WhisperX adapter and character alignments
- an explicit checkpoint selected independently of WhisperX's default language-model table
- convenient for long-form speech mining.

The checkpoint's model-card ASR metrics are not forced-alignment boundary metrics. Before a later
backend change, compare it against Canary CTC and MMS on manually word-aligned Ukrainian audio.
Canary's 80 ms encoder-frame quantum may be sufficient for word cropping but is coarser than this
Wav2Vec2 model. Generic MMS forced alignment is not a default because it requires a separate
Ukrainian romanization contract and currently depends on deprecated TorchAudio alignment APIs. A
future bake-off SHOULD also include NVIDIA NeMo Forced Aligner with
`nvidia/stt_ua_fastconformer_hybrid_large_pc` in CTC mode. The production default MUST change only
when boundary-error measurements on manually aligned Ukrainian audio show an improvement.

### 9.2 Fine alignment

Support two backends:

- `mfa`: slower, useful for research-quality phone boundaries and gold-set preparation.
- `ctc_phone` or equivalent: preferred long-term scalable backend.

Backend-neutral outputs feed the feature layer.

### 9.3 Validation

A target is unusable when:
- word span is invalid
- expected vowel count cannot be reconciled with alignment
- vowels are outside the word crop beyond tolerance
- alignment confidence is below configured minimum
- ASR/reference identity mismatch exceeds configured policy.

Do not repair suspicious alignment by inventing timestamps.

## 10. Bootstrap Acoustic Dataset

Purpose: teach the model acoustic correlates of stress without ambiguous text labels.

Algorithm:

```text
for each aligned word occurrence:
    candidates = lexicon.lookup(word)
    if len(candidates) != 1: skip
    if vowel_count < 2: skip by default
    if alignment invalid: skip
    gold_vowel = candidates[0].vowel_index
    extract features
    write bootstrap example
```

Store multiple occurrences of the same lexeme, but build evaluation splits that prevent lexical memorization from masquerading as acoustic competence.

## 11. Acoustic Features

### 11.1 SSL representation

Use a configurable Ukrainian-capable Wav2Vec2/XLS-R family encoder.

Default training sequence:

1. freeze all SSL layers
2. train pooling/projection/ranker head
3. if validation improves, unfreeze only the final N SSL transformer layers
4. record N in experiment manifest.

### 11.2 Per-vowel pooling

For each vowel interval:
- map time bounds to SSL frames
- extend by configurable context padding, default experiment value 40 ms
- attention-pool or mean-pool hidden states
- preserve an interval-validity mask.

Attention pooling is preferred, but mean pooling MUST be available as a deterministic baseline.

### 11.3 Prosodic features

Minimum feature set:
- raw vowel duration
- log duration
- duration / mean duration of other vowels in the same word
- duration / utterance median vowel duration when available
- RMS energy mean/max
- relative energy within word
- F0 median
- F0 range
- F0 slope
- F0 validity mask.

Optional:
- spectral tilt
- F1/F2
- syllable duration
- pretonic/post-tonic relative-duration descriptors.

Normalize continuous prosody features using statistics computed only from the training split.

## 12. Vowel Stress Ranker

Reference head:

```text
SSL pooled embedding --------+
                             |
prosodic feature vector -----+--> projection --> 256d vowel vector
                                                 |
                              [V1, V2, ..., Vn] --+
                                                 v
                                      2-layer Transformer Encoder
                                      hidden=256, heads=4, FFN=768
                                                 |
                                                 v
                                        scalar logit per vowel
                                                 |
                                       dictionary candidate mask
                                                 |
                                                 v
                                         masked softmax / CE
```

These dimensions are initial defaults, not immutable API requirements. All must be configuration-driven.

### Loss

For logits `s_i`, eligible mask `m_i`, and gold vowel `g`:

```text
s_i' = s_i if m_i else -inf
L = -log softmax(s')_g
```

No class label should encode a specific lexeme.

## 13. Anti-Memorization Measures

Training and evaluation MUST distinguish acoustic skill from lexical lookup.

Required evaluation partitions:

### A. Speaker-held-out
No evaluation speaker occurs in training.

### B. Lexeme-held-out
Evaluation target forms/lemmas do not occur in acoustic training.

### C. Ambiguous same-spelling/manual set
The identical surface form occurs with multiple verified stress readings.

### D. Unmasked evaluation
Candidate masking disabled to measure raw acoustic position inference.

Optional training regularization:
- randomly mask lexical token identity if token embeddings are ever introduced
- never feed the stressed spelling to the primary VowelRanker
- balance sampling so frequent lexemes do not dominate batches.

## 14. Manual Ambiguous Gold Dataset

Before trusting large-scale mining, create a manually verified set of ambiguous occurrences.

Recommended stages:

### MVP calibration set
At least 2,000 verified ambiguous target occurrences across many forms, speakers, and contexts.

### Production calibration set
Target 10,000+ verified occurrences, with both readings represented for ambiguous forms where data exists.

These are engineering targets, not linguistic constants.

Where recording controlled speech:
- each speaker SHOULD read both variants of the same surface form
- contexts SHOULD make the intended lexical reading unambiguous
- split by speaker before model tuning
- retain raw audio and human labels independently from model predictions.

## 15. Optional Candidate-Conditioned Audio/Text Scorer

Purpose: provide an error-independent second opinion for mining.

Inputs:
- target audio crop
- candidate stressed form.

Outputs:
- compatibility score per candidate.

Possible implementation:
- speech encoder -> audio embedding
- small character/stress-aware text encoder -> candidate embedding
- contrastive or ranking objective
- normalized similarity per candidate.

Do not block the first VowelRanker milestone on this model. Add it after the primary model has a reliable manual-gold baseline.

## 16. Stress-Aware CTC Adapter

Provide an adapter interface around existing Ukrainian stress-aware CTC models.

Responsibilities:
- run or consume stress-aware transcript output
- normalize stress notation
- locate the target occurrence
- map its reading to lexicon candidates
- return `candidate_index | None` and model confidence when available.

The CTC model is advisory; it is not treated as gold because its training labels may originate from text stressifiers.

## 17. Calibration and Ensemble

Raw neural probability is not a mining confidence.

Calibration options:
- temperature scaling baseline
- isotonic regression if enough held-out data exists
- calibration model chosen only on held-out manual gold.

Acceptance policy input:

```python
class AcceptanceEvidence:
    ranker_confidence: float
    candidate_margin: float
    alignment_quality: float
    identity_quality: float
    candidate_scorer_agrees: bool | None
    stress_ctc_agrees: bool | None
```

Suggested default high-precision profile:
- calibrated ranker confidence >= configured threshold
- candidate margin >= configured threshold
- alignment quality >= configured threshold
- trusted target identity
- require candidate-scorer agreement when that scorer is enabled and validated
- CTC disagreement lowers confidence or rejects, but policy remains configurable.

Thresholds MUST be selected from precision/coverage curves on manual gold, not invented as universal constants.

## 18. Mining Workflow

```text
load source shard
  -> decode/normalize audio
  -> transcript/ASR
  -> word alignment
  -> find words with >1 lexicon candidate
  -> crop target audio
  -> fine vowel alignment
  -> validate
  -> features
  -> VowelRanker
  -> optional independent scorers
  -> calibrate
  -> acceptance policy
  -> write candidate + accepted/rejected datasets
```

Do not process all words through heavy stress inference. The lexicon discovery stage should reduce work to ambiguous targets plus explicitly requested diagnostic samples.

## 19. Output Dataset Partitioning

Recommended paths:

```text
artifacts/
  bootstrap/<dataset-version>/part-*.parquet
  candidates/<run-id>/part-*.parquet
  accepted/<run-id>/part-*.parquet
  rejected/<run-id>/part-*.parquet
  text-resolver/<run-id>/part-*.parquet
  reports/<run-id>/metrics.json
  reports/<run-id>/report.md
  manifests/<run-id>.yaml
```

No output path itself is authoritative; the experiment manifest and dataset fingerprint are.

## 20. Text-Resolver Export

Accepted example:

```json
{
  "context": "Працівник перевірив <w>замок</w>.",
  "target": "замок",
  "candidates": ["за́мок", "замо́к"],
  "gold": "замо́к",
  "confidence": 0.997,
  "source_type": "audio_mined"
}
```

Do not export rejected examples as positive training gold. They may be exported separately for hard-negative/manual-review workflows.

## 21. Training Sampler

Global random sampling would overrepresent frequent words and majority readings.

For ambiguous supervised fine-tuning, prefer hierarchical sampling:

```text
choose surface form
  -> choose stress variant approximately uniformly
     -> choose occurrence
```

For bootstrap unambiguous training, cap per-lexeme occurrences per epoch or use inverse-frequency weighting.

## 22. Evaluation

Report at minimum:
- vowel stress-position accuracy
- unrestricted lexeme-held-out accuracy
- speaker-held-out accuracy
- manual ambiguous micro accuracy
- manual ambiguous macro accuracy by surface form
- macro-F1 over variants when defined
- per-form confusion
- reliability/calibration plot data
- accepted-set precision
- accepted-set coverage
- rejection reason distribution.

The production mining gate defaults to 99% accepted-set precision on manual ambiguous gold, but the report MUST include sample count and uncertainty. If data is too small to support the claim, mark the profile unvalidated rather than extrapolating.

## 23. Reproducibility

Every CLI run writes an experiment/run manifest before committing final outputs.

Recommended fields:

```yaml
run_id: ...
git_commit: ...
command: ...
config_fingerprint: ...
random_seeds: ...
inputs:
  corpus_fingerprint: ...
  lexicon_fingerprint: ...
models:
  ssl_encoder: ...
  ranker_checkpoint: ...
  candidate_scorer: ...
  stress_ctc: ...
alignment:
  word_backend: ...
  fine_backend: ...
environment:
  lock_fingerprint: ...
outputs:
  dataset_fingerprints: ...
```

## 24. CLI Surface

Reference commands:

```bash
ukstress ingest --config configs/experiments/commonvoice.yaml
ukstress align --run <run-id>
ukstress build-bootstrap --run <run-id>
ukstress train-ranker --config configs/models/vowel-ranker.yaml
ukstress evaluate --checkpoint <path> --dataset <dataset-id>
ukstress calibrate --checkpoint <path> --dataset <manual-gold-id>
ukstress mine --config configs/mining/high-precision.yaml
ukstress export-text-resolver --run <mining-run-id>
```

CLI names may change, but equivalent non-interactive commands are required for automation.

## 25. Configuration

Use structured YAML plus typed validation. Avoid configuration hidden in Python constants.

Example:

```yaml
sample_rate: 16000

alignment:
  word_backend: whisperx
  fine_backend: mfa
  min_quality: 0.8

features:
  ssl_model: ${UKSTRESS_SSL_MODEL}
  vowel_context_ms: 40
  prosody:
    duration: true
    energy: true
    f0: true

model:
  hidden_size: 256
  transformer_layers: 2
  attention_heads: 4
  ffn_size: 768
  trainable_ssl_layers: 0

mining:
  require_candidate_scorer_agreement: false
  min_calibrated_confidence: null  # selected by calibration, not hard-coded
  min_candidate_margin: null
```

## 26. Error Handling

Expected data-quality failures are records, not exceptions that kill a corpus job.

Use structured rejection reasons such as:
- `audio_decode_failed`
- `asr_failed`
- `target_identity_mismatch`
- `word_alignment_failed`
- `vowel_alignment_failed`
- `vowel_count_mismatch`
- `lexicon_missing`
- `lexicon_unambiguous`
- `low_alignment_quality`
- `low_stress_confidence`
- `low_candidate_margin`
- `model_disagreement`.

Unexpected programmer/system failures should still fail the shard and preserve diagnostics.

## 27. Performance Strategy

Separate CPU-heavy and GPU-heavy stages so artifacts can be cached:

1. audio decode/normalization
2. ASR/alignment
3. SSL feature extraction
4. ranker inference
5. calibration/filtering.

Cache deterministic intermediate data using content/config fingerprints. Do not serialize enormous duplicated waveforms into Parquet; reference source media plus time ranges unless the export policy explicitly requires clips.

Batch SSL inference by similar crop duration.

## 28. Security and Data Governance

- Treat corpus manifests and media as untrusted input.
- Do not execute metadata or derive shell commands from filenames.
- Validate paths/URIs and codecs.
- Preserve licensing metadata.
- Provide a policy layer for excluding non-exportable sources.
- Avoid storing unnecessary personal metadata from speakers.
- Never infer missing licenses.

## 29. Implementation Order

1. schemas/provenance/lexicon
2. ingestion + normalization
3. word alignment adapter
4. fine vowel alignment + validation
5. bootstrap dataset
6. SSL/prosody features
7. VowelRanker training
8. leakage-safe evaluation
9. manual-gold tooling
10. calibration + mining
11. candidate scorer
12. stress-CTC adapter
13. ensemble high-precision profile
14. text-resolver export
15. scale/performance work.

## 30. Key Architectural Decision

The primary acoustic model predicts **which aligned vowel is stressed**, not a stressed character string and not a lexical class.

This is intentional because it:
- generalizes to unseen words
- prevents a fixed global homograph-class vocabulary
- directly uses Ukrainian prosodic evidence
- supports candidate masking without requiring it
- allows unmasked evaluation of true acoustic competence
- makes dictionary-unambiguous natural speech usable as large-scale bootstrap supervision.
