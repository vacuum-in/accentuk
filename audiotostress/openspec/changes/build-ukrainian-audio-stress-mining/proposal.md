# Build Ukrainian Audio Stress Mining Pipeline

## Why

Current Ukrainian text stressification systems are limited on context-dependent stress variants and homographs. Text-only pseudo-labels are useful for unambiguous words but can propagate the same errors that the final contextual model must learn to correct.

This change introduces an audio-derived stress mining pipeline whose primary purpose is to create high-precision training records of the form:

`(sentence, target word, candidate stress variants, selected variant, confidence, audio evidence)`.

The pipeline will learn acoustic lexical-stress cues from naturally spoken Ukrainian, bootstrap from dictionary-unambiguous words, evaluate on manually verified ambiguous forms, and mine only high-confidence ambiguous-word examples from large audio corpora. The resulting dataset will be suitable for training a later text-only contextual stress resolver.

## What Changes

- Add manifest-driven ingestion for Ukrainian speech corpora with explicit provenance and licensing metadata.
- Add ASR/transcript normalization and word-level forced alignment.
- Add phone/vowel alignment with pluggable alignment backends.
- Add a canonical Ukrainian stress lexicon interface that exposes all valid stress variants for a surface form.
- Build training examples from dictionary-unambiguous words without using ambiguous forms as pseudo-gold.
- Implement an acoustic vowel stress ranker that scores the valid vowel positions of a spoken word.
- Extract SSL speech embeddings and explicit prosodic features for each aligned vowel.
- Add candidate masking so the model ranks only dictionary-valid stress positions when variants are known.
- Add an optional candidate-conditioned audio/text scorer as an independent second model.
- Add an adapter for an existing stress-aware Ukrainian CTC model as an optional third vote.
- Add calibrated ensemble scoring and precision-first acceptance thresholds.
- Write accepted and rejected mining records to versioned Parquet datasets with traceable evidence.
- Add speaker-held-out, lexeme-held-out, and ambiguous minimal-pair evaluation protocols.
- Add reproducible training, dataset versioning, experiment manifests, and deterministic split generation.

## Capabilities

### New Capabilities

- `audio-ingestion-alignment`: ingest Ukrainian audio, obtain normalized transcript, word boundaries, and vowel/phone boundaries.
- `acoustic-stress-inference`: train and run the acoustic vowel stress ranker and optional independent stress scorers.
- `stress-dataset-mining`: discover target words, score candidate stresses, filter by calibrated confidence, and emit canonical training records.
- `evaluation-reproducibility`: evaluate true acoustic stress recognition without lexical/speaker leakage and preserve reproducible experiment metadata.

## Out of Scope

- Production text-only contextual stressification at TTS inference time.
- Training a Ukrainian RUAccent-style text resolver from the mined records.
- End-user UI.
- General-purpose Ukrainian ASR quality optimization beyond what is needed for reliable target-word identity and alignment.
- Automatic acquisition of copyrighted audio without explicit provenance and permitted use.

## Success Criteria

1. The system can ingest a corpus manifest and produce aligned word and vowel intervals for Ukrainian speech.
2. The bootstrap dataset excludes all dictionary-ambiguous forms from automatic gold-label generation.
3. The acoustic ranker can infer a stress position for previously unseen lexemes using only audio-derived evidence plus candidate constraints.
4. Evaluation includes speaker-held-out, lexeme-held-out, and manually verified ambiguous-word subsets.
5. Mining acceptance is precision-first: thresholds are calibrated on manual gold data, and low-confidence/disagreeing examples are retained as rejected records rather than silently accepted.
6. Every accepted record is traceable to source audio, transcript/alignment evidence, model versions, dictionary version, and confidence components.
7. The output dataset can be transformed directly into text-resolver training samples without re-running audio processing.

## Assumptions

- Implementation language: Python 3.11+.
- Training stack: PyTorch with Hugging Face-compatible speech encoders.
- Storage format: Parquet for canonical datasets and JSON/YAML for manifests/configuration.
- Initial word alignment backend: WhisperX-compatible Ukrainian CTC alignment when available.
- Initial fine phone/vowel alignment backend: pluggable; MFA may be used for research-quality alignment, with a CTC backend supported for scalable processing.
- The stress lexicon is external/versioned data and MUST NOT be hard-coded into model code.
