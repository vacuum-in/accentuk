# Implementation Tasks

## 1. Project Foundation

- [x] 1.1 Create Python package, `pyproject.toml`, lockfile strategy, lint/type/test configuration.
- [x] 1.2 Add typed configuration loading and validation.
- [x] 1.3 Add run/experiment manifest schema and content fingerprints.
- [x] 1.4 Add structured logging with `run_id`, `shard_id`, `source_id`, and `record_id` context.
- [x] 1.5 Add deterministic random-seed utilities.

## 2. Canonical Schemas and Storage

- [x] 2.1 Implement `CorpusManifestRecord`, normalized utterance, `WordAlignment`, `VowelInterval`, `StressCandidate`, feature example, and `MiningRecord` models.
- [x] 2.2 Add Arrow/Parquet schemas with explicit `schema_version`.
- [x] 2.3 Implement stable record IDs and dataset fingerprints.
- [x] 2.4 Implement atomic/sharded Parquet writing and resume detection.
- [x] 2.5 Add schema migration/version rejection tests.

## 3. Ukrainian Stress Lexicon

- [x] 3.1 Implement lexicon adapter interface.
- [x] 3.2 Implement canonical stress-mark normalization and vowel indexing.
- [x] 3.3 Load a versioned Ukrainian stress dictionary from configured external data.
- [x] 3.4 Return all valid variants for ambiguous forms without resolving them.
- [x] 3.5 Add lexicon fingerprinting and provenance.
- [x] 3.6 Add tests for unambiguous, ambiguous, OOV, uppercase, apostrophe, hyphen, and Unicode-normalization cases.

## 4. Audio and Transcript Ingestion

- [x] 4.1 Implement corpus manifest loader and validator.
- [x] 4.2 Implement safe audio decode and canonical mono/16-kHz waveform conversion.
- [x] 4.3 Implement deterministic Ukrainian transcript normalization.
- [x] 4.4 Preserve original text separately from normalized matching text.
- [x] 4.5 Implement structured rejection records for decode/metadata failures.
- [x] 4.6 Add a tiny fixture corpus for CI that requires no large-model download.

## 5. ASR Backend

- [x] 5.1 Define ASR backend interface and normalized result schema.
- [x] 5.2 Implement configurable Whisper-compatible backend.
- [x] 5.3 Preserve segment/token confidence where available.
- [x] 5.4 Implement transcript/reference cross-check utilities.
- [x] 5.5 Add tests with mocked ASR results.

## 6. Word Alignment

- [x] 6.1 Define backend-neutral word alignment interface.
- [x] 6.2 Implement Ukrainian WhisperX/CTC-compatible word alignment backend.
- [x] 6.3 Normalize backend output into `WordAlignment` records.
- [x] 6.4 Add alignment quality scoring/normalization.
- [x] 6.5 Add bounds/order validation and target-identity checks.
- [x] 6.6 Persist backend/model identifiers in records and manifests.

## 7. Fine Phone/Vowel Alignment

- [x] 7.1 Define fine-alignment backend interface.
- [x] 7.2 Implement MFA adapter capable of producing phone/vowel intervals.
- [x] 7.3 Implement mapping from phones/graphemes to ordered Ukrainian vowel indices.
- [x] 7.4 Validate vowel count, ordering, overlap, and word-bound containment.
- [x] 7.5 Persist explicit rejection reasons for unusable alignments.
- [x] 7.6 Add an interface placeholder and fixtures for a future scalable CTC fine aligner.

## 8. Bootstrap Dataset Builder

- [x] 8.1 Join aligned occurrences with lexicon candidates.
- [x] 8.2 Include only words with exactly one stress candidate.
- [x] 8.3 Exclude ambiguous words from automatic gold labels.
- [x] 8.4 Exclude one-vowel words by default with a configuration override.
- [x] 8.5 Add per-lexeme occurrence caps/weights to reduce memorization.
- [x] 8.6 Write bootstrap Parquet shards with audio references and vowel intervals.
- [x] 8.7 Produce build statistics: accepted, skipped, OOV, ambiguous, alignment rejected, by source.

## 9. SSL Feature Extraction

- [x] 9.1 Implement configurable Hugging Face-compatible speech encoder loader.
- [x] 9.2 Implement time-to-frame mapping.
- [x] 9.3 Implement mean pooling baseline over vowel intervals.
- [x] 9.4 Implement learnable attention pooling with configurable context padding.
- [x] 9.5 Batch examples by duration and add device/mixed-precision controls.
- [x] 9.6 Cache deterministic feature artifacts keyed by encoder/config/input fingerprints.

## 10. Prosodic Feature Extraction

- [x] 10.1 Implement vowel duration and log-duration features.
- [x] 10.2 Implement within-word relative-duration features.
- [x] 10.3 Implement utterance-relative duration normalization.
- [x] 10.4 Implement RMS/peak/relative energy features.
- [x] 10.5 Implement F0 median/range/slope with explicit validity mask.
- [x] 10.6 Add optional spectral/formant feature hooks without making them mandatory.
- [x] 10.7 Fit normalization statistics from training split only and persist them.

## 11. Vowel Stress Ranker

- [x] 11.1 Implement per-vowel SSL+prosody projection.
- [x] 11.2 Implement variable-length vowel sequence batching and masks.
- [x] 11.3 Implement configurable 2-layer Transformer reference head.
- [x] 11.4 Implement one scalar logit per vowel.
- [x] 11.5 Implement lexicon candidate masking.
- [x] 11.6 Implement masked categorical cross-entropy.
- [x] 11.7 Implement unmasked inference/evaluation mode.
- [x] 11.8 Support fully frozen SSL encoder and selective last-N-layer unfreezing.
- [x] 11.9 Save/load checkpoints with full model/feature/config metadata.

## 12. Leakage-Safe Splits

- [x] 12.1 Implement deterministic random/base split utility.
- [x] 12.2 Implement speaker-held-out split and leakage validator.
- [x] 12.3 Implement lexeme/form-held-out split and leakage validator.
- [x] 12.4 Implement per-lexeme occurrence balancing/capping for training.
- [x] 12.5 Generate machine-readable split reports.

## 13. Ranker Training Pipeline

- [x] 13.1 Implement training DataLoader over bootstrap Parquet.
- [x] 13.2 Add optimizer/scheduler/gradient clipping/mixed precision configuration.
- [x] 13.3 Log train/validation masked and unmasked metrics.
- [x] 13.4 Implement early stopping/checkpoint selection using leakage-safe validation.
- [x] 13.5 Save experiment manifest, metrics, normalization stats, and checkpoint fingerprint.
- [x] 13.6 Add a CPU/small-fixture smoke-training test.

## 14. Manual Ambiguous Gold Tooling

- [x] 14.1 Define schema for manually verified ambiguous occurrences.
- [x] 14.2 Implement import from a simple CSV/JSONL annotation format.
- [x] 14.3 Validate that selected stress is one of the lexicon candidates.
- [x] 14.4 Add duplicate/speaker/context checks.
- [x] 14.5 Add reports for per-form/per-variant counts and speaker balance.
- [x] 14.6 Document protocol requiring both readings per speaker where controlled recording is used.

## 15. Evaluation Suite

- [x] 15.1 Implement stress-position accuracy.
- [x] 15.2 Implement macro accuracy by ambiguous surface form.
- [x] 15.3 Implement macro-F1/per-variant metrics where defined.
- [x] 15.4 Implement speaker-held-out report.
- [x] 15.5 Implement lexeme-held-out masked and unmasked reports.
- [x] 15.6 Implement same-spelling ambiguous/minimal-pair report.
- [x] 15.7 Implement per-form confusion tables.
- [x] 15.8 Implement rejection/alignment/identity quality statistics.
- [x] 15.9 Emit JSON metrics and human-readable Markdown report.

## 16. Confidence Calibration

- [x] 16.1 Implement temperature-scaling calibration baseline.
- [x] 16.2 Optionally implement isotonic calibration for sufficiently large gold sets.
- [x] 16.3 Fit calibration only on held-out manually verified data.
- [x] 16.4 Generate reliability data and precision-versus-coverage curves.
- [x] 16.5 Persist calibration artifact and fingerprint.
- [x] 16.6 Implement validation gate for configured accepted-set precision target.

## 17. Candidate Discovery and Mining

- [x] 17.1 Find aligned transcript tokens with multiple stress candidates.
- [x] 17.2 Run fine vowel alignment only for target candidates plus requested diagnostics.
- [x] 17.3 Extract features and run VowelRanker.
- [x] 17.4 Compute candidate margin, alignment quality, and identity quality.
- [x] 17.5 Apply calibrated confidence.
- [x] 17.6 Implement configurable precision-first acceptance policy.
- [x] 17.7 Write all candidates, accepted records, and rejected/manual-review records.
- [x] 17.8 Add idempotent shard resume/reprocessing behavior.
- [x] 17.9 Generate per-source and overall mining statistics.

## 18. Candidate-Conditioned Audio/Text Scorer

- [x] 18.1 Implement candidate text representation with explicit stress position.
- [x] 18.2 Implement audio embedding from target crop.
- [x] 18.3 Implement candidate compatibility/ranking objective.
- [x] 18.4 Train using trusted unambiguous and manual ambiguous examples without treating ambiguous text pseudo-labels as gold.
- [x] 18.5 Evaluate correlation/error overlap versus VowelRanker.
- [x] 18.6 Add scorer output to mining evidence.

## 19. Stress-Aware CTC Adapter

- [x] 19.1 Define normalized adapter interface.
- [x] 19.2 Integrate a configurable existing Ukrainian stress-aware CTC checkpoint.
- [x] 19.3 Map its stress output to lexicon candidate indices.
- [x] 19.4 Return `None` on ambiguous/unreliable target mapping rather than guessing.
- [x] 19.5 Record model/checkpoint version and confidence when available.

## 20. Ensemble Acceptance

- [x] 20.1 Add optional requirement for VowelRanker/CandidateScorer agreement.
- [x] 20.2 Add configurable treatment of stress-CTC agreement/disagreement.
- [x] 20.3 Recalibrate ensemble confidence on manual gold.
- [x] 20.4 Select and persist a `high-precision` profile from measured precision/coverage curves.
- [x] 20.5 Prevent profile from being marked validated when the configured precision gate is not met.

## 21. Text-Resolver Dataset Export

- [x] 21.1 Transform accepted mining records into context/target/candidates/gold records.
- [x] 21.2 Mark the target span deterministically, e.g. `<w>...</w>` in an export field.
- [x] 21.3 Preserve confidence and source category.
- [x] 21.4 Implement license-aware export policy that can omit audio references/media.
- [x] 21.5 Write versioned Parquet/JSONL export and dataset fingerprint.

## 22. Performance and Scaling

- [x] 22.1 Profile ASR, alignment, feature extraction, and ranker stages independently.
- [x] 22.2 Add cache reuse for deterministic ASR/alignment/features.
- [x] 22.3 Add GPU batching by duration for SSL extraction.
- [x] 22.4 Add shard-level parallel execution contract.
- [x] 22.5 Add bounded queues/backpressure for streaming large manifests.
- [x] 22.6 Confirm restart from completed shard boundaries without duplicate output.

## 23. Data Governance

- [x] 23.1 Preserve source license/provenance in every derived record.
- [x] 23.2 Implement configurable export policy by license flags.
- [x] 23.3 Avoid persisting unnecessary speaker PII.
- [x] 23.4 Validate media/path inputs and prevent command/path injection through metadata.
- [x] 23.5 Document which artifacts may or may not contain redistributed audio.

## 24. CI and Acceptance

- [x] 24.1 Unit-test normalization, stress parsing, vowel indexing, masking, loss, confidence policy, IDs, fingerprints, and schemas.
- [x] 24.2 Integration-test ingestion -> mocked alignment -> features -> inference -> Parquet on a tiny fixture.
- [x] 24.3 Add optional GPU integration test profile outside default CI.
- [x] 24.4 Run OpenSpec/spec acceptance review against all scenarios.
- [x] 24.5 Document reproducible commands for bootstrap build, training, evaluation, calibration, mining, and export.
- [x] 24.6 Produce first evaluation report before enabling high-confidence large-scale export.
