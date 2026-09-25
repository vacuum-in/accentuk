# Delta for Evaluation and Reproducibility

## ADDED Requirements

### Requirement: Deterministic dataset splits
The system MUST generate deterministic split assignments from stable identifiers and an explicit split seed/version.

#### Scenario: Regenerate split
- GIVEN identical source manifests, split configuration, and seed
- WHEN splits are regenerated
- THEN every record receives the same split assignment.

### Requirement: Speaker-held-out evaluation
Where speaker identifiers are available, the evaluation suite MUST include a split whose evaluation speakers do not appear in training.

#### Scenario: Detect speaker overlap
- GIVEN a speaker present in both training and speaker-held-out evaluation
- WHEN split validation runs
- THEN validation fails with the overlapping speaker identifiers.

### Requirement: Lexeme-held-out evaluation
The evaluation suite MUST include a split where normalized target lexemes/forms selected for evaluation are absent from acoustic-ranker training.

#### Scenario: Detect lexeme leakage
- GIVEN an evaluation lexeme present in bootstrap training
- WHEN lexeme-held-out validation runs
- THEN the split is invalid and MUST NOT be reported as unseen-lexeme performance.

### Requirement: Manually verified ambiguous evaluation set
Production mining thresholds MUST be evaluated on a manually verified ambiguous-word dataset containing both readings for the same surface forms where possible.

The dataset SHOULD contain multiple speakers and multiple contexts per reading.

#### Scenario: Evaluate ambiguous form fairly
- GIVEN a form with two stress variants
- WHEN included in the manual evaluation set
- THEN both variants are represented when verified examples exist
- AND metrics report per-variant as well as aggregated performance.

### Requirement: Minimal-pair/same-spelling evaluation
The system MUST separately report accuracy for same-spelling occurrences whose correct stress differs by context.

#### Scenario: Frequency shortcut exposed
- GIVEN a model that always predicts the majority stress variant for a homograph
- WHEN same-spelling evaluation runs
- THEN minority-variant errors reduce macro/per-variant metrics even if micro accuracy remains high.

### Requirement: Evaluation metrics
The evaluation suite MUST report at least:
- overall stress-position accuracy
- macro accuracy across ambiguous surface forms
- macro-F1 across stress variants where meaningful
- per-form/per-variant accuracy
- accepted-set precision at the active mining threshold
- accepted-set coverage/recall proxy
- calibration error or reliability summary
- alignment rejection rate
- ASR/identity rejection rate.

#### Scenario: Precision-coverage curve
- GIVEN calibrated candidate predictions
- WHEN threshold evaluation runs
- THEN the report shows accepted precision and coverage across multiple thresholds
- AND the chosen production threshold is recorded.

### Requirement: Production mining gate
A production high-confidence mining profile MUST NOT be designated "validated" until its manually verified accepted-set precision meets the configured target.

The default target SHALL be 99% precision for ambiguous-word accepted records, but MUST remain configuration-driven and reported with sample count and confidence interval.

#### Scenario: Gate fails below target
- GIVEN measured accepted-set precision below the configured target
- WHEN validation runs
- THEN the profile is marked unvalidated
- AND large-scale accepted-data export requires an explicit override.

### Requirement: Experiment manifest
Every train/evaluate/mine run MUST persist an immutable experiment manifest containing at least:
- git commit when available
- command/config snapshot
- random seeds
- input dataset versions/fingerprints
- lexicon version/fingerprint
- model/checkpoint identifiers
- feature/alignment backend versions
- environment/package lock fingerprint
- output artifact identifiers.

#### Scenario: Reproduce reported model
- GIVEN a model checkpoint and its experiment manifest
- WHEN inspected
- THEN the exact input versions and configuration required to reproduce the run are identifiable.

### Requirement: Unit and integration test coverage
The repository MUST include tests for deterministic text normalization, lexicon candidate mapping, alignment validation, candidate masking, masked loss, confidence policy, stable record IDs, Parquet schema, and end-to-end processing of a small fixture corpus.

#### Scenario: End-to-end fixture
- GIVEN a small checked-in or generated fixture with transcript, alignments, and expected lexicon candidates
- WHEN CI runs
- THEN the pipeline produces schema-valid deterministic outputs without downloading a large model.
