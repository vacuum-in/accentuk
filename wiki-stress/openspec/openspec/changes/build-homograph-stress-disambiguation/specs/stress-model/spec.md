# Delta for Stress Model

## ADDED Requirements

### Requirement: Reproducible training runs

A training run SHALL be described completely enough to be repeated, and SHALL
record what it actually consumed.

#### Scenario: Training run completes

- **WHEN** a training run finishes
- **THEN** it records the corpus version, inventory content hash, tokenizer
  artifact checksums, model configuration, hyperparameters, random seed, library
  versions, and hardware
- **AND** it records measured training duration and the selected checkpoint

#### Scenario: Training run is repeated

- **GIVEN** the same corpus version, configuration, seed, and library versions
- **WHEN** training is repeated on equivalent hardware
- **THEN** reported evaluation metrics agree within a declared tolerance
- **AND** the tolerance is stated rather than exactness being claimed

#### Scenario: Checkpoint is selected

- **WHEN** the best checkpoint is chosen
- **THEN** it is selected on group-macro accuracy over the development split
- **AND** not on loss or on a translation-quality metric

### Requirement: Tokenizer construction

The tokenizer SHALL cover Ukrainian text and stressed forms without loss, and
SHALL be versioned with the model.

#### Scenario: Tokenizer is built

- **WHEN** tokenizer artifacts are produced
- **THEN** they are built from the corpus version's training split only
- **AND** their checksums are recorded with the model

#### Scenario: Text round-trips

- **WHEN** any corpus sentence or target form is encoded and decoded
- **THEN** the result equals the input exactly, including the combining acute
  and canonical apostrophes

#### Scenario: Unseen character appears at inference

- **WHEN** input contains a character absent from the training corpus
- **THEN** encoding succeeds through character fallback
- **AND** the request does not fail

### Requirement: Constrained candidate scoring

The model SHALL select among supplied candidates and SHALL NOT generate a
stressed form freely.

#### Scenario: Candidates are supplied

- **WHEN** a disambiguation is requested with a candidate set
- **THEN** every candidate is scored under forced decoding
- **AND** the highest-scoring candidate is returned
- **AND** no output outside the candidate set can be produced

#### Scenario: Scores are reported

- **WHEN** a candidate is selected
- **THEN** the response includes a comparable score for every candidate and the
  margin between the best two

#### Scenario: A single candidate is supplied

- **WHEN** the candidate set contains one entry
- **THEN** it is returned without invoking the model

#### Scenario: The candidate set is empty

- **WHEN** no candidates are supplied
- **THEN** the request is rejected as invalid rather than answered by generation

#### Scenario: Candidates differ only by normalization

- **WHEN** two supplied candidates share a canonical lookup key and stress
  signature
- **THEN** they are merged before scoring
- **AND** the merge is reported in the response

### Requirement: Declared evaluation

Model quality SHALL be reported on the declared evaluation sets, and SHALL never
be reported as a single aggregate figure.

#### Scenario: Evaluation report is produced

- **WHEN** a model is evaluated
- **THEN** the report includes group-macro accuracy, accuracy on the natural-text
  set, accuracy on the same-part-of-speech subset, and accuracy on the
  unseen-group set
- **AND** each figure names the evaluation set and its row count

#### Scenario: Baselines are compared

- **WHEN** a model is evaluated
- **THEN** the report includes the accuracy of selecting each group's default
  sense
- **AND** the accuracy of the encoder-classifier baseline on the same sets

#### Scenario: A model does not beat the default-sense baseline

- **WHEN** the model does not exceed the default-sense baseline on group-macro
  accuracy
- **THEN** the model is not eligible for release
- **AND** the comparison is retained in the report

#### Scenario: Per-group results are reported

- **WHEN** evaluation completes
- **THEN** the report lists the worst-performing groups with their confusion
  counts
- **AND** groups below the configured floor are named in a known-limitations
  artifact

#### Scenario: Calibration is measured

- **WHEN** evaluation completes
- **THEN** the report includes a calibration measurement over the score margin
- **AND** an abstention threshold derived from it

#### Scenario: Metrics are measured, not assumed

- **WHEN** any accuracy, latency, or throughput figure is published
- **THEN** it originates from a recorded evaluation run on named hardware
- **AND** no figure is pre-populated or estimated

### Requirement: Release gating

A model version SHALL NOT be released without meeting declared gates and
recording its provenance.

#### Scenario: Model is released

- **WHEN** a model version is marked releasable
- **THEN** its evaluation report meets every configured gate
- **AND** it records the corpus version, inventory hash, tokenizer checksums, and
  training run identifier

#### Scenario: A gate is not met

- **WHEN** any configured gate is not met
- **THEN** release is refused
- **AND** the unmet gates are named with their measured values

#### Scenario: Known limitations are published

- **WHEN** a model is released
- **THEN** the groups it resolves poorly are listed with the release
- **AND** no claim of complete homograph coverage is made

### Requirement: Inference artifact conversion

The released artifact SHALL be converted for serving and SHALL be verified to
agree with the training-framework model.

#### Scenario: Artifact is converted

- **WHEN** a released model is converted for serving
- **THEN** the conversion records the source checkpoint, converter version, and
  quantization setting

#### Scenario: Converted artifact is verified

- **WHEN** the converted artifact is evaluated on the development split
- **THEN** its selections agree with the training-framework model within a
  declared tolerance
- **AND** a larger divergence blocks the artifact from serving
