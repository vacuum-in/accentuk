# Delta for Training Corpus

## ADDED Requirements

### Requirement: Corpus row validity

Every corpus row SHALL be verifiable against the frozen inventory without
consulting the model or the annotation service.

#### Scenario: Row is emitted

- **WHEN** a row is written to a corpus version
- **THEN** it carries the sentence, the target character offsets, the target
  surface form, the assigned sense, the stressed form, and the stress signature
- **AND** the assigned sense exists in the frozen inventory

#### Scenario: Target span does not match

- **WHEN** the recorded offsets do not delimit the recorded surface form in the
  sentence
- **THEN** the row is rejected

#### Scenario: Stressed form is not a form of the sense

- **WHEN** the recorded stressed form is absent from the assigned sense's
  paradigm
- **THEN** the row is rejected
- **AND** the rejection is reported against the sense

#### Scenario: Stressed form fails stress validation

- **WHEN** the recorded stressed form places an acute anywhere other than
  immediately after a Ukrainian vowel
- **THEN** the row is rejected

#### Scenario: Unstressed projection disagrees

- **WHEN** removing the acute from the stressed form does not yield the recorded
  surface form under the canonical lookup key
- **THEN** the row is rejected

### Requirement: Deduplication

The corpus SHALL NOT contain repeated or near-repeated sentences within or
across senses.

#### Scenario: Identical sentence recurs

- **WHEN** the same sentence appears more than once
- **THEN** one instance is retained
- **AND** the retained instance records every source it was observed in

#### Scenario: Near-duplicate sentences occur

- **WHEN** two sentences exceed the configured similarity threshold
- **THEN** one is removed
- **AND** removal is deterministic for a given corpus version

#### Scenario: A sentence is assigned to two senses

- **WHEN** the same sentence carries different sense labels
- **THEN** both rows are quarantined rather than one being chosen
- **AND** the conflict is reported

### Requirement: Provenance

Every corpus row SHALL record where it came from and how it was labelled.

#### Scenario: Row provenance is recorded

- **WHEN** a row is written
- **THEN** it records the source tier, corpus name, source location, licence
  tag, label origin, and, where applicable, the annotation job identifier and
  model version
- **AND** it records the inventory content hash it was built against

#### Scenario: Publishable subset is requested

- **WHEN** a publishable subset is derived
- **THEN** it is produced by filtering on licence tags
- **AND** rows without a licence tag are excluded

### Requirement: Balanced composition

The corpus SHALL bound the imbalance a model can exploit, without erasing the
natural distribution from evaluation.

#### Scenario: Sense frequency is skewed within a group

- **WHEN** a group's training rows are assembled
- **THEN** the ratio between its most and least represented senses does not
  exceed the configured cap

#### Scenario: A sense is below the floor

- **WHEN** a sense has fewer training rows than the configured floor
- **THEN** it is reported as under-covered
- **AND** the corpus version records the shortfall rather than concealing it

#### Scenario: Natural evaluation set is assembled

- **WHEN** the natural-text evaluation set is assembled
- **THEN** it is drawn only from mined sources with human-confirmed labels
- **AND** its sense distribution is not balanced

### Requirement: Reproducible splits

Splits SHALL be deterministic and SHALL support measuring generalization beyond
the inventory.

#### Scenario: Splits are produced

- **GIVEN** the same corpus version and seed
- **WHEN** splits are produced
- **THEN** the assignment of every row is identical

#### Scenario: Training and evaluation overlap is checked

- **WHEN** splits are produced
- **THEN** no sentence appears in more than one split
- **AND** near-duplicates do not straddle splits

#### Scenario: Held-out groups are reserved

- **WHEN** splits are produced
- **THEN** a configured number of whole groups is reserved for an unseen-group
  evaluation set
- **AND** no row from those groups appears in training

#### Scenario: Every trained group is represented

- **WHEN** splits are produced
- **THEN** every group not reserved as unseen appears in the training split
- **AND** groups absent from training are reported

### Requirement: Model input encoding

The corpus SHALL be rendered into model input and target strings by a single
documented encoder.

#### Scenario: Row is encoded

- **WHEN** a row is encoded
- **THEN** the source string contains the sentence with the target span
  delimited by reserved markers
- **AND** the target string is the stressed surface form alone

#### Scenario: Sentence contains a reserved marker

- **WHEN** a sentence already contains a reserved span marker character
- **THEN** the row is rejected or the marker is escaped by the documented rule
- **AND** the behaviour is the same for every corpus version

#### Scenario: Encoding round-trips

- **WHEN** an encoded source string is decoded
- **THEN** the original sentence and target offsets are recovered exactly

### Requirement: Versioned corpus artifacts

Corpus versions SHALL be immutable and self-describing.

#### Scenario: Corpus version is written

- **WHEN** a corpus version is written
- **THEN** it records the inventory content hash, source corpora and their
  checksums, annotation job identifiers, filter configuration, split seed, row
  counts per split, and per-sense coverage
- **AND** the artifact is not modified after being written

#### Scenario: Corpus is rebuilt

- **GIVEN** the same inputs and configuration recorded in a corpus manifest
- **WHEN** the corpus is rebuilt
- **THEN** the row set and split assignment are identical

### Requirement: Quarantine retention

Rejected rows SHALL be retained rather than discarded.

#### Scenario: Row is rejected

- **WHEN** a row fails any validation, verification, or deduplication rule
- **THEN** it is written to a quarantine artifact with the failing rule recorded
- **AND** quarantine volume per rule is reported

#### Scenario: Quarantine rate is high

- **WHEN** the share of rows rejected by a single rule exceeds the configured
  threshold
- **THEN** the assembly reports it as a probable upstream defect rather than
  proceeding silently
