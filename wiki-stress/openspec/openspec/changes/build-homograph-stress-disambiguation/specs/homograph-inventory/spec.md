# Delta for Homograph Inventory

## ADDED Requirements

### Requirement: Complete sense records

The pipeline SHALL maintain a versioned inventory in which every homograph sense
carries the fields required to describe it to an annotator: stressed form, part
of speech, grammatical features, definition, register, and a contrast statement
distinguishing it from its sibling senses.

#### Scenario: Sense record is complete

- **WHEN** a sense record is emitted to the frozen inventory
- **THEN** it carries a stressed form, a part of speech, a definition, a
  contrast statement, and a review status
- **AND** it records the source of each field

#### Scenario: Definition is missing

- **GIVEN** a sense has no definition from any dictionary source
- **WHEN** the inventory is built
- **THEN** the sense is marked incomplete
- **AND** it is excluded from corpus generation
- **AND** it is counted in a completion report

#### Scenario: Definition originates from a language model

- **WHEN** a definition or contrast statement was proposed by a language model
- **THEN** it is stored with a review status indicating it is unconfirmed
- **AND** it SHALL NOT be used as the basis for generated sentences until a
  human confirms it

### Requirement: Duplicate stress collapse

Senses within a group that share an identical stressed form SHALL NOT be
presented as distinguishable labels, because no stress model can separate them.

#### Scenario: Two senses share a stressed form

- **GIVEN** a group contains two sense rows whose canonical stressed forms are
  equal
- **WHEN** the inventory is built
- **THEN** the senses are merged into one label
- **AND** both definitions are retained on the merged record
- **AND** the merge is recorded in a report

#### Scenario: Group reduces to one label

- **WHEN** collapsing leaves a group with a single distinct stressed form
- **THEN** the group is removed from the ambiguous set
- **AND** the removal is reported, because exact lookup already resolves it

### Requirement: Paradigm expansion from an external dictionary

The pipeline SHALL expand every sense to its inflected forms using a pinned,
vendored external morphological dictionary, and SHALL NOT depend on the
PostgreSQL lexicon for paradigms.

#### Scenario: Dictionary is acquired

- **WHEN** the morphological dictionary is downloaded
- **THEN** its release identifier, source URL, and SHA-256 checksum are recorded
- **AND** a later build with the same recorded checksum produces the same
  paradigms

#### Scenario: Sense matches one paradigm

- **WHEN** a sense matches exactly one dictionary paradigm by lemma, part of
  speech, and grammatical features
- **THEN** its inflected forms are emitted with grammatical tags
- **AND** the paradigm source is recorded as dictionary-derived

#### Scenario: Sense matches several paradigms

- **WHEN** more than one paradigm matches a sense
- **THEN** the recorded grammatical features are used to select one
- **AND** if they cannot select one, the sense is flagged for review rather than
  resolved arbitrarily

#### Scenario: Sense matches no paradigm

- **GIVEN** a proper noun, toponymic adjective, or dialectal entry absent from
  the dictionary
- **WHEN** paradigm expansion runs
- **THEN** a declension-class rule may generate the paradigm
- **AND** the paradigm source is recorded as generated
- **AND** generated paradigms are distinguishable from dictionary-derived ones
  in every downstream artifact

### Requirement: Stress transfer across a paradigm

Stress SHALL be transferred to inflected forms only where the transfer is
determinate, and SHALL NOT be inferred where the paradigm has mobile stress.

#### Scenario: Fixed-stress paradigm

- **WHEN** a paradigm has fixed stress
- **THEN** the accent position is carried to every form by vowel ordinal using
  the existing stress-signature rules
- **AND** each generated form passes stress validation

#### Scenario: Mobile-stress paradigm

- **WHEN** a paradigm has mobile stress
- **THEN** accent positions are taken from accented dictionary data or manual
  entry
- **AND** if neither is available the forms are flagged as unknown-stress
- **AND** unknown-stress forms are excluded from labels

#### Scenario: Transferred form fails validation

- **WHEN** a transferred form places an acute anywhere other than immediately
  after a Ukrainian vowel
- **THEN** the form is rejected
- **AND** the rejection is recorded with its sense and paradigm cell

### Requirement: Ambiguous form surface

The pipeline SHALL compute, per group, the set of unstressed surface forms that
collide across two or more senses, and SHALL treat only that set as requiring
training data.

#### Scenario: Form collides across senses

- **WHEN** two senses of a group produce the same unstressed form under the
  canonical lookup key
- **THEN** the form is emitted with every candidate stressed realization and its
  stress signature
- **AND** each candidate names its sense

#### Scenario: Form separates orthographically

- **WHEN** an inflected form of one sense has no collision within its group
- **THEN** it is excluded from the ambiguous surface
- **AND** the exclusion is counted, because exact lookup already resolves it

#### Scenario: Ambiguity is limited to part of a paradigm

- **WHEN** a group collides in some paradigm cells and not others
- **THEN** the report distinguishes groups ambiguous in the nominative only from
  groups ambiguous across the paradigm
- **AND** the distinction is available for sizing per-group corpus budgets

### Requirement: Canonical normalization reuse

The inventory SHALL use the project's existing canonicalization rules, and SHALL
NOT define its own.

#### Scenario: Forms are compared

- **WHEN** two forms are compared for collision, deduplication, or matching
- **THEN** comparison uses the existing canonical lookup key, which applies NFD,
  canonical apostrophes, canonical internal hyphens, lowercasing, and removal of
  only the acute accent
- **AND** no other normalization is applied

#### Scenario: Composed input is supplied

- **GIVEN** a source record contains precomposed accented characters
- **WHEN** it enters the inventory
- **THEN** it is decomposed before storage
- **AND** stress signatures computed from it match those computed from the
  decomposed equivalent

### Requirement: Frozen, versioned inventory

The inventory SHALL be immutable once frozen and SHALL be identifiable by
content hash from every artifact derived from it.

#### Scenario: Inventory is frozen

- **WHEN** an inventory version is written
- **THEN** it records a content hash, the dictionary release checksum, the
  normalization version, and creation metadata
- **AND** downstream artifacts record that content hash

#### Scenario: Inventory changes after corpus generation

- **WHEN** a sense boundary changes after a corpus was built
- **THEN** a new inventory version is created rather than editing the frozen one
- **AND** artifacts built from the previous version remain attributable to it
