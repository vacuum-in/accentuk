# Delta for Stress Lexicon

## ADDED Requirements

### Requirement: Canonical Ukrainian normalization

The Python pipeline SHALL produce deterministic canonical lookup keys and
stressed forms for supported Ukrainian lexical records.

#### Scenario: Combining acute accent

- **WHEN** a stressed Ukrainian form contains an acute accent
- **THEN** the stored stressed representation uses decomposed `U+0301`
- **AND** the unstressed lookup key removes only configured stress marks

#### Scenario: Apostrophe variants

- **WHEN** equivalent Ukrainian apostrophe characters occur in a form
- **THEN** the canonical key uses `U+02BC`
- **AND** equivalent input variants resolve to the same canonical key

#### Scenario: Hyphen variants

- **WHEN** a lexical internal dash is one of the configured equivalent dash
  characters
- **THEN** the canonical key uses `-`
- **AND** the pipeline does not split the hyphenated lexical form

#### Scenario: Normalization repeats

- **WHEN** normalization is applied to an already normalized form
- **THEN** the result is unchanged

### Requirement: Stress validity

The pipeline SHALL validate stress marks against Ukrainian vowel structure.

#### Scenario: Valid stressed word

- **WHEN** each acute accent directly follows a Ukrainian vowel
- **THEN** the candidate may pass stress validation

#### Scenario: Isolated or misplaced accent

- **WHEN** an acute accent is isolated or follows a non-vowel
- **THEN** the candidate is rejected or quarantined
- **AND** the reason is reported

#### Scenario: Stress signature

- **WHEN** a valid stressed form is accepted
- **THEN** its stress signature is calculated from Unicode vowel ordinals
- **AND** byte positions are not used

### Requirement: Structured lemma extraction

The pipeline SHALL prefer explicitly structured Ukrainian entry data when
identifying lexemes and stressed lemmas.

#### Scenario: Explicit headword template

- **WHEN** a supported Ukrainian headword template provides a stressed lemma
- **THEN** the lemma is extracted with structured-source provenance
- **AND** receives the configured structured-source confidence

#### Scenario: Multiple lexemes on one page

- **WHEN** one page represents multiple parts of speech, homonyms, or lexical
  entries
- **THEN** separate lexemes are retained
- **AND** page title alone is not treated as a unique lexeme identifier

### Requirement: Inflected form extraction

The pipeline SHALL extract all forms represented by supported Ukrainian
inflection templates and morphology tables.

#### Scenario: Supported morphology table

- **WHEN** a supported table contains stressed forms
- **THEN** each lexical form is associated with its lexeme
- **AND** canonical grammatical tags are derived from row, column, template, and
  heading context

#### Scenario: Multiple forms in one cell

- **WHEN** a table cell structurally contains multiple forms
- **THEN** legitimate separators are handled
- **AND** hyphenated words are not split
- **AND** notes, footnotes, references, and formatting are removed

#### Scenario: Reflexive verb

- **WHEN** a Ukrainian verb form ends in `-ся` or `-сь`
- **THEN** the suffix remains part of the lexical form

#### Scenario: Unsupported structure

- **WHEN** a page appears to contain forms but uses an unsupported structure
- **THEN** the pipeline does not claim those forms were extracted
- **AND** records a coverage signal for future handler development

### Requirement: Ambiguity preservation

The lexicon SHALL preserve all legitimate stress and lexical variants instead
of forcing one answer.

#### Scenario: Homographic stress variants

- **GIVEN** `за́мок` and `замо́к` normalize to `замок`
- **WHEN** they are imported
- **THEN** both stressed forms remain queryable
- **AND** neither overwrites the other

#### Scenario: Same form across lexemes

- **WHEN** the same normalized form belongs to multiple lexemes or grammatical
  analyses
- **THEN** each supported analysis remains represented
- **AND** unrelated homonyms are not merged solely by spelling

### Requirement: Provenance and confidence

Every accepted lexical record SHALL retain source and extraction-quality
metadata.

#### Scenario: Record is accepted

- **WHEN** a lemma, form, or stress variant is accepted
- **THEN** it records dataset, page, revision, source kind, parser version, and
  confidence where available

#### Scenario: Regex fallback

- **WHEN** a candidate is emitted by controlled fallback extraction
- **THEN** its confidence is lower than equivalent structured extraction
- **AND** its fallback provenance is visible

### Requirement: Versioned reproducible dataset

Every database build SHALL be traceable to source and implementation versions.

#### Scenario: Build completes

- **WHEN** a dataset is built
- **THEN** the manifest records dump checksum, dump metadata, parser version,
  normalization version, schema version, Git commit, command options, start and
  finish times, and statistics

#### Scenario: Same inputs are rebuilt

- **GIVEN** identical source bytes, parser version, normalization version, and
  configuration
- **WHEN** the pipeline is rerun
- **THEN** stable natural keys and canonical record content are reproducible

### Requirement: Bulk database loading

The Python pipeline SHALL load large datasets without per-record SQL inserts.

#### Scenario: Staging import

- **WHEN** normalized staging records are imported
- **THEN** PostgreSQL `COPY` or an equivalently measured bulk method is used
- **AND** validation occurs before publication

### Requirement: Atomic dataset publication

The system SHALL expose either the previous valid dataset or the new valid
dataset, never a partially built dataset.

#### Scenario: New build passes validation

- **WHEN** a new dataset passes schema and quality gates
- **THEN** it becomes active through one atomic publication transaction
- **AND** subsequent API lookups can discover the new dataset

#### Scenario: New build fails

- **WHEN** parsing, import, projection, indexing, or quality validation fails
- **THEN** the active dataset remains unchanged
- **AND** the failed run is recorded

#### Scenario: Operator rolls back

- **WHEN** an operator activates a previous published dataset
- **THEN** the change is atomic
- **AND** the prior dataset becomes available to new lookups without rebuilding
  it

### Requirement: Transparent quality reporting

The pipeline SHALL report coverage and quality limitations without asserting
complete Ukrainian-language coverage.

#### Scenario: Import report is generated

- **WHEN** a build finishes
- **THEN** reports include page counts, Ukrainian sections, lexemes, forms,
  stressed forms, ambiguity, confidence, rejections, errors, duplicates,
  unsupported templates, and database sizes

#### Scenario: Incomplete source coverage

- **WHEN** Wiktionary omits a word or a supported entry lacks a stressed form
- **THEN** the system reports no extracted result
- **AND** does not synthesize an unsupported answer
