# Delta for Corpus Acquisition

## ADDED Requirements

### Requirement: Licence resolution before acquisition

The pipeline SHALL NOT acquire content from a source whose terms have not been
recorded, and SHALL record a licence tag on every acquired sentence.

#### Scenario: Source terms are recorded

- **WHEN** a source is registered for acquisition
- **THEN** its licence, attribution requirement, and redistribution status are
  recorded in the source registry
- **AND** acquisition is permitted only for sources present in that registry

#### Scenario: Source terms are unresolved

- **GIVEN** a source whose terms are unknown or disputed
- **WHEN** acquisition is attempted
- **THEN** the command fails with the unresolved source named
- **AND** no request is issued to that source

#### Scenario: Redistribution status differs across the corpus

- **WHEN** sentences from sources with different licences are assembled
- **THEN** each sentence retains its own licence tag
- **AND** a publishable subset is derivable by filtering on those tags

### Requirement: Sense-bound citation acquisition

The pipeline SHALL extract dictionary citations that are already attached to a
numbered sense, and SHALL preserve that attachment as a label.

#### Scenario: Citation carries a sense number

- **WHEN** a dictionary entry supplies an illustrative quotation under a
  numbered sense
- **THEN** the sentence is emitted with that sense reference, the entry
  identifier, and the source location
- **AND** the label provenance is recorded as human-authored

#### Scenario: Sense number cannot be mapped

- **WHEN** a citation's sense number cannot be mapped to an inventory sense
- **THEN** the citation is quarantined rather than assigned to a guessed sense
- **AND** the unmapped case is reported

#### Scenario: Wiktionary usage examples are extracted

- **WHEN** the already-acquired Ukrainian Wiktionary dump is processed for usage
  examples
- **THEN** examples are extracted only from confirmed Ukrainian sections
- **AND** no template, Lua module, or dump-provided code is executed

### Requirement: Dump-based corpus acquisition

The pipeline SHALL acquire MediaWiki corpora using the existing verified,
resumable download and bounded-memory streaming behaviour.

#### Scenario: Corpus dump is downloaded

- **WHEN** a configured corpus dump is downloaded
- **THEN** it is written through an atomic temporary-file process
- **AND** its SHA-256 checksum, source URL, byte size, and acquisition timestamp
  are recorded

#### Scenario: Corpus dump is streamed

- **WHEN** a multi-gigabyte compressed dump is processed
- **THEN** it is read as a stream without full expansion to disk
- **AND** processed page elements are released
- **AND** memory growth remains approximately constant relative to dump size

#### Scenario: Article markup is converted to prose

- **WHEN** an article page is processed
- **THEN** markup, tables, infoboxes, references, and navigation fragments are
  removed before sentence splitting
- **AND** fragments that are not running prose are excluded

### Requirement: Bounded targeted crawling

The pipeline MUST enforce the rate, robots.txt, authentication, and page-cap
limits below whenever it performs a targeted HTTP crawl. Where a sense remains
without natural examples after dump and bulk-corpus acquisition, the pipeline
MAY perform such a crawl.

#### Scenario: Crawl is executed

- **WHEN** a targeted crawl runs against a registered host
- **THEN** it honours that host's robots.txt
- **AND** issues at most the configured request rate per host
- **AND** sends a descriptive User-Agent identifying the project
- **AND** stops at the configured per-host page cap

#### Scenario: Access requires authentication or payment

- **WHEN** a resource requires a login, a paywall bypass, or any access
  restriction circumvention
- **THEN** the resource is skipped
- **AND** the skip is recorded

#### Scenario: Crawl is interrupted

- **WHEN** a crawl is interrupted
- **THEN** the frontier and fetched-page cache persist
- **AND** a resumed crawl does not re-fetch cached pages

#### Scenario: Raw response is retained

- **WHEN** a page is fetched
- **THEN** the raw response is cached before parsing
- **AND** a parser change can be re-applied without re-fetching

### Requirement: Single-pass ambiguous form matching

The pipeline SHALL locate ambiguous forms in each corpus in a single streaming
pass with bounded memory, matching whole tokens rather than substrings.

#### Scenario: Corpus is scanned

- **WHEN** a corpus is scanned for the ambiguous form surface
- **THEN** all surface forms are matched in one pass over the corpus
- **AND** the matcher is built from the frozen inventory's ambiguous form set

#### Scenario: Match occurs inside a longer word

- **GIVEN** an ambiguous form appears as a substring of a longer token
- **WHEN** the sentence is scanned
- **THEN** it is not reported as a match

#### Scenario: Matching is normalization-consistent

- **WHEN** a candidate token is compared to an ambiguous form
- **THEN** comparison uses the existing canonical lookup key
- **AND** the original surface string, not the normalized string, is retained in
  the corpus
- **AND** the character offsets recorded refer to the original surface string

#### Scenario: Sentence already carries stress marks

- **WHEN** a candidate sentence contains acute accents
- **THEN** it is excluded from mined candidates
- **AND** the exclusion is counted

#### Scenario: Sentence contains several ambiguous forms

- **WHEN** a sentence contains more than one occurrence of ambiguous forms
- **THEN** it is excluded from the default candidate set
- **AND** the exclusion is counted, so a later change can reconsider it

### Requirement: Per-sense sampling during the pass

The pipeline SHALL bound the number of candidates retained per sense during
scanning, and SHALL NOT write the full hit set to disk.

#### Scenario: Frequent form is scanned

- **GIVEN** a form occurring tens of thousands of times in a corpus
- **WHEN** the corpus is scanned
- **THEN** at most the configured candidate cap is retained for that form
- **AND** retained candidates are sampled across the corpus rather than taken
  from its beginning

#### Scenario: Sampling is reproducible

- **GIVEN** the same corpus, inventory version, cap, and random seed
- **WHEN** the scan is repeated
- **THEN** the retained candidate set is identical

### Requirement: Coverage reporting

The pipeline SHALL report acquired candidate counts per sense, and that report
SHALL be the input to the annotation budget.

#### Scenario: Coverage report is produced

- **WHEN** acquisition completes for a corpus
- **THEN** the report records candidates per sense, per group, and per source
  tier
- **AND** records the corpus checksum, byte size, and scan configuration

#### Scenario: Senses remain starved

- **WHEN** a sense has fewer candidates than the configured floor
- **THEN** it is listed as starved
- **AND** the generation budget is derived from that list rather than from an
  estimate
