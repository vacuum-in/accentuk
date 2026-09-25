# Delta for Dump Ingestion

## ADDED Requirements

### Requirement: Verified dump acquisition

The Python pipeline SHALL acquire or accept an official Ukrainian Wiktionary
article dump and SHALL record enough metadata to reproduce the input dataset.

#### Scenario: Download completes successfully

- **WHEN** an operator downloads a configured dump
- **THEN** the pipeline stores the file through an atomic temporary-file process
- **AND** calculates and records its SHA-256 checksum
- **AND** records the source URL, filename, and acquisition timestamp

#### Scenario: Checksum does not match

- **GIVEN** an expected checksum was supplied
- **WHEN** the downloaded file has a different checksum
- **THEN** the command fails
- **AND** the invalid file is not accepted as an import source

### Requirement: Bounded-memory XML processing

The Python pipeline SHALL process the compressed dump with approximately
constant memory growth relative to dump size.

#### Scenario: Large dump is processed

- **WHEN** the pipeline parses a multi-gigabyte `.xml.bz2` dump
- **THEN** it reads the compressed stream without first expanding the whole dump
  to disk
- **AND** does not load the whole XML document into memory
- **AND** releases processed page elements

#### Scenario: Page exceeds configured limit

- **WHEN** an individual page exceeds the configured uncompressed page limit
- **THEN** the page is rejected or quarantined
- **AND** processing continues unless fail-fast mode is enabled

### Requirement: Main-namespace page extraction

The pipeline SHALL extract page and revision metadata from lexical pages without
processing unrelated namespaces as lexical entries.

#### Scenario: Main namespace page

- **WHEN** a main-namespace page is encountered
- **THEN** its page ID, title, redirect target, revision ID, revision timestamp,
  and wikitext are made available to downstream parsers

#### Scenario: Non-main namespace page

- **WHEN** a category, template, discussion, help, file, or project page is
  encountered
- **THEN** it is excluded from lexical extraction by default

### Requirement: Ukrainian section isolation

The pipeline SHALL extract lexical candidates only from confirmed Ukrainian
language sections.

#### Scenario: Multilingual page

- **GIVEN** a page contains Ukrainian and non-Ukrainian sections
- **WHEN** the page is parsed
- **THEN** candidates are extracted only from the Ukrainian section
- **AND** forms from other languages are not emitted

#### Scenario: Unknown language marker

- **WHEN** the parser cannot confidently classify a language section
- **THEN** it does not treat the section as Ukrainian
- **AND** records the marker for analysis

### Requirement: Safe MediaWiki parsing

The pipeline SHALL treat MediaWiki content as untrusted data and parse supported
syntax without executing it.

#### Scenario: Template invocation is present

- **WHEN** the page contains a template or Lua-backed template invocation
- **THEN** the invocation is parsed as data
- **AND** no template, Lua module, remote code, or shell command is executed

#### Scenario: Unsupported template is present

- **WHEN** no registered handler supports a template
- **THEN** the template is counted
- **AND** representative page titles and bounded invocation samples are reported
- **AND** the import continues

### Requirement: Deterministic parallel parsing

The Python pipeline SHALL permit bounded parallel page parsing without changing
the logical output.

#### Scenario: Worker count changes

- **GIVEN** the same dump, parser version, and configuration
- **WHEN** it is parsed with different supported worker counts
- **THEN** the normalized record set and stable natural keys are identical

#### Scenario: Worker fails

- **WHEN** a parsing worker encounters an unhandled exception
- **THEN** the failure is propagated to the coordinator
- **AND** partial output is not silently marked complete
- **AND** resume metadata remains consistent
