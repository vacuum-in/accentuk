# Delta for Operations

## ADDED Requirements

### Requirement: Reproducible local deployment

The project SHALL provide documented commands to run PostgreSQL, Python ETL, and
the Go API locally.

#### Scenario: Fresh checkout

- **WHEN** an operator follows the documented Docker Compose workflow
- **THEN** PostgreSQL becomes healthy
- **AND** Python can migrate and build a dataset
- **AND** Go can serve lookups from the published dataset

### Requirement: Automated verification

The project SHALL provide automated tests for linguistic correctness, database
publication, and API contracts.

#### Scenario: Continuous integration runs

- **WHEN** a change is proposed for merge
- **THEN** Python linting, typing, unit tests, and integration tests run
- **AND** Go formatting, vetting, unit tests, integration tests, and race tests
  run
- **AND** database migrations and API contracts are validated

### Requirement: Measured performance

The project SHALL generate performance reports from executable benchmarks and
SHALL NOT fabricate results.

#### Scenario: ETL benchmark runs

- **WHEN** an ETL benchmark completes
- **THEN** the report records hardware, input dataset, command, throughput,
  memory, import timings, projection timings, and storage sizes

#### Scenario: API benchmark runs

- **WHEN** an API benchmark completes
- **THEN** the report records hardware, PostgreSQL configuration, active dataset,
  concurrency, sample count, throughput, and latency percentiles

#### Scenario: Target is missed

- **WHEN** a measured target is not achieved
- **THEN** the report states the measured result
- **AND** the task is not marked successful by substituting the target value

### Requirement: Safe failure and recovery

The system SHALL provide documented recovery for interrupted imports, failed
publication, database restore, and dataset rollback.

#### Scenario: Parsing is interrupted

- **WHEN** the ETL process is interrupted after a valid checkpoint
- **THEN** it may resume without duplicating accepted natural keys
- **AND** an incomplete run cannot become active

#### Scenario: Published dataset is defective

- **WHEN** an operator selects a previous retained dataset
- **THEN** rollback is performed atomically
- **AND** the Go API discovers the selected dataset

### Requirement: Source attribution and limitations

The project SHALL preserve source attribution metadata and document dataset
limitations.

#### Scenario: Dataset is distributed

- **WHEN** a database or export is released
- **THEN** its documentation identifies the source dump, dump date/checksum, and
  available page/revision provenance
- **AND** states that extraction coverage is incomplete and structure-dependent
- **AND** does not claim that every Ukrainian word or stress is present
