# Delta for Lookup API

## ADDED Requirements

### Requirement: Go-only public lookup service

The public HTTP lookup service SHALL be implemented in Go and SHALL perform
read-only lookups against PostgreSQL.

#### Scenario: API request is handled

- **WHEN** a client sends a valid lookup request
- **THEN** the Go service validates the request
- **AND** queries PostgreSQL
- **AND** serializes the response

#### Scenario: ETL behavior is requested from Go

- **WHEN** an implementation change would add dump parsing, morphology
  extraction, import, migration, publication, or linguistic rule authoring to Go
- **THEN** the change violates the architecture boundary
- **AND** the behavior remains in Python tooling

### Requirement: Exact word lookup

The API SHALL return exact matches for a canonical normalized input key.

#### Scenario: One candidate exists

- **WHEN** a normalized word has exactly one valid candidate
- **THEN** the response status is `found`
- **AND** the candidate includes stressed form, lemma, grammatical metadata,
  confidence, and dataset identifier

#### Scenario: Multiple candidates exist

- **WHEN** a normalized word has multiple valid candidates
- **THEN** the response status is `ambiguous`
- **AND** all candidates up to the configured result limit are returned
- **AND** the API does not claim contextual correctness

#### Scenario: No candidate exists

- **WHEN** no exact record exists in the active dataset
- **THEN** the response status is `not_found`
- **AND** no stress form is invented

### Requirement: Deterministic candidate ordering

The API SHALL order identical lookup results deterministically.

#### Scenario: Ambiguous result is returned repeatedly

- **WHEN** the same word is queried against the same dataset
- **THEN** candidate ordering is stable
- **AND** ranking prioritizes configured usage, confidence, source quality, and
  stable lexical ordering

### Requirement: Best-candidate mode

The API SHALL support a ranked best-candidate mode while preserving evidence of
ambiguity.

#### Scenario: Best mode has alternatives

- **WHEN** `mode=best` is requested for an ambiguous word
- **THEN** the top-ranked candidate is returned
- **AND** the response indicates that alternatives exist
- **AND** the response does not imply sentence-context disambiguation

### Requirement: Batch lookup

The API SHALL support efficient lookup of multiple words while preserving input
identity.

#### Scenario: Batch contains duplicates

- **WHEN** a batch contains the same word more than once
- **THEN** every occurrence is represented in the output
- **AND** original input order is preserved

#### Scenario: Maximum batch is accepted

- **WHEN** a request contains no more than the configured maximum, initially
  10,000 words
- **THEN** the service performs a set-oriented PostgreSQL lookup
- **AND** does not issue one database round trip per word

#### Scenario: Batch exceeds limit

- **WHEN** a request exceeds the configured batch limit
- **THEN** the API rejects it with a bounded error response
- **AND** no database lookup is attempted

### Requirement: Lemma form retrieval

The API SHALL permit exact retrieval of stored forms for a lemma.

#### Scenario: Lemma has many forms

- **WHEN** a client requests all forms for an exact normalized lemma
- **THEN** results are paginated
- **AND** each result includes its stressed form and grammatical tags

### Requirement: Active dataset visibility

The API SHALL query one published dataset version per request.

#### Scenario: Dataset switches during traffic

- **WHEN** Python publishes a new active dataset while requests are in flight
- **THEN** each request completes against one immutable dataset ID
- **AND** later requests discover the new active dataset without an API restart
- **AND** no response mixes rows from two dataset versions

### Requirement: Conservative request normalization

The Go service SHALL apply only the versioned lookup-key normalization required
to match Python-produced keys.

#### Scenario: Equivalent apostrophe input

- **WHEN** a client submits a configured equivalent apostrophe character
- **THEN** the Go service produces the same lookup key as Python

#### Scenario: Conformance drift

- **WHEN** Go normalization differs from a Python-generated conformance vector
- **THEN** automated tests fail
- **AND** the service is not considered release-ready

### Requirement: Bounded API behavior

The API SHALL enforce configured bounds on inputs and outputs.

#### Scenario: Unsupported characters

- **WHEN** a lookup token contains unsupported controls or exceeds length limits
- **THEN** the request is rejected
- **AND** the invalid value is not interpolated into SQL

#### Scenario: Excessive result set

- **WHEN** a word has more candidates than the configured result limit
- **THEN** the response is bounded
- **AND** `truncated` is true

### Requirement: Read-only database access

The Go API SHALL use credentials that cannot modify lexicon datasets.

#### Scenario: API attempts a write

- **WHEN** the API database role attempts to insert, update, delete, migrate, or
  publish data
- **THEN** PostgreSQL denies the operation

### Requirement: Stable error contract

The API SHALL return structured errors without exposing internal implementation
details.

#### Scenario: Invalid request

- **WHEN** request syntax or validation fails
- **THEN** the response includes a stable error code and request ID
- **AND** does not contain SQL, credentials, or a stack trace

#### Scenario: PostgreSQL unavailable

- **WHEN** a lookup cannot run because PostgreSQL is unavailable
- **THEN** the API returns a service-unavailable response
- **AND** readiness reports unavailable

### Requirement: Operational endpoints

The API SHALL expose liveness, readiness, version, and metrics endpoints.

#### Scenario: No active dataset

- **WHEN** PostgreSQL is reachable but no dataset is active
- **THEN** liveness remains healthy
- **AND** readiness fails

#### Scenario: Metrics are collected

- **WHEN** lookups are processed
- **THEN** bounded-cardinality metrics record latency, hit, miss, ambiguity,
  truncation, pool state, and active dataset
- **AND** raw words are not used as metric labels
