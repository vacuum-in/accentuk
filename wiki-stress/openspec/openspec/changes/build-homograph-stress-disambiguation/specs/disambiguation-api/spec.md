# Delta for Disambiguation API

## MODIFIED Requirements

### Requirement: Component responsibility boundary

Python SHALL own all offline processing and SHALL additionally own an
**internal** model inference service that is not exposed outside the deployment
network. Go SHALL remain the only public HTTP interface and SHALL NOT acquire
any part of the model lifecycle.

This modifies the prior boundary, under which the Python component contained no
HTTP service. The change is deliberate and limited to an internal service.

#### Scenario: Inference service is deployed

- **WHEN** the inference service is deployed
- **THEN** it is reachable only from within the deployment network
- **AND** it is not published as a public endpoint
- **AND** the public contract remains the Go `/v1` API

#### Scenario: Go component scope is checked

- **WHEN** the Go component is reviewed
- **THEN** it contains no dump parsing, morphology inference, corpus processing,
  annotation, tokenizer construction, model loading, or model training code
- **AND** its only model-related behaviour is calling the internal service with
  candidates it obtained from PostgreSQL

#### Scenario: Inference service receives an external request

- **WHEN** a request arrives at the inference service from outside the
  deployment network
- **THEN** it is refused

## ADDED Requirements

### Requirement: Internal disambiguation interface

The inference service SHALL accept a sentence with target spans and their
candidates, and SHALL return one selected candidate per span.

#### Scenario: Request is served

- **WHEN** a request supplies a sentence and one or more spans with candidate
  stressed forms
- **THEN** the response returns, for each span, the selected candidate, its
  stress signature, a score for every candidate, and the margin

#### Scenario: Several spans are supplied

- **WHEN** a request contains several spans
- **THEN** all spans are processed in one request
- **AND** the response preserves input span order

#### Scenario: Span offsets are invalid

- **WHEN** a span's offsets fall outside the sentence or do not delimit a token
- **THEN** the request is rejected with the offending span identified
- **AND** other spans in the request are not silently answered

#### Scenario: Model version is reported

- **WHEN** any response is returned
- **THEN** it names the model version and the inventory hash the model was
  trained against

#### Scenario: Service reports readiness

- **WHEN** the service is queried for health
- **THEN** it reports whether the model artifact is loaded and which version

### Requirement: Lookup integration preserving the ambiguity contract

The Go API SHALL continue to return every valid candidate and SHALL present a
model selection as additional information rather than as a replacement.

#### Scenario: Token is unambiguous

- **WHEN** exact lookup returns one distinct stressed realization for a token
- **THEN** the response is unchanged from the pre-change behaviour
- **AND** the inference service is not called

#### Scenario: Token is ambiguous and context is supplied

- **GIVEN** a request supplies sentence context
- **WHEN** exact lookup returns several distinct stressed realizations
- **THEN** the ambiguous candidates are sent to the inference service in one
  batched call
- **AND** the response retains all candidates
- **AND** the selected candidate is marked, with the resolution source recorded
  as the model

#### Scenario: Token is ambiguous and no context is supplied

- **WHEN** an ambiguous token is looked up without sentence context
- **THEN** the response is the existing ambiguous response
- **AND** the inference service is not called

#### Scenario: Selection margin is below the threshold

- **WHEN** the margin between the two best candidates is below the configured
  threshold
- **THEN** no candidate is marked as selected by the model
- **AND** the default sense is marked instead, with the resolution source
  recorded as the default

#### Scenario: Existing clients are unaffected

- **WHEN** a client that predates this change reads a response
- **THEN** every previously defined field retains its meaning
- **AND** ignoring the added fields yields the pre-change behaviour

### Requirement: Degradation and isolation

An unavailable or slow inference service SHALL reduce answer specificity, never
availability.

#### Scenario: Inference service is unreachable

- **WHEN** the inference service cannot be reached
- **THEN** the API returns the existing ambiguous response
- **AND** the request does not fail
- **AND** the degradation is recorded in metrics

#### Scenario: Inference service exceeds its timeout

- **WHEN** the call exceeds the configured timeout
- **THEN** the call is abandoned
- **AND** the ambiguous response is returned within the API's own latency budget

#### Scenario: Feature is disabled

- **WHEN** model resolution is disabled by configuration
- **THEN** the API behaves exactly as it did before this change
- **AND** no call to the inference service is made

#### Scenario: Model version differs from the active lexicon dataset

- **WHEN** the model's inventory hash does not correspond to the active dataset's
  homograph inventory
- **THEN** the mismatch is reported through health and metrics
- **AND** operators can disable model resolution without redeploying the API

### Requirement: Measured serving performance

Serving performance claims SHALL be measured on named hardware and SHALL
separate dictionary-only from model-assisted requests.

#### Scenario: Benchmark is executed

- **WHEN** the serving benchmark runs
- **THEN** it reports latency distributions for dictionary-only lookups, for
  model-assisted lookups, and for the inference service in isolation
- **AND** it records the hardware, model version, quantization, and batch sizes

#### Scenario: Performance figures are published

- **WHEN** a performance figure appears in documentation
- **THEN** it originates from a recorded benchmark run
- **AND** no figure is pre-populated

### Requirement: Untrusted input handling

The disambiguation path SHALL treat request text as data.

#### Scenario: Sentence contains directive text

- **WHEN** a request sentence contains text resembling instructions
- **THEN** it is processed only as model input
- **AND** no part of it alters service configuration or control flow

#### Scenario: Oversized input is supplied

- **WHEN** a sentence or span count exceeds configured limits
- **THEN** the request is rejected with an explicit limit error
- **AND** the service remains available for subsequent requests
