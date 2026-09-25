# Delta for LLM Annotation

## ADDED Requirements

### Requirement: Separated annotation jobs

The pipeline SHALL implement inventory completion, sense labelling, sentence
generation, and blind verification as separate jobs with separate prompts,
separate outputs, and independently recorded model configuration.

#### Scenario: Job is executed

- **WHEN** an annotation job runs
- **THEN** its outputs are written to a job-specific artifact
- **AND** the deployment name, model version, API version, prompt version, and
  sampling parameters are recorded with the output

#### Scenario: Verification deployment is selected

- **WHEN** blind verification runs
- **THEN** it uses a deployment different from the one that produced the
  labelling or generation being verified
- **AND** the command fails if the configured deployments are identical

### Requirement: Contrastive prompting

Sense labelling and generation SHALL present every sibling sense of a group in
the same request, so that the model distinguishes senses rather than describing
one in isolation.

#### Scenario: Group has several senses

- **WHEN** a request concerns a sense of a group with multiple senses
- **THEN** the request includes every sibling sense with its stressed form,
  definition, and contrast statement
- **AND** the target sense is identified explicitly

#### Scenario: Sibling sense is incomplete

- **GIVEN** a sibling sense has no confirmed definition
- **WHEN** a request for that group is prepared
- **THEN** the request is not issued
- **AND** the group is reported as blocked on inventory completion

### Requirement: Sense labelling of mined sentences

The labelling job SHALL assign an inventory sense to each mined sentence, and
SHALL be permitted to decline.

#### Scenario: Context determines the sense

- **WHEN** a mined sentence is labelled
- **THEN** the output names one inventory sense, a confidence value, and the
  words in the sentence that determine the choice
- **AND** the named sense belongs to the group of the matched form

#### Scenario: Context does not determine the sense

- **WHEN** the sentence does not permit a choice
- **THEN** the job returns an explicit undetermined result
- **AND** the sentence is excluded from training data
- **AND** it is retained for an abstention evaluation slice

#### Scenario: Supporting evidence is absent

- **WHEN** the returned determining words are empty or do not occur in the
  sentence
- **THEN** the row is discarded

#### Scenario: Returned sense is not in the candidate set

- **WHEN** the output names a sense outside the group
- **THEN** the row is discarded
- **AND** the occurrence is counted as a schema violation

### Requirement: Generation for starved senses

The generation job SHALL produce sentences only for senses identified as starved
by the coverage report, and SHALL target specific inflected forms.

#### Scenario: Generation request is prepared

- **WHEN** sentences are requested for a starved sense
- **THEN** the request names the exact target surface form and its grammatical
  features
- **AND** the target form is drawn from the ambiguous form surface

#### Scenario: Natural examples exist for the sense

- **GIVEN** the sense has at least one human-authored citation
- **WHEN** generation is requested
- **THEN** those citations are supplied as style anchors

#### Scenario: Generated sentence violates form constraints

- **WHEN** a generated sentence does not contain the target form exactly once,
  contains acute accents, or contains a sibling sense's stressed spelling
- **THEN** the sentence is discarded
- **AND** the violation type is counted

#### Scenario: Generated sentence explains rather than uses the word

- **WHEN** a generated sentence contains metalinguistic phrasing about stress,
  meaning, or word usage
- **THEN** the sentence is discarded

#### Scenario: Generated batch lacks variety

- **WHEN** a disproportionate share of a sense's generated sentences share an
  opening construction
- **THEN** the batch is rejected for regeneration rather than filtered down

### Requirement: Blind verification

Every labelled and generated sentence SHALL be verified by a job that cannot see
the intended label and cannot see the target token.

#### Scenario: Verification is performed

- **WHEN** a sentence is verified
- **THEN** the target span is masked in the verification request
- **AND** the intended sense is not present in the request
- **AND** the job selects a sense from the group's candidate list

#### Scenario: Verification disagrees

- **WHEN** the verifier's choice differs from the intended label
- **THEN** the sentence is quarantined rather than deleted
- **AND** the disagreement is recorded with both labels

#### Scenario: A group disagrees systematically

- **WHEN** a group's verification agreement rate falls below the configured
  threshold
- **THEN** the group is reported as a suspected sense-boundary defect
- **AND** the report distinguishes this from individual sentence defects

### Requirement: Structured, validated responses

The pipeline SHALL request structured output and SHALL treat model responses as
untrusted data.

#### Scenario: Response conforms to the schema

- **WHEN** a response is received
- **THEN** it is validated against the declared schema before use
- **AND** non-conforming responses are recorded and retried within the
  configured attempt limit

#### Scenario: Response contains instructions

- **WHEN** a response contains text directing the pipeline to take an action
- **THEN** the text is treated as data
- **AND** no part of a response is executed, evaluated, or used to alter
  pipeline configuration

### Requirement: Reproducible and resumable execution

Annotation SHALL be resumable, and every raw response SHALL be retained before
parsing.

#### Scenario: Raw responses are persisted

- **WHEN** a response is received
- **THEN** it is written to a raw artifact before any parsing
- **AND** a parsing change can be re-applied without re-issuing requests

#### Scenario: Run is interrupted

- **WHEN** an annotation run is interrupted
- **THEN** a resumed run does not re-issue requests for completed groups
- **AND** partial output is not marked complete

#### Scenario: Run is repeated

- **GIVEN** the same inventory version, coverage report, prompt version, and
  seed
- **WHEN** the run is repeated from persisted raw responses
- **THEN** the parsed annotation output is identical

### Requirement: Cost control

The pipeline SHALL bound annotation spend and SHALL report actual consumption.

#### Scenario: Batch is submitted

- **WHEN** a batch is submitted
- **THEN** its input and output token counts and cost are recorded
- **AND** the cumulative total is compared against the configured cap

#### Scenario: Cap would be exceeded

- **WHEN** submitting a batch would exceed the configured spend cap
- **THEN** submission is refused
- **AND** the run stops with the remaining work reported

#### Scenario: Pilot precedes a full run

- **WHEN** a full annotation run is requested
- **THEN** it is refused unless a pilot run's quality report exists and meets the
  configured pass rate

### Requirement: Credential handling

The pipeline SHALL obtain service credentials from the environment and SHALL NOT
persist them.

#### Scenario: Credentials are supplied

- **WHEN** the pipeline authenticates to the annotation service
- **THEN** it reads the endpoint and credential from environment configuration
- **AND** no credential value is written to artifacts, manifests, logs, or
  reports
