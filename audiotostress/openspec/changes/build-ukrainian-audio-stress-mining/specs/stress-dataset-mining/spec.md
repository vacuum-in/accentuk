# Delta for Stress Dataset Mining

## ADDED Requirements

### Requirement: Candidate target discovery
The mining pipeline MUST identify aligned words whose lexicon lookup returns multiple valid stress variants.

Unambiguous words MAY be retained for acoustic-model training but MUST be distinguishable from ambiguous mining targets.

#### Scenario: Discover homograph occurrence
- GIVEN a transcript containing a lexically ambiguous word
- WHEN candidate discovery runs
- THEN the occurrence is emitted as a mining candidate with all valid stress variants.

### Requirement: Evidence-preserving mining record
Every candidate occurrence MUST have a canonical record containing sufficient information to audit the decision.

The record MUST include at least:
- stable `record_id`
- source/utterance/audio identifiers
- optional speaker identifier
- original and normalized sentence
- target surface form and character/token position
- word audio start/end
- aligned vowel intervals
- candidate stress variants and vowel positions
- predictions from every enabled scorer
- alignment quality
- ASR/reference-text match quality when applicable
- calibrated final confidence
- accepted/rejected status and reason
- model/config/lexicon/dataset versions
- licensing/provenance metadata.

#### Scenario: Audit accepted record
- GIVEN an accepted ambiguous-word record
- WHEN inspected later
- THEN a reviewer can identify the exact source segment, candidates, model outputs, and acceptance reason without re-running inference.

### Requirement: Precision-first acceptance
The mining pipeline MUST treat precision as more important than coverage.

It MUST support an acceptance policy that considers:
- calibrated primary-model confidence
- top-1 versus runner-up margin
- alignment quality
- target identity / ASR agreement
- agreement among independent scorers when enabled.

#### Scenario: Low margin is rejected
- GIVEN top candidate probability 0.55 and runner-up probability 0.44
- WHEN the configured minimum margin exceeds 0.11
- THEN the record is rejected from training output
- AND retained with reason `low_candidate_margin`.

#### Scenario: Model disagreement is rejected
- GIVEN a primary ranker selecting candidate A
- AND an enabled independent scorer selecting candidate B
- WHEN the policy requires agreement
- THEN the occurrence is rejected or routed to manual review
- AND MUST NOT silently enter the high-confidence training set.

### Requirement: Calibrated confidence
Acceptance thresholds MUST be selected using held-out manually verified data rather than raw neural probabilities alone.

The system MUST support confidence calibration and persist the calibration artifact/version.

#### Scenario: Apply calibrated probability
- GIVEN raw model logits and an active calibration artifact
- WHEN confidence is produced
- THEN acceptance uses the calibrated value
- AND the calibration version is written to the record.

### Requirement: Exact target identity gating
The mining pipeline MUST reject occurrences when target-word identity is uncertain beyond configured tolerance.

Where a trusted transcript exists, ASR/alignment token identity SHOULD be cross-checked against it.

#### Scenario: Transcript disagreement
- GIVEN trusted reference token `замовк`
- AND ASR/alignment token `замок`
- WHEN target validation runs
- THEN the occurrence cannot be accepted as a `замок` stress example.

### Requirement: Canonical Parquet outputs
The system MUST produce versioned Parquet datasets for:
- bootstrap unambiguous acoustic training examples
- ambiguous mining candidates
- accepted high-confidence ambiguous records
- rejected/manual-review records.

#### Scenario: Write mining shard
- GIVEN a completed processing batch
- WHEN output is committed
- THEN each dataset has a schema version
- AND batch metadata identifies source inputs and pipeline versions
- AND failed/rejected records are not lost.

### Requirement: Idempotent processing
Re-running the same pipeline version/configuration over the same source record MUST NOT create semantically duplicate accepted records.

#### Scenario: Reprocess same utterance
- GIVEN an utterance already processed under the same fingerprint
- WHEN the job is rerun
- THEN its stable record identifiers are reused or deterministically regenerated
- AND duplicate accepted rows are not appended.

### Requirement: Resume and shard processing
The pipeline MUST support resumable corpus processing and independent shards suitable for local or distributed execution.

#### Scenario: Resume failed batch
- GIVEN a 100-shard mining job where 80 shards completed
- WHEN processing resumes
- THEN successfully committed shards are not recomputed unless explicitly requested.

### Requirement: License-aware output filtering
The system MUST preserve source licensing/provenance fields and support excluding records from export based on license policy.

#### Scenario: Non-redistributable source
- GIVEN a source whose metadata forbids audio redistribution
- WHEN a text-only training export is created
- THEN the policy can omit audio bytes/paths from the exported artifact while preserving internal provenance references
- AND the exporter does not falsely relabel the source license.

### Requirement: Text-resolver export
The system MUST provide a transformation from accepted audio-mined records to text-resolver examples.

The export MUST include:
- context sentence
- marked target span
- candidate stress variants
- selected gold variant
- mining confidence
- source category.

#### Scenario: Export contextual sample
- GIVEN accepted record for `Працівник перевірив замок.` with selected `замо́к`
- WHEN text-resolver export runs
- THEN output contains the sentence with the target span identified
- AND candidates include every lexicon-valid variant
- AND gold equals `замо́к`.
