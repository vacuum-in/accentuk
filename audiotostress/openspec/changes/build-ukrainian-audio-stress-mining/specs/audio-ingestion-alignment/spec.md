# Delta for Audio Ingestion and Alignment

## ADDED Requirements

### Requirement: Corpus manifest ingestion
The system MUST ingest speech corpora through a versioned manifest instead of corpus-specific hard-coded paths.

The manifest MUST support at least:
- `source_id`
- `audio_uri` or local path
- optional reference transcript
- optional `speaker_id`
- optional segment start/end offsets
- source/license/provenance metadata
- redistribution and commercial-use flags when known

#### Scenario: Ingest local audio with transcript
- GIVEN a manifest record with a readable local audio path and Ukrainian transcript
- WHEN the ingestion pipeline runs
- THEN the system creates a normalized internal utterance record
- AND preserves the original transcript and provenance fields
- AND records the audio duration and sample rate.

#### Scenario: Reject unreadable audio
- GIVEN a manifest record whose audio cannot be decoded
- WHEN ingestion runs
- THEN the record is marked rejected with a machine-readable error reason
- AND downstream alignment MUST NOT receive that record.

### Requirement: Canonical audio normalization
The system MUST expose a canonical mono waveform representation at a configurable sample rate, defaulting to 16 kHz, without overwriting the original media.

#### Scenario: Normalize stereo audio
- GIVEN a stereo source file
- WHEN audio normalization runs
- THEN the downstream waveform is mono
- AND the transformation is recorded in metadata
- AND original timing remains recoverable relative to the source segment.

### Requirement: Ukrainian transcript normalization
The system MUST normalize transcript text using deterministic, versioned rules suitable for matching spoken words while retaining the original text.

Normalization MUST NOT insert lexical stress marks into ambiguous words.

#### Scenario: Preserve ambiguous spelling
- GIVEN a transcript containing `замок`
- WHEN transcript normalization runs
- THEN the normalized token remains unstressed `замок`
- AND no stress variant is selected by the normalizer.

### Requirement: ASR fallback
For records without a trusted transcript, the system MUST support generating a Ukrainian transcript through a configurable ASR backend.

#### Scenario: Generate transcript for audio-only record
- GIVEN valid Ukrainian audio with no transcript
- WHEN ASR processing runs
- THEN the utterance receives a transcript
- AND the ASR backend/model version is recorded
- AND available token/segment confidence is retained.

### Requirement: Word-level forced alignment
The system MUST align normalized transcript tokens to audio and return word start/end times plus an alignment confidence or quality indicator.

#### Scenario: Align target word
- GIVEN transcript `Працівник перевірив замок.` and matching speech
- WHEN word alignment succeeds
- THEN the record for `замок` contains start/end offsets within the utterance
- AND offsets satisfy `0 <= start < end <= utterance_duration`.

#### Scenario: Alignment cannot identify target
- GIVEN a transcript token that cannot be aligned confidently
- WHEN alignment finishes
- THEN that token is marked unaligned or low-quality
- AND it MUST NOT be automatically accepted as a mined stress example.

### Requirement: Pluggable alignment backends
The word/phone alignment layer MUST use a backend interface so implementations such as WhisperX/CTC and MFA can be selected by configuration.

The rest of the pipeline MUST consume a backend-neutral alignment schema.

#### Scenario: Switch alignment backend
- GIVEN two configured alignment backends
- WHEN the backend setting changes
- THEN downstream dataset generation requires no code changes
- AND records identify which backend/model produced the alignment.

### Requirement: Phone and vowel intervals
The system MUST be able to produce phone-level or vowel-level intervals for a target word.

Each vowel interval MUST contain:
- vowel/grapheme or normalized phone identity
- start time
- end time
- position within the target word
- alignment quality when available.

#### Scenario: Align a two-vowel word
- GIVEN an aligned occurrence of `замок`
- WHEN fine alignment succeeds
- THEN exactly the vowels corresponding to `а` and `о` are represented in order
- AND each has a valid non-overlapping time interval.

### Requirement: Alignment validation
The system MUST validate alignment topology before feature extraction.

At minimum it MUST reject or flag:
- negative durations
- intervals outside the utterance
- reversed intervals
- overlapping vowel intervals beyond configured tolerance
- vowel count inconsistent with the normalized target form, unless an explicitly supported pronunciation rule explains the mismatch.

#### Scenario: Invalid vowel count
- GIVEN a target whose normalized form contains two expected vowels
- AND the aligner returns one usable vowel interval
- WHEN validation runs
- THEN the occurrence is not used as automatic training gold
- AND the rejection reason is persisted.
