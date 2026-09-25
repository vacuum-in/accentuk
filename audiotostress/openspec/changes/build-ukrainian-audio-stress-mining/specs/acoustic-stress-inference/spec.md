# Delta for Acoustic Stress Inference

## ADDED Requirements

### Requirement: Stress lexicon interface
The system MUST expose a versioned lexicon API that maps a normalized Ukrainian surface form to zero, one, or multiple valid stress variants.

Each variant MUST resolve to a stress-bearing vowel index.

#### Scenario: Unambiguous form
- GIVEN a surface form with one lexicon stress variant
- WHEN queried
- THEN the API returns exactly one candidate and its vowel index.

#### Scenario: Ambiguous form
- GIVEN a surface form such as `замок` with multiple stress variants
- WHEN queried
- THEN all configured valid variants are returned
- AND no variant is silently selected.

### Requirement: Bootstrap labels use only unambiguous words
Automatic bootstrap training labels MUST be derived only from lexicon entries with exactly one valid stress position.

#### Scenario: Exclude ambiguous token
- GIVEN an aligned occurrence whose lexicon lookup returns two stress variants
- WHEN bootstrap dataset construction runs
- THEN the occurrence is excluded from automatic gold labels
- AND may be retained for later mining/evaluation.

### Requirement: Minimum training-word eligibility
The bootstrap trainer MUST require at least two vowel candidates unless a specific experiment explicitly enables monosyllabic examples.

#### Scenario: Skip monosyllable by default
- GIVEN an aligned single-vowel word
- WHEN bootstrap examples are produced under default configuration
- THEN the word is skipped.

### Requirement: Per-vowel acoustic representation
The feature pipeline MUST create one feature vector per aligned vowel.

Each vector MUST support:
- pooled self-supervised speech-encoder representation
- vowel duration
- duration normalized relative to the word/utterance
- energy statistics
- F0 statistics when measurable
- a missing-value mask for unavailable prosodic features.

The implementation SHOULD support additional features such as spectral tilt and formant estimates behind configuration flags.

#### Scenario: F0 unavailable
- GIVEN a vowel segment for which reliable F0 cannot be estimated
- WHEN features are generated
- THEN the sample remains usable
- AND F0-derived values are marked missing rather than fabricated.

### Requirement: Context window around vowels
The SSL pooling implementation MUST support configurable temporal padding around the aligned vowel interval to capture syllabic context.

#### Scenario: Pool padded vowel interval
- GIVEN an aligned vowel and `vowel_context_ms=40`
- WHEN embeddings are pooled
- THEN frames up to 40 ms before and after the vowel are eligible
- AND the effective interval is clipped to the target-word/utterance boundaries.

### Requirement: Acoustic vowel stress ranker
The primary model MUST score the vowel sequence jointly and output one stress score per eligible vowel.

The reference implementation SHALL use:
- configurable pretrained Ukrainian/multilingual SSL speech encoder
- per-vowel pooling
- feature projection
- a small sequence encoder over vowel vectors
- one scalar logit per vowel
- masked softmax over eligible positions.

#### Scenario: Two-vowel inference
- GIVEN features for a word with two eligible vowels
- WHEN the ranker runs
- THEN it returns two finite logits
- AND normalized probabilities sum to 1 within numeric tolerance.

### Requirement: Candidate masking
When the lexicon supplies valid stress variants, inference MUST mask stress positions not represented by those variants.

#### Scenario: Mask impossible stress
- GIVEN a three-vowel word where the lexicon allows stress only on vowel 1 or vowel 2
- WHEN inference runs
- THEN vowel 3 receives zero posterior probability after masking
- AND selection occurs only between vowels 1 and 2.

### Requirement: Unrestricted acoustic evaluation mode
The model MUST support an evaluation mode without candidate masking to measure whether it has learned acoustic stress rather than only dictionary constraints.

#### Scenario: Lexeme-held-out evaluation
- GIVEN a word not seen during training
- WHEN unrestricted evaluation runs
- THEN all aligned vowels are eligible
- AND the predicted stress is computed from acoustic features without a dictionary-valid-position mask.

### Requirement: Frozen-encoder bootstrap training
The training system MUST support training the stress head with the SSL encoder frozen.

It MUST also support later selective unfreezing of a configurable number of final encoder layers.

#### Scenario: Freeze speech encoder
- GIVEN configuration `trainable_ssl_layers=0`
- WHEN training runs
- THEN no SSL encoder parameter receives an optimizer update.

### Requirement: Stress loss
The primary training objective MUST optimize categorical selection of one stressed vowel among the eligible vowels.

#### Scenario: Compute masked cross-entropy
- GIVEN a gold stress vowel and a valid-candidate mask
- WHEN loss is computed
- THEN invalid candidate positions do not contribute to the softmax denominator.

### Requirement: Optional candidate-conditioned scorer
The system SHOULD provide an independent model that scores `(audio crop, stress candidate)` compatibility.

Its representation and training MUST be independent enough from the vowel-ranker head to provide useful ensemble disagreement signals.

#### Scenario: Rank two stress candidates
- GIVEN one target-word audio crop and candidates `за́мок`, `замо́к`
- WHEN candidate scoring runs
- THEN the scorer returns a comparable score for each candidate
- AND the selected candidate is the maximum-scoring candidate.

### Requirement: Optional stress-aware CTC adapter
The system MUST support an optional adapter that converts an externally trained Ukrainian stress-aware ASR/CTC model into a normalized vote over candidate stresses.

#### Scenario: CTC output matches one candidate
- GIVEN stress-aware CTC output that maps unambiguously to one lexicon candidate
- WHEN the adapter runs
- THEN that candidate receives the CTC vote
- AND the external model version is recorded.

### Requirement: No lexical-identity shortcut in acoustic evaluation
Evaluation tooling MUST make it possible to detect lexical memorization.

At minimum, it MUST report an unseen-lexeme split and an ambiguous-same-spelling split.

#### Scenario: Same spelling, opposite stress
- GIVEN two manually verified occurrences of the same surface form with different stress variants
- WHEN evaluated
- THEN both readings are scored separately
- AND the system cannot receive full credit by always selecting the globally more frequent variant.
