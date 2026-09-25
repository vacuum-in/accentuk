"""Build the unambiguous, multi-vowel bootstrap set from aligned utterances."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass

from ukstress.alignment.fine_interface import FineAlignmentResult
from ukstress.alignment.fine_validation import validate_vowel_intervals
from ukstress.alignment.interface import WordAlignmentResult
from ukstress.config.models import AlignmentConfig
from ukstress.datasets import FeatureExample, NormalizedUtterance
from ukstress.datasets.ids import stable_record_id
from ukstress.datasets.parquet import ShardWriteResult, write_parquet_shard
from ukstress.lexicon.interface import StressLexicon
from ukstress.lexicon.stress import vowel_count


@dataclass(frozen=True)
class BootstrapStats:
    """Counters emitted alongside bootstrap examples."""

    total_words: int = 0
    accepted: int = 0
    skipped: int = 0
    oov: int = 0
    ambiguous: int = 0
    one_vowel: int = 0
    alignment_rejected: int = 0
    capped: int = 0


def build_bootstrap_examples(
    utterance: NormalizedUtterance,
    word_alignment: WordAlignmentResult,
    fine_alignments: Mapping[int, FineAlignmentResult],
    lexicon: StressLexicon,
    *,
    config_fingerprint: str,
    alignment_config: AlignmentConfig | None = None,
    exclude_one_vowel: bool = True,
    occurrence_counts: MutableMapping[str, int] | None = None,
    max_occurrences_per_lexeme: int | None = None,
) -> tuple[list[FeatureExample], BootstrapStats]:
    """Join word/fine alignments with uniquely stressed lexicon entries.

    ``fine_alignments`` is keyed by the zero-based index in ``word_alignment.words``.  Missing
    entries are treated as alignment rejection rather than silently producing incomplete labels.
    """

    if max_occurrences_per_lexeme is not None and max_occurrences_per_lexeme < 1:
        raise ValueError("max_occurrences_per_lexeme must be positive")
    min_quality = alignment_config.min_quality if alignment_config is not None else 0.8
    overlap_tolerance = (
        alignment_config.overlap_tolerance_s if alignment_config is not None else 0.005
    )
    examples: list[FeatureExample] = []
    counts = {
        "total_words": len(word_alignment.words),
        "accepted": 0,
        "oov": 0,
        "ambiguous": 0,
        "one_vowel": 0,
        "alignment_rejected": 0,
        "capped": 0,
    }
    for index, word in enumerate(word_alignment.words):
        candidates = lexicon.lookup(word.normalized_token)
        if not candidates:
            counts["oov"] += 1
            continue
        if len(candidates) != 1:
            counts["ambiguous"] += 1
            continue
        candidate = candidates[0]
        if exclude_one_vowel and vowel_count(word.normalized_token) < 2:
            counts["one_vowel"] += 1
            continue
        if (
            max_occurrences_per_lexeme is not None
            and occurrence_counts is not None
            and occurrence_counts.get(word.normalized_token, 0) >= max_occurrences_per_lexeme
        ):
            counts["capped"] += 1
            continue
        fine = fine_alignments.get(index)
        if fine is None or fine.rejection_reasons:
            counts["alignment_rejected"] += 1
            continue
        validation = validate_vowel_intervals(
            fine.vowels,
            word.normalized_token,
            word_start_s=word.start_s,
            word_end_s=word.end_s,
            utterance_duration_s=utterance.duration_s,
            overlap_tolerance_s=overlap_tolerance,
        )
        if not validation.valid or fine.quality < min_quality:
            counts["alignment_rejected"] += 1
            continue
        record_id = stable_record_id(
            source_id=utterance.source_id,
            utterance_id=utterance.utterance_id,
            normalized_target=word.normalized_token,
            word_start_s=word.start_s,
            word_end_s=word.end_s,
        )
        examples.append(
            FeatureExample(
                record_id=record_id,
                source_id=utterance.source_id,
                utterance_id=utterance.utterance_id,
                speaker_id=utterance.speaker_id,
                audio_uri=utterance.audio_uri,
                target_word=word.normalized_token,
                word_start_s=word.start_s,
                word_end_s=word.end_s,
                vowels=fine.vowels,
                gold_vowel_index=candidate.vowel_index,
                lexicon_fingerprint=lexicon.fingerprint,
                config_fingerprint=config_fingerprint,
                license_id=utterance.license_id,
                provenance={
                    **utterance.provenance,
                    "word_alignment_backend": word_alignment.backend,
                    "word_alignment_model": word_alignment.model_id or "",
                    "fine_alignment_backend": fine.backend,
                    "fine_alignment_model": fine.model_id or "",
                },
            )
        )
        counts["accepted"] += 1
        if occurrence_counts is not None:
            occurrence_counts[word.normalized_token] = (
                occurrence_counts.get(word.normalized_token, 0) + 1
            )

    stats = BootstrapStats(
        **counts,
        skipped=counts["total_words"] - counts["accepted"],
    )
    return examples, stats


def write_bootstrap_shard(
    examples: list[FeatureExample],
    output_dir: str,
    *,
    shard_id: str,
    input_fingerprint: str,
    force: bool = False,
) -> ShardWriteResult:
    """Persist bootstrap examples through the canonical atomic shard writer."""

    return write_parquet_shard(
        examples,
        output_dir,
        shard_id=shard_id,
        input_fingerprint=input_fingerprint,
        force=force,
    )
