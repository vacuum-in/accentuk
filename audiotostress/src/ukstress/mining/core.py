"""Precision-first ambiguous-stress mining primitives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ukstress.alignment.fine_interface import FineAlignmentBackend, FineAlignmentResult
from ukstress.alignment.interface import WordAlignmentResult
from ukstress.datasets import MiningRecord, NormalizedUtterance, StressCandidate, WordAlignment
from ukstress.datasets.ids import stable_record_id
from ukstress.datasets.parquet import ShardWriteResult, write_parquet_shard
from ukstress.lexicon.interface import StressLexicon


@dataclass(frozen=True)
class AmbiguousCandidate:
    word_index: int
    alignment: WordAlignment
    candidates: tuple[StressCandidate, ...]

    @property
    def target_word(self) -> str:
        return self.alignment.normalized_token


def discover_ambiguous_candidates(
    alignment: WordAlignmentResult, lexicon: StressLexicon
) -> list[AmbiguousCandidate]:
    """Find only aligned tokens with multiple lexicon readings."""

    discovered: list[AmbiguousCandidate] = []
    for index, word in enumerate(alignment.words):
        candidates = lexicon.lookup(word.normalized_token)
        if len(candidates) > 1:
            discovered.append(AmbiguousCandidate(index, word, tuple(candidates)))
    return discovered


def align_candidate_targets(
    candidates: Sequence[AmbiguousCandidate],
    backend: FineAlignmentBackend,
    waveform: np.ndarray,
    sample_rate: int,
) -> dict[int, FineAlignmentResult]:
    """Run fine alignment only for discovered target words."""

    if waveform.ndim != 1 or sample_rate <= 0:
        raise ValueError("fine alignment requires a one-dimensional waveform and positive rate")
    return {
        candidate.word_index: backend.align(
            waveform, sample_rate, candidate.alignment, candidate.target_word
        )
        for candidate in candidates
    }


@dataclass(frozen=True)
class AcceptanceEvidence:
    ranker_confidence: float
    candidate_margin: float
    alignment_quality: float
    identity_quality: float
    candidate_scorer_agrees: bool | None = None
    stress_ctc_agrees: bool | None = None

    def __post_init__(self) -> None:
        for name in (
            "ranker_confidence",
            "candidate_margin",
            "alignment_quality",
            "identity_quality",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class AcceptanceDecision:
    accepted: bool
    confidence: float
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class AcceptancePolicy:
    min_confidence: float = 0.0
    min_margin: float = 0.0
    min_alignment_quality: float = 0.0
    min_identity_quality: float = 1.0
    require_candidate_scorer_agreement: bool = False
    ctc_disagreement: str = "reject"
    ctc_penalty: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "min_confidence",
            "min_margin",
            "min_alignment_quality",
            "min_identity_quality",
            "ctc_penalty",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.ctc_disagreement not in {"reject", "penalize", "ignore"}:
            raise ValueError("ctc_disagreement must be reject, penalize, or ignore")

    def evaluate(self, evidence: AcceptanceEvidence) -> AcceptanceDecision:
        reasons: list[str] = []
        confidence = evidence.ranker_confidence
        if confidence < self.min_confidence:
            reasons.append("low_calibrated_confidence")
        if evidence.candidate_margin < self.min_margin:
            reasons.append("low_candidate_margin")
        if evidence.alignment_quality < self.min_alignment_quality:
            reasons.append("low_alignment_quality")
        if evidence.identity_quality < self.min_identity_quality:
            reasons.append("low_identity_quality")
        if self.require_candidate_scorer_agreement and evidence.candidate_scorer_agrees is not True:
            reasons.append("candidate_scorer_disagreement")
        if evidence.stress_ctc_agrees is False:
            if self.ctc_disagreement == "reject":
                reasons.append("stress_ctc_disagreement")
            elif self.ctc_disagreement == "penalize":
                confidence = max(0.0, confidence - self.ctc_penalty)
                if confidence < self.min_confidence:
                    reasons.append("low_calibrated_confidence")
        unique = tuple(dict.fromkeys(reasons))
        return AcceptanceDecision(not unique, confidence, unique)


def score_ranker_candidates(
    ranker: Any,
    batch: Any,
    candidate_mask: np.ndarray,
    *,
    device: str = "cpu",
) -> tuple[np.ndarray, int, float, float]:
    """Return candidate probabilities, selected index, confidence, and top-two margin."""

    probabilities = np.asarray(
        ranker.probabilities(batch, candidate_mask=candidate_mask, device=device)
    )[0]
    if probabilities.ndim != 1 or not len(probabilities):
        raise ValueError("ranker must return one probability vector")
    order = np.argsort(probabilities)[::-1]
    confidence = float(probabilities[order[0]])
    runner_up = float(probabilities[order[1]]) if len(order) > 1 else 0.0
    return probabilities, int(order[0]), confidence, confidence - runner_up


def build_mining_record(
    utterance: NormalizedUtterance,
    candidate: AmbiguousCandidate,
    fine_alignment: FineAlignmentResult,
    *,
    lexicon: StressLexicon,
    config_fingerprint: str,
    ranker_probs: Sequence[float],
    policy: AcceptancePolicy,
    identity_quality: float,
    candidate_scorer_probs: Sequence[float] | None = None,
    stress_ctc_candidate: int | None = None,
    alignment_quality: float | None = None,
    model_versions: Mapping[str, str] | None = None,
    calibration_version: str | None = None,
) -> MiningRecord:
    """Create an evidence-preserving canonical record and apply the acceptance policy."""

    probabilities = np.asarray(ranker_probs, dtype=np.float64)
    if probabilities.shape != (len(candidate.candidates),):
        raise ValueError("ranker_probs must contain one value per lexicon candidate")
    order = np.argsort(probabilities)[::-1]
    predicted = int(order[0])
    confidence = float(probabilities[predicted])
    margin = confidence - (float(probabilities[order[1]]) if len(order) > 1 else 0.0)
    scorer_agrees = None
    if candidate_scorer_probs is not None:
        scorer = np.asarray(candidate_scorer_probs, dtype=np.float64)
        if scorer.shape != probabilities.shape:
            raise ValueError("candidate_scorer_probs must match ranker_probs")
        scorer_agrees = int(np.argmax(scorer)) == predicted
    ctc_agrees = None if stress_ctc_candidate is None else stress_ctc_candidate == predicted
    evidence = AcceptanceEvidence(
        ranker_confidence=confidence,
        candidate_margin=margin,
        alignment_quality=fine_alignment.quality
        if alignment_quality is None
        else alignment_quality,
        identity_quality=identity_quality,
        candidate_scorer_agrees=scorer_agrees,
        stress_ctc_agrees=ctc_agrees,
    )
    decision = policy.evaluate(evidence)
    versions = dict(model_versions or {})
    if calibration_version is not None:
        versions["calibration"] = calibration_version
    record_id = stable_record_id(
        source_id=utterance.source_id,
        utterance_id=utterance.utterance_id,
        normalized_target=candidate.target_word,
        word_start_s=candidate.alignment.start_s,
        word_end_s=candidate.alignment.end_s,
    )
    return MiningRecord(
        record_id=record_id,
        source_id=utterance.source_id,
        utterance_id=utterance.utterance_id,
        speaker_id=utterance.speaker_id,
        audio_uri=utterance.audio_uri,
        sentence_original=utterance.transcript_original,
        sentence_normalized=utterance.transcript_normalized,
        target_word=candidate.target_word,
        target_char_start=candidate.alignment.char_start,
        target_char_end=candidate.alignment.char_end,
        word_start_s=candidate.alignment.start_s,
        word_end_s=candidate.alignment.end_s,
        vowels=fine_alignment.vowels,
        candidates=list(candidate.candidates),
        ranker_probs=probabilities.tolist(),
        candidate_scorer_probs=(
            list(map(float, candidate_scorer_probs)) if candidate_scorer_probs is not None else None
        ),
        stress_ctc_candidate=stress_ctc_candidate,
        predicted_candidate=predicted,
        calibrated_confidence=decision.confidence,
        candidate_margin=margin,
        alignment_score=evidence.alignment_quality,
        identity_score=identity_quality,
        accepted=decision.accepted,
        rejection_reasons=list(decision.rejection_reasons),
        lexicon_fingerprint=lexicon.fingerprint,
        model_versions=versions,
        config_fingerprint=config_fingerprint,
        license_id=utterance.license_id,
        commercial_use=utterance.commercial_use,
        redistribution=utterance.redistribution,
        provenance=utterance.provenance,
    )


@dataclass(frozen=True)
class MiningOutput:
    candidates: ShardWriteResult
    accepted: ShardWriteResult | None
    rejected: ShardWriteResult | None


def deduplicate_records(records: Sequence[MiningRecord]) -> list[MiningRecord]:
    """Keep one deterministic record per stable ID, rejecting conflicting duplicates."""

    seen: dict[str, MiningRecord] = {}
    for record in records:
        previous = seen.get(record.record_id)
        if previous is not None and previous.model_dump(mode="json") != record.model_dump(
            mode="json"
        ):
            raise ValueError(f"conflicting duplicate mining record {record.record_id}")
        seen[record.record_id] = record
    return [seen[key] for key in sorted(seen)]


def write_mining_outputs(
    records: Sequence[MiningRecord],
    output_dir: str | Path,
    *,
    shard_id: str,
    input_fingerprint: str,
) -> MiningOutput:
    """Write all candidates plus accepted/rejected partitions atomically and resumably."""

    unique = deduplicate_records(records)
    if not unique:
        raise ValueError("at least one mining candidate is required")
    root = Path(output_dir)
    candidates_result = write_parquet_shard(
        unique, root / "candidates", shard_id=shard_id, input_fingerprint=input_fingerprint
    )
    accepted = [record for record in unique if record.accepted]
    rejected = [record for record in unique if not record.accepted]
    accepted_result = (
        write_parquet_shard(
            accepted, root / "accepted", shard_id=shard_id, input_fingerprint=input_fingerprint
        )
        if accepted
        else None
    )
    rejected_result = (
        write_parquet_shard(
            rejected, root / "rejected", shard_id=shard_id, input_fingerprint=input_fingerprint
        )
        if rejected
        else None
    )
    return MiningOutput(candidates_result, accepted_result, rejected_result)


def mining_statistics(records: Sequence[MiningRecord]) -> dict[str, object]:
    """Summarize accepted/rejected outcomes for a shard or complete run."""

    reasons: dict[str, int] = {}
    by_source: dict[str, dict[str, int]] = {}
    for record in records:
        source = by_source.setdefault(
            record.source_id, {"candidates": 0, "accepted": 0, "rejected": 0}
        )
        source["candidates"] += 1
        source["accepted" if record.accepted else "rejected"] += 1
        for reason in record.rejection_reasons:
            reasons[reason] = reasons.get(reason, 0) + 1
    accepted = sum(record.accepted for record in records)
    return {
        "candidates": len(records),
        "accepted": accepted,
        "rejected": len(records) - accepted,
        "acceptance_rate": accepted / len(records) if records else 0.0,
        "rejection_reasons": dict(sorted(reasons.items())),
        "by_source": dict(sorted(by_source.items())),
    }
