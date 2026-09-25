"""Candidate discovery, scoring evidence, acceptance, and shard output."""

from ukstress.mining.core import (
    AcceptanceDecision,
    AcceptanceEvidence,
    AcceptancePolicy,
    AmbiguousCandidate,
    MiningOutput,
    align_candidate_targets,
    build_mining_record,
    deduplicate_records,
    discover_ambiguous_candidates,
    mining_statistics,
    score_ranker_candidates,
    write_mining_outputs,
)

__all__ = [
    "AcceptanceDecision",
    "AcceptanceEvidence",
    "AcceptancePolicy",
    "AmbiguousCandidate",
    "MiningOutput",
    "align_candidate_targets",
    "build_mining_record",
    "deduplicate_records",
    "discover_ambiguous_candidates",
    "mining_statistics",
    "score_ranker_candidates",
    "write_mining_outputs",
]
