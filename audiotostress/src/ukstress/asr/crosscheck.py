"""Transcript/reference identity checks kept separate from stress confidence."""

from __future__ import annotations

from ukstress.text import normalize_transcript


def _edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def transcript_identity_score(reference: str, hypothesis: str) -> float:
    """Return one minus token error rate, clipped into [0, 1]."""

    reference_tokens = normalize_transcript(reference).split()
    hypothesis_tokens = normalize_transcript(hypothesis).split()
    if not reference_tokens:
        return 1.0 if not hypothesis_tokens else 0.0
    distance = _edit_distance(reference_tokens, hypothesis_tokens)
    return max(0.0, 1.0 - distance / len(reference_tokens))


def target_identity_matches(reference_token: str, aligned_token: str) -> bool:
    return normalize_transcript(reference_token) == normalize_transcript(aligned_token)
