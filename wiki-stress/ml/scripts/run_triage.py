"""Triage homograph groups: prior-resolvable vs context-dependent.

The served model does not have to be consulted for every ambiguous span. Where
one sense accounts for essentially all natural usage, the dictionary's default
is already right and calling a model can only add error and latency. This script
estimates the natural sense distribution per group from mined, blind-verified
labels and classifies each group.

Classification uses the **lower bound** of a Wilson interval, not the point
estimate: routing a group to "never call the model" on the strength of 24
observations would be a guess dressed as a decision.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ukstress_ml import ambiguity, annotate

MIN_OBSERVATIONS = 8
PRIOR_ONLY = 0.95
PRIOR_DOMINANT = 0.80
OUT = Path("output/ml/triage.json")


def wilson_lower(successes: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 0.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (centre - margin) / denominator


def required_observations(target: float, observed: float = 1.0, limit: int = 5000) -> int | None:
    for total in range(1, limit):
        if wilson_lower(round(observed * total), total) >= target:
            return total
    return None


def natural_distribution() -> dict[int, Counter[str]]:
    """Sense counts from mined sentences whose label survived blind verification."""
    mined: dict[str, dict[str, Any]] = {}
    with Path("output/ml/mined/ukwiki/candidates.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            mined[row["sentence_id"]] = row

    labels = annotate.parse_raw(Path("output/ml/raw/label_ukwiki.jsonl"))
    verifications = annotate.parse_raw(Path("output/ml/raw/verify_ukwiki.jsonl"))

    distribution: dict[int, Counter[str]] = defaultdict(Counter)
    for sentence_id, label in labels.items():
        if label["unclear"] or sentence_id not in mined:
            continue
        verification = verifications.get(sentence_id)
        if verification is None or verification["unclear"]:
            continue
        if verification["sense_id"] != label["sense_id"]:
            continue
        distribution[mined[sentence_id]["group_id"]][label["sense_id"]] += 1
    return distribution


def classify(lower: float) -> str:
    if lower >= PRIOR_ONLY:
        return "prior-only"
    if lower >= PRIOR_DOMINANT:
        return "prior-dominant"
    return "context-needed"


def main() -> None:
    forms = {f.group_id: f for f in ambiguity.load(Path("output/ml/ambiguous_forms.jsonl"))}
    occurrences = json.loads(
        Path("output/ml/mined/ukwiki/coverage.json").read_text(encoding="utf-8")
    )["occurrences_seen_per_form"]
    distribution = natural_distribution()

    groups: list[dict[str, Any]] = []
    for group_id, counts in distribution.items():
        total = sum(counts.values())
        if total < MIN_OBSERVATIONS:
            continue
        form = forms.get(group_id)
        if form is None:
            continue
        dominant_sense, dominant = counts.most_common(1)[0]
        lower = wilson_lower(dominant, total)
        groups.append(
            {
                "group_id": group_id,
                "form": form.form,
                "pos": sorted({c.pos for c in form.candidates}),
                "observations": total,
                "senses_observed": len(counts),
                "dominant_sense": dominant_sense,
                "dominant_share": round(dominant / total, 4),
                "dominant_share_ci_low": round(lower, 4),
                "corpus_occurrences": occurrences.get(form.form, 0),
                "routing": classify(lower),
                "routing_point_estimate": classify(dominant / total),
            }
        )

    by_routing = Counter(g["routing"] for g in groups)
    by_point = Counter(g["routing_point_estimate"] for g in groups)
    weighted: Counter[str] = Counter()
    for group in groups:
        weighted[group["routing"]] += group["corpus_occurrences"]

    by_pos: dict[str, Counter[str]] = defaultdict(Counter)
    for group in groups:
        by_pos["+".join(group["pos"])][group["routing"]] += 1

    report = {
        "method": {
            "source": "mined ukwiki sentences, label confirmed by blind verification",
            "min_observations": MIN_OBSERVATIONS,
            "thresholds": {"prior_only": PRIOR_ONLY, "prior_dominant": PRIOR_DOMINANT},
            "statistic": "Wilson 95% lower bound on the dominant sense's share",
        },
        "groups_analysed": len(groups),
        "by_routing_ci_lower_bound": dict(by_routing),
        "by_routing_point_estimate": dict(by_point),
        "corpus_occurrences_by_routing": dict(weighted),
        "by_part_of_speech": {pos: dict(c) for pos, c in by_pos.items()},
        "observations_needed_to_certify": {
            "prior_only_perfect_record": required_observations(PRIOR_ONLY, 1.0),
            "prior_only_98pct_record": required_observations(PRIOR_ONLY, 0.98),
            "prior_dominant_perfect_record": required_observations(PRIOR_DOMINANT, 1.0),
        },
        "groups": sorted(groups, key=lambda g: -g["corpus_occurrences"]),
    }
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"groups analysed: {len(groups)}")
    print(f"by CI lower bound: {dict(by_routing)}")
    print(f"by point estimate: {dict(by_point)}")
    print(f"corpus occurrences by routing: {dict(weighted)}")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
