"""Deterministic bounded quality-report generation."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_ISSUE_REPORTS = (
    "parse_errors",
    "rejected_forms",
    "unhandled_templates",
    "duplicates",
    "ambiguous_forms",
)


def bounded_text(value: str, limit: int = 512) -> str:
    if limit < 1:
        raise ValueError("limit must be positive")
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_quality_reports(
    output_dir: Path,
    *,
    issues: dict[str, list[dict[str, Any]]],
    accepted: Iterable[dict[str, Any]],
    fragment_limit: int = 512,
    aggregate_counts: Mapping[str, Counter[str]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for report_name in _ISSUE_REPORTS:
        bounded_rows: list[dict[str, Any]] = []
        for row in issues.get(report_name, []):
            bounded = dict(row)
            for field in ("source_fragment", "error_message", "sample_invocation"):
                if field in bounded:
                    bounded[field] = bounded_text(str(bounded[field]), fragment_limit)
            bounded_rows.append(bounded)
        bounded_rows.sort(key=lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True))
        _write_json(output_dir / f"{report_name}.json", bounded_rows)

    counters = {
        "part_of_speech": Counter[str](),
        "source_kind": Counter[str](),
        "confidence": Counter[str](),
        "grammatical_feature": Counter[str](),
    }
    if aggregate_counts is not None:
        for name in counters:
            counters[name].update(aggregate_counts.get(name, Counter()))
    else:
        for row in accepted:
            counters["part_of_speech"][str(row.get("part_of_speech", "unknown"))] += 1
            counters["source_kind"][str(row.get("source_kind", "unknown"))] += 1
            counters["confidence"][f"{float(row.get('confidence', 0.0)):.2f}"] += 1
            counters["grammatical_feature"].update(
                str(tag) for tag in row.get("grammatical_tags", [])
            )
    _write_json(
        output_dir / "counts.json",
        {
            name: dict(sorted(counter.items())) for name, counter in counters.items()
        },
    )
