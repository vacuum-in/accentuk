"""Position, form, F1, split, confusion, and quality reporting."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast


@dataclass(frozen=True)
class EvaluationRow:
    record_id: str
    target_word: str
    gold_vowel_index: int
    predicted_vowel_index: int
    speaker_id: str | None = None
    accepted: bool = True
    rejection_reasons: tuple[str, ...] = ()
    alignment_score: float | None = None
    identity_score: float | None = None
    masked_predicted_vowel_index: int | None = None
    unmasked_predicted_vowel_index: int | None = None


def evaluate_rows(rows: list[EvaluationRow]) -> dict[str, object]:
    if not rows:
        raise ValueError("evaluation requires at least one row")
    accuracy = sum(row.gold_vowel_index == row.predicted_vowel_index for row in rows) / len(rows)
    by_form: dict[str, list[EvaluationRow]] = defaultdict(list)
    for row in rows:
        by_form[row.target_word].append(row)
    form_accuracy = {
        form: (
            sum(item.gold_vowel_index == item.predicted_vowel_index for item in values)
            / len(values)
        )
        for form, values in sorted(by_form.items())
    }
    labels = sorted(
        {row.gold_vowel_index for row in rows} | {row.predicted_vowel_index for row in rows}
    )
    f1 = {str(label): _f1(rows, label) for label in labels}
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        confusion[str(row.gold_vowel_index)][str(row.predicted_vowel_index)] += 1
    ambiguous = {
        form: values
        for form, values in by_form.items()
        if len({item.gold_vowel_index for item in values}) > 1
    }
    quality = _quality(rows)
    return {
        "records": len(rows),
        "stress_position_accuracy": accuracy,
        "macro_form_accuracy": sum(form_accuracy.values()) / len(form_accuracy),
        "per_form_accuracy": form_accuracy,
        "macro_f1": sum(f1.values()) / len(f1),
        "per_variant_f1": f1,
        "speaker_report": _group_accuracy(rows, lambda item: item.speaker_id or "unknown"),
        "lexeme_report": {
            "masked": _group_accuracy(
                rows,
                lambda item: item.target_word,
                lambda item: (
                    item.masked_predicted_vowel_index
                    if item.masked_predicted_vowel_index is not None
                    else item.predicted_vowel_index
                ),
            ),
            "unmasked": _group_accuracy(
                rows,
                lambda item: item.target_word,
                lambda item: (
                    item.unmasked_predicted_vowel_index
                    if item.unmasked_predicted_vowel_index is not None
                    else item.predicted_vowel_index
                ),
            ),
        },
        "same_spelling_report": (
            _group_accuracy(
                [item for values in ambiguous.values() for item in values],
                lambda item: item.target_word,
            )
            if ambiguous
            else {}
        ),
        "confusion": {gold: dict(predicted) for gold, predicted in sorted(confusion.items())},
        "quality": quality,
    }


def write_evaluation_report(directory: str | Path, report: dict[str, object]) -> tuple[Path, Path]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "metrics.json"
    markdown_path = root / "report.md"
    _atomic_text(json_path, json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    markdown = (
        "\n".join(
            [
                "# Evaluation report",
                "",
                f"- Records: {report['records']}",
                "- Stress-position accuracy: "
                f"{float(cast(float, report['stress_position_accuracy'])):.4f}",
                f"- Macro form accuracy: {float(cast(float, report['macro_form_accuracy'])):.4f}",
                f"- Macro F1: {float(cast(float, report['macro_f1'])):.4f}",
            ]
        )
        + "\n"
    )
    _atomic_text(markdown_path, markdown)
    return json_path, markdown_path


def _f1(rows: list[EvaluationRow], label: int) -> float:
    true_positive = sum(
        row.gold_vowel_index == label and row.predicted_vowel_index == label for row in rows
    )
    false_positive = sum(
        row.gold_vowel_index != label and row.predicted_vowel_index == label for row in rows
    )
    false_negative = sum(
        row.gold_vowel_index == label and row.predicted_vowel_index != label for row in rows
    )
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if not denominator else 2 * true_positive / denominator


def _group_accuracy(
    rows: list[EvaluationRow],
    key: Callable[[EvaluationRow], str],
    prediction: Callable[[EvaluationRow], int] | None = None,
) -> dict[str, float]:
    prediction = prediction or (lambda item: item.predicted_vowel_index)
    groups: dict[str, list[EvaluationRow]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return {
        name: sum(item.gold_vowel_index == prediction(item) for item in values) / len(values)
        for name, values in sorted(groups.items())
    }


def _quality(rows: list[EvaluationRow]) -> dict[str, object]:
    rejected = [row for row in rows if not row.accepted]
    reasons = Counter(reason for row in rejected for reason in row.rejection_reasons)
    alignment = [row.alignment_score for row in rows if row.alignment_score is not None]
    identity = [row.identity_score for row in rows if row.identity_score is not None]
    return {
        "accepted": len(rows) - len(rejected),
        "rejected": len(rejected),
        "rejection_reasons": dict(sorted(reasons.items())),
        "mean_alignment_score": sum(alignment) / len(alignment) if alignment else None,
        "mean_identity_score": sum(identity) / len(identity) if identity else None,
    }


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
