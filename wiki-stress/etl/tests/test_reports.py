import json
from pathlib import Path

from ukstress.reports import bounded_text, write_quality_reports


def test_quality_reports_are_bounded_deterministic_and_aggregated(tmp_path: Path) -> None:
    issues = {
        "parse_errors": [
            {"page": "мова", "error_message": "x" * 20, "source_fragment": "y" * 20}
        ],
        "unhandled_templates": [
            {"name": "unknown", "sample_invocation": "{{" + "z" * 30}
        ],
    }
    accepted = [
        {
            "part_of_speech": "noun",
            "source_kind": "template",
            "confidence": 1.0,
            "grammatical_tags": ["nominative", "singular"],
        },
        {
            "part_of_speech": "noun",
            "source_kind": "table",
            "confidence": 0.95,
            "grammatical_tags": ["genitive", "singular"],
        },
    ]

    write_quality_reports(tmp_path, issues=issues, accepted=accepted, fragment_limit=10)

    errors = json.loads((tmp_path / "parse_errors.json").read_text())
    counts = json.loads((tmp_path / "counts.json").read_text())
    assert errors[0]["error_message"] == "xxxxxxxxx…"
    assert errors[0]["source_fragment"] == "yyyyyyyyy…"
    assert counts["part_of_speech"] == {"noun": 2}
    assert counts["grammatical_feature"]["singular"] == 2
    assert (tmp_path / "rejected_forms.json").read_text() == "[]\n"
    assert bounded_text("abc", 3) == "abc"

