from ukstress.evaluation import EvaluationRow, evaluate_rows, write_evaluation_report


def test_evaluation_reports_accuracy_f1_confusion_quality_and_files(tmp_path) -> None:
    rows = [
        EvaluationRow("a", "замок", 0, 0, "s1", alignment_score=0.9, identity_score=1.0),
        EvaluationRow(
            "b", "замок", 1, 0, "s2", accepted=False, rejection_reasons=("low_alignment_quality",)
        ),
        EvaluationRow("c", "молоко", 2, 2, "s2"),
    ]
    report = evaluate_rows(rows)
    json_path, markdown_path = write_evaluation_report(tmp_path, report)

    assert report["stress_position_accuracy"] == 2 / 3
    assert report["same_spelling_report"]["замок"] == 0.5
    assert report["quality"]["rejected"] == 1
    assert json_path.is_file() and markdown_path.is_file()
