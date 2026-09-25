import pytest

from ukstress_ml import annotate


def test_verification_refuses_the_labelling_deployment(tmp_path) -> None:
    # A model must never confirm its own labels.
    with pytest.raises(RuntimeError, match="same model that produced the labels"):
        annotate.run_job(
            job="verify",
            rows=[],
            forms={},
            deployment=annotate.LABEL_DEPLOYMENT,
            config={},
            raw_path=tmp_path / "verify.jsonl",
        )


def test_verify_deployment_differs_from_label_deployment() -> None:
    assert annotate.VERIFY_DEPLOYMENT.name != annotate.LABEL_DEPLOYMENT.name


def test_completed_keys_ignore_failed_batches(tmp_path) -> None:
    # A transient failure must be retried on resume, not treated as done.
    path = tmp_path / "raw.jsonl"
    path.write_text(
        '{"batch_key": "label:a:0", "content": "{}"}\n'
        '{"batch_key": "label:a:1", "content": null}\n',
        encoding="utf-8",
    )
    assert annotate._completed_keys(path) == {"label:a:0"}


def test_mask_sentence_hides_only_the_target_span() -> None:
    row = {"sentence": "Він відчинив замок ключем.", "start": 13, "end": 18}
    masked = annotate.mask_sentence(row)
    assert "замок" not in masked
    assert masked.startswith("Він відчинив ")
    assert masked.endswith(" ключем.")


@pytest.mark.parametrize(
    "content",
    [
        '{"items": [{"i": 0, "sense_id": "1.a"}]}',
        '```json\n{"items": [{"i": 0, "sense_id": "1.a"}]}\n```',
        'Ось відповідь: {"items": [{"i": 0, "sense_id": "1.a"}]}',
    ],
)
def test_json_extraction_tolerates_wrapping(content: str) -> None:
    parsed = annotate._extract_json(content)
    assert parsed is not None
    assert parsed["items"][0]["sense_id"] == "1.a"
