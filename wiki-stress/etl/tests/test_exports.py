import json
from pathlib import Path

from ukstress.exports import write_export

ROWS = [
    {
        "form_normalized": "замок",
        "stressed_form": "за́мок",
        "grammatical_tags": ["nominative"],
    },
    {
        "form_normalized": "мова",
        "stressed_form": "мо́ва",
        "grammatical_tags": [],
    },
]


def test_writes_jsonl_tsv_and_compact_tts(tmp_path: Path) -> None:
    jsonl = tmp_path / "lexicon.jsonl"
    tsv = tmp_path / "lexicon.tsv"
    tts = tmp_path / "lexicon.tts"
    assert write_export(jsonl, ROWS, "jsonl") == 2
    assert write_export(tsv, ROWS, "tsv") == 2
    assert write_export(tts, ROWS, "tts") == 2
    assert json.loads(jsonl.read_text(encoding="utf-8").splitlines()[0])[
        "stressed_form"
    ] == "за́мок"
    assert tsv.read_text(encoding="utf-8").startswith("form_normalized\t")
    assert tts.read_text(encoding="utf-8") == "замок\tза́мок\nмова\tмо́ва\n"


def test_unknown_export_format_does_not_publish_output(tmp_path: Path) -> None:
    output = tmp_path / "bad"
    try:
        write_export(output, ROWS, "xml")
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported format was accepted")
    assert not output.exists()
