import json
from pathlib import Path

import pytest

from ukstress.config.models import LexiconConfig
from ukstress.lexicon import FileStressLexicon


def _write_lexicon(path: Path) -> None:
    entries = [
        {"stressed_form": "молоко́", "source": "dictionary-a"},
        {"stressed_form": "за́мок", "source": "dictionary-a"},
        {"surface": "замок", "vowel_index": 1, "source": "dictionary-a"},
        {"stressed_form": "ОБ’Є́КТ", "source": "dictionary-a"},
        {"stressed_form": "БУДЬ-Я́КИЙ", "source": "dictionary-a"},
        {"stressed_form": "і\N{COMBINING DIAERESIS}\N{COMBINING ACUTE ACCENT}жа"},
        {"stressed_form": "за́мок", "source": "dictionary-b"},
    ]
    path.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
        encoding="utf-8",
    )


def test_lexicon_returns_unambiguous_ambiguous_and_oov_results(tmp_path: Path) -> None:
    path = tmp_path / "stress.jsonl"
    _write_lexicon(path)
    lexicon = FileStressLexicon.load(path, version="2026.09")

    milk = lexicon.lookup("МОЛОКО")
    castle = lexicon.lookup("замок")

    assert len(milk) == 1
    assert milk[0].vowel_index == 2
    assert [candidate.stressed_form for candidate in castle] == ["за́мок", "замо́к"]
    assert castle[0].source == "dictionary-a;dictionary-b"
    assert lexicon.lookup("відсутнє") == ()


def test_lexicon_lookup_normalizes_apostrophe_hyphen_and_unicode(tmp_path: Path) -> None:
    path = tmp_path / "stress.jsonl"
    _write_lexicon(path)
    lexicon = FileStressLexicon.from_config(
        LexiconConfig(path=path, version="2026.09", source_name="fixture")
    )

    assert lexicon.lookup("об'єкт")[0].vowel_index == 1
    assert lexicon.lookup("будь—який")[0].vowel_index == 1
    assert lexicon.lookup("їжа")[0].stressed_form == "ї́жа"


def test_lexicon_fingerprint_and_provenance_bind_version_and_bytes(tmp_path: Path) -> None:
    path = tmp_path / "stress.jsonl"
    _write_lexicon(path)
    first = FileStressLexicon.load(path, version="1")
    same = FileStressLexicon.load(path, version="1")
    new_version = FileStressLexicon.load(path, version="2")

    assert first.fingerprint == same.fingerprint
    assert first.fingerprint != new_version.fingerprint
    assert first.provenance["version"] == "1"
    assert first.provenance["source_fingerprint"].startswith("sha256:")


def test_embedded_version_must_match_configuration(tmp_path: Path) -> None:
    path = tmp_path / "stress.yaml"
    path.write_text("version: '1'\nentries:\n  - stressed_form: за́мок\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        FileStressLexicon.load(path, version="2")


def test_invalid_entry_reports_row_number(tmp_path: Path) -> None:
    path = tmp_path / "stress.tsv"
    path.write_text("surface\tvowel_index\nзамок\t9\n", encoding="utf-8")

    with pytest.raises(ValueError, match="entry 1"):
        FileStressLexicon.load(path, version="1")
