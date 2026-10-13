import json
from pathlib import Path
import pytest
import marisa_trie

from accentuk.errors import AssetError
from accentuk.lexicon import Lexicon


def test_lexicon_lookup_and_context_manager(tmp_path: Path):
    trie_path = tmp_path / "lexicon.marisa"
    payload = {
        "candidates": ["0", "1"],
        "proper": {"0": False, "1": True},
        "readings": [[["Case=Nom"], [2]]],
    }
    entries = [("мова", json.dumps(payload).encode("utf-8"))]
    trie = marisa_trie.BytesTrie(entries)
    trie.save(str(trie_path))

    with Lexicon(trie_path) as lex:
        assert len(lex) == 1
        res = lex.lookup("Мо́ва")  # Normalized to "мова"
        assert res is not None
        assert res["candidates"] == ["0", "1"]
        assert res["proper"] == {"0": False, "1": True}
        assert res["readings"] == [[["Case=Nom"], [2]]]

        assert lex.lookup("невідоме") is None

    # After close, lookup raises RuntimeError
    with pytest.raises(RuntimeError):
        lex.lookup("мова")


def test_lexicon_malformed_payload(tmp_path: Path):
    trie_path = tmp_path / "lexicon_malformed.marisa"
    entries = [("слово", b"not-a-json")]
    trie = marisa_trie.BytesTrie(entries)
    trie.save(str(trie_path))

    with Lexicon(trie_path) as lex:
        with pytest.raises(AssetError) as exc_info:
            lex.lookup("слово")
        assert "Malformed JSON" in str(exc_info.value)


def test_lexicon_invalid_schema(tmp_path: Path):
    trie_path = tmp_path / "lexicon_schema.marisa"
    entries = [
        ("слово1", json.dumps({"candidates": "not-a-list", "proper": {}, "readings": []}).encode("utf-8")),
        ("слово2", json.dumps({"candidates": [], "proper": "not-a-dict", "readings": []}).encode("utf-8")),
        ("слово3", json.dumps({"candidates": [], "proper": {}, "readings": "not-a-list"}).encode("utf-8")),
    ]
    trie = marisa_trie.BytesTrie(entries)
    trie.save(str(trie_path))

    with Lexicon(trie_path) as lex:
        for key in ("слово1", "слово2", "слово3"):
            with pytest.raises(AssetError):
                lex.lookup(key)


def test_lexicon_duplicate_payload(tmp_path: Path):
    trie_path = tmp_path / "lexicon_dup.marisa"
    p1 = json.dumps({"candidates": ["0"], "proper": {}, "readings": []}).encode("utf-8")
    p2 = json.dumps({"candidates": ["1"], "proper": {}, "readings": []}).encode("utf-8")
    entries = [("слово", p1), ("слово", p2)]
    trie = marisa_trie.BytesTrie(entries)
    trie.save(str(trie_path))

    with Lexicon(trie_path) as lex:
        with pytest.raises(AssetError) as exc_info:
            lex.lookup("слово")
        assert "multiple payloads" in str(exc_info.value)
