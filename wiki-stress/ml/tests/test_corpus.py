import pytest
from ukstress.normalizer import lookup_key, stress_signature

from ukstress_ml.corpus import (
    EncodingError,
    apply_signature,
    decode_source,
    encode_source,
)


def test_apply_signature_preserves_surface_casing() -> None:
    assert apply_signature("замок", "0") == "за́мок"
    assert apply_signature("замок", "1") == "замо́к"
    assert apply_signature("Вона", "1") == "Вона́"


def test_apply_signature_round_trips_through_the_signature() -> None:
    for surface, signature in [("замок", "0"), ("замок", "1"), ("країна", "1")]:
        stressed = apply_signature(surface, signature)
        assert stress_signature(stressed) == signature
        assert lookup_key(stressed) == lookup_key(surface)


def test_apply_signature_handles_decomposing_vowels() -> None:
    # `ї` is a vowel and takes an ordinal even though it decomposes to two
    # code points.
    assert stress_signature(apply_signature("країна", "1")) == "1"


def test_apply_signature_rejects_a_signature_that_does_not_fit() -> None:
    with pytest.raises(EncodingError):
        apply_signature("замок", "5")


def test_apply_signature_rejects_an_already_stressed_surface() -> None:
    with pytest.raises(EncodingError):
        apply_signature("за́мок", "0")


def test_source_encoding_round_trips() -> None:
    sentence = "Він відчинив замок старим ключем."
    start = sentence.index("замок")
    end = start + len("замок")
    source = encode_source(sentence, start, end)
    assert source == "Він відчинив ⟦замок⟧ старим ключем."
    assert decode_source(source) == (sentence, start, end)


def test_source_encoding_rejects_a_reserved_marker() -> None:
    with pytest.raises(EncodingError):
        encode_source("Текст ⟦ з маркером", 0, 5)


# Exemplar retrieval: score a sentence against real labelled usages of a sense
# rather than against its dictionary definition. See STATE.md for why —
# supervision volume does not reach the decision under the definition form.

def _row(sentence: str, form: str, sense: str) -> dict:
    start = sentence.index(form)
    return {"sentence": sentence, "start": start, "end": start + len(form),
            "form": form, "gold_sense": sense, "sense_id": sense, "group_id": 1,
            "candidates": [{"sense_id": "1.a"}, {"sense_id": "1.b"}]}


def test_exemplars_are_collected_per_sense() -> None:
    from ukstress_ml.crossencoder import build_exemplars

    rows = [_row("Старовинний замок на горі.", "замок", "1.a"),
            _row("Він зламав замок дверей.", "замок", "1.b"),
            _row("Високий замок над містом.", "замок", "1.a")]
    got = build_exemplars(rows, per_sense=3)
    assert len(got["1.a"]) == 2
    assert len(got["1.b"]) == 1
    # The target span is marked, as it is on the input side.
    assert all("⟦замок⟧" in s for s in got["1.a"])


def test_exemplars_are_capped_and_deterministic() -> None:
    from ukstress_ml.crossencoder import build_exemplars

    rows = [_row(f"Речення номер {n} має замок тут.", "замок", "1.a") for n in range(20)]
    first = build_exemplars(rows, per_sense=3)
    assert len(first["1.a"]) == 3
    assert build_exemplars(rows, per_sense=3) == first


def test_a_row_never_appears_among_its_own_evidence() -> None:
    from ukstress_ml.crossencoder import build_exemplars, candidate_exemplars, marked_sentence

    row = _row("Старовинний замок на горі.", "замок", "1.a")
    exemplars = build_exemplars([row], per_sense=3)
    pairs = candidate_exemplars(row, {}, exemplars)
    assert marked_sentence(row) not in dict(pairs)["1.a"]


def test_a_sense_without_exemplars_keeps_its_definition() -> None:
    from ukstress_ml.crossencoder import candidate_exemplars

    row = _row("Старовинний замок на горі.", "замок", "1.a")
    glosses = {"1.a": {"stressed": "за́мок", "definition": "Укріплене житло."},
               "1.b": {"stressed": "замо́к", "definition": "Пристрій для замикання."}}
    pairs = dict(candidate_exemplars(row, glosses, {}))
    assert pairs["1.a"] == "за́мок: Укріплене житло."
    assert pairs["1.b"] == "замо́к: Пристрій для замикання."


def test_the_dataset_falls_back_to_glosses_when_given_none() -> None:
    from ukstress_ml.crossencoder import PairDataset

    row = _row("Старовинний замок на горі.", "замок", "1.a")
    glosses = {"1.a": {"stressed": "за́мок", "definition": "Укріплене житло."},
               "1.b": {"stressed": "замо́к", "definition": "Пристрій."}}
    plain = PairDataset([row], glosses)[0]
    assert "Укріплене житло" in dict(plain["pairs"])["1.a"]
    assert plain["gold"] == 0


def test_a_corpus_form_missing_from_the_inventory_is_skipped_not_fatal(tmp_path) -> None:
    """Corpus and inventory are separate artifacts and need not agree."""
    import json as _json
    import sys

    sys.path.insert(0, "ml/scripts")
    from run_finetune_inflected import load_rows

    inventory = tmp_path / "inv.jsonl"
    inventory.write_text(_json.dumps({
        "form": "замок", "group_id": 1, "complete": True, "feats": "",
        "paradigm_source": "lemma",
        "candidates": [
            {"sense_id": "1.a", "signature": "0", "stressed": "за́мок",
             "definition": "Фортеця.", "pos": "noun", "priority": 1},
            {"sense_id": "1.b", "signature": "1", "stressed": "замо́к",
             "definition": "Пристрій.", "pos": "noun", "priority": 2},
        ],
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    silver = tmp_path / "silver.json"
    silver.write_text(_json.dumps([
        {"sentence": "Старовинний замок.", "form": "замок", "start": 12, "end": 17,
         "gold_sense": "1.a", "group_id": 1},
        {"sentence": "Тут макариха була.", "form": "макариха", "start": 4, "end": 12,
         "gold_sense": "9.a", "group_id": 9},
    ], ensure_ascii=False), encoding="utf-8")

    rows, glosses = load_rows(silver, inventory)
    assert len(rows) == 1
    assert rows[0]["form"] == "замок"
    assert set(glosses) == {"1.a", "1.b"}
