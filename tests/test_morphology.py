import os
from pathlib import Path
import pytest

from accentuk.morphology import (
    SpacyMorphologyTier,
    agreement_repair,
    apply_accents,
    counted_form_feats,
    governed_by_counted_numeral,
    imperfective_infinitive,
    resolve,
)


def test_apply_accents():
    # Character offsets
    assert apply_accents("коси", (2,)) == "ко́си"
    assert apply_accents("коси", (4,)) == "коси́"
    assert apply_accents("колеса", (2,)) == "ко́леса"


def test_resolve_grammatical_features():
    # колеса: Nom Plur vs Gen Sing
    readings = [
        (["Number=Plur", "Case=Nom", "Gender=Neut", "upos=NOUN"], [2]),
        (["Number=Sing", "Case=Gen", "Gender=Neut", "upos=NOUN"], [4]),
    ]

    res = resolve(readings, "NOUN", "Case=Nom|Number=Plur|Gender=Neut")
    assert res is not None
    assert res.accents == (2,)

    res = resolve(readings, "NOUN", "Case=Gen|Number=Sing|Gender=Neut")
    assert res is not None
    assert res.accents == (4,)

    # Disagreement / unknown tags
    assert resolve(readings, "NOUN", "Case=Dat|Number=Sing") is None

    # Unreliable POS split (noun vs verb): коси
    unreliable_readings = [
        (["Number=Plur", "Case=Nom", "upos=NOUN"], [2]),
        (["Mood=Imp", "Number=Sing", "Person=2", "upos=VERB"], [4]),
    ]
    assert resolve(unreliable_readings, "NOUN", "Case=Nom|Number=Plur") is None


def test_aspect_imperfective_infinitive():
    # засипати: [2] perfective, [6] imperfective
    readings = [
        (["VerbForm=Inf", "upos=VERB"], [2]),
        (["VerbForm=Inf", "upos=VERB"], [6]),
    ]
    preceding = [("VERB", "Aspect=Perf", "почала")]
    chosen = imperfective_infinitive("засипати", readings, preceding)
    assert chosen == (6,)


def test_counted_form_helpers():
    # 2/3/4 governance
    preceding = [("NUM", "Case=Nom", "три")]
    assert governed_by_counted_numeral(preceding) is True

    # Oblique case does NOT govern counted form
    preceding_oblique = [("NUM", "Case=Gen", "трьох")]
    assert governed_by_counted_numeral(preceding_oblique) is False

    # Counted form rewrite: Nom Plur -> Gen Sing
    rewritten = counted_form_feats("NOUN", "Animacy=Inan|Case=Nom|Gender=Fem|Number=Plur")
    assert rewritten == "Animacy=Inan|Case=Gen|Gender=Fem|Number=Sing"


def test_real_spacy_morphology_tier():
    spacy_dir = (Path(os.environ.get("ACCENTUK_TEST_ASSETS", "/nonexistent")) / "spacy-uk_core_news_trf_380")
    if not spacy_dir.is_dir():
        pytest.skip("spacy model fixture not present")

    tier = SpacyMorphologyTier(spacy_dir)
    res = tier.parse("Мої сестри.")
    assert len(res) > 0

    # Test agreement repair on "Мої сестри."
    readings = [
        (["Number=Plur", "Case=Nom", "Gender=Fem", "upos=NOUN"], [2]),
        (["Number=Sing", "Case=Gen", "Gender=Fem", "upos=NOUN"], [6]),
    ]
    start = "Мої сестри.".index("сестри")
    repaired = agreement_repair(readings, "Мої сестри.", res, start)
    # се́стри is Nom Plur ([2])
    assert repaired == (2,)
