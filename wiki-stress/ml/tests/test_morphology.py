from ukstress_ml.morphology import (
    apply_accents,
    apply_counted_form,
    counted_form_feats,
    governed_by_counted_numeral,
    resolve,
)

NOUN_NOM_PL = ["Case=Nom", "Gender=Fem", "Number=Plur", "upos=NOUN"]
NOUN_GEN_SG = ["Case=Gen", "Gender=Fem", "Number=Sing", "upos=NOUN"]
VERB = ["Number=Sing", "upos=VERB"]


def test_case_split_resolves_from_the_parse() -> None:
    """ко́леса (nom.pl) vs коле́са (gen.sg) — the parse decides."""
    readings = [(NOUN_NOM_PL, [1]), (NOUN_GEN_SG, [3])]
    got = resolve(readings, "NOUN", "Case=Gen|Gender=Fem|Number=Sing")
    assert got is not None and got.accents == (3,)
    got = resolve(readings, "NOUN", "Case=Nom|Gender=Fem|Number=Plur")
    assert got is not None and got.accents == (1,)


def test_extra_features_in_the_parse_do_not_block_a_match() -> None:
    """Stanza emits Animacy and others the trie never records."""
    readings = [(NOUN_NOM_PL, [1]), (NOUN_GEN_SG, [3])]
    got = resolve(readings, "NOUN", "Animacy=Inan|Case=Gen|Gender=Fem|Number=Sing")
    assert got is not None and got.accents == (3,)


def test_noun_verb_split_is_refused() -> None:
    """Stanza tags a Ukrainian imperative as a nominative plural noun, so
    matching here would return the noun reading with false confidence."""
    assert resolve([(NOUN_NOM_PL, [1]), (VERB, [3])], "NOUN",
                   "Case=Nom|Gender=Fem|Number=Plur") is None


def test_pos_only_split_is_refused() -> None:
    noun = ["Case=Dat", "Gender=Masc", "Number=Sing", "upos=NOUN"]
    propn = ["Case=Dat", "Gender=Masc", "Number=Sing", "upos=PROPN"]
    assert resolve([(noun, [3]), (propn, [5])], "PROPN",
                   "Case=Dat|Gender=Masc|Number=Sing") is None


def test_conflicting_matches_return_nothing() -> None:
    """When the parse satisfies readings that disagree, morphology has decided
    nothing; guessing would be a coin flip."""
    assert resolve([(NOUN_NOM_PL, [1]), (NOUN_NOM_PL, [3])], "NOUN",
                   "Case=Nom|Gender=Fem|Number=Plur") is None


def test_no_matching_reading_returns_nothing() -> None:
    assert resolve([(NOUN_NOM_PL, [1]), (NOUN_GEN_SG, [3])], "VERB",
                   "Mood=Imp|Number=Sing") is None


def test_single_reading_is_not_this_tier() -> None:
    assert resolve([(NOUN_NOM_PL, [1])], "NOUN", "Case=Nom") is None


def test_apply_accents_uses_character_positions_not_vowel_ordinals() -> None:
    """The trie stores character offsets: `коси` -> [2] and [4], impossible as
    ordinals for a two-vowel word. `stress_signature` uses the opposite
    convention, and conflating them marks the wrong syllable."""
    assert apply_accents("коси", (2,)) == "ко́си"
    assert apply_accents("коси", (4,)) == "коси́"
    assert apply_accents("колеса", (2,)) == "ко́леса"
    assert apply_accents("колеса", (4,)) == "коле́са"


# The counted form: `дві сестри́` is tagged Case=Nom|Number=Plur, identically to
# `мої се́стри`, so the parse alone cannot separate them. These cover the rewrite
# that makes it separable.

SESTRY = [
    (["Number=Plur", "Case=Nom", "upos=NOUN", "Gender=Fem"], [2]),
    (["Number=Sing", "Case=Gen", "upos=NOUN", "Gender=Fem"], [6]),
    (["Number=Plur", "Case=Voc", "upos=NOUN", "Gender=Fem"], [2]),
]


def test_nominative_plural_noun_is_rewritten_as_genitive_singular() -> None:
    assert counted_form_feats("NOUN", "Animacy=Anim|Case=Nom|Gender=Fem|Number=Plur") == (
        "Animacy=Anim|Case=Gen|Gender=Fem|Number=Sing"
    )


def test_other_cases_are_left_alone() -> None:
    assert counted_form_feats("NOUN", "Case=Ins|Number=Plur") is None
    assert counted_form_feats("NOUN", "Case=Nom|Number=Sing") is None
    assert counted_form_feats("VERB", "Case=Nom|Number=Plur") is None
    assert counted_form_feats("NOUN", "") is None


def test_a_counted_numeral_governs_the_next_noun() -> None:
    assert governed_by_counted_numeral([("NUM", "Case=Nom|NumType=Card", "Три")])


def test_an_agreeing_adjective_does_not_break_the_chain() -> None:
    # `три смугасті полотна́`
    assert governed_by_counted_numeral(
        [("NUM", "Case=Nom", "Три"), ("ADJ", "Case=Nom|Number=Plur", "смугасті")]
    )


def test_a_numeral_above_four_governs_a_genitive_plural_instead() -> None:
    assert not governed_by_counted_numeral([("NUM", "Case=Nom", "П'ять")])


def test_an_oblique_numeral_does_not_make_a_counted_form() -> None:
    # `двома сестрами` — the noun agrees normally.
    assert not governed_by_counted_numeral([("NUM", "Case=Ins", "двома")])


def test_an_intervening_noun_breaks_the_chain() -> None:
    assert not governed_by_counted_numeral(
        [("NUM", "Case=Nom", "Три"), ("NOUN", "Case=Gen|Number=Sing", "роки")]
    )


def test_nothing_before_the_token_is_not_a_counted_form() -> None:
    assert not governed_by_counted_numeral([])


def test_the_rewrite_flips_which_reading_resolve_picks() -> None:
    plural = "Animacy=Anim|Case=Nom|Gender=Fem|Number=Plur"
    assert resolve(SESTRY, "NOUN", plural).accents == (2,)          # се́стри
    rewritten = counted_form_feats("NOUN", plural)
    assert resolve(SESTRY, "NOUN", rewritten).accents == (6,)       # сестри́


def test_apply_counted_form_edits_only_the_governed_token(monkeypatch) -> None:
    monkeypatch.setenv("COUNTED_FORM", "1")
    # `Три сестри прийшли.`
    spans = {
        (0, 3): ("NUM", "Case=Nom"),
        (4, 10): ("NOUN", "Case=Nom|Number=Plur"),
        (11, 19): ("VERB", "Number=Plur"),
    }
    tokens = [
        ((0, 3), "NUM", "Case=Nom", "Три"),
        ((4, 10), "NOUN", "Case=Nom|Number=Plur", "сестри"),
        ((11, 19), "VERB", "Number=Plur", "прийшли"),
    ]
    apply_counted_form(spans, tokens)
    assert spans[(4, 10)] == ("NOUN", "Case=Gen|Number=Sing")
    assert spans[(0, 3)] == ("NUM", "Case=Nom")
    assert spans[(11, 19)] == ("VERB", "Number=Plur")


def test_the_counted_form_rewrite_is_off_unless_asked_for(monkeypatch) -> None:
    spans = {(4, 10): ("NOUN", "Case=Nom|Number=Plur")}
    tokens = [
        ((0, 3), "NUM", "Case=Nom", "Три"),
        ((4, 10), "NOUN", "Case=Nom|Number=Plur", "сестри"),
    ]
    monkeypatch.delenv("COUNTED_FORM", raising=False)
    apply_counted_form(spans, tokens)
    assert spans[(4, 10)] == ("NOUN", "Case=Nom|Number=Plur")

    monkeypatch.setenv("COUNTED_FORM", "1")
    apply_counted_form(spans, tokens)
    assert spans[(4, 10)] == ("NOUN", "Case=Gen|Number=Sing")


def test_the_list_restricts_which_forms_are_rewritten(monkeypatch, tmp_path) -> None:
    import json as _json

    import ukstress_ml.morphology as m

    listing = tmp_path / "counted.json"
    listing.write_text(_json.dumps([
        {"form": "сестри", "verdict": "counted"},
        {"form": "площини", "verdict": "nominative"},
    ]), encoding="utf-8")
    monkeypatch.setenv("COUNTED_FORM", "1")
    monkeypatch.setenv("COUNTED_FORM_LIST", str(listing))
    monkeypatch.setattr(m, "_COUNTED_FORMS", None)

    def spans_for(word: str) -> dict:
        spans = {(4, 4 + len(word)): ("NOUN", "Case=Nom|Number=Plur")}
        tokens = [((0, 3), "NUM", "Case=Nom", "Три"),
                  ((4, 4 + len(word)), "NOUN", "Case=Nom|Number=Plur", word)]
        m.apply_counted_form(spans, tokens)
        return spans[(4, 4 + len(word))]

    assert spans_for("сестри") == ("NOUN", "Case=Gen|Number=Sing")
    # On the list as `nominative`, so it is left alone.
    assert spans_for("площини") == ("NOUN", "Case=Nom|Number=Plur")
    # Not on the list at all.
    assert spans_for("полотна") == ("NOUN", "Case=Nom|Number=Plur")
