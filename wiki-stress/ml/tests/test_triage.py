import json
from pathlib import Path

from ukstress_ml.triage import (
    ARTIFACT,
    FREE_VARIATION,
    GRAMMATICAL,
    HOMOGRAPH,
    NOT_IN_SOURCE,
    PRIMARY_LEXICON,
    RELIABLE_SEPARATING_FEATURES,
    classify_readings,
    is_proper_noun_suspect,
    needs_model,
    needs_review,
    separating_features,
    triage_forms,
    write_surface,
)

NOUN_NOM = ["Number=Sing", "Case=Nom", "upos=NOUN", "Gender=Masc"]
NOUN_GEN = ["Number=Sing", "Case=Gen", "upos=NOUN", "Gender=Masc"]
ADJ_NOM = ["Number=Sing", "Case=Nom", "Gender=Fem", "upos=ADJ"]


class FakeDictionary:
    def __init__(self, readings: dict[str, list]) -> None:
        self._readings = readings

    def readings(self, form: str) -> list:
        return self._readings.get(form, [])


def test_one_reading_carrying_two_accents_is_free_variation() -> None:
    # помилка: the dictionary itself records по́милка and поми́лка as acceptable.
    triage_class, shape = classify_readings([([], [2, 4])])
    assert triage_class == FREE_VARIATION
    assert shape["multi_accent_readings"] == 1


def test_same_tagset_with_different_accents_is_a_homograph() -> None:
    # замок: NOUN Sing Nom Masc occurs with both accents, so morphology cannot decide.
    triage_class, shape = classify_readings([(NOUN_NOM, [2]), (NOUN_NOM, [4])])
    assert triage_class == HOMOGRAPH
    assert shape["tagsets"] == 1
    assert shape["distinct_accents"] == 2


def test_tagsets_that_separate_the_accents_are_grammatical() -> None:
    # Morphology decides; the existing Stanza tag matching already resolves these.
    triage_class, _ = classify_readings([(NOUN_NOM, [2]), (NOUN_GEN, [4])])
    assert triage_class == GRAMMATICAL


def test_three_accents_under_one_tagset_is_an_artifact() -> None:
    # березинська: три accents for one adjective form is over-merging.
    triage_class, shape = classify_readings(
        [(ADJ_NOM, [4]), (ADJ_NOM, [6]), (ADJ_NOM, [11])]
    )
    assert triage_class == ARTIFACT
    assert shape["distinct_accents"] == 3


def test_source_offering_one_stress_cannot_explain_the_ambiguity() -> None:
    """`августа` is ambiguous in Wiktionary (А́вгуста/Авгу́ста) but the source
    dictionary gives a single stress. Calling that `grammatical` would assert
    morphology resolves it, which nothing supports."""
    triage_class, shape = classify_readings([([], [1])])
    assert triage_class == PRIMARY_LEXICON
    assert shape["accent_positions"] == 1


def test_readings_separated_by_case_are_morphologically_resolvable() -> None:
    features = separating_features([(NOUN_NOM, [2]), (NOUN_GEN, [4])])
    assert "Case" in features
    assert features & RELIABLE_SEPARATING_FEATURES


def test_readings_separated_only_by_pos_are_not_treated_as_resolved() -> None:
    # Абра́мович (NOUN, patronymic) vs Абрамо́вич (PROPN, surname): both are
    # proper names in running text, so upos is not a reliable router.
    noun = ["Case=Dat", "Gender=Masc", "Number=Sing", "upos=NOUN"]
    propn = ["Case=Dat", "Gender=Masc", "Number=Sing", "upos=PROPN"]
    features = separating_features([(noun, [4]), (propn, [6])])
    assert features == {"upos"}
    assert not (features & RELIABLE_SEPARATING_FEATURES)


def test_absent_from_source_is_reported_rather_than_guessed() -> None:
    triage_class, shape = classify_readings([])
    assert triage_class == NOT_IN_SOURCE
    assert shape["readings"] == 0


def test_capitalised_free_variation_is_flagged_for_review() -> None:
    # Бе́рестове / Берестове́ are two different villages, not one word with
    # two acceptable stresses.
    assert is_proper_noun_suspect(FREE_VARIATION, ("Бе́рестове", "Берестове́"))
    assert not is_proper_noun_suspect(FREE_VARIATION, ("по́милка", "поми́лка"))


def test_capitalisation_only_matters_for_free_variation() -> None:
    # Other classes keep their readings apart regardless of capitalisation.
    assert not is_proper_noun_suspect(HOMOGRAPH, ("Ве́дмедиці", "ведмеди́ці"))


def test_triage_forms_reports_class_and_review_reason() -> None:
    dictionary = FakeDictionary(
        {
            "замок": [(NOUN_NOM, [2]), (NOUN_NOM, [4])],
            "помилка": [([], [2, 4])],
            "берестове": [([], [2, 6])],
            "невідоме": [],
        }
    )
    rows = {
        row.form_normalized: row
        for row in triage_forms(
            [
                ("замок", ("за́мок", "замо́к")),
                ("помилка", ("по́милка", "поми́лка")),
                ("берестове", ("Бе́рестове", "Берестове́")),
                ("невідоме", ("неві́доме", "невідо́ме")),
            ],
            dictionary=dictionary,  # type: ignore[arg-type]
        )
    }

    assert rows["замок"].triage_class == HOMOGRAPH
    assert rows["замок"].review_reason is None
    assert rows["помилка"].triage_class == FREE_VARIATION
    assert not rows["помилка"].proper_noun_suspect
    assert rows["берестове"].proper_noun_suspect
    assert rows["берестове"].review_reason is not None
    assert rows["невідоме"].triage_class == NOT_IN_SOURCE


def test_pos_only_separation_routes_to_the_model_not_to_morphology() -> None:
    noun = ["Case=Dat", "Gender=Masc", "upos=NOUN"]
    propn = ["Case=Dat", "Gender=Masc", "upos=PROPN"]
    dictionary = FakeDictionary({"абрамовичеві": [(noun, [4]), (propn, [6])]})
    (row,) = triage_forms(
        [("абрамовичеві", ("абра́мовичеві", "абрамо́вичеві"))],
        dictionary=dictionary,  # type: ignore[arg-type]
    )

    assert row.triage_class == GRAMMATICAL
    assert row.separating_features == ("upos",)
    assert not row.morphologically_resolvable
    assert needs_model(row), "upos alone cannot route this; it needs context"
    assert row.review_reason is not None


def test_case_separation_is_kept_out_of_the_model_tier() -> None:
    dictionary = FakeDictionary({"абдула": [(NOUN_NOM, [6]), (NOUN_GEN, [4])]})
    (row,) = triage_forms(
        [("абдула", ("абду́ла", "абдула́"))],
        dictionary=dictionary,  # type: ignore[arg-type]
    )

    assert row.morphologically_resolvable
    assert not needs_model(row)
    assert not needs_review(row)


def test_write_surface_counts_every_class_and_the_model_tier(tmp_path: Path) -> None:
    dictionary = FakeDictionary(
        {
            "замок": [(NOUN_NOM, [2]), (NOUN_NOM, [4])],
            "обєднання": [(NOUN_NOM, [4]), (NOUN_NOM, [7])],
            "помилка": [([], [2, 4])],
            "ведмедиці": [(NOUN_NOM, [2]), (NOUN_GEN, [4])],
        }
    )
    path = tmp_path / "surface.jsonl"
    counts = write_surface(
        path,
        triage_forms(
            [
                ("замок", ("за́мок", "замо́к")),
                ("обєднання", ("обʼє́днання", "обʼєдна́ння")),
                ("помилка", ("по́милка", "поми́лка")),
                ("ведмедиці", ("ведме́диці", "ведмеди́ці")),
            ],
            dictionary=dictionary,  # type: ignore[arg-type]
        ),
    )

    assert counts["total"] == 4
    assert counts[HOMOGRAPH] == 2
    assert counts["model_tier"] == 2, "only homographs need the contextual model"
    assert counts[FREE_VARIATION] == 1
    assert counts[GRAMMATICAL] == 1

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    assert {row["triage_class"] for row in rows} == {HOMOGRAPH, FREE_VARIATION, GRAMMATICAL}


def test_identical_glosses_mean_free_variation_not_homography() -> None:
    """андру́шівська / андруші́вська are both "relating to Andrushivka" — one
    referent, two attested stresses. The tag structure cannot tell that apart
    from за́мок/замо́к; the gloss can."""
    from ukstress_ml.triage import is_free_variation_by_gloss

    assert is_free_variation_by_gloss(
        [
            "той, що стосується Андрушівки — міста в Житомирській області",
            "той, що стосується Андрушівки — міста в Житомирській області",
        ]
    )


def test_distinct_glosses_remain_a_homograph() -> None:
    from ukstress_ml.triage import is_free_variation_by_gloss

    assert not is_free_variation_by_gloss(
        ["пристрій для замикання дверей", "укріплена споруда феодальної доби"]
    )


def test_whitespace_differences_do_not_hide_an_identical_gloss() -> None:
    from ukstress_ml.triage import is_free_variation_by_gloss

    assert is_free_variation_by_gloss(["село в  Україні", " село в Україні "])


def test_a_single_or_missing_gloss_is_not_free_variation() -> None:
    from ukstress_ml.triage import is_free_variation_by_gloss

    assert not is_free_variation_by_gloss(["село в Україні"])
    assert not is_free_variation_by_gloss(["село в Україні", ""])


def test_two_distinct_verbs_under_one_tagset_are_a_homograph_not_free_variation() -> None:
    """`плачу` is пла́чу (cry) and плачу́ (pay) — different verbs, identical
    1sg tags — while its NOUN readings use variant notation [3,5]. Checking
    free variation first saw the variant notation and misfiled it."""
    verb = ["Number=Sing", "upos=VERB"]
    noun = ["Case=Gen", "Gender=Masc", "Number=Sing", "upos=NOUN"]
    triage_class, _ = classify_readings(
        [(verb, [3]), (verb, [5]), (noun, [3, 5]), (noun, [3])]
    )
    assert triage_class == HOMOGRAPH


def test_variant_notation_beside_a_plain_reading_stays_free_variation() -> None:
    """ба́тьківщи́на: [2,9] and [2] under one tag set is one word with an
    optional stress, not two words."""
    tags = ["Case=Nom", "Gender=Fem", "Number=Sing", "upos=NOUN"]
    triage_class, _ = classify_readings([(tags, [2, 9]), (tags, [2])])
    assert triage_class == FREE_VARIATION


def test_noun_beside_verb_is_not_routed_on_morphology() -> None:
    """ко́си "braids" / коси́ "mow!" — the trie's tags separate them, but Stanza
    tags the imperative as a nominative plural noun, identically to the real
    noun. Tag matching would inherit the parse error every time."""
    from ukstress_ml.triage import has_unreliable_pos_split

    noun = ["Case=Nom", "Gender=Fem", "Number=Plur", "upos=NOUN"]
    verb = ["Number=Sing", "upos=VERB"]
    assert has_unreliable_pos_split([(noun, [2]), (verb, [4])])


def test_case_split_within_one_pos_stays_routable() -> None:
    from ukstress_ml.triage import has_unreliable_pos_split

    assert not has_unreliable_pos_split([(NOUN_NOM, [2]), (NOUN_GEN, [4])])


def test_noun_verb_split_reaches_the_model_tier() -> None:
    noun = ["Case=Nom", "Gender=Fem", "Number=Plur", "upos=NOUN"]
    verb = ["Number=Sing", "upos=VERB"]
    dictionary = FakeDictionary({"коси": [(noun, [2]), (verb, [4])]})
    (row,) = triage_forms(
        [("коси", ("ко́си", "коси́"))],
        dictionary=dictionary,  # type: ignore[arg-type]
    )
    assert row.triage_class == GRAMMATICAL
    assert not row.morphologically_resolvable
    assert needs_model(row)
    assert "imperative" in (row.review_reason or "")
