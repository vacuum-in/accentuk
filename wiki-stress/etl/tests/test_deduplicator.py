from ukstress.deduplicator import (
    FormObservation,
    SourceProvenance,
    deduplicate_observations,
    lexeme_key,
    stable_natural_key,
)


def _source(page_id: int) -> SourceProvenance:
    return SourceProvenance(page_id, page_id + 100, "template", "bounded fragment")


def test_stable_natural_keys_are_deterministic_and_homonym_aware() -> None:
    first = lexeme_key("dataset", "замок", "noun", "building", "замок", "1")
    repeated = lexeme_key("dataset", "замок", "noun", "building", "замок", "1")
    homonym = lexeme_key("dataset", "замок", "noun", "lock", "замок", "2")
    assert first == repeated
    assert first != homonym
    assert stable_natural_key("source", 1, 2) == stable_natural_key("source", 1, 2)


def test_preserves_stress_ambiguity_and_merges_duplicate_provenance() -> None:
    lexeme = lexeme_key("dataset", "замок", "noun", "", "замок", "noun")
    source = _source(1)
    merged = deduplicate_observations(
        [
            FormObservation(lexeme, "за́мок", "nom;sg", source),
            FormObservation(lexeme, "за́мок", "nom;sg", source),
            FormObservation(lexeme, "замо́к", "nom;sg", _source(2)),
        ]
    )

    assert len(merged) == 1
    assert merged[0].stress_variants == (("за́мок", "0"), ("замо́к", "1"))
    assert merged[0].sources == (_source(1), _source(2))


def test_same_spelling_for_unrelated_lexemes_is_not_merged() -> None:
    building = lexeme_key("dataset", "замок", "noun", "building", "замок", "1")
    lock = lexeme_key("dataset", "замок", "noun", "lock", "замок", "2")
    merged = deduplicate_observations(
        [
            FormObservation(building, "за́мок", "nom;sg", _source(1)),
            FormObservation(lock, "замо́к", "nom;sg", _source(2)),
        ]
    )
    assert len(merged) == 2

