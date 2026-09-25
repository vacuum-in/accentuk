from pathlib import Path

from ukstress.table_parser import extract_table_forms


def test_extracts_table_cells_with_context_without_splitting_hyphens() -> None:
    table = """{|
|-
| Н. || [[мо́ва]]<br/>пів-мо́ви<ref>x</ref>
|-
| Р. || мо́ви, мо́в
|}"""

    forms = extract_table_forms(table)

    assert [(form.stressed_form, form.row, form.column) for form in forms] == [
        ("мо́ва", 1, 1),
        ("пів-мо́ви", 1, 1),
        ("мо́ви", 2, 1),
        ("мо́в", 2, 1),
    ]


def test_representative_wiktionary_fixture_extracts_attested_table_forms() -> None:
    fixture = Path(__file__).parent / "fixtures" / "ukwiktionary_mova_multilingual.wiki"

    forms = extract_table_forms(fixture.read_text(encoding="utf-8"))

    assert [form.stressed_form for form in forms] == ["мо́ва", "мо́ви"]
