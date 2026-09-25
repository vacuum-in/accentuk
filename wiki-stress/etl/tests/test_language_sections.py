from pathlib import Path

from ukstress.language_sections import ukrainian_entries


def test_extracts_multiple_entries_only_from_ukrainian_section() -> None:
    text = """== Англійська ==
=== noun ===
'''word'''
== Українська ==
=== Іменник ===
'''мо́ва'''
=== Дієслово ===
'''мова́ти'''
== Польська ==
=== noun ===
'''mowa'''
"""

    entries, unknown = ukrainian_entries(text)

    assert [entry.heading for entry in entries] == ["Іменник", "Дієслово"]
    assert "мо́ва" in entries[0].content
    assert "mowa" not in "".join(entry.content for entry in entries)
    assert unknown == []


def test_accepts_known_template_marker_and_reports_unknown_ukrainian_marker() -> None:
    entries, unknown = ukrainian_entries("== {{-uk-}} ==\n'''сло́во'''\n== Українська діалектна ==")

    assert len(entries) == 1
    assert "сло́во" in entries[0].content
    assert unknown == ["Українська діалектна"]


def test_single_equals_language_heading_used_by_live_dump_is_supported() -> None:
    entries, unknown = ukrainian_entries(
        "= {{-uk-}} =\n== Іменник ==\n{{імен uk 1a f una|мо́ва}}"
    )

    assert [entry.heading for entry in entries] == ["Іменник"]
    assert unknown == []


def test_standalone_template_marker_used_by_most_live_pages_is_supported() -> None:
    entries, unknown = ukrainian_entries(
        "{{=uk=|{{PAGENAME}}}}\n{{імен uk|мо́ва}}\n{{=ru=|слово}}"
    )

    assert len(entries) == 1
    assert "мо́ва" in entries[0].content
    assert "слово" not in entries[0].content
    assert unknown == []


def test_non_ukrainian_sections_never_emit_entries() -> None:
    entries, _ = ukrainian_entries("== Російська ==\n=== Іменник ===\n'''сло́во'''")
    assert entries == []


def test_representative_multilingual_wiktionary_fixture_isolated() -> None:
    fixture = Path(__file__).parent / "fixtures" / "ukwiktionary_mova_multilingual.wiki"
    entries, unknown = ukrainian_entries(fixture.read_text(encoding="utf-8"))

    assert [entry.heading for entry in entries] == ["Морфосинтаксичні ознаки", "Іменник"]
    assert "мо́ва" in "".join(entry.content for entry in entries)
    assert "mowa" not in "".join(entry.content for entry in entries)
    assert unknown == []
