from __future__ import annotations

import mwparserfromhell

from ukstress.morphology import MorphologyCatalog


def test_noun_template_definition_expands_all_explicit_cases() -> None:
    catalog = MorphologyCatalog(
        {
            "імен-uk-1a-f-una": """
{{inflection_імен_uk
|nom-sg={{{1}}}а
|gen-sg={{{1}}}и
|dat-sg={{{1}}}і
|nom-pl={{{1}}}и
|gen-pl={{{2|{{{1}}}}}}
}}"""
        }
    )
    invocation = mwparserfromhell.parse(
        "{{імен uk 1a f una|мо́в|мо́в}}"
    ).filter_templates()[0]

    forms = catalog.extract(invocation)

    assert {form.stressed_form for form in forms} == {
        "мо́ва",
        "мо́ви",
        "мо́ві",
        "мо́в",
    }
    assert any(form.grammatical_tags[:2] == ("nominative", "singular") for form in forms)


def test_verb_template_expands_conditionals_and_optional_endings() -> None:
    catalog = MorphologyCatalog(
        {
            "дієсл-uk-док-4a-ш": """
{{дієсл-блок
|Я (майб.)={{{2}}}{{#if:{{{ю|}}}|ю|у}}
|Ми (майб.)={{{1}}}им(о)
}}"""
        }
    )
    invocation = mwparserfromhell.parse(
        "{{дієсл uk док 4a-ш|званта́ж|званта́ж}}"
    ).filter_templates()[0]

    forms = catalog.extract(invocation)

    assert {form.stressed_form for form in forms} == {
        "званта́жу",
        "званта́жим",
        "званта́жимо",
    }
