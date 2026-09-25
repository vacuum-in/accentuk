from ukstress.wikicode import (
    UnhandledTemplateRegistry,
    extract_bold_headwords,
    extract_controlled_fallback,
    extract_templates,
    normalized_template_name,
)


def test_templates_are_data_and_registered_headwords_are_extracted() -> None:
    candidates, unhandled = extract_templates("{{uk-noun|head=мо́ва}}{{unknown|{{#invoke:x|run}}}}")

    assert candidates[0].stressed_form == "мо́ва"
    assert candidates[0].part_of_speech == "noun"
    assert candidates[0].confidence == 1.0
    assert unhandled == {"unknown": 1, "#invoke:x": 1}


def test_inflection_templates_preserve_reflexive_forms_and_all_values() -> None:
    candidates, _ = extract_templates("{{uk-conj|form1=вчу́ся|form2=вчи́шся}}")
    assert [candidate.stressed_form for candidate in candidates] == ["вчу́ся", "вчи́шся"]
    assert {candidate.confidence for candidate in candidates} == {0.95}


def test_registered_inflection_families_extract_supported_forms() -> None:
    markup = "".join(
        [
            "{{uk-decl-noun|n=мо́ви}}",
            "{{uk-decl-adj|n=краси́ва}}",
            "{{uk-conj|n=роби́ть}}",
            "{{uk-pron-decl|n=ко́гось}}",
            "{{uk-num-decl|n=дво́х}}",
            "{{uk-part-decl|n=зро́блений}}",
            "{{uk-adv|head=ви́ще}}",
            "{{uk-cmpr|n=кра́щий}}",
            "{{uk-supr|n=найкра́щий}}",
        ]
    )
    candidates, unhandled = extract_templates(markup)

    assert [candidate.stressed_form for candidate in candidates] == [
        "мо́ви",
        "краси́ва",
        "роби́ть",
        "ко́гось",
        "дво́х",
        "зро́блений",
        "ви́ще",
        "кра́щий",
        "найкра́щий",
    ]
    assert [candidate.part_of_speech for candidate in candidates] == [
        "noun",
        "adjective",
        "verb",
        "pronoun",
        "numeral",
        "participle",
        "adverb",
        "comparative",
        "superlative",
    ]
    assert unhandled == {}


def test_template_names_and_bold_fallback_are_normalized() -> None:
    assert normalized_template_name(" Uk_noun ") == "uk-noun"
    candidates = extract_bold_headwords("'''за́мок''' '''замо́к'''")
    assert [candidate.stressed_form for candidate in candidates] == ["за́мок", "замо́к"]
    assert {candidate.confidence for candidate in candidates} == {0.9}


def test_controlled_fallback_is_bounded_deduplicated_and_lower_confidence() -> None:
    candidates = extract_controlled_fallback(
        "Неструктуровані форми: мо́ва, мо́ва; слова́. unstressed.", max_candidates=2
    )

    assert [candidate.stressed_form for candidate in candidates] == ["мо́ва", "слова́"]
    assert {candidate.source_kind for candidate in candidates} == {"controlled_fallback"}
    assert {candidate.confidence for candidate in candidates} == {0.5}


def test_unknown_template_registry_counts_and_bounds_representative_samples() -> None:
    registry = UnhandledTemplateRegistry(sample_limit=1, fragment_limit=20)
    extract_templates(
        "{{Unknown_template|first sample}}",
        page_title="мова",
        unhandled_registry=registry,
    )
    extract_templates(
        "{{unknown template|second sample}}",
        page_title="слово",
        unhandled_registry=registry,
    )

    report = registry.reports()[0]
    assert report.normalized_name == "unknown-template"
    assert report.occurrences == 2
    assert report.sample_page_titles == ("мова",)
    assert report.sample_invocations == ("{{Unknown_template|f",)


def test_live_ukrainian_headword_and_pronunciation_templates_are_supported() -> None:
    candidates, unhandled = extract_templates(
        "{{імен uk|мо́ва}}{{імен uk 1a f una|склади={{склади|мо́|ва}}|мо́в}}"
        "{{transcription-uk|мо́ва}}{{transcriptions-uk|мо́ва|мо́ви|Uk-мова.ogg}}"
    )

    assert unhandled == {"склади": 1}
    assert {candidate.stressed_form for candidate in candidates} >= {"мо́в", "мо́ва", "мо́ви"}
