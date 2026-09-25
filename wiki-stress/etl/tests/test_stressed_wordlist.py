import hashlib
import json
from pathlib import Path

import pytest

from ukstress.normalizer import ACUTE, stress_signature
from ukstress.sources.merge import merge_wordlist_into_staging
from ukstress.sources.stressed_wordlist import (
    SPACING_ACUTE,
    WordlistStats,
    audit_wordlist,
    build_form_records,
    iter_wordlist_forms,
    split_stress_alternatives,
    to_combining_acute,
)
from ukstress.staging import iter_jsonl_zst, write_jsonl_zst


def _write_wordlist(directory: Path, lines: list[str]) -> Path:
    path = directory / "wordlist.dict"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_spacing_acute_is_converted_to_the_canonical_combining_acute() -> None:
    converted = to_combining_acute(f"за{SPACING_ACUTE}мок")
    assert SPACING_ACUTE not in converted
    assert converted == f"за{ACUTE}мок"


def test_multi_acute_token_splits_into_one_reading_per_acute() -> None:
    alternatives = split_stress_alternatives(to_combining_acute("за´мо´к"))
    assert alternatives == [f"за{ACUTE}мок", f"замо{ACUTE}к"]
    assert [stress_signature(value) for value in alternatives] == ["0", "1"]


def test_single_acute_token_is_returned_unchanged() -> None:
    single = to_combining_acute("восьмидеся´та")
    assert split_stress_alternatives(single) == [single]


def test_unstressed_and_duplicate_lines_are_counted_not_silently_dropped(
    tmp_path: Path,
) -> None:
    path = _write_wordlist(tmp_path, ["за´мо´к", "аакувата", "", "восьмидеся´та"])
    stats = WordlistStats()
    forms = list(iter_wordlist_forms(path, stats))

    assert [form.form_normalized for form in forms] == ["замок", "восьмидесята"]
    assert stats.lines == 4
    assert stats.unstressed == 1
    assert stats.blank == 1
    assert stats.rejected == 0


def test_ambiguous_form_keeps_every_reading_and_declines_a_lemma_stress(
    tmp_path: Path,
) -> None:
    path = _write_wordlist(tmp_path, ["за´мо´к"])
    (form,) = list(iter_wordlist_forms(path, WordlistStats()))
    assert form.is_ambiguous

    lexeme, word_form, variants, source_ref = build_form_records(form, dataset_key="d")

    # Picking one reading here would invent the answer the model tier supplies.
    assert lexeme["stressed_lemma"] is None
    assert {variant["stress_signature"] for variant in variants} == {"0", "1"}
    assert word_form["form_normalized"] == "замок"
    assert source_ref["word_form_natural_key"] == word_form["natural_key"]


def test_unambiguous_form_records_its_single_reading_as_the_lemma_stress(
    tmp_path: Path,
) -> None:
    path = _write_wordlist(tmp_path, ["восьмидеся´та"])
    (form,) = list(iter_wordlist_forms(path, WordlistStats()))
    lexeme, _, variants, _ = build_form_records(form, dataset_key="d")

    assert lexeme["stressed_lemma"] == f"восьмидеся{ACUTE}та"
    assert len(variants) == 1


def test_wordlist_source_rank_sorts_below_the_wiktionary_parser(tmp_path: Path) -> None:
    path = _write_wordlist(tmp_path, ["восьмидеся´та"])
    (form,) = list(iter_wordlist_forms(path, WordlistStats()))
    _, word_form, _, _ = build_form_records(form, dataset_key="d")

    # The Wiktionary parser emits confidence 0.95 -> source_rank 5, and
    # stress_lookup orders by confidence DESC, source_rank.
    assert word_form["confidence"] < 0.95
    assert int(str(word_form["source_rank"])) > 5


def test_audit_detects_multi_token_entries(tmp_path: Path) -> None:
    path = _write_wordlist(tmp_path, ["за´мок", "а´льфа-ро´зпад", "два сло´ва"])
    audit = audit_wordlist(path)
    assert audit == {"forms": 3, "multi_token": 1, "hyphenated": 1}


def _staging_fixture(directory: Path) -> Path:
    """A minimal primary-source staging directory."""
    base = directory / "base"
    base.mkdir()
    lexeme = {
        "natural_key": "a" * 64,
        "lemma": "замок",
        "lemma_normalized": "замок",
        "stressed_lemma": f"за{ACUTE}мок",
        "part_of_speech": "noun",
        "sense_key": "0",
        "source_title": "замок",
        "source_section": "Ukrainian",
        "is_multiword": False,
        "is_obsolete": False,
        "confidence": 0.95,
    }
    word_form = {
        "natural_key": "b" * 64,
        "lexeme_natural_key": "a" * 64,
        "form": f"за{ACUTE}мок",
        "form_normalized": "замок",
        "grammatical_tags": [],
        "morphology_key": "",
        "is_lemma": True,
        "is_variant": False,
        "confidence": 0.95,
        "source_rank": 5,
    }
    variant = {
        "natural_key": "c" * 64,
        "word_form_natural_key": "b" * 64,
        "stressed_form": f"за{ACUTE}мок",
        "stress_signature": "0",
        "variant_type": "primary",
        "confidence": 0.95,
    }
    write_jsonl_zst(base / "lexemes.jsonl.zst", [lexeme])
    write_jsonl_zst(base / "word_forms.jsonl.zst", [word_form])
    write_jsonl_zst(base / "stress_variants.jsonl.zst", [variant])
    write_jsonl_zst(base / "source_refs.jsonl.zst", [])
    write_jsonl_zst(base / "parse_errors.jsonl.zst", [])
    (base / "aliases.json").write_text("{}", encoding="utf-8")
    (base / "reports").mkdir()
    (base / "reports" / "unhandled_templates.json").write_text("[]", encoding="utf-8")
    (base / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_key": "primary",
                "dump_url": "https://example.invalid/dump.xml.bz2",
                "dump_sha256": "0" * 64,
                "dump_timestamp": None,
                "parser_version": "0.1.0",
                "normalization_version": "1",
                "schema_version": "006",
                "git_commit": None,
            }
        ),
        encoding="utf-8",
    )
    return base


def _merge(tmp_path: Path, lines: list[str]) -> tuple[dict, Path]:
    base = _staging_fixture(tmp_path)
    wordlist = _write_wordlist(tmp_path, lines)
    output = tmp_path / "merged"
    stats = merge_wordlist_into_staging(
        base,
        wordlist,
        output,
        dataset_key="merged",
        wordlist_sha256=hashlib.sha256(wordlist.read_bytes()).hexdigest(),
    )
    return stats, output


def test_primary_wins_so_a_conflicting_wordlist_stress_never_reaches_the_dataset(
    tmp_path: Path,
) -> None:
    # The wordlist disagrees with the primary source about `замок`.
    stats, output = _merge(tmp_path, ["замо´к", "восьмидеся´та"])

    assert stats["conflicts"] == 1
    assert stats["skipped_present_in_primary"] == 1
    assert stats["accepted_forms"] == 1

    variants = list(iter_jsonl_zst(output / "stress_variants.jsonl.zst"))
    by_form = {variant["stressed_form"] for variant in variants}
    assert f"за{ACUTE}мок" in by_form
    assert f"замо{ACUTE}к" not in by_form

    report = (output / "reports" / "stress_source_conflicts.tsv").read_text(encoding="utf-8")
    assert "замок\t0\t1\tprimary_wins" in report


def test_agreeing_forms_are_counted_and_not_duplicated(tmp_path: Path) -> None:
    stats, output = _merge(tmp_path, ["за´мок"])

    assert stats["agreements"] == 1
    assert stats["conflicts"] == 0
    assert stats["accepted_forms"] == 0
    # The primary row is kept exactly once, not re-added by the wordlist.
    assert len(list(iter_jsonl_zst(output / "word_forms.jsonl.zst"))) == 1


def test_new_forms_are_added_with_referential_integrity(tmp_path: Path) -> None:
    stats, output = _merge(tmp_path, ["восьмидеся´та", "жо´вні´"])

    assert stats["accepted_forms"] == 2
    assert stats["ambiguous_forms"] == 1

    lexemes = {row["natural_key"] for row in iter_jsonl_zst(output / "lexemes.jsonl.zst")}
    word_forms = {
        row["natural_key"]: row for row in iter_jsonl_zst(output / "word_forms.jsonl.zst")
    }
    for row in word_forms.values():
        assert row["lexeme_natural_key"] in lexemes
    for row in iter_jsonl_zst(output / "stress_variants.jsonl.zst"):
        assert row["word_form_natural_key"] in word_forms
        assert row["stress_signature"] == stress_signature(str(row["stressed_form"]))


def test_multi_token_wordlist_is_refused_rather_than_split_wrongly(tmp_path: Path) -> None:
    base = _staging_fixture(tmp_path)
    wordlist = _write_wordlist(tmp_path, ["а´льфа-ро´зпад"])
    with pytest.raises(ValueError, match="single tokens"):
        merge_wordlist_into_staging(
            base,
            wordlist,
            tmp_path / "merged",
            dataset_key="merged",
            wordlist_sha256="0" * 64,
        )


def test_merged_directory_satisfies_every_file_the_importer_reads(tmp_path: Path) -> None:
    """Guards the whole importer file contract, not just the record streams.

    `import_staging_dataset` reads `parse_errors.jsonl.zst` *after* copying
    ~10M staging rows and rebuilding the projection, so a missing auxiliary
    file surfaces only at the end of a multi-hour import.
    """
    _, output = _merge(tmp_path, ["восьмидеся´та"])

    for name in ("manifest.json", "parse_errors.jsonl.zst", "aliases.json"):
        assert (output / name).exists(), f"importer reads {name}"
    for name in ("lexemes", "word_forms", "stress_variants", "source_refs"):
        assert (output / f"{name}.jsonl.zst").exists()
    assert (output / "reports" / "unhandled_templates.json").exists()
    # The merge's own reports sit alongside the carried-forward ones.
    assert (output / "reports" / "stress_source_conflicts.tsv").exists()
    assert (output / "reports" / "wordlist_merge.json").exists()


def test_missing_required_auxiliary_file_fails_before_any_work(tmp_path: Path) -> None:
    base = _staging_fixture(tmp_path)
    (base / "parse_errors.jsonl.zst").unlink()
    wordlist = _write_wordlist(tmp_path, ["восьмидеся´та"])

    with pytest.raises(FileNotFoundError, match="parse_errors.jsonl.zst"):
        merge_wordlist_into_staging(
            base,
            wordlist,
            tmp_path / "merged",
            dataset_key="merged",
            wordlist_sha256="0" * 64,
        )


def test_manifest_records_conflict_policy_and_wordlist_provenance(tmp_path: Path) -> None:
    _, output = _merge(tmp_path, ["восьмидеся´та"])
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    options = manifest["command_options"]
    assert options["conflict_policy"] == "primary_wins"
    assert options["merge_source"] == "stressed_wordlist"
    assert options["primary_dataset_key"] == "primary"
    assert len(options["wordlist_sha256"]) == 64
