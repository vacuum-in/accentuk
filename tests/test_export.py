import csv
import gzip
import hashlib
import json
from pathlib import Path
import pytest

from accentuk.export import (
    build_bundle,
    drop_subsumed_free_variation,
    prefer_trie_default,
)
from accentuk.lexicon import Lexicon


def test_drop_subsumed_free_variation():
    # If specific member is present, drop 0|1
    assert drop_subsumed_free_variation(["0|1", "1"]) == ["1"]
    assert drop_subsumed_free_variation(["1", "0|1"]) == ["1"]
    # If no specific member, keep free variation
    assert drop_subsumed_free_variation(["0|1"]) == ["0|1"]
    # Multi-segment signatures
    assert drop_subsumed_free_variation(["0:0|1:0", "0"]) == ["0:0|1:0", "0"]


def test_prefer_trie_default():
    assert prefer_trie_default(["0", "1", "2"], "1") == ["1", "0", "2"]
    assert prefer_trie_default(["0", "1"], "0") == ["0", "1"]
    assert prefer_trie_default(["0", "1"], "3") == ["0", "1"]
    assert prefer_trie_default(["0"], "1") == ["0"]


def test_synthetic_bundle_export(tmp_path: Path):
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()

    # 1. active_dataset.csv.gz
    with gzip.open(archive_dir / "active_dataset.csv.gz", "wt", encoding="utf-8") as f:
        f.write("singleton,dataset_id,activated_at\nt,2,2026-08-23 20:47:16+00\n")

    # 2. stress_lookup.csv.gz
    # Columns: dataset_id,form_normalized,stressed_form,stress_signature,lemma_normalized,stressed_lemma,part_of_speech,grammatical_tags,lexeme_id,word_form_id,is_lemma,is_variant,is_obsolete,confidence,source_rank
    rows = [
        # Inactive dataset 99 -> should be excluded
        (99, "слово", "сло́во", "0", "слово", "сло́во", "noun", "{}", 1, 1, "t", "f", "f", 1.0, 1),
        # Word with free variation: 0|1 from ds 2, 1 from reviewed ds 12
        (2, "держави", "де́ржа́ви", "0|1", "держава", "держа́ва", "noun", "{}", 2, 2, "f", "f", "f", 0.9, 1),
        (12, "держави", "держа́ви", "1", "держава", "держа́ви", "noun", "{}", 2, 3, "f", "f", "f", 0.95, 2),
        # Capitalization / proper noun: русі
        # Proper reading: "Русі́" (sig 1), reviewed ds 12, conf 1.0, rank 1
        (12, "русі", "Русі́", "1", "Русь", "Русь", "noun", "{}", 3, 4, "f", "f", "f", 1.0, 1),
        # Common reading: "ру́сі" (sig 0), ds 2, conf 0.8, rank 2
        (2, "русі", "ру́сі", "0", "рух", "рух", "noun", "{}", 3, 5, "f", "f", "f", 0.8, 2),
        # Exclusive reading: маю (sig 0 in exclusive ds 13)
        (2, "маю", "маю́", "1", "мати", "мати", "verb", "{}", 4, 6, "f", "f", "f", 0.9, 1),
        (13, "маю", "ма́ю", "0", "мати", "мати", "verb", "{}", 4, 7, "f", "f", "f", 1.0, 1),
        # Trie default word: замок
        (2, "замок", "замо́к", "1", "замок", "замо́к", "noun", "{}", 5, 8, "f", "f", "f", 0.8, 2),
        (2, "замок", "за́мок", "0", "замок", "за́мок", "noun", "{}", 5, 9, "f", "f", "f", 0.8, 2),
    ]

    lookup_file = archive_dir / "stress_lookup.csv.gz"
    with gzip.open(lookup_file, "wt", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset_id", "form_normalized", "stressed_form", "stress_signature",
            "lemma_normalized", "stressed_lemma", "part_of_speech", "grammatical_tags",
            "lexeme_id", "word_form_id", "is_lemma", "is_variant", "is_obsolete",
            "confidence", "source_rank"
        ])
        for r in rows:
            writer.writerow(r)

    # 3. contextual_stress_candidate.csv.gz
    # Form "замок": matching inventory hash adds candidate "0" (already present)
    # Form "чужий": matching inventory adds candidate "1", mismatching hash ignored
    matching_hash = "abc123hash"
    contextual_file = archive_dir / "contextual_stress_candidate.csv.gz"
    with gzip.open(contextual_file, "wt", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["inventory_hash", "model_version", "form_normalized", "stress_signature", "stressed_form", "group_id", "sense_id"])
        writer.writerow([matching_hash, "v3", "чужий", "1", "чужи́й", 1, 1])
        writer.writerow(["wrong_hash", "v3", "чужий", "0", "чу́жий", 1, 2])

    # 4. Write MANIFEST.tsv
    manifest_tsv = archive_dir / "MANIFEST.tsv"
    with manifest_tsv.open("wt", encoding="utf-8") as f:
        for tname in ("active_dataset", "stress_lookup", "contextual_stress_candidate"):
            fpath = archive_dir / f"{tname}.csv.gz"
            h = hashlib.sha256(fpath.read_bytes()).hexdigest()
            f.write(f"{tname}\t1\t1\t{h}\n")

    # Assets dir with trie defaults
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    # For "замок", trie default is "0"
    (assets_dir / "trie_defaults_map.json").write_text(json.dumps({"замок": "0"}), encoding="utf-8")
    (assets_dir / "suffix_table.json").write_text("{}", encoding="utf-8")
    (assets_dir / "counted_forms.json").write_text("[]", encoding="utf-8")

    bundle_dir = tmp_path / "bundle"
    model_manifest = {"inventory_hash": matching_hash, "model_version": "v3-test"}

    build_bundle(
        archive=archive_dir,
        model_manifest=model_manifest,
        output=bundle_dir,
        assets_dir=assets_dir,
    )

    # Verify bundle structure
    assert (bundle_dir / "lexicon.marisa").is_file()
    assert (bundle_dir / "manifest.json").is_file()
    assert (bundle_dir / "suffix_table.json").is_file()
    assert (bundle_dir / "counted_forms.json").is_file()
    assert (bundle_dir / "trie_defaults_map.json").is_file()

    # Verify contents with Lexicon
    with Lexicon(bundle_dir / "lexicon.marisa") as lex:
        # 1. слово was in excluded dataset 99 -> None
        assert lex.lookup("слово") is None

        # 2. держави: 0|1 was subsumed by 1
        res = lex.lookup("держави")
        assert res is not None
        assert res["candidates"] == ["1"]

        # 3. русі: 0 is common, 1 is proper (reviewed). Reviewed 1 preferred
        res = lex.lookup("русі")
        assert res is not None
        assert res["candidates"] == ["1", "0"]
        assert res["proper"]["1"] is True
        assert res["proper"]["0"] is False

        # 4. маю: 0 is exclusive in ds 13, so it completely collapsed the list
        res = lex.lookup("маю")
        assert res is not None
        assert res["candidates"] == ["0"]

        # 5. замок: trie default was 0, so 0 leads 1
        res = lex.lookup("замок")
        assert res is not None
        assert res["candidates"] == ["0", "1"]

        # 6. чужий: matching contextual row was included
        res = lex.lookup("чужий")
        assert res is not None
        assert res["candidates"] == ["1"]
