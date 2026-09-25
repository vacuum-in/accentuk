import bz2
import os
import threading
from pathlib import Path

import psycopg
import pytest

from ukstress.database import (
    QualityGateError,
    QualityGates,
    apply_migrations,
    build_lookup_projection,
    cleanup_retained_datasets,
    copy_staging_records,
    enforce_quality_gates,
    finalize_dataset,
    import_staging_dataset,
    merge_staging,
    publish_dataset,
    rollback_dataset,
    validate_staging,
)
from ukstress.exports import (
    database_statistics,
    export_rows,
    unhandled_template_report,
)
from ukstress.pipeline import parse_dump

DATABASE_URL = os.environ.get("UKSTRESS_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(DATABASE_URL is None, reason="integration database not configured")


def test_copy_and_staging_validation_are_streaming_and_idempotent() -> None:
    assert DATABASE_URL is not None
    migrations = Path(__file__).resolve().parents[2] / "db" / "migrations"
    apply_migrations(DATABASE_URL, migrations)
    with psycopg.connect(DATABASE_URL) as connection:
        dataset_id = connection.execute(
            """
            INSERT INTO import_run (
                dataset_key, dump_url, dump_sha256, parser_version,
                normalization_version, schema_version
            )
            VALUES ('copy-fixture', 'fixture', %s, '1', '1', '1')
            RETURNING id
            """,
            ("0" * 64,),
        ).fetchone()
        assert dataset_id is not None
        identifier = int(dataset_id[0])

    record = {
        "natural_key": "lexeme-1",
        "lemma": "мова",
        "lemma_normalized": "мова",
        "stressed_lemma": "мо́ва",
        "part_of_speech": "noun",
        "sense_key": "",
        "source_title": "мова",
        "source_section": "Іменник",
        "is_multiword": False,
        "is_obsolete": False,
        "confidence": 1.0,
    }
    assert copy_staging_records(
        DATABASE_URL, "staging_lexeme", identifier, iter([record])
    ) == 1
    with psycopg.connect(DATABASE_URL) as connection:
        counts = validate_staging(connection, identifier)
    assert counts["staging_lexeme"] == 1

    with pytest.raises(psycopg.errors.UniqueViolation):
        copy_staging_records(
            DATABASE_URL, "staging_lexeme", identifier, iter([record])
        )

    copy_staging_records(
        DATABASE_URL,
        "staging_word_form",
        identifier,
        [
            {
                "natural_key": "form-1",
                "lexeme_natural_key": "lexeme-1",
                "form": "мова",
                "form_normalized": "мова",
                "grammatical_tags": ["nominative", "singular"],
                "morphology_key": "nom;sg",
                "is_lemma": True,
                "is_variant": False,
                "confidence": 1.0,
                "source_rank": 1,
            }
        ],
    )
    copy_staging_records(
        DATABASE_URL,
        "staging_stress_variant",
        identifier,
        [
            {
                "natural_key": "variant-1",
                "word_form_natural_key": "form-1",
                "stressed_form": "мо́ва",
                "stress_signature": "0",
                "variant_type": "primary",
                "confidence": 1.0,
            }
        ],
    )
    copy_staging_records(
        DATABASE_URL,
        "staging_source_ref",
        identifier,
        [
            {
                "natural_key": "source-1",
                "lexeme_natural_key": "lexeme-1",
                "word_form_natural_key": "form-1",
                "page_id": 1,
                "revision_id": 2,
                "page_title": "мова",
                "source_kind": "template",
                "source_section": "Іменник",
                "source_fragment": "{{uk-noun|head=мо́ва}}",
                "source_offset": 0,
            }
        ],
    )
    with psycopg.connect(DATABASE_URL) as connection:
        merge_staging(connection, identifier)
        assert build_lookup_projection(connection, identifier) == 1
        finalize_dataset(connection, identifier)
        normalized_counts = [
            connection.execute(f"SELECT count(*) FROM {table}").fetchone()
            for table in (
                "lexeme",
                "word_form",
                "stress_variant",
                "source_ref",
                "stress_lookup",
            )
        ]
        status = connection.execute(
            "SELECT status FROM import_run WHERE id = %s", (identifier,)
        ).fetchone()
    assert normalized_counts == [(1,), (1,), (1,), (1,), (1,)]
    assert status == ("validated",)

    with psycopg.connect(DATABASE_URL) as connection:
        assert enforce_quality_gates(connection, identifier, QualityGates())[
            "lookup_rows"
        ] == 1.0
        with pytest.raises(QualityGateError, match="below minimum"):
            enforce_quality_gates(
                connection, identifier, QualityGates(minimum_lookup_rows=2)
            )

    assert publish_dataset(DATABASE_URL, identifier) is None
    with psycopg.connect(DATABASE_URL) as connection:
        failed = connection.execute(
            """
            INSERT INTO import_run (
                dataset_key, status, dump_url, dump_sha256, parser_version,
                normalization_version, schema_version
            )
            VALUES ('failed-fixture', 'building', 'fixture', %s, '1', '1', '1')
            RETURNING id
            """,
            ("2" * 64,),
        ).fetchone()
        assert failed is not None
        failed_id = int(failed[0])
    with psycopg.connect(DATABASE_URL) as connection:
        with pytest.raises(RuntimeError, match="injected"):
            with connection.transaction():
                connection.execute(
                    "UPDATE active_dataset SET dataset_id = %s WHERE singleton",
                    (failed_id,),
                )
                raise RuntimeError("injected publication failure")
        active_after_failure = connection.execute(
            "SELECT dataset_id FROM active_dataset WHERE singleton"
        ).fetchone()
        connection.execute(
            "UPDATE import_run SET status = 'failed' WHERE id = %s", (failed_id,)
        )
    assert active_after_failure == (identifier,)

    with psycopg.connect(DATABASE_URL) as connection:
        second = connection.execute(
            """
            INSERT INTO import_run (
                dataset_key, status, dump_url, dump_sha256, parser_version,
                normalization_version, schema_version
            )
            VALUES ('second-fixture', 'validated', 'fixture', %s, '1', '1', '1')
            RETURNING id
            """,
            ("1" * 64,),
        ).fetchone()
        assert second is not None
        second_id = int(second[0])

    reader_ready = threading.Event()
    publication_complete = threading.Event()
    observed: list[int] = []

    def read_across_publication() -> None:
        with psycopg.connect(DATABASE_URL) as reader:
            reader.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            first = reader.execute(
                "SELECT dataset_id FROM active_dataset WHERE singleton"
            ).fetchone()
            assert first is not None
            observed.append(int(first[0]))
            reader_ready.set()
            assert publication_complete.wait(timeout=5)
            repeated = reader.execute(
                "SELECT dataset_id FROM active_dataset WHERE singleton"
            ).fetchone()
            assert repeated is not None
            observed.append(int(repeated[0]))

    reader_thread = threading.Thread(target=read_across_publication)
    reader_thread.start()
    assert reader_ready.wait(timeout=5)
    assert publish_dataset(DATABASE_URL, second_id) == identifier
    publication_complete.set()
    reader_thread.join(timeout=5)
    assert not reader_thread.is_alive()
    assert observed == [identifier, identifier]
    with psycopg.connect(DATABASE_URL) as connection:
        newly_active = connection.execute(
            "SELECT dataset_id FROM active_dataset WHERE singleton"
        ).fetchone()
    assert newly_active == (second_id,)

    assert rollback_dataset(DATABASE_URL, identifier) == second_id
    assert cleanup_retained_datasets(DATABASE_URL, retain_count=0) == [
        second_id,
        failed_id,
    ]
    with psycopg.connect(DATABASE_URL) as connection:
        active = connection.execute(
            "SELECT dataset_id FROM active_dataset WHERE singleton"
        ).fetchone()
    assert active == (identifier,)


def test_parsed_staging_imports_as_validated_dataset(tmp_path: Path) -> None:
    assert DATABASE_URL is not None
    migrations = Path(__file__).resolve().parents[2] / "db" / "migrations"
    apply_migrations(DATABASE_URL, migrations)
    dump = tmp_path / "fixture.xml.bz2"
    xml = """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
<page><title>мова</title><ns>0</ns><id>91</id><revision><id>92</id>
<text>==Українська==\n===Іменник===\n{{uk-noun|мо́ва}}{{unknown-fixture|x}}</text>
</revision></page>
<page><title>замок</title><ns>0</ns><id>93</id><revision><id>94</id>
<text>==Українська==\n===Іменник===\n{{uk-noun|head=за́мок|lemma=замо́к}}</text>
</revision></page></mediawiki>"""
    dump.write_bytes(bz2.compress(xml.encode()))
    staging = parse_dump(
        dump, tmp_path / "output", dataset_key="full-import-fixture"
    ).output_dir

    dataset_id, quality = import_staging_dataset(DATABASE_URL, staging)

    assert quality["lookup_rows"] == 3.0
    with psycopg.connect(DATABASE_URL) as connection:
        status = connection.execute(
            "SELECT status FROM import_run WHERE id = %s", (dataset_id,)
        ).fetchone()
        rows = connection.execute(
            """
            SELECT form_normalized, stressed_form
            FROM stress_lookup
            WHERE dataset_id = %s
            ORDER BY form_normalized, stressed_form
            """,
            (dataset_id,),
        ).fetchall()
    assert status == ("validated",)
    assert rows == [("замок", "за́мок"), ("замок", "замо́к"), ("мова", "мо́ва")]

    publish_dataset(DATABASE_URL, dataset_id)
    all_rows = export_rows(DATABASE_URL)
    best_rows = export_rows(DATABASE_URL, best_only=True)
    assert len(all_rows) == 3
    assert len(best_rows) == 2
    assert database_statistics(DATABASE_URL)["counts"]["stress_lookup"] == 3
    assert unhandled_template_report(DATABASE_URL)[0]["normalized_name"] == "unknown-fixture"
