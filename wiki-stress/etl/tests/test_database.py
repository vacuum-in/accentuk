from pathlib import Path

import pytest

from ukstress.database import apply_migrations, cleanup_retained_datasets, copy_staging_records


def test_migration_runner_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no SQL migrations"):
        apply_migrations("postgresql://unused", tmp_path)


def test_project_migrations_are_contiguous_and_self_recording() -> None:
    migrations = Path(__file__).resolve().parents[2] / "db" / "migrations"
    paths = sorted(migrations.glob("*.sql"))

    assert [path.name[:3] for path in paths] == [
        f"{number:03d}" for number in range(1, len(paths) + 1)
    ]
    for path in paths:
        assert f"VALUES ('{path.stem}')" in path.read_text(encoding="utf-8")


def test_copy_rejects_unrecognized_table_before_connecting() -> None:
    with pytest.raises(ValueError, match="unsupported staging table"):
        copy_staging_records("postgresql://unused", "lexeme", 1, [])


def test_retention_rejects_negative_count_before_connecting() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        cleanup_retained_datasets("postgresql://unused", -1)
