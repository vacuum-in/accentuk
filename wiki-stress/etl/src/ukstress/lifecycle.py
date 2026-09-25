"""High-level ETL orchestration used by the build command."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ukstress.database import (
    QualityGates,
    apply_migrations,
    import_staging_dataset,
    publish_dataset,
)
from ukstress.pipeline import parse_dump, validate_staging_directory


def build_dataset(
    *,
    dump_path: Path,
    output_root: Path,
    database_url: str,
    migrations_dir: Path,
    workers: int = 1,
    dataset_key: str | None = None,
    publish: bool = False,
    gates: QualityGates | None = None,
) -> dict[str, Any]:
    applied = apply_migrations(database_url, migrations_dir)
    parsed = parse_dump(
        dump_path,
        output_root,
        workers=workers,
        dataset_key=dataset_key,
    )
    validation = validate_staging_directory(parsed.output_dir)
    dataset_id, quality = import_staging_dataset(
        database_url, parsed.output_dir, gates=gates
    )
    previous = publish_dataset(database_url, dataset_id) if publish else None
    return {
        "dataset_id": dataset_id,
        "dataset_key": parsed.dataset_key,
        "status": "published" if publish else "validated",
        "previous_dataset_id": previous,
        "migrations_applied": applied,
        "staging_counts": validation,
        "quality": quality,
    }
