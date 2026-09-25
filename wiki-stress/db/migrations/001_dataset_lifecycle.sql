-- Forward-only migration: dataset lifecycle and migration ledger.
CREATE TABLE schema_migration (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE import_run (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_key TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'building',
    dump_url TEXT NOT NULL,
    dump_sha256 TEXT NOT NULL CHECK (dump_sha256 ~ '^[0-9a-f]{64}$'),
    dump_timestamp TIMESTAMPTZ,
    parser_version TEXT NOT NULL,
    normalization_version TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    git_commit TEXT,
    statistics JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    CHECK (status IN ('building', 'validated', 'published', 'superseded', 'failed'))
);

CREATE TABLE active_dataset (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    dataset_id BIGINT NOT NULL REFERENCES import_run(id),
    activated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO schema_migration (version) VALUES ('001_dataset_lifecycle');

