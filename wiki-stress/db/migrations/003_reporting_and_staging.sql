-- Forward-only migration: bounded diagnostics and persistent import staging.
CREATE TABLE parse_error (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id BIGINT REFERENCES import_run(id) ON DELETE CASCADE,
    page_id BIGINT,
    revision_id BIGINT,
    page_title TEXT,
    error_type TEXT NOT NULL,
    error_message TEXT NOT NULL,
    source_fragment TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE rejected_form (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id BIGINT NOT NULL REFERENCES import_run(id) ON DELETE CASCADE,
    page_id BIGINT,
    page_title TEXT,
    candidate TEXT NOT NULL,
    rejection_reason TEXT NOT NULL,
    source_fragment TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE unhandled_template (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id BIGINT NOT NULL REFERENCES import_run(id) ON DELETE CASCADE,
    normalized_name TEXT NOT NULL,
    occurrences BIGINT NOT NULL DEFAULT 1 CHECK (occurrences > 0),
    sample_page_title TEXT NOT NULL,
    sample_invocation TEXT NOT NULL,
    UNIQUE (dataset_id, normalized_name)
);

CREATE TABLE staging_lexeme (
    dataset_id BIGINT NOT NULL REFERENCES import_run(id) ON DELETE CASCADE,
    natural_key TEXT NOT NULL,
    payload JSONB NOT NULL,
    PRIMARY KEY (dataset_id, natural_key)
);

CREATE TABLE staging_word_form (
    dataset_id BIGINT NOT NULL REFERENCES import_run(id) ON DELETE CASCADE,
    natural_key TEXT NOT NULL,
    payload JSONB NOT NULL,
    PRIMARY KEY (dataset_id, natural_key)
);

CREATE TABLE staging_stress_variant (
    dataset_id BIGINT NOT NULL REFERENCES import_run(id) ON DELETE CASCADE,
    natural_key TEXT NOT NULL,
    payload JSONB NOT NULL,
    PRIMARY KEY (dataset_id, natural_key)
);

CREATE TABLE staging_source_ref (
    dataset_id BIGINT NOT NULL REFERENCES import_run(id) ON DELETE CASCADE,
    natural_key TEXT NOT NULL,
    payload JSONB NOT NULL,
    PRIMARY KEY (dataset_id, natural_key)
);

INSERT INTO schema_migration (version) VALUES ('003_reporting_and_staging');

