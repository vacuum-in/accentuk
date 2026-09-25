-- Forward-only migration: read-optimized active-dataset lookup projection.
CREATE TABLE stress_lookup (
    dataset_id BIGINT NOT NULL REFERENCES import_run(id) ON DELETE CASCADE,
    form_normalized TEXT NOT NULL,
    stressed_form TEXT NOT NULL,
    stress_signature TEXT NOT NULL,
    lemma_normalized TEXT NOT NULL,
    stressed_lemma TEXT,
    part_of_speech TEXT,
    grammatical_tags TEXT[] NOT NULL DEFAULT '{}',
    lexeme_id BIGINT NOT NULL REFERENCES lexeme(id) ON DELETE CASCADE,
    word_form_id BIGINT NOT NULL REFERENCES word_form(id) ON DELETE CASCADE,
    is_lemma BOOLEAN NOT NULL,
    is_variant BOOLEAN NOT NULL,
    is_obsolete BOOLEAN NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    source_rank SMALLINT NOT NULL,
    PRIMARY KEY (dataset_id, form_normalized, stressed_form, lexeme_id, word_form_id)
);

CREATE INDEX stress_lookup_exact_idx
    ON stress_lookup (dataset_id, form_normalized, is_obsolete, confidence DESC, source_rank)
    INCLUDE (stressed_form, stress_signature, lemma_normalized, stressed_lemma,
             part_of_speech, grammatical_tags, lexeme_id, word_form_id, is_lemma, is_variant);
CREATE INDEX stress_lookup_lemma_idx
    ON stress_lookup (dataset_id, lemma_normalized)
    INCLUDE (form_normalized, stressed_form, grammatical_tags, confidence);

INSERT INTO schema_migration (version) VALUES ('004_lookup_projection');

