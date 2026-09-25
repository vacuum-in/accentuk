-- Forward-only migration: normalized, dataset-scoped lexicon records.
CREATE TABLE part_of_speech (
    id SMALLINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    code TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL
);

INSERT INTO part_of_speech (code, description) VALUES
    ('noun', 'noun'), ('verb', 'verb'), ('adjective', 'adjective'),
    ('adverb', 'adverb'), ('pronoun', 'pronoun'), ('numeral', 'numeral'),
    ('participle', 'participle'), ('converb', 'converb'),
    ('preposition', 'preposition'), ('conjunction', 'conjunction'),
    ('particle', 'particle'), ('interjection', 'interjection'),
    ('abbreviation', 'abbreviation'), ('proper_noun', 'proper noun'),
    ('phrase', 'phrase'), ('unknown', 'unknown');

CREATE TABLE grammatical_feature (
    code TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    description TEXT NOT NULL
);

CREATE TABLE lexeme (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id BIGINT NOT NULL REFERENCES import_run(id) ON DELETE CASCADE,
    lemma TEXT NOT NULL,
    lemma_normalized TEXT NOT NULL,
    stressed_lemma TEXT,
    part_of_speech_id SMALLINT REFERENCES part_of_speech(id),
    sense_key TEXT NOT NULL DEFAULT '',
    source_title TEXT NOT NULL,
    source_section TEXT NOT NULL DEFAULT '',
    is_multiword BOOLEAN NOT NULL DEFAULT FALSE,
    is_obsolete BOOLEAN NOT NULL DEFAULT FALSE,
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (dataset_id, lemma_normalized, part_of_speech_id, sense_key, source_title, source_section)
);

CREATE TABLE word_form (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id BIGINT NOT NULL REFERENCES import_run(id) ON DELETE CASCADE,
    lexeme_id BIGINT NOT NULL REFERENCES lexeme(id) ON DELETE CASCADE,
    form TEXT NOT NULL,
    form_normalized TEXT NOT NULL,
    grammatical_tags TEXT[] NOT NULL DEFAULT '{}',
    morphology_key TEXT NOT NULL DEFAULT '',
    is_lemma BOOLEAN NOT NULL DEFAULT FALSE,
    is_variant BOOLEAN NOT NULL DEFAULT FALSE,
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    source_rank SMALLINT NOT NULL,
    UNIQUE (dataset_id, lexeme_id, form_normalized, morphology_key, is_variant)
);

CREATE TABLE stress_variant (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id BIGINT NOT NULL REFERENCES import_run(id) ON DELETE CASCADE,
    word_form_id BIGINT NOT NULL REFERENCES word_form(id) ON DELETE CASCADE,
    stressed_form TEXT NOT NULL,
    stress_signature TEXT NOT NULL,
    variant_type TEXT NOT NULL DEFAULT 'primary',
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    UNIQUE (dataset_id, word_form_id, stressed_form, stress_signature)
);

CREATE TABLE source_ref (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id BIGINT NOT NULL REFERENCES import_run(id) ON DELETE CASCADE,
    lexeme_id BIGINT REFERENCES lexeme(id) ON DELETE CASCADE,
    word_form_id BIGINT REFERENCES word_form(id) ON DELETE CASCADE,
    page_id BIGINT,
    revision_id BIGINT,
    page_title TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    source_section TEXT,
    source_fragment TEXT NOT NULL,
    source_offset INTEGER,
    CHECK (lexeme_id IS NOT NULL OR word_form_id IS NOT NULL)
);

INSERT INTO schema_migration (version) VALUES ('002_lexicon_tables');

