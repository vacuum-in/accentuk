-- Forward-only migration: persist staging identities for deterministic set-based merges.
ALTER TABLE lexeme ADD COLUMN natural_key TEXT;
UPDATE lexeme SET natural_key = 'legacy:lexeme:' || id;
ALTER TABLE lexeme ALTER COLUMN natural_key SET NOT NULL;
ALTER TABLE lexeme ADD CONSTRAINT lexeme_dataset_natural_key UNIQUE (dataset_id, natural_key);

ALTER TABLE word_form ADD COLUMN natural_key TEXT;
UPDATE word_form SET natural_key = 'legacy:word_form:' || id;
ALTER TABLE word_form ALTER COLUMN natural_key SET NOT NULL;
ALTER TABLE word_form
    ADD CONSTRAINT word_form_dataset_natural_key UNIQUE (dataset_id, natural_key);

ALTER TABLE stress_variant ADD COLUMN natural_key TEXT;
UPDATE stress_variant SET natural_key = 'legacy:stress_variant:' || id;
ALTER TABLE stress_variant ALTER COLUMN natural_key SET NOT NULL;
ALTER TABLE stress_variant
    ADD CONSTRAINT stress_variant_dataset_natural_key UNIQUE (dataset_id, natural_key);

ALTER TABLE source_ref ADD COLUMN natural_key TEXT;
UPDATE source_ref SET natural_key = 'legacy:source_ref:' || id;
ALTER TABLE source_ref ALTER COLUMN natural_key SET NOT NULL;
ALTER TABLE source_ref
    ADD CONSTRAINT source_ref_dataset_natural_key UNIQUE (dataset_id, natural_key);

INSERT INTO schema_migration (version) VALUES ('006_natural_keys');
