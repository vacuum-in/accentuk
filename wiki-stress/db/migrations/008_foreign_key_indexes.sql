-- Index every foreign-key column that points at `lexeme` or `word_form`.
--
-- PostgreSQL indexes the *referenced* side of a foreign key automatically and
-- the *referencing* side never. Every child table here is indexed on
-- `(dataset_id, ...)`, so a cascade lookup by `lexeme_id` or `word_form_id`
-- alone cannot use a prefix of any of them and falls back to a sequential
-- scan -- of `stress_lookup`, which holds 2.9M rows.
--
-- The cost is per deleted row, so it is invisible on a fresh import and
-- crippling on a rewrite: re-writing a 6,183-row dataset ran for over an hour
-- before being killed. Dataset rewrites are how the generated and recovered
-- datasets are maintained, so this is the normal path, not an edge case.
CREATE INDEX IF NOT EXISTS word_form_lexeme_fk_idx ON word_form (lexeme_id);
CREATE INDEX IF NOT EXISTS stress_variant_word_form_fk_idx ON stress_variant (word_form_id);
CREATE INDEX IF NOT EXISTS source_ref_lexeme_fk_idx ON source_ref (lexeme_id);
CREATE INDEX IF NOT EXISTS source_ref_word_form_fk_idx ON source_ref (word_form_id);
CREATE INDEX IF NOT EXISTS stress_lookup_lexeme_fk_idx ON stress_lookup (lexeme_id);
CREATE INDEX IF NOT EXISTS stress_lookup_word_form_fk_idx ON stress_lookup (word_form_id);

INSERT INTO schema_migration (version) VALUES ('008_foreign_key_indexes');
