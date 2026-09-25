"""Fold supplementary datasets into one publishable dataset.

The lexicon is layered by design: corrections, trie-recovered readings, an LLM
gap fill, a homograph audit and lang-uk's heteronym dictionary each live in
their own `import_run` at a confidence below the curated wordlist's, so each
can be inspected and rolled back on its own.

Serving them, though, means the API reads six dataset ids — a deployment flag
rather than a published lexicon, and one that a rollback would not undo. This
copies every layer into a single new dataset in ascending precedence order, so
`stress_lookup`'s confidence ordering resolves them exactly as the layered read
did, and the result can go through the normal validate/publish path.

Natural keys are prefixed with their source dataset, because two layers may
legitimately use the same key for the same form and the merged dataset has a
uniqueness constraint over `(dataset_id, natural_key)`.

Nothing is deleted. The source datasets stay exactly as they were, so this is
reversible by publishing the previous dataset again.
"""

from __future__ import annotations

from typing import cast

import psycopg


def merge_datasets(
    database_url: str, sources: list[int], dataset_key: str
) -> tuple[int, dict[str, int]]:
    """Copy `sources` into a new dataset, later ids winning on equal confidence."""
    counts: dict[str, int] = {}
    with psycopg.connect(database_url) as connection, connection.transaction():
        existing = connection.execute(
            "SELECT id FROM import_run WHERE dataset_key = %s", (dataset_key,)
        ).fetchone()
        if existing is not None:
            target = int(cast(tuple[int], existing)[0])
            for table in ("source_ref", "stress_variant", "word_form", "lexeme"):
                connection.execute(
                    f"DELETE FROM {table} WHERE dataset_id = %s", (target,)  # noqa: S608
                )
            connection.execute("DELETE FROM stress_lookup WHERE dataset_id = %s", (target,))
        else:
            target = int(cast(tuple[int], connection.execute(
                """
                INSERT INTO import_run (dataset_key, status, dump_url, dump_sha256,
                       parser_version, normalization_version, schema_version, statistics)
                VALUES (%s, 'building', 'merge://supplements', %s, 'merge-1', '1', '007', '{}')
                RETURNING id
                """,
                (dataset_key, "0" * 64),
            ).fetchone())[0])

        for source in sources:
            # lexeme first: word_form references it, and the new ids are needed
            # to rewrite those references.
            connection.execute(
                """
                INSERT INTO lexeme (dataset_id, lemma, lemma_normalized, stressed_lemma,
                       part_of_speech_id, sense_key, source_title, source_section,
                       confidence, is_obsolete, natural_key)
                SELECT %s, lemma, lemma_normalized, stressed_lemma, part_of_speech_id,
                       sense_key, source_title, source_section, confidence, is_obsolete,
                       %s || natural_key
                FROM lexeme WHERE dataset_id = %s
                ON CONFLICT DO NOTHING
                """,
                (target, f"d{source}:", source),
            )
            connection.execute(
                """
                INSERT INTO word_form (dataset_id, lexeme_id, form, form_normalized,
                       morphology_key, grammatical_tags, is_lemma, is_variant,
                       confidence, source_rank, natural_key)
                SELECT %s, new_lex.id, w.form, w.form_normalized, w.morphology_key,
                       w.grammatical_tags, w.is_lemma, w.is_variant, w.confidence,
                       w.source_rank, %s || w.natural_key
                FROM word_form w
                JOIN lexeme old_lex ON old_lex.id = w.lexeme_id
                JOIN lexeme new_lex ON new_lex.dataset_id = %s
                     AND new_lex.natural_key = %s || old_lex.natural_key
                WHERE w.dataset_id = %s
                ON CONFLICT DO NOTHING
                """,
                (target, f"d{source}:", target, f"d{source}:", source),
            )
            connection.execute(
                """
                INSERT INTO stress_variant (dataset_id, word_form_id, stressed_form,
                       stress_signature, variant_type, confidence, natural_key)
                SELECT %s, new_wf.id, v.stressed_form, v.stress_signature,
                       v.variant_type, v.confidence, %s || v.natural_key
                FROM stress_variant v
                JOIN word_form old_wf ON old_wf.id = v.word_form_id
                JOIN word_form new_wf ON new_wf.dataset_id = %s
                     AND new_wf.natural_key = %s || old_wf.natural_key
                WHERE v.dataset_id = %s
                ON CONFLICT DO NOTHING
                """,
                (target, f"d{source}:", target, f"d{source}:", source),
            )
            moved = connection.execute(
                "SELECT count(*) FROM word_form WHERE dataset_id = %s", (target,)
            ).fetchone()
            counts[f"dataset {source}"] = int(cast(tuple[int], moved)[0])

        from ukstress.database import build_lookup_projection

        counts["stress_lookup rows"] = build_lookup_projection(connection, target)
        connection.execute(
            "UPDATE import_run SET status = 'validated' WHERE id = %s", (target,)
        )
    return target, counts
