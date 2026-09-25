-- Database-backed candidates for the frozen contextual model inventory.
-- These rows supplement missing Wiktionary forms without changing the
-- published 363,272-form dataset or allowing the model to invent spellings.
CREATE TABLE contextual_stress_candidate (
    inventory_hash TEXT NOT NULL,
    model_version TEXT NOT NULL,
    form_normalized TEXT NOT NULL,
    stress_signature TEXT NOT NULL,
    stressed_form TEXT NOT NULL,
    group_id TEXT NOT NULL,
    sense_id TEXT NOT NULL,
    PRIMARY KEY (
        inventory_hash, form_normalized, stress_signature, stressed_form, sense_id
    )
);

CREATE INDEX contextual_stress_candidate_lookup_idx
    ON contextual_stress_candidate (inventory_hash, form_normalized)
    INCLUDE (stress_signature, stressed_form);

ALTER TABLE contextual_stress_candidate OWNER TO ukstress_etl;
GRANT SELECT ON contextual_stress_candidate TO ukstress_api;

INSERT INTO schema_migration (version) VALUES ('007_contextual_serving_inventory');
