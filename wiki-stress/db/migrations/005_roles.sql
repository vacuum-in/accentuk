-- Run by the migration-time owner. Application passwords are injected at deployment.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'ukstress_etl') THEN
        CREATE ROLE ukstress_etl LOGIN NOINHERIT;
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'ukstress_api') THEN
        CREATE ROLE ukstress_api LOGIN NOINHERIT;
    END IF;
END
$$;

REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT USAGE, CREATE ON SCHEMA public TO ukstress_etl;
GRANT USAGE ON SCHEMA public TO ukstress_api;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO ukstress_etl;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO ukstress_etl;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO ukstress_api;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO ukstress_api;

-- The importer owns the read projection because finalization maintains its
-- indexes and refreshes planner statistics after every dataset load.
ALTER TABLE stress_lookup OWNER TO ukstress_etl;

INSERT INTO schema_migration (version) VALUES ('005_roles');
