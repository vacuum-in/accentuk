# Database migrations

Migrations are append-only SQL files applied in lexical filename order by the
Python-owned `ukstress migrate` command. Connect it as the migration-time owner
(`ukstress_owner` in local Compose); the ETL and API roles never run migrations.

