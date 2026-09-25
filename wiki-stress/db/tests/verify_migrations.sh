#!/bin/sh
# Requires a migration-owner connection and an API-role connection.
set -eu

: "${DATABASE_URL:?DATABASE_URL is required}"
: "${API_DATABASE_URL:?API_DATABASE_URL is required}"

for migration in "$(dirname "$0")"/../migrations/*.sql; do
    psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f "$migration"
done

if psql "$API_DATABASE_URL" -v ON_ERROR_STOP=1 -c 'DELETE FROM import_run'; then
    echo 'API database role unexpectedly modified lexicon data' >&2
    exit 1
fi

