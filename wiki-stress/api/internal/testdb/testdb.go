// Package testdb provides shared PostgreSQL setup for integration tests
// across the repository and httpapi packages. It is only ever imported
// from _test.go files, never from production code, so it never reaches
// the built binary despite importing "testing".
//
// Tests using it are skipped unless UKSTRESS_TEST_DATABASE_URL is set,
// mirroring the Python convention in
// etl/tests/test_database_integration.py. Point it at a throwaway
// database. EnsureMigrated tracks applied versions in schema_migration,
// so it is safe whether the target is empty or was already migrated by
// another test binary run moments earlier.
//
// Run integration tests with `make test-integration` (go test -p 1
// ./...), not plain `go test ./...`. The repository and httpapi packages
// both mutate the single global active_dataset row against the same live
// database; Go's default per-package test parallelism runs their test
// binaries as concurrent OS processes, which race on that shared row.
// -p 1 forces packages to run one at a time, which is all this needs —
// tests within a single package already run sequentially by default.
package testdb

import (
	"context"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
)

// APIRolePassword is a fixed, hardcoded test constant, not user input,
// so setting it via a literal ALTER ROLE statement is safe.
const APIRolePassword = "ukstress-integration-test-password"

var migrateOnce sync.Once

// OwnerDatabaseURL returns UKSTRESS_TEST_DATABASE_URL or skips the test.
func OwnerDatabaseURL(t *testing.T) string {
	t.Helper()
	dbURL := os.Getenv("UKSTRESS_TEST_DATABASE_URL")
	if dbURL == "" {
		t.Skip("UKSTRESS_TEST_DATABASE_URL not set; skipping integration tests")
	}
	return dbURL
}

// EnsureMigrated applies every db/migrations file not yet recorded in
// schema_migration, and sets a known password on the ukstress_api role so
// tests can connect through it (production credentials are injected at
// deployment, not baked into the migration).
//
// It is safe to run from independent test binaries against the same live
// database: `go test ./...` runs each package as its own process, so the
// repository and httpapi packages each carry their own process-local
// migrateOnce. The schema_migration check below is what makes the second
// process's run a no-op instead of a failed CREATE TABLE; migrateOnce
// only avoids redundant round trips within a single binary's own tests.
func EnsureMigrated(t *testing.T, ctx context.Context, ownerURL string) {
	t.Helper()
	migrateOnce.Do(func() {
		dir, err := filepath.Abs(filepath.Join("..", "..", "..", "db", "migrations"))
		if err != nil {
			t.Fatalf("resolve migrations directory: %v", err)
		}
		entries, err := os.ReadDir(dir)
		if err != nil {
			t.Fatalf("read migrations directory: %v", err)
		}
		names := make([]string, 0, len(entries))
		for _, entry := range entries {
			if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".sql") {
				names = append(names, entry.Name())
			}
		}
		sort.Strings(names)

		cfg, err := pgx.ParseConfig(ownerURL)
		if err != nil {
			t.Fatalf("parse owner database url: %v", err)
		}
		// Migration files contain multiple ;-separated statements
		// (including dollar-quoted DO blocks); the simple protocol lets
		// PostgreSQL itself parse statement boundaries instead of the
		// client naively splitting on ";".
		cfg.DefaultQueryExecMode = pgx.QueryExecModeSimpleProtocol
		conn, err := pgx.ConnectConfig(ctx, cfg)
		if err != nil {
			t.Fatalf("connect as owner: %v", err)
		}
		defer conn.Close(ctx)

		applied := appliedMigrationVersions(t, ctx, conn)

		for _, name := range names {
			version := strings.TrimSuffix(name, ".sql")
			if applied[version] {
				continue
			}
			content, err := os.ReadFile(filepath.Join(dir, name))
			if err != nil {
				t.Fatalf("read migration %s: %v", name, err)
			}
			if _, err := conn.Exec(ctx, string(content)); err != nil {
				t.Fatalf("apply migration %s: %v", name, err)
			}
		}

		alter := fmt.Sprintf("ALTER ROLE ukstress_api WITH PASSWORD '%s'", APIRolePassword)
		if _, err := conn.Exec(ctx, alter); err != nil {
			t.Fatalf("set ukstress_api password: %v", err)
		}
	})
}

// appliedMigrationVersions returns the empty set on a fresh database
// (schema_migration does not exist yet, so the first migration file is
// the one that creates it) and the recorded version set otherwise.
func appliedMigrationVersions(t *testing.T, ctx context.Context, conn *pgx.Conn) map[string]bool {
	t.Helper()
	var tableExists bool
	err := conn.QueryRow(ctx, "SELECT to_regclass('public.schema_migration') IS NOT NULL").Scan(&tableExists)
	if err != nil {
		t.Fatalf("check schema_migration existence: %v", err)
	}
	applied := make(map[string]bool)
	if !tableExists {
		return applied
	}
	rows, err := conn.Query(ctx, "SELECT version FROM schema_migration")
	if err != nil {
		t.Fatalf("read applied migrations: %v", err)
	}
	defer rows.Close()
	for rows.Next() {
		var version string
		if err := rows.Scan(&version); err != nil {
			t.Fatalf("scan applied migration version: %v", err)
		}
		applied[version] = true
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("iterate applied migrations: %v", err)
	}
	return applied
}

// APIDatabaseURL rewrites ownerURL's credentials to the ukstress_api role.
func APIDatabaseURL(t *testing.T, ownerURL string) string {
	t.Helper()
	parsed, err := url.Parse(ownerURL)
	if err != nil {
		t.Fatalf("parse owner database url: %v", err)
	}
	parsed.User = url.UserPassword("ukstress_api", APIRolePassword)
	return parsed.String()
}

// SeedDataset creates one import_run, publishes it as active, and inserts
// a fixed fixture: an unambiguous lexeme ("мова") and an ambiguous group
// of two ("замок" -> за́мок / замо́к). It returns the dataset ID.
func SeedDataset(t *testing.T, ctx context.Context, ownerURL string) int64 {
	t.Helper()
	conn, err := pgx.Connect(ctx, ownerURL)
	if err != nil {
		t.Fatalf("connect as owner: %v", err)
	}
	defer conn.Close(ctx)

	var datasetID int64
	datasetKey := fmt.Sprintf("go-integration-%s-%d", t.Name(), time.Now().UnixNano())
	err = conn.QueryRow(ctx, `
		INSERT INTO import_run (dataset_key, status, dump_url, dump_sha256,
		                        parser_version, normalization_version, schema_version)
		VALUES ($1, 'published', 'https://example.invalid/dump', $2, 'test', 'ukstress-nfd-v1', '006')
		RETURNING id`,
		datasetKey, strings.Repeat("a", 64),
	).Scan(&datasetID)
	if err != nil {
		t.Fatalf("insert import_run: %v", err)
	}

	if _, err := conn.Exec(ctx, `
		INSERT INTO active_dataset (singleton, dataset_id) VALUES (true, $1)
		ON CONFLICT (singleton) DO UPDATE SET dataset_id = EXCLUDED.dataset_id`,
		datasetID,
	); err != nil {
		t.Fatalf("publish active dataset: %v", err)
	}

	insertWord := func(formNormalized, stressedForm, signature string, sourceRank int16, natural string) {
		t.Helper()
		_, err := conn.Exec(ctx, `
			WITH new_lexeme AS (
				INSERT INTO lexeme (dataset_id, lemma, lemma_normalized, stressed_lemma,
				                    source_title, confidence, natural_key)
				VALUES ($1, $2, $2, $3, $2, 1.0, $6 || '-lex')
				RETURNING id
			), new_form AS (
				INSERT INTO word_form (dataset_id, lexeme_id, form, form_normalized,
				                       is_lemma, is_variant, confidence, source_rank, natural_key)
				SELECT $1, id, $3, $2, true, false, 1.0, $4, $6 || '-form' FROM new_lexeme
				RETURNING id, lexeme_id
			)
			INSERT INTO stress_lookup (dataset_id, form_normalized, stressed_form,
			                           stress_signature, lemma_normalized, stressed_lemma,
			                           grammatical_tags, lexeme_id, word_form_id,
			                           is_lemma, is_variant, is_obsolete, confidence, source_rank)
			SELECT $1, $2, $3, $5, $2, $3, '{}', lexeme_id, id, true, false, false, 1.0, $4
			FROM new_form`,
			datasetID, formNormalized, stressedForm, sourceRank, signature, natural,
		)
		if err != nil {
			t.Fatalf("insert fixture word %s: %v", natural, err)
		}
	}

	insertWord("мова", "мо́ва", "0", 0, datasetKey+"-mova")
	insertWord("замок", "за́мок", "0", 0, datasetKey+"-zamok1")
	insertWord("замок", "замо́к", "1", 1, datasetKey+"-zamok2")

	return datasetID
}

// SwitchActiveDataset republishes datasetID as the active dataset using a
// fresh connection, for tests that simulate a live publish.
func SwitchActiveDataset(t *testing.T, ctx context.Context, ownerURL string, datasetID int64) {
	t.Helper()
	conn, err := pgx.Connect(ctx, ownerURL)
	if err != nil {
		t.Fatalf("connect as owner: %v", err)
	}
	defer conn.Close(ctx)
	if _, err := conn.Exec(ctx, `
		INSERT INTO active_dataset (singleton, dataset_id) VALUES (true, $1)
		ON CONFLICT (singleton) DO UPDATE SET dataset_id = EXCLUDED.dataset_id`, datasetID,
	); err != nil {
		t.Fatalf("switch active dataset: %v", err)
	}
}
