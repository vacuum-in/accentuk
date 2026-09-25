package config

import (
	"testing"
	"time"
)

func TestFromEnvDefaultsAndValidation(t *testing.T) {
	t.Setenv("DATABASE_URL", "postgresql://reader@localhost/dictionary")
	cfg, err := FromEnv()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.MaxBatchSize != 10_000 || cfg.ListenAddress != ":8080" {
		t.Fatalf("unexpected defaults: %#v", cfg)
	}
}

func TestFromEnvRejectsMissingDatabaseURL(t *testing.T) {
	t.Setenv("DATABASE_URL", "")
	if _, err := FromEnv(); err == nil {
		t.Fatal("missing URL was accepted")
	}
}

func TestFromEnvRejectsInvalidBounds(t *testing.T) {
	t.Setenv("DATABASE_URL", "postgresql://unused")
	t.Setenv("DB_MIN_CONNECTIONS", "3")
	t.Setenv("DB_MAX_CONNECTIONS", "2")
	if _, err := FromEnv(); err == nil {
		t.Fatal("invalid pool bounds were accepted")
	}
}

// MorphologyTimeout was declared and passed to the resolver but never read
// from the environment, so `MORPHOLOGY_TIMEOUT` did nothing and the parser
// ran with http.Client{Timeout: 0} — no deadline at all.
func TestFromEnvReadsMorphologyTimeout(t *testing.T) {
	t.Setenv("DATABASE_URL", "postgresql://unused")
	cfg, err := FromEnv()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.MorphologyTimeout != 20*time.Second {
		t.Fatalf("default morphology timeout = %v, want 20s", cfg.MorphologyTimeout)
	}
	t.Setenv("MORPHOLOGY_TIMEOUT", "45s")
	cfg, err = FromEnv()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.MorphologyTimeout != 45*time.Second {
		t.Fatalf("configured morphology timeout = %v, want 45s", cfg.MorphologyTimeout)
	}
}

func TestFromEnvRejectsUnparsableMorphologyTimeout(t *testing.T) {
	t.Setenv("DATABASE_URL", "postgresql://unused")
	t.Setenv("MORPHOLOGY_TIMEOUT", "twenty seconds")
	if _, err := FromEnv(); err == nil {
		t.Fatal("unparsable morphology timeout was accepted")
	}
}
