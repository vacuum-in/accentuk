// Package config contains validated API configuration parsing.
package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

type Config struct {
	DatabaseURL      string
	ListenAddress    string
	MaxConnections   int32
	MinConnections   int32
	ConnectTimeout   time.Duration
	StatementTimeout time.Duration
	DatasetRefresh   time.Duration
	MaxTokenBytes    int
	MaxBatchSize     int
	MaxResults       int
	MaxPageSize      int
	MaxBodyBytes     int64
	ModelURL         string
	ModelManifest    string
	InventoryHash    string
	ModelTimeout     time.Duration
	// MorphologyTimeout is separate from ModelTimeout on purpose: a batched
	// cross-encoder pass and a morphological parse have different latency
	// profiles, and holding the parser to the model's 2s budget made every
	// morphology call time out and degrade silently to a dictionary default.
	MorphologyTimeout time.Duration

	// SupplementaryDatasets are read alongside the active dataset.
	//
	// The lexicon is layered by design: corrections, trie-recovered second
	// readings and LLM gap-fill each live in their own dataset at a confidence
	// below the curated wordlist's, so they can be inspected and rolled back
	// independently. The evaluation harness has always read them
	// (`run_gold_eval.py --extra-dataset`); the API never did, so measured
	// heteronym accuracy was 72.96% against 55.29% served.
	SupplementaryDatasets []int64
	// ReviewedDatasets hold readings a person decided. They outrank the
	// trie's statistical default; see Repository.SetReviewedDatasets.
	ReviewedDatasets []int64
	// ExclusiveDatasets hold reviewed readings that are the only admissible
	// one, so the form is served with a single candidate.
	ExclusiveDatasets []int64

	// SuffixTable is the analogy fallback written by
	// ml/scripts/run_suffix_fallback.py. Empty leaves the fallback off.
	SuffixTable string
	// Combiner lets the learned combiner overrule the tier order's answer for
	// ambiguous tokens (COMBINER_ENABLED). The model service must load it.
	Combiner bool

	// TrieDefaults orders ambiguous candidates; see
	// repository.TrieDefaults for the measurement behind it.
	TrieDefaults string
}

func FromEnv() (Config, error) {
	cfg := Config{
		DatabaseURL:           os.Getenv("DATABASE_URL"),
		ListenAddress:         envString("LISTEN_ADDRESS", ":8080"),
		MaxConnections:        int32(envInt("DB_MAX_CONNECTIONS", 10)),
		MinConnections:        int32(envInt("DB_MIN_CONNECTIONS", 1)),
		ConnectTimeout:        envDuration("DB_CONNECT_TIMEOUT", 5*time.Second),
		StatementTimeout:      envDuration("DB_STATEMENT_TIMEOUT", 2*time.Second),
		DatasetRefresh:        envDuration("DATASET_REFRESH_INTERVAL", 5*time.Second),
		MaxTokenBytes:         envInt("MAX_TOKEN_BYTES", 256),
		MaxBatchSize:          envInt("MAX_BATCH_SIZE", 10_000),
		MaxResults:            envInt("MAX_RESULTS", 50),
		MaxPageSize:           envInt("MAX_PAGE_SIZE", 500),
		MaxBodyBytes:          int64(envInt("MAX_BODY_BYTES", 2_000_000)),
		ModelURL:              os.Getenv("MODEL_URL"),
		ModelManifest:         os.Getenv("MODEL_MANIFEST"),
		InventoryHash:         os.Getenv("ACTIVE_INVENTORY_HASH"),
		ModelTimeout:          envDuration("MODEL_TIMEOUT", 2*time.Second),
		MorphologyTimeout:     envDuration("MORPHOLOGY_TIMEOUT", 20*time.Second),
		SupplementaryDatasets: envIntList("SUPPLEMENTARY_DATASETS"),
		ReviewedDatasets:      envIntList("REVIEWED_DATASETS"),
		ExclusiveDatasets:     envIntList("REVIEWED_EXCLUSIVE_DATASETS"),
		SuffixTable:           os.Getenv("SUFFIX_TABLE"),
		Combiner:              os.Getenv("COMBINER_ENABLED") == "1" || os.Getenv("COMBINER_ENABLED") == "true",
		TrieDefaults:          os.Getenv("TRIE_DEFAULTS"),
	}
	if cfg.DatabaseURL == "" {
		return Config{}, fmt.Errorf("DATABASE_URL is required")
	}
	if cfg.MaxConnections < 1 || cfg.MinConnections < 0 ||
		cfg.MinConnections > cfg.MaxConnections {
		return Config{}, fmt.Errorf("database connection bounds are invalid")
	}
	if cfg.ConnectTimeout <= 0 || cfg.StatementTimeout <= 0 || cfg.DatasetRefresh <= 0 ||
		cfg.ModelTimeout <= 0 || cfg.MorphologyTimeout <= 0 {
		return Config{}, fmt.Errorf("timeouts and refresh interval must be positive")
	}
	if (cfg.ModelURL == "") != (cfg.ModelManifest == "") {
		return Config{}, fmt.Errorf("MODEL_URL and MODEL_MANIFEST must be configured together")
	}
	if cfg.MaxTokenBytes < 1 || cfg.MaxBatchSize < 1 || cfg.MaxResults < 1 ||
		cfg.MaxPageSize < 1 || cfg.MaxBodyBytes < 1 {
		return Config{}, fmt.Errorf("request and response bounds must be positive")
	}
	return cfg, nil
}

// envIntList parses a comma-separated list of dataset ids; empty means none.
func envIntList(key string) []int64 {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return nil
	}
	var out []int64
	for _, part := range strings.Split(raw, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		value, err := strconv.ParseInt(part, 10, 64)
		if err != nil {
			continue
		}
		out = append(out, value)
	}
	return out
}

func envString(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}

func envInt(name string, fallback int) int {
	value := os.Getenv(name)
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return -1
	}
	return parsed
}

func envDuration(name string, fallback time.Duration) time.Duration {
	value := os.Getenv(name)
	if value == "" {
		return fallback
	}
	parsed, err := time.ParseDuration(value)
	if err != nil {
		return -1
	}
	return parsed
}
