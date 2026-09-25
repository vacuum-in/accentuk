// Command stress-api serves read-only Ukrainian stress lookups over
// PostgreSQL. It contains no dump parsing, morphology extraction, import,
// or migration logic; all of that stays in the Python etl package.
package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/ukstress/ukstress/api/internal/config"
	"github.com/ukstress/ukstress/api/internal/httpapi"
	"github.com/ukstress/ukstress/api/internal/metrics"
	"github.com/ukstress/ukstress/api/internal/repository"
)

// version is overridden at build time via -ldflags "-X main.version=...".
var version = "dev"

const (
	readHeaderTimeout = 5 * time.Second
	readTimeout       = 10 * time.Second
	writeTimeout      = 10 * time.Second
	idleTimeout       = 60 * time.Second
	shutdownTimeout   = 10 * time.Second
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	if err := run(logger); err != nil {
		logger.Error("stress-api exited with error", "error", err)
		os.Exit(1)
	}
}

func run(logger *slog.Logger) error {
	cfg, err := config.FromEnv()
	if err != nil {
		return fmt.Errorf("load configuration: %w", err)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	repo, err := repository.Connect(ctx, cfg.DatabaseURL, cfg.MaxConnections, cfg.MinConnections,
		cfg.ConnectTimeout, cfg.StatementTimeout, cfg.MaxResults, cfg.InventoryHash)
	if err != nil {
		return fmt.Errorf("connect to database: %w", err)
	}
	defer repo.Close()

	refreshCtx, stopRefresh := context.WithCancel(context.Background())
	defer stopRefresh()
	// Corrections, recovered readings and gap-fill live in their own datasets
	// so they stay inspectable and reversible. Without this the API serves the
	// curated wordlist alone: measured on lang-uk's benchmark, 55.29%
	// heteronym accuracy served against 72.96% with the supplements read.
	repo.SetSupplementaryDatasets(cfg.SupplementaryDatasets)
	repo.SetReviewedDatasets(cfg.ReviewedDatasets)
	repo.SetExclusiveDatasets(cfg.ExclusiveDatasets)
	if len(cfg.ReviewedDatasets) > 0 {
		logger.Info("reviewed datasets lead the trie default", "ids", cfg.ReviewedDatasets)
	}
	if len(cfg.ExclusiveDatasets) > 0 {
		logger.Info("exclusive datasets collapse the candidate list",
			"ids", cfg.ExclusiveDatasets)
	}
	if len(cfg.SupplementaryDatasets) > 0 {
		logger.Info("supplementary datasets active", "ids", cfg.SupplementaryDatasets)
	}

	go repo.RunRefresh(refreshCtx, cfg.DatasetRefresh)

	if defaults, defaultsErr := repository.LoadTrieDefaults(cfg.TrieDefaults); defaultsErr != nil {
		logger.Error("trie defaults unreadable", "path", cfg.TrieDefaults, "error", defaultsErr)
		os.Exit(1)
	} else if defaults != nil {
		repo.SetTrieDefaults(defaults)
		logger.Info("trie defaults enabled", "forms", len(defaults))
	}

	server := httpapi.New(repo, metrics.New(), logger, version,
		cfg.MaxTokenBytes, cfg.MaxBatchSize, cfg.MaxPageSize, cfg.MaxBodyBytes)
	// A word no tier reaches is served unstressed, which is wrong for a TTS
	// caller. The table guesses from the ending at 74.5% on held-out lexicon
	// forms. A bad path is fatal: silently serving without it would make the
	// API disagree with the evaluator.
	if suffixes, suffixErr := httpapi.LoadSuffixTable(cfg.SuffixTable); suffixErr != nil {
		logger.Error("suffix table unreadable", "path", cfg.SuffixTable, "error", suffixErr)
		os.Exit(1)
	} else if suffixes != nil {
		server.SetSuffixTable(suffixes)
		logger.Info("suffix fallback enabled", "path", cfg.SuffixTable)
	}
	if cfg.ModelURL != "" {
		resolver, resolverErr := httpapi.NewHTTPContextResolverWithTimeouts(
			cfg.ModelURL, cfg.ModelManifest, cfg.InventoryHash, cfg.ModelTimeout,
			cfg.MorphologyTimeout,
		)
		if resolverErr != nil {
			// Inventory/model mismatch must not take down database-only stressing.
			logger.Error("contextual model disabled", "error", resolverErr)
		} else {
			server.SetContextResolver(resolver)
			server.SetCombiner(cfg.Combiner)
			logger.Info("learned combiner available per request", "default", cfg.Combiner)
		}
	}

	httpServer := &http.Server{
		Addr:              cfg.ListenAddress,
		Handler:           server.Handler(),
		ReadHeaderTimeout: readHeaderTimeout,
		ReadTimeout:       readTimeout,
		WriteTimeout:      writeTimeout,
		IdleTimeout:       idleTimeout,
	}

	errCh := make(chan error, 1)
	go func() {
		logger.Info("stress-api listening", "address", cfg.ListenAddress, "version", version)
		if serveErr := httpServer.ListenAndServe(); serveErr != nil && !errors.Is(serveErr, http.ErrServerClosed) {
			errCh <- serveErr
			return
		}
		errCh <- nil
	}()

	select {
	case <-ctx.Done():
		logger.Info("stress-api shutting down")
	case serveErr := <-errCh:
		if serveErr != nil {
			return fmt.Errorf("http server: %w", serveErr)
		}
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
	defer cancel()
	if err := httpServer.Shutdown(shutdownCtx); err != nil {
		return fmt.Errorf("graceful shutdown: %w", err)
	}
	return nil
}
