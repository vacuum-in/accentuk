// Package httpapi implements the public read-only HTTP lookup service.
// It validates and serializes requests only; all lookup logic lives in
// internal/repository, and all linguistic rules live in Python, per the
// project's Python/Go architecture boundary.
package httpapi

import (
	"errors"
	"log/slog"
	"net/http"

	"github.com/ukstress/ukstress/api/internal/metrics"
	"github.com/ukstress/ukstress/api/internal/repository"
)

type Server struct {
	repo     *repository.Repository
	metrics  *metrics.Metrics
	logger   *slog.Logger
	version  string
	resolver ContextResolver

	maxTokenBytes int
	maxBatchSize  int
	maxPageSize   int
	maxBodyBytes  int64

	// suffixes stresses words no other tier reaches. Nil when SUFFIX_TABLE is
	// unset, in which case those words are served unstressed as before.
	suffixes *suffixTable

	// combine is the default for requests that do not say: ask the learned
	// combiner about every ambiguous token once the tier order has answered,
	// and take its reading where it departs. A request's "combiner" wins.
	combine bool
}

func New(
	repo *repository.Repository,
	m *metrics.Metrics,
	logger *slog.Logger,
	version string,
	maxTokenBytes, maxBatchSize, maxPageSize int,
	maxBodyBytes int64,
) *Server {
	return &Server{
		repo:          repo,
		metrics:       m,
		logger:        logger,
		version:       version,
		maxTokenBytes: maxTokenBytes,
		maxBatchSize:  maxBatchSize,
		maxPageSize:   maxPageSize,
		maxBodyBytes:  maxBodyBytes,
	}
}

// SetSuffixTable installs the analogy fallback. Loading is the caller's job so
// a missing or unreadable file fails at start-up rather than per request.
func (s *Server) SetSuffixTable(table *suffixTable) { s.suffixes = table }

// LoadSuffixTable reads the table at path; an empty path or missing file yields
// a nil table and no error, leaving the pipeline exactly as it was.
func LoadSuffixTable(path string) (*suffixTable, error) { return loadSuffixTable(path) }

// Handler returns the fully wired HTTP handler, including middleware.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/lookup", s.handleLookup)
	mux.HandleFunc("POST /v1/lookup:batch", s.handleLookupBatch)
	mux.HandleFunc("POST /v1/stress", s.handleStress)
	mux.HandleFunc("GET /v1/lemmas/{lemma}/forms", s.handleLemmaForms)
	mux.HandleFunc("GET /health/live", s.handleLive)
	mux.HandleFunc("GET /health/ready", s.handleReady)
	mux.HandleFunc("GET /metrics", s.handleMetrics)
	mux.HandleFunc("GET /version", s.handleVersion)
	return s.withMiddleware(mux)
}

// SetCombiner sets whether requests that do not ask get the combiner.
func (s *Server) SetCombiner(enabled bool) { s.combine = enabled }

// SetContextResolver enables contextual resolution. With no resolver the
// endpoint remains database-only and preserves ambiguous tokens.
func (s *Server) SetContextResolver(resolver ContextResolver) { s.resolver = resolver }

// handleRepositoryError maps repository failures to the stable
// service-unavailable contract without leaking SQL or connection details.
func (s *Server) handleRepositoryError(w http.ResponseWriter, r *http.Request, err error) {
	requestID := requestIDFromContext(r.Context())
	if errors.Is(err, repository.ErrNoActiveDataset) {
		s.logger.Warn("lookup attempted with no active dataset", "request_id", requestID)
		writeError(w, r, http.StatusServiceUnavailable, "no_active_dataset",
			"no dataset is currently published")
		return
	}
	s.logger.Error("repository error", "request_id", requestID, "error", err)
	writeError(w, r, http.StatusServiceUnavailable, "database_unavailable",
		"the lookup service is temporarily unavailable")
}
