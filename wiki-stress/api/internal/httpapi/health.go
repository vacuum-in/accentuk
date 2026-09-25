package httpapi

import (
	"net/http"
	"runtime"
)

// handleLive implements GET /health/live. Liveness stays healthy even with
// no active dataset — the process is up, whether or not there is
// currently anything to serve.
func (s *Server) handleLive(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "live"})
}

// handleReady implements GET /health/ready. Readiness fails when
// PostgreSQL is unreachable or no dataset is published. Healthy() and
// ActiveDataset() are checked separately: a connection failure must not
// be masked by a dataset ID cached from before PostgreSQL became
// unreachable.
func (s *Server) handleReady(w http.ResponseWriter, r *http.Request) {
	if !s.repo.Healthy() {
		writeError(w, r, http.StatusServiceUnavailable, "not_ready", "database is unreachable")
		return
	}
	if _, err := s.repo.ActiveDataset(); err != nil {
		writeError(w, r, http.StatusServiceUnavailable, "not_ready", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

func (s *Server) handleVersion(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{
		"version":    s.version,
		"go_version": runtime.Version(),
	})
}

func (s *Server) handleMetrics(w http.ResponseWriter, _ *http.Request) {
	stats := s.repo.PoolStats()
	s.metrics.SetPoolStats(stats.AcquiredConns(), stats.IdleConns(), stats.MaxConns())
	if id, err := s.repo.ActiveDataset(); err == nil {
		s.metrics.SetActiveDataset(id)
	}
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	s.metrics.Render(w)
}
