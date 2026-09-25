package httpapi

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"net/http"
	"time"
)

type contextKey int

const requestIDKey contextKey = iota

func newRequestID() string {
	buf := make([]byte, 8)
	// crypto/rand.Read never fails on supported platforms; a zeroed buffer
	// degrades to a duplicate-prone but still-present request ID rather
	// than panicking.
	_, _ = rand.Read(buf)
	return hex.EncodeToString(buf)
}

func requestIDFromContext(ctx context.Context) string {
	if id, ok := ctx.Value(requestIDKey).(string); ok {
		return id
	}
	return ""
}

type statusRecorder struct {
	http.ResponseWriter
	status      int
	wroteHeader bool
}

func (rec *statusRecorder) WriteHeader(status int) {
	if rec.wroteHeader {
		return
	}
	rec.status = status
	rec.wroteHeader = true
	rec.ResponseWriter.WriteHeader(status)
}

func (rec *statusRecorder) Write(b []byte) (int, error) {
	if !rec.wroteHeader {
		rec.WriteHeader(http.StatusOK)
	}
	return rec.ResponseWriter.Write(b)
}

// withMiddleware wraps mux with request ID assignment, panic recovery,
// structured access logging, and per-request metrics. It never logs raw
// query parameters or request bodies, so lookup words never reach logs.
func (s *Server) withMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestID := newRequestID()
		r = r.WithContext(context.WithValue(r.Context(), requestIDKey, requestID))
		w.Header().Set("X-Request-Id", requestID)

		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		start := time.Now()

		defer func() {
			if rerr := recover(); rerr != nil {
				s.logger.Error("panic recovered",
					"request_id", requestID, "panic", rerr)
				if !rec.wroteHeader {
					writeError(rec, r, http.StatusInternalServerError,
						"internal_error", "an unexpected error occurred")
				}
			}
			duration := time.Since(start)
			// r.Pattern is set by ServeMux after routing and already
			// includes the method verb for method-specific patterns
			// (e.g. "GET /v1/lookup"); it is empty for unmatched routes.
			// A fixed placeholder (not the raw path) keeps the metric
			// label set bounded regardless of what a client requests.
			endpoint := r.Pattern
			if endpoint == "" {
				endpoint = "unmatched"
			}
			s.metrics.RecordRequest(endpoint, rec.status)
			s.logger.Info("request handled",
				"request_id", requestID,
				"method", r.Method,
				"path", r.URL.Path,
				"status", rec.status,
				"duration_ms", duration.Milliseconds())
		}()

		next.ServeHTTP(rec, r)
	})
}
