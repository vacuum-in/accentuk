package httpapi

import (
	"encoding/json"
	"net/http"
)

// apiError is the stable error contract: a code clients can branch on, a
// human-readable message, and a request ID for support correlation. It
// never carries SQL text, credentials, or a stack trace.
type apiError struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	RequestID string `json:"request_id"`
}

func writeError(w http.ResponseWriter, r *http.Request, status int, code, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(apiError{
		Code:      code,
		Message:   message,
		RequestID: requestIDFromContext(r.Context()),
	})
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}
