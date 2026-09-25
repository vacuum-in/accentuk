package httpapi

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/ukstress/ukstress/api/internal/normalize"
	"github.com/ukstress/ukstress/api/internal/repository"
)

func validMode(mode string) (string, bool) {
	if mode == "" {
		return "exact", true
	}
	if mode == "exact" || mode == "best" {
		return mode, true
	}
	return "", false
}

// handleLookup implements GET /v1/lookup?word=...&mode=exact|best.
func (s *Server) handleLookup(w http.ResponseWriter, r *http.Request) {
	word := r.URL.Query().Get("word")
	mode, ok := validMode(r.URL.Query().Get("mode"))
	if !ok {
		writeError(w, r, http.StatusBadRequest, "invalid_mode", "mode must be 'exact' or 'best'")
		return
	}
	if err := validateToken(word, s.maxTokenBytes); err != nil {
		writeError(w, r, http.StatusBadRequest, "invalid_word", err.Error())
		return
	}

	normalized := normalize.LookupKey(word)
	result, err := s.repo.Lookup(r.Context(), word, normalized, mode)
	if err != nil {
		s.handleRepositoryError(w, r, err)
		return
	}
	s.metrics.RecordLookup(result.Status, result.Truncated)
	writeJSON(w, http.StatusOK, result)
}

type batchRequest struct {
	Words []string `json:"words"`
	Mode  string   `json:"mode"`
}

type batchResponse struct {
	Results   []repository.LookupResult `json:"results"`
	DatasetID int64                     `json:"dataset_id"`
}

// handleLookupBatch implements POST /v1/lookup:batch. Input order and
// duplicates are preserved; the whole batch is resolved in one
// set-oriented PostgreSQL query rather than one round trip per word.
func (s *Server) handleLookupBatch(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, s.maxBodyBytes)

	var req batchRequest
	decoder := json.NewDecoder(r.Body)
	if err := decoder.Decode(&req); err != nil {
		writeError(w, r, http.StatusBadRequest, "invalid_body", "request body must be valid JSON")
		return
	}

	mode, ok := validMode(req.Mode)
	if !ok {
		writeError(w, r, http.StatusBadRequest, "invalid_mode", "mode must be 'exact' or 'best'")
		return
	}
	if len(req.Words) == 0 {
		writeError(w, r, http.StatusBadRequest, "empty_batch", "words must not be empty")
		return
	}
	if len(req.Words) > s.maxBatchSize {
		writeError(w, r, http.StatusBadRequest, "batch_too_large",
			fmt.Sprintf("batch exceeds the maximum of %d words", s.maxBatchSize))
		return
	}

	normalized := make([]string, len(req.Words))
	for i, word := range req.Words {
		if err := validateToken(word, s.maxTokenBytes); err != nil {
			writeError(w, r, http.StatusBadRequest, "invalid_word",
				fmt.Sprintf("word at index %d: %s", i, err.Error()))
			return
		}
		normalized[i] = normalize.LookupKey(word)
	}

	results, err := s.repo.BatchLookup(r.Context(), req.Words, normalized)
	if err != nil {
		s.handleRepositoryError(w, r, err)
		return
	}

	if mode == "best" {
		for i := range results {
			if len(results[i].Candidates) > 1 {
				results[i].Candidates = results[i].Candidates[:1]
			}
			s.metrics.RecordLookup(results[i].Status, results[i].Truncated)
		}
	} else {
		for i := range results {
			s.metrics.RecordLookup(results[i].Status, results[i].Truncated)
		}
	}

	datasetID, err := s.repo.ActiveDataset()
	if err != nil {
		s.handleRepositoryError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, batchResponse{Results: results, DatasetID: datasetID})
}
