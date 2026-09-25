package httpapi

import (
	"net/http"
	"strconv"

	"github.com/ukstress/ukstress/api/internal/normalize"
	"github.com/ukstress/ukstress/api/internal/repository"
)

type lemmaFormsResponse struct {
	Lemma  string                 `json:"lemma"`
	Forms  []repository.LemmaForm `json:"forms"`
	Offset int                    `json:"offset"`
	Limit  int                    `json:"limit"`
}

func parsePagination(r *http.Request, maxPageSize int) (offset, limit int, err error) {
	offset = 0
	limit = maxPageSize

	if raw := r.URL.Query().Get("offset"); raw != "" {
		offset, err = strconv.Atoi(raw)
		if err != nil || offset < 0 {
			return 0, 0, errInvalidOffset
		}
	}
	if raw := r.URL.Query().Get("limit"); raw != "" {
		limit, err = strconv.Atoi(raw)
		if err != nil || limit < 1 {
			return 0, 0, errInvalidLimit
		}
	}
	if limit > maxPageSize {
		limit = maxPageSize
	}
	return offset, limit, nil
}

// handleLemmaForms implements GET /v1/lemmas/{lemma}/forms?offset=&limit=.
func (s *Server) handleLemmaForms(w http.ResponseWriter, r *http.Request) {
	lemma := r.PathValue("lemma")
	if err := validateToken(lemma, s.maxTokenBytes); err != nil {
		writeError(w, r, http.StatusBadRequest, "invalid_lemma", err.Error())
		return
	}
	offset, limit, err := parsePagination(r, s.maxPageSize)
	if err != nil {
		writeError(w, r, http.StatusBadRequest, "invalid_pagination", err.Error())
		return
	}

	normalized := normalize.LookupKey(lemma)
	forms, err := s.repo.LemmaForms(r.Context(), normalized, offset, limit)
	if err != nil {
		s.handleRepositoryError(w, r, err)
		return
	}

	writeJSON(w, http.StatusOK, lemmaFormsResponse{
		Lemma: lemma, Forms: forms, Offset: offset, Limit: limit,
	})
}
