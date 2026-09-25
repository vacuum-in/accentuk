// Contract tests exercise the server over real HTTP and decode responses
// into structs matching openapi.yaml, so a shape drift between the
// handler and the documented contract fails a test rather than surfacing
// only in a client's parser. They share PostgreSQL setup with
// internal/repository's integration tests via internal/testdb, and are
// skipped under the same condition (UKSTRESS_TEST_DATABASE_URL unset).
package httpapi_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ukstress/ukstress/api/internal/httpapi"
	"github.com/ukstress/ukstress/api/internal/metrics"
	"github.com/ukstress/ukstress/api/internal/repository"
	"github.com/ukstress/ukstress/api/internal/testdb"
)

const (
	testMaxTokenBytes = 256
	testMaxBatchSize  = 10_000
	testMaxPageSize   = 500
	testMaxBodyBytes  = 2_000_000
)

// candidate and lookupResult mirror the Candidate/LookupResult schemas in
// openapi.yaml field-for-field, independent of repository's own struct
// tags, so this test actually proves the wire contract rather than
// re-decoding into the same type the handler already used to encode it.
type candidate struct {
	StressedForm    string   `json:"stressed_form"`
	StressSignature string   `json:"stress_signature"`
	Lemma           string   `json:"lemma"`
	StressedLemma   *string  `json:"stressed_lemma"`
	PartOfSpeech    *string  `json:"part_of_speech"`
	GrammaticalTags []string `json:"grammatical_tags"`
	IsLemma         bool     `json:"is_lemma"`
	IsVariant       bool     `json:"is_variant"`
	IsObsolete      bool     `json:"is_obsolete"`
	Confidence      float64  `json:"confidence"`
}

type lookupResult struct {
	Input      string      `json:"input"`
	Status     string      `json:"status"`
	Candidates []candidate `json:"candidates"`
	Ambiguous  bool        `json:"ambiguous"`
	Truncated  bool        `json:"truncated"`
	DatasetID  int64       `json:"dataset_id"`
}

type batchResponse struct {
	Results   []lookupResult `json:"results"`
	DatasetID int64          `json:"dataset_id"`
}

type lemmaForm struct {
	Form            string   `json:"form"`
	StressedForm    string   `json:"stressed_form"`
	GrammaticalTags []string `json:"grammatical_tags"`
	Confidence      float64  `json:"confidence"`
}

type lemmaFormsResponse struct {
	Lemma  string      `json:"lemma"`
	Forms  []lemmaForm `json:"forms"`
	Offset int         `json:"offset"`
	Limit  int         `json:"limit"`
}

type apiError struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	RequestID string `json:"request_id"`
}

func newTestServer(t *testing.T) (base string, datasetID int64) {
	t.Helper()
	ctx := context.Background()
	ownerURL := testdb.OwnerDatabaseURL(t)
	testdb.EnsureMigrated(t, ctx, ownerURL)
	datasetID = testdb.SeedDataset(t, ctx, ownerURL)

	repo, err := repository.Connect(ctx, testdb.APIDatabaseURL(t, ownerURL),
		5, 1, 5*time.Second, 2*time.Second, 50, "")
	if err != nil {
		t.Fatalf("connect repository: %v", err)
	}
	t.Cleanup(repo.Close)

	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	server := httpapi.New(repo, metrics.New(), logger, "contract-test",
		testMaxTokenBytes, testMaxBatchSize, testMaxPageSize, testMaxBodyBytes)

	httpServer := httptest.NewServer(server.Handler())
	t.Cleanup(httpServer.Close)
	return httpServer.URL, datasetID
}

func decodeJSON[T any](t *testing.T, resp *http.Response) T {
	t.Helper()
	defer resp.Body.Close()
	var value T
	if err := json.NewDecoder(resp.Body).Decode(&value); err != nil {
		t.Fatalf("decode response body: %v", err)
	}
	return value
}

func TestContractLookupEndpoint(t *testing.T) {
	base, datasetID := newTestServer(t)

	resp, err := http.Get(base + "/v1/lookup?word=" + "мова") //nolint:noctx
	if err != nil {
		t.Fatalf("GET /v1/lookup: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /v1/lookup status = %d, want 200", resp.StatusCode)
	}
	if id := resp.Header.Get("X-Request-Id"); id == "" {
		t.Error("response missing X-Request-Id header")
	}
	result := decodeJSON[lookupResult](t, resp)
	if result.Status != "found" || len(result.Candidates) != 1 {
		t.Errorf("lookup result = %+v, want one found candidate", result)
	}
	if result.DatasetID != datasetID {
		t.Errorf("lookup result.DatasetID = %d, want %d", result.DatasetID, datasetID)
	}
	if result.Candidates[0].StressedForm != "мо́ва" {
		t.Errorf("candidate.StressedForm = %q, want мо́ва", result.Candidates[0].StressedForm)
	}
}

func TestContractLookupAmbiguousAndBestMode(t *testing.T) {
	base, _ := newTestServer(t)

	resp, err := http.Get(base + "/v1/lookup?word=замок") //nolint:noctx
	if err != nil {
		t.Fatalf("GET /v1/lookup: %v", err)
	}
	ambiguous := decodeJSON[lookupResult](t, resp)
	if ambiguous.Status != "ambiguous" || len(ambiguous.Candidates) != 2 || !ambiguous.Ambiguous {
		t.Errorf("ambiguous lookup = %+v, want two ambiguous candidates", ambiguous)
	}

	resp, err = http.Get(base + "/v1/lookup?word=замок&mode=best") //nolint:noctx
	if err != nil {
		t.Fatalf("GET /v1/lookup?mode=best: %v", err)
	}
	best := decodeJSON[lookupResult](t, resp)
	if len(best.Candidates) != 1 || !best.Ambiguous {
		t.Errorf("best-mode lookup = %+v, want one candidate with ambiguous still true", best)
	}
}

func TestContractLookupBatchEndpoint(t *testing.T) {
	base, datasetID := newTestServer(t)

	body, _ := json.Marshal(map[string]any{"words": []string{"мова", "замок", "мова", "нема"}})
	resp, err := http.Post(base+"/v1/lookup:batch", "application/json", bytes.NewReader(body)) //nolint:noctx
	if err != nil {
		t.Fatalf("POST /v1/lookup:batch: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("POST /v1/lookup:batch status = %d, want 200", resp.StatusCode)
	}
	batch := decodeJSON[batchResponse](t, resp)
	if batch.DatasetID != datasetID {
		t.Errorf("batch.DatasetID = %d, want %d", batch.DatasetID, datasetID)
	}
	if len(batch.Results) != 4 {
		t.Fatalf("batch.Results has %d entries, want 4", len(batch.Results))
	}
	wantInputs := []string{"мова", "замок", "мова", "нема"}
	for i, want := range wantInputs {
		if batch.Results[i].Input != want {
			t.Errorf("batch.Results[%d].Input = %q, want %q (order/duplicates not preserved)",
				i, batch.Results[i].Input, want)
		}
	}
}

func TestContractStressEndpointDatabaseOnly(t *testing.T) {
	base, datasetID := newTestServer(t)
	body, _ := json.Marshal(map[string]string{"text": "Мова, стіл, замок і мо́ва."})
	resp, err := http.Post(base+"/v1/stress", "application/json", bytes.NewReader(body)) //nolint:noctx
	if err != nil {
		t.Fatalf("POST /v1/stress: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("POST /v1/stress status = %d, want 200", resp.StatusCode)
	}
	var result struct {
		Text      string `json:"text"`
		DatasetID int64  `json:"dataset_id"`
		Tokens    []struct {
			Text   string `json:"text"`
			Status string `json:"status"`
			Start  int    `json:"start"`
			End    int    `json:"end"`
		} `json:"tokens"`
	}
	defer resp.Body.Close()
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}
	if result.DatasetID != datasetID || result.Text != "Мо́ва, стіл, замок і мо́ва." {
		t.Errorf("stress response = %+v", result)
	}
	wantStatuses := []string{"stressed", "not_required", "model_ineligible", "not_required", "already_stressed"}
	if len(result.Tokens) != len(wantStatuses) {
		t.Fatalf("got %d tokens, want %d", len(result.Tokens), len(wantStatuses))
	}
	for i, status := range wantStatuses {
		if result.Tokens[i].Status != status {
			t.Errorf("token %d status = %q, want %q", i, result.Tokens[i].Status, status)
		}
	}
}

func TestContractLemmaFormsEndpoint(t *testing.T) {
	base, _ := newTestServer(t)

	resp, err := http.Get(base + "/v1/lemmas/замок/forms?limit=1") //nolint:noctx
	if err != nil {
		t.Fatalf("GET /v1/lemmas/{lemma}/forms: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /v1/lemmas/{lemma}/forms status = %d, want 200", resp.StatusCode)
	}
	page := decodeJSON[lemmaFormsResponse](t, resp)
	if page.Lemma != "замок" || len(page.Forms) != 1 || page.Limit != 1 {
		t.Errorf("lemma forms page = %+v, want one row respecting limit=1", page)
	}
}

func TestContractHealthAndVersionEndpoints(t *testing.T) {
	base, _ := newTestServer(t)

	for _, path := range []string{"/health/live", "/health/ready", "/version"} {
		resp, err := http.Get(base + path) //nolint:noctx
		if err != nil {
			t.Fatalf("GET %s: %v", path, err)
		}
		if resp.StatusCode != http.StatusOK {
			t.Errorf("GET %s status = %d, want 200", path, resp.StatusCode)
		}
		resp.Body.Close()
	}

	resp, err := http.Get(base + "/metrics") //nolint:noctx
	if err != nil {
		t.Fatalf("GET /metrics: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("GET /metrics status = %d, want 200", resp.StatusCode)
	}
	contentType := resp.Header.Get("Content-Type")
	if contentType == "" {
		t.Error("GET /metrics returned no Content-Type")
	}
}

func TestContractErrorShape(t *testing.T) {
	base, _ := newTestServer(t)

	resp, err := http.Get(base + "/v1/lookup?word=мова&mode=bogus") //nolint:noctx
	if err != nil {
		t.Fatalf("GET /v1/lookup: %v", err)
	}
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}
	apiErr := decodeJSON[apiError](t, resp)
	if apiErr.Code == "" || apiErr.Message == "" || apiErr.RequestID == "" {
		t.Errorf("error body = %+v, want all of code/message/request_id populated", apiErr)
	}
	// The stable error contract must never leak SQL, connection strings,
	// or Go internals into a client-facing message.
	for _, leak := range []string{"SELECT", "pgx", "postgres://", ".go:"} {
		if strings.Contains(apiErr.Message, leak) {
			t.Errorf("error message leaks internal detail %q: %q", leak, apiErr.Message)
		}
	}
}
