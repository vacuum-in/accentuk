package httpapi

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func testResolver(t *testing.T, handler http.HandlerFunc) *HTTPContextResolver {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	manifest := map[string]any{
		"model_version": "v3-xenc", "inventory_hash": "inventory", "threshold": 1.25,
		"forms": map[string]any{"замок": map[string]any{"signatures": []string{"0", "1"}}},
	}
	content, _ := json.Marshal(manifest)
	path := filepath.Join(t.TempDir(), "manifest.json")
	if err := os.WriteFile(path, content, 0o600); err != nil {
		t.Fatal(err)
	}
	resolver, err := NewHTTPContextResolver(server.URL, path, "inventory", time.Second)
	if err != nil {
		t.Fatal(err)
	}
	return resolver
}

func TestResolverRejectsCandidateMismatchWithoutInference(t *testing.T) {
	called := false
	resolver := testResolver(t, func(http.ResponseWriter, *http.Request) { called = true })
	decisions, err := resolver.Resolve(t.Context(), []ModelTarget{{
		Form: "замок", Candidates: []string{"0", "2"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if called || decisions[0].Status != "candidate_mismatch" {
		t.Fatalf("called=%v decisions=%+v", called, decisions)
	}
}

func TestResolverAppliesCalibratedMargin(t *testing.T) {
	resolver := testResolver(t, func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model_version": "v3-xenc", "inventory_hash": "inventory",
			"decisions": []map[string]any{{"index": 0, "signature": "1", "margin": 1.0,
				"scores": map[string]float64{"0": 0.0, "1": 1.0}}},
		})
	})
	decisions, err := resolver.Resolve(t.Context(), []ModelTarget{{
		Form: "замок", Candidates: []string{"1", "0"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if decisions[0].Status != "low_margin" {
		t.Fatalf("decision = %+v, want low_margin", decisions[0])
	}
}
