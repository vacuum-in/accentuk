package normalize

import (
	"encoding/json"
	"os"
	"testing"
)

// conformanceVectorsPath points at the Python-generated conformance
// artifact. Go must never define its own normalization rules; this test is
// the mechanism that catches drift between the two implementations.
const conformanceVectorsPath = "../../../artifacts/normalization-conformance.json"

type conformanceVector struct {
	Input                 string `json:"input"`
	CanonicalStressedForm string `json:"canonical_stressed_form"`
	LookupKey             string `json:"lookup_key"`
}

type conformanceFile struct {
	Version string              `json:"version"`
	Vectors []conformanceVector `json:"vectors"`
}

func TestConformanceVectors(t *testing.T) {
	data, err := os.ReadFile(conformanceVectorsPath)
	if err != nil {
		t.Fatalf("read conformance vectors: %v", err)
	}
	var file conformanceFile
	if err := json.Unmarshal(data, &file); err != nil {
		t.Fatalf("parse conformance vectors: %v", err)
	}
	if len(file.Vectors) == 0 {
		t.Fatal("no conformance vectors found")
	}
	for _, vector := range file.Vectors {
		if got := CanonicalStressedForm(vector.Input); got != vector.CanonicalStressedForm {
			t.Errorf("CanonicalStressedForm(%q) = %q, want %q",
				vector.Input, got, vector.CanonicalStressedForm)
		}
		if got := LookupKey(vector.Input); got != vector.LookupKey {
			t.Errorf("LookupKey(%q) = %q, want %q", vector.Input, got, vector.LookupKey)
		}
	}
}

func TestHyphenAtBoundaryIsNotConverted(t *testing.T) {
	// A hyphen-variant at the very start or end of the string has no
	// preceding/following non-space character, so it must not be converted
	// (matches the (?<=\S)...(?=\S) lookaround in the Python implementation).
	input := "—half"
	if got := CanonicalStressedForm(input); got != input {
		t.Errorf("CanonicalStressedForm(%q) = %q, want unchanged", input, got)
	}
}

func TestLookupKeyIsIdempotent(t *testing.T) {
	input := "мова"
	once := LookupKey(input)
	twice := LookupKey(once)
	if once != twice {
		t.Errorf("LookupKey is not idempotent: %q -> %q -> %q", input, once, twice)
	}
}
