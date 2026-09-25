package httpapi

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func writeTable(t *testing.T, entries map[string]int) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "suffix.json")
	raw, err := json.Marshal(entries)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	return path
}

func TestLoadSuffixTableMissingPathIsNotAnError(t *testing.T) {
	for _, path := range []string{"", filepath.Join(t.TempDir(), "absent.json")} {
		table, err := loadSuffixTable(path)
		if err != nil {
			t.Fatalf("path %q: unexpected error %v", path, err)
		}
		if table != nil {
			t.Fatalf("path %q: expected no table", path)
		}
		if got := table.stress("будинок"); got != "" {
			t.Fatalf("nil table answered %q", got)
		}
	}
}

func TestSuffixTableStress(t *testing.T) {
	// Keys are NFD-decomposed lowercase word endings; values are the stressed
	// vowel's distance from the last vowel. NFD matters: `й` is stored as
	// `и` plus a combining breve, so a key written composed never matches.
	path := writeTable(t, map[string]int{"рти": 1, "а": 0, "ний": 0})
	table, err := loadSuffixTable(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	for _, testCase := range []struct {
		name, word, want string
	}{
		{"longest suffix wins", "натюрморти", "2"},
		// `й` occupies a vowel ordinal, as the lexicon signatures count it, so
		// натюрмортний has five: а, ю, о, и and the и inside й.
		{"decomposed key matches a composed word", "натюрмортний", "4"},
		{"single vowel is left alone", "дім", ""},
		{"unknown ending", "будинок", ""},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			if got := table.stress(testCase.word); got != testCase.want {
				t.Fatalf("stress(%q) = %q, want %q", testCase.word, got, testCase.want)
			}
		})
	}
}

func TestApplySuffixFallbackOnlyFillsNotFound(t *testing.T) {
	path := writeTable(t, map[string]int{"рти": 1})
	table, err := loadSuffixTable(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	work := []tokenWork{
		{stressToken: stressToken{Text: "натюрморти", Status: "not_found"}},
		{stressToken: stressToken{Text: "натюрморти", Status: "found", OutputText: "kept"}},
	}
	applySuffixFallback(work, table, func(surface, signature string) (string, error) {
		return surface + "/" + signature, nil
	})
	if work[0].Status != "suffix" || work[0].OutputText != "натюрморти/2" {
		t.Fatalf("not_found token: status %q text %q", work[0].Status, work[0].OutputText)
	}
	if work[1].Status != "found" || work[1].OutputText != "kept" {
		t.Fatalf("a decided token must not be overwritten: %+v", work[1].stressToken)
	}
}

func TestPositionalDefaultStressesPenultimateVowel(t *testing.T) {
	work := []tokenWork{
		{stressToken: stressToken{Text: "сент-кіттс", Status: "not_found"}},
		{stressToken: stressToken{Text: "нью", Status: "not_found"}},
		{stressToken: stressToken{Text: "мова", Status: "stressed", OutputText: "мо́ва"}},
	}
	applyPositionalDefault(work, applySignature)
	if work[0].Status != "default_position" || work[0].OutputText != "сент-кі́ттс" {
		t.Fatalf("expected the last segment of сент-кіттс stressed, got %q %q", work[0].Status, work[0].OutputText)
	}
	if work[1].Status != "not_found" {
		t.Fatalf("a one-vowel word must stay unanswered, got %q", work[1].Status)
	}
	if work[2].OutputText != "мо́ва" {
		t.Fatalf("a decided word must not be touched, got %q", work[2].OutputText)
	}
}
