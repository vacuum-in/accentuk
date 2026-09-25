package repository

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestLoadTrieDefaultsMissingPathIsNotAnError(t *testing.T) {
	for _, path := range []string{"", filepath.Join(t.TempDir(), "absent.json")} {
		defaults, err := LoadTrieDefaults(path)
		if err != nil {
			t.Fatalf("path %q: unexpected error %v", path, err)
		}
		if defaults != nil {
			t.Fatalf("path %q: expected no map", path)
		}
	}
}

func TestLoadTrieDefaults(t *testing.T) {
	path := filepath.Join(t.TempDir(), "defaults.json")
	raw, err := json.Marshal(map[string]string{"гори": "1"})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	defaults, err := LoadTrieDefaults(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if defaults["гори"] != "1" {
		t.Fatalf("got %q, want %q", defaults["гори"], "1")
	}
}

func TestPreferTrieDefault(t *testing.T) {
	for _, testCase := range []struct {
		name   string
		in     []string
		wanted string
		want   []string
	}{
		{"moves the wanted reading first", []string{"0", "1", "2"}, "2", []string{"2", "0", "1"}},
		{"keeps the rest in order", []string{"0", "1", "2"}, "1", []string{"1", "0", "2"}},
		{"already first", []string{"1", "0"}, "1", []string{"1", "0"}},
		{"not offered leaves order alone", []string{"0", "1"}, "5", []string{"0", "1"}},
		{"no preference", []string{"0", "1"}, "", []string{"0", "1"}},
		{"single candidate", []string{"0"}, "0", []string{"0"}},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			got := preferTrieDefault(append([]string(nil), testCase.in...), testCase.wanted)
			if !reflect.DeepEqual(got, testCase.want) {
				t.Fatalf("got %q, want %q", got, testCase.want)
			}
		})
	}
}
