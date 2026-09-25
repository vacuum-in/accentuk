package repository

import (
	"encoding/json"
	"os"
)

// TrieDefaults names, for a form the lexicon leaves ambiguous, the single
// reading the source trie records.
//
// When several readings exist and nothing else decides — the form is outside
// the serving manifest and the parser could not resolve it — the pipeline
// serves the highest-confidence entry, and that order says nothing about which
// reading is right. The trie names exactly one reading for 53,995 of the 61,474
// ambiguous forms, and on lang-uk's benchmark it is the better default: where
// the two disagree the trie is right 26 times against the lexicon order's 12.
//
// This only reorders. The set of signatures a form offers is untouched, so the
// contextual model still chooses between exactly the same candidates.
type TrieDefaults map[string]string

// LoadTrieDefaults reads the map written by ml/scripts/run_trie_defaults.py.
// An empty path, or a missing file, yields nil and no error: the preference is
// optional and the pipeline orders candidates as before without it.
func LoadTrieDefaults(path string) (TrieDefaults, error) {
	if path == "" {
		return nil, nil
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var decoded TrieDefaults
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return nil, err
	}
	return decoded, nil
}

// SetTrieDefaults installs the preference. Safe to call with nil.
func (r *Repository) SetTrieDefaults(defaults TrieDefaults) { r.trieDefaults = defaults }

// preferTrieDefault moves the trie's reading to the front of the candidate
// list, leaving the rest in the order the query produced.
func preferTrieDefault(signatures []string, wanted string) []string {
	if wanted == "" || len(signatures) < 2 || signatures[0] == wanted {
		return signatures
	}
	for index, signature := range signatures {
		if signature != wanted {
			continue
		}
		copy(signatures[1:index+1], signatures[:index])
		signatures[0] = wanted
		return signatures
	}
	return signatures
}
