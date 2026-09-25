package httpapi

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// suffixTable guesses the stress of a word no tier covers, by analogy with the
// endings of words the lexicon does hold.
//
// A multi-vowel word emitted with no stress is simply wrong — for a TTS caller
// as much as for lang-uk's Rule 4 — and 2,342 of the benchmark's 12,228 tokens
// were reaching output bare. Ukrainian stress is largely carried by the ending,
// so the table is keyed on the word's final characters and stores the stressed
// vowel's distance from the *last* vowel; counting from the end is what lets
// the analogy transfer between words of different length. Measured on held-out
// lexicon forms the table has never seen: 74.5% correct at 99.9% coverage,
// against the 0% an unstressed word scores.
type suffixTable struct {
	fromEnd   map[string]int
	maxSuffix int
}

// loadSuffixTable reads the table written by ml/scripts/run_suffix_fallback.py.
// An absent path is not an error: the fallback is optional and the pipeline
// behaves exactly as before without it.
func loadSuffixTable(path string) (*suffixTable, error) {
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
	var decoded map[string]json.Number
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return nil, err
	}
	table := &suffixTable{fromEnd: make(map[string]int, len(decoded)), maxSuffix: 13}
	for suffix, value := range decoded {
		position, err := strconv.Atoi(value.String())
		if err != nil {
			continue
		}
		table.fromEnd[suffix] = position
	}
	return table, nil
}

// stress returns the signature to apply to surface, or "" when the table has
// nothing for this ending or the word has fewer than two vowels.
func (t *suffixTable) stress(surface string) string {
	if t == nil || len(t.fromEnd) == 0 {
		return ""
	}
	decomposed := []rune(strings.ToLower(norm.NFD.String(surface)))
	vowels := 0
	for _, r := range decomposed {
		if ukrainianVowels[unicode.ToLower(r)] {
			vowels++
		}
	}
	if vowels < 2 {
		return ""
	}
	limit := t.maxSuffix
	if limit > len(decomposed) {
		limit = len(decomposed)
	}
	for length := limit; length > 0; length-- {
		position, ok := t.fromEnd[string(decomposed[len(decomposed)-length:])]
		if !ok || position >= vowels {
			continue
		}
		return strconv.Itoa(vowels - 1 - position)
	}
	return ""
}

// applySuffixFallback stresses whatever the lexicon and the compound fallback
// both failed to reach. It runs last, so it never displaces a real answer.
func applySuffixFallback(work []tokenWork, table *suffixTable,
	stress func(string, string) (string, error)) {
	if table == nil {
		return
	}
	for i := range work {
		if work[i].Status != "not_found" {
			continue
		}
		signature := table.stress(work[i].Text)
		if signature == "" {
			continue
		}
		stressed, err := stress(work[i].Text, signature)
		if err != nil {
			continue
		}
		work[i].Status, work[i].OutputText = "suffix", stressed
	}
}

// applyPositionalDefault is the last resort: a word with two or more vowels
// that every tier, the lexicon, the compound and the suffix fallbacks all
// left unanswered gets its penultimate vowel stressed.
//
// A multi-vowel word emitted with no mark is wrong for a TTS caller (and
// scores as wrong on lang-uk's Rule 4), so a guess is never worse than
// silence. The penultimate vowel is the guess because what reaches here is
// foreign names and ad-hoc compounds — «сент-кіттс», «нью-йорк» — where it
// is the commonest pattern. The LLM gap-fill dataset used to cover a
// thousand such forms; measured on lang-uk it was worth 0.2 points of word
// accuracy, and this rule is what stands in for it.
func applyPositionalDefault(work []tokenWork, stress func(string, string) (string, error)) {
	for i := range work {
		if work[i].Status != "not_found" {
			continue
		}
		if vowelCount(work[i].Text) < 2 {
			continue
		}
		// Signatures address a vowel inside a hyphen segment, so the guess
		// goes on the last segment: its penultimate vowel, or its only one.
		segment, perSegment := 0, []int{0}
		for _, r := range work[i].Text {
			if isSegmentBreak(r) {
				segment++
				perSegment = append(perSegment, 0)
			} else if countsAsVowel(r) {
				perSegment[segment]++
			}
		}
		last := len(perSegment) - 1
		for last > 0 && perSegment[last] == 0 {
			last--
		}
		ordinal := perSegment[last] - 2
		if ordinal < 0 {
			ordinal = 0
		}
		stressed, err := stress(work[i].Text, fmt.Sprintf("%d:%d", last, ordinal))
		if err != nil {
			continue
		}
		work[i].Status, work[i].OutputText = "default_position", stressed
	}
}
