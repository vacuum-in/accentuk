// Package normalize implements the Go side of the versioned lookup-key
// normalization. It must stay in exact agreement with
// etl/src/ukstress/normalizer.py; conformance is proven by
// normalize_test.go against artifacts/normalization-conformance.json.
package normalize

import (
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

const (
	acute               rune = '́'
	canonicalApostrophe rune = 'ʼ'
)

// apostropheVariants mirrors APOSTROPHE_VARIANTS in
// etl/src/ukstress/normalizer.py: '`‘’ʻ＇
var apostropheVariants = map[rune]bool{
	'\'':   true,
	'\x60': true,
	'‘':    true,
	'’':    true,
	'ʻ':    true,
	'＇':    true,
}

// hyphenVariants mirrors HYPHEN_VARIANTS in
// etl/src/ukstress/normalizer.py: ‐‑‒–—−
var hyphenVariants = map[rune]bool{
	'‐': true,
	'‑': true,
	'‒': true,
	'–': true,
	'—': true,
	'−': true,
}

// CanonicalStressedForm returns NFD text with canonical apostrophes and
// canonical internal hyphens, matching canonical_stressed_form in
// etl/src/ukstress/normalizer.py.
func CanonicalStressedForm(value string) string {
	runes := []rune(norm.NFD.String(value))
	for i, r := range runes {
		if apostropheVariants[r] {
			runes[i] = canonicalApostrophe
		}
	}
	for i, r := range runes {
		if !hyphenVariants[r] {
			continue
		}
		if i == 0 || i == len(runes)-1 {
			continue
		}
		if unicode.IsSpace(runes[i-1]) || unicode.IsSpace(runes[i+1]) {
			continue
		}
		runes[i] = '-'
	}
	return string(runes)
}

// LookupKey returns the lowercase key with only the acute accent removed,
// matching lookup_key in etl/src/ukstress/normalizer.py.
func LookupKey(value string) string {
	canonical := strings.ToLower(CanonicalStressedForm(value))
	return strings.ReplaceAll(canonical, string(acute), "")
}
