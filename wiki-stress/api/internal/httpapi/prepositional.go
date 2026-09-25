package httpapi

import (
	"strings"
	"unicode"
)

// prepositionalPronouns are the personal and reflexive pronouns whose stress
// moves to the first syllable when a preposition governs them: «до ме́не»,
// «у ньо́го», «з се́бе» — against «він знає мене́», «виказали себе́».
//
// The value is the vowel ordinal to stress in that position, which for this
// class is always the first.
//
// This cannot live in the lexicon. A stress that depends on the preceding word
// has no home in a table keyed by one word, which is why the source dictionary
// records only мене́ and why the pipeline served it everywhere. Measured on
// lang-uk the pattern is categorical in both directions: after a preposition
// 39 of 40 tokens take the first syllable, and with no preposition 0 of 11 do,
// across six different pronouns.
//
// `них` and `ним` are absent on purpose: one vowel, and a one-vowel word
// carries no mark anywhere else in this pipeline.
var prepositionalPronouns = map[string]string{
	"мене":  "0",
	"тебе":  "0",
	"себе":  "0",
	"нього": "0",
	"неї":   "0",
	// Same paradigm, no occurrence in the benchmark to confirm them; the rule
	// is the pronoun class, not the individual word, so they are included.
	"нею":  "0",
	"ними": "0",
}

// governingPrepositions are those that can take the pronouns above.
var governingPrepositions = map[string]bool{
	"без": true, "біля": true, "в": true, "від": true, "для": true, "до": true,
	"за": true, "з": true, "зі": true, "із": true, "коло": true, "крізь": true,
	"круг": true, "між": true, "на": true, "над": true, "о": true, "об": true,
	"округ": true, "перед": true, "під": true, "по": true, "повз": true,
	"поза": true, "поміж": true, "попри": true, "при": true, "про": true,
	"серед": true, "у": true, "через": true, "од": true,
}

// applyPrepositionalShift moves the stress of a governed pronoun to its first
// syllable.
//
// It runs after the tiers and overrides whatever they decided, because for
// these words nothing decided anything: the lexicon holds one reading, so the
// token arrives already "stressed" with a single candidate and no tier is
// consulted at all.
//
// Adjacency is checked against the source text rather than by token position:
// tokens skip punctuation, so «до, мене» would otherwise read as governed.
func applyPrepositionalShift(work []tokenWork, text string,
	apply func(string, string) (string, error)) {
	for i := 1; i < len(work); i++ {
		signature, governed := prepositionalPronouns[strings.ToLower(work[i].Text)]
		if !governed {
			continue
		}
		if !governingPrepositions[strings.ToLower(work[i-1].Text)] {
			continue
		}
		if !onlySpaceBetween(text, work[i-1].byteEnd, work[i].byteStart) {
			continue
		}
		stressed, err := apply(work[i].Text, signature)
		if err != nil {
			continue
		}
		work[i].Status, work[i].OutputText = "prepositional", stressed
	}
}

// onlySpaceBetween reports whether nothing but whitespace separates two tokens.
func onlySpaceBetween(text string, from, to int) bool {
	if from < 0 || to > len(text) || from > to {
		return false
	}
	return strings.TrimFunc(text[from:to], unicode.IsSpace) == ""
}
