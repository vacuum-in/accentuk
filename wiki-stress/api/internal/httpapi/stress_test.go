package httpapi

import (
	"strings"
	"testing"
)

func TestWordTokensTreatsCompoundsAsCompleteWords(t *testing.T) {
	tokens := wordTokens("Так, ПІВ-ЯБЛУКА й п'ять.")
	want := []string{"Так", "ПІВ-ЯБЛУКА", "й", "п'ять"}
	if len(tokens) != len(want) {
		t.Fatalf("got %d tokens, want %d", len(tokens), len(want))
	}
	for i := range want {
		if tokens[i].Text != want[i] {
			t.Errorf("token %d = %q, want %q", i, tokens[i].Text, want[i])
		}
	}
}

func TestApplySignaturePreservesOrthography(t *testing.T) {
	tests := []struct{ surface, signature, want string }{
		{"МОВА", "0", "МО́ВА"},
		{"пів-яблука", "1:0", "пів-я́блука"},
		// The apostrophe is NOT a segment boundary: `stress_signature` splits
		// on `[\s-]+` only, and `stress_lookup` stores `пʼятьо́х` as "1", not
		// "1:0". This case previously asserted "1:0", which codified the
		// reader's disagreement with the producer instead of catching it.
		{"п'ятьох", "1", "п'ятьо́х"},
		{"їхати", "0", "ї́хати"},
	}
	for _, test := range tests {
		got, err := applySignature(test.surface, test.signature)
		if err != nil {
			t.Errorf("applySignature(%q): %v", test.surface, err)
			continue
		}
		if got != test.want {
			t.Errorf("applySignature(%q) = %q, want %q", test.surface, got, test.want)
		}
		if strings.ReplaceAll(got, string(acute), "") != test.surface {
			t.Errorf("removing acute from %q did not restore %q", got, test.surface)
		}
	}
}

func TestOneVowelWordsNeedNoStress(t *testing.T) {
	for _, word := range []string{"шум", "стіл", "внук", "ШУМ", "ЇЖ"} {
		if got := vowelCount(word); got != 1 {
			t.Errorf("vowelCount(%q) = %d, want 1", word, got)
		}
	}
}

func TestDictionaryDefaultFillsWordsNothingDecided(t *testing.T) {
	work := []tokenWork{
		{stressToken: stressToken{Text: "замок", OutputText: "замок",
			Status: "ambiguous", Candidates: []string{"1", "0"}}},
		{stressToken: stressToken{Text: "коси", OutputText: "коси",
			Status: "model_ineligible", Candidates: []string{"0", "1"}}},
		{stressToken: stressToken{Text: "мова", OutputText: "мо́ва",
			Status: "stressed", Candidates: []string{"0"}}},
		{stressToken: stressToken{Text: "абра", OutputText: "абра",
			Status: "not_found"}},
		{stressToken: stressToken{Text: "мова", OutputText: "мова",
			Status: "ambiguous", Candidates: []string{"9"}}},
	}
	applyDictionaryDefault(work)

	// The first candidate is the lexicon's own preference, not the lowest
	// signature: the query orders by confidence, so "1" beating "0" here is the
	// case that would regress if the ordering were dropped.
	if work[0].Status != "dictionary_default" || work[0].OutputText != "замо́к" {
		t.Errorf("ambiguous token = %q/%q, want dictionary_default/замо́к",
			work[0].Status, work[0].OutputText)
	}
	if work[1].Status != "dictionary_default" || work[1].OutputText != "ко́си" {
		t.Errorf("ineligible token = %q/%q, want dictionary_default/ко́си",
			work[1].Status, work[1].OutputText)
	}
	if work[2].Status != "stressed" || work[2].OutputText != "мо́ва" {
		t.Errorf("decided token was rewritten: %q/%q", work[2].Status, work[2].OutputText)
	}
	if work[3].Status != "not_found" || work[3].OutputText != "абра" {
		t.Errorf("token with no candidates was rewritten: %q/%q",
			work[3].Status, work[3].OutputText)
	}
	if work[4].Status != "invalid_candidate" || work[4].OutputText != "мова" {
		t.Errorf("unusable signature = %q/%q, want invalid_candidate and the word intact",
			work[4].Status, work[4].OutputText)
	}
}

func TestMorphologyIsOfferedEveryAmbiguousToken(t *testing.T) {
	// The tier used to see only "model_ineligible" tokens, so a form inside the
	// serving manifest never reached a parser and a low-margin abstention did
	// not fall through to one either. On lang-uk that cost 8.29 points of
	// heteronym accuracy.
	work := []tokenWork{
		{stressToken: stressToken{Text: "коси", Status: "model_ineligible",
			Candidates: []string{"0", "1"}}},
		{stressToken: stressToken{Text: "замок", Status: "ambiguous",
			Candidates: []string{"1", "0"}}},
		{stressToken: stressToken{Text: "руки", Status: "stressed",
			OutputText: "ру́ки", Candidates: []string{"0", "1"}}}, // offered; only a rule may override
		{stressToken: stressToken{Text: "мова", Status: "stressed",
			OutputText: "мо́ва", Candidates: []string{"0"}}},
		{stressToken: stressToken{Text: "абра", Status: "not_found"}},
	}
	targets := morphologyTargets(work, "text")

	got := make([]int, 0, len(targets))
	for _, target := range targets {
		got = append(got, target.Index)
	}
	want := []int{0, 1, 2}
	if len(got) != len(want) {
		t.Fatalf("targets = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("targets = %v, want %v", got, want)
		}
	}
}
