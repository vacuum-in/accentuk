package httpapi

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestSignatureOfOutputCountsLikeStressSignature(t *testing.T) {
	cases := map[string]string{
		"за́мок":   "0",
		"замо́к":   "1",
		"райо́н":   "2", // й decomposes to и + breve and occupies an ordinal
		"Сестри́": "1",
		"замок":   "",
	}
	for output, want := range cases {
		if got := signatureOfOutput(output); got != want {
			t.Errorf("signatureOfOutput(%q) = %q, want %q", output, got, want)
		}
	}
}

func TestCombinerTargetsOfferOnlyRealChoices(t *testing.T) {
	work := []tokenWork{
		{stressToken: stressToken{Text: "Замок", OutputText: "Замо́к", Status: "token_model",
			Candidates: []string{"1", "0"}, Start: 0, End: 5}, normalized: "замок"},
		{stressToken: stressToken{Text: "на", OutputText: "на", Status: "not_required",
			Candidates: []string{"0"}}, normalized: "на"},
		{stressToken: stressToken{Text: "держава", OutputText: "держа́ва", Status: "stressed",
			Candidates: []string{"0|1"}}, normalized: "держава"},
	}
	targets := combinerTargets(work, "Замок на держава")
	if len(targets) != 1 {
		t.Fatalf("want one target, got %d", len(targets))
	}
	got := targets[0]
	if got.Index != 0 || got.Pipeline != "1" || got.Status != "token_model" || got.Form != "замок" {
		t.Errorf("unexpected target %+v", got)
	}
}

func TestByCapitalisationUsesTheCapitalLetter(t *testing.T) {
	proper := map[string]bool{"русі\x001": true}
	both := []string{"1", "0"}
	if got := byCapitalisation(both, proper, "русі", "русі", false); len(got) != 1 || got[0] != "0" {
		t.Errorf("lowercase русі should be the common noun, got %v", got)
	}
	if got := byCapitalisation(both, proper, "русі", "Русі", false); len(got) != 1 || got[0] != "1" {
		t.Errorf("capitalised mid-sentence Русі should be the proper noun, got %v", got)
	}
	if got := byCapitalisation(both, proper, "русі", "Русі", true); len(got) != 2 {
		t.Errorf("a capital opening a sentence says nothing, got %v", got)
	}
	three := []string{"2", "1", "0"}
	threeProper := map[string]bool{"форма\x002": true}
	if got := byCapitalisation(three, threeProper, "форма", "форма", false); len(got) != 3 || got[2] != "2" {
		t.Errorf("a partial narrowing must reorder, not drop: got %v", got)
	}
}

func TestSentenceInitial(t *testing.T) {
	cases := []struct {
		text  string
		word  string
		first bool
	}{
		{"Русі", "Русі", true},
		{"Він жив у Київській Русі.", "Русі", false},
		{"Кінець. Русі багато.", "Русі", true},
		{"— «Русі", "Русі", true},
	}
	for _, c := range cases {
		at := strings.Index(c.text, c.word)
		if got := sentenceInitial(c.text, at); got != c.first {
			t.Errorf("sentenceInitial(%q, %q) = %v, want %v", c.text, c.word, got, c.first)
		}
	}
}

func TestStressRequestCarriesTheCombinerChoice(t *testing.T) {
	var off, on, unset stressRequest
	for body, into := range map[string]*stressRequest{
		`{"text":"x","combiner":false}`: &off, `{"text":"x","combiner":true}`: &on, `{"text":"x"}`: &unset} {
		if err := json.Unmarshal([]byte(body), into); err != nil {
			t.Fatal(err)
		}
	}
	if off.Combiner == nil || *off.Combiner || on.Combiner == nil || !*on.Combiner || unset.Combiner != nil {
		t.Errorf("combiner: off=%v on=%v unset=%v", off.Combiner, on.Combiner, unset.Combiner)
	}
}
