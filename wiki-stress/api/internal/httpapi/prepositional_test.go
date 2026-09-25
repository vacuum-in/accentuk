package httpapi

import "testing"

func TestPrepositionalShiftMovesGovernedPronouns(t *testing.T) {
	text := "Заходи до мене, він знає мене, і взяв усе на себе."
	work := []tokenWork{
		{stressToken: stressToken{Text: "Заходи"}, byteStart: 0, byteEnd: 12},
		{stressToken: stressToken{Text: "до"}, byteStart: 13, byteEnd: 17},
		{stressToken: stressToken{Text: "мене", Status: "stressed",
			OutputText: "мене́", Candidates: []string{"1"}}, byteStart: 18, byteEnd: 26},
		{stressToken: stressToken{Text: "він"}, byteStart: 28, byteEnd: 34},
		{stressToken: stressToken{Text: "знає"}, byteStart: 35, byteEnd: 43},
		{stressToken: stressToken{Text: "мене", Status: "stressed",
			OutputText: "мене́", Candidates: []string{"1"}}, byteStart: 44, byteEnd: 52},
		{stressToken: stressToken{Text: "на"}, byteStart: 73, byteEnd: 77},
		{stressToken: stressToken{Text: "себе", Status: "stressed",
			OutputText: "себе́", Candidates: []string{"1"}}, byteStart: 78, byteEnd: 86},
	}
	applyPrepositionalShift(work, text, applySignature)

	if work[2].Status != "prepositional" || work[2].OutputText != "ме́не" {
		t.Errorf("governed pronoun = %q/%q, want prepositional/ме́не",
			work[2].Status, work[2].OutputText)
	}
	if work[5].Status != "stressed" || work[5].OutputText != "мене́" {
		t.Errorf("ungoverned pronoun was rewritten: %q/%q",
			work[5].Status, work[5].OutputText)
	}
	if work[7].Status != "prepositional" || work[7].OutputText != "се́бе" {
		t.Errorf("governed reflexive = %q/%q, want prepositional/се́бе",
			work[7].Status, work[7].OutputText)
	}
}

func TestPrepositionalShiftNeedsAdjacency(t *testing.T) {
	// Tokens skip punctuation, so "до, мене" puts the preposition in the
	// previous slot without governing anything.
	text := "до, мене"
	work := []tokenWork{
		{stressToken: stressToken{Text: "до"}, byteStart: 0, byteEnd: 4},
		{stressToken: stressToken{Text: "мене", Status: "stressed",
			OutputText: "мене́"}, byteStart: 6, byteEnd: 14},
	}
	applyPrepositionalShift(work, text, applySignature)

	if work[1].Status != "stressed" || work[1].OutputText != "мене́" {
		t.Errorf("comma-separated pronoun was shifted: %q/%q",
			work[1].Status, work[1].OutputText)
	}
}
