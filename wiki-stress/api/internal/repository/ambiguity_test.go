package repository

import "testing"

func candidates(signatures ...string) []Candidate {
	result := make([]Candidate, 0, len(signatures))
	for _, signature := range signatures {
		result = append(result, Candidate{StressSignature: signature})
	}
	return result
}

func TestIsAmbiguous(t *testing.T) {
	cases := []struct {
		name       string
		candidates []Candidate
		truncated  bool
		want       bool
	}{
		{
			// "а" has 33 rows in merged-wordlist-v1, every one of them "а́".
			// Row counting called this ambiguous and sent a word with exactly
			// one possible pronunciation to the model tier.
			name:       "many lexemes agreeing on one stress are not ambiguous",
			candidates: candidates("0", "0", "0", "0"),
		},
		{
			name:       "за́мок / замо́к genuinely disagree",
			candidates: candidates("0", "1"),
			want:       true,
		},
		{
			name:       "disagreement anywhere in the list counts",
			candidates: candidates("2", "2", "2", "5"),
			want:       true,
		},
		{
			name:       "a single candidate is never ambiguous",
			candidates: candidates("3"),
		},
		{
			name:       "no candidates is not ambiguous",
			candidates: candidates(),
		},
		{
			// Signatures past the limit are unknown, so agreement among the
			// ones returned proves nothing.
			name:       "a truncated result stays ambiguous even when the visible signatures agree",
			candidates: candidates("0", "0"),
			truncated:  true,
			want:       true,
		},
		{
			name:      "truncation with no visible candidates is still ambiguous",
			truncated: true,
			want:      true,
		},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			if got := isAmbiguous(testCase.candidates, testCase.truncated); got != testCase.want {
				t.Fatalf("isAmbiguous() = %v, want %v", got, testCase.want)
			}
		})
	}
}
