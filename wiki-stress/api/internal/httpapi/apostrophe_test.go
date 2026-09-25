package httpapi

import "testing"

// The signature producer (Python `stress_signature`) splits segments on
// hyphens and whitespace only. The reader must agree, or every apostrophe
// word is served unstressed.
func TestApostropheIsNotASegmentBoundary(t *testing.T) {
	for _, tc := range []struct{ surface, signature, want string }{
		{"м'ясо", "0", "м'я́со"},
		{"здоров'я", "1", "здоро́в'я"},
		{"сім'я", "0", "сі́м'я"},
		{"полум'ям", "0", "по́лум'ям"},
		{"пів-яблука", "1:0", "пів-я́блука"},
	} {
		got, err := applySignature(tc.surface, tc.signature)
		if err != nil {
			t.Errorf("applySignature(%q, %q): %v", tc.surface, tc.signature, err)
			continue
		}
		if got != tc.want {
			t.Errorf("applySignature(%q, %q) = %q, want %q",
				tc.surface, tc.signature, got, tc.want)
		}
	}
}

// Free variation is one acceptable stress chosen from several, not several
// stresses emitted at once.
func TestFreeVariationEmitsOneStress(t *testing.T) {
	for _, tc := range []struct{ surface, signature, want string }{
		// These are the signatures `stress_lookup` actually stores: the
		// wordlist records `де́ржа́ва` with two acutes meaning either is
		// acceptable, and serving must pick one.
		{"держава", "0|1", "де́ржава"},
		{"дівчина", "0|1", "ді́вчина"},
		{"років", "0|1", "ро́ків"},
		// A multi-token signature still marks each token.
		{"пів-яблука", "1:0", "пів-я́блука"},
	} {
		got, err := applySignature(tc.surface, tc.signature)
		if err != nil {
			t.Errorf("applySignature(%q, %q): %v", tc.surface, tc.signature, err)
			continue
		}
		if got != tc.want {
			t.Errorf("applySignature(%q, %q) = %q, want %q",
				tc.surface, tc.signature, got, tc.want)
		}
	}
}

// `й` decomposes to `и` plus a breve, and the signature generator counts that
// `и` as a vowel. The reader must agree or every ordinal after a `й` shifts.
func TestYotCountsAsAVowelOrdinal(t *testing.T) {
	for _, tc := range []struct{ surface, signature, want string }{
		{"район", "2", "райо́н"},
		{"району", "2", "райо́ну"},
		{"війна", "2", "війна́"},
		{"їжак", "0", "ї́жак"},
		{"мова", "0", "мо́ва"},
	} {
		got, err := applySignature(tc.surface, tc.signature)
		if err != nil {
			t.Errorf("applySignature(%q, %q): %v", tc.surface, tc.signature, err)
			continue
		}
		if got != tc.want {
			t.Errorf("applySignature(%q, %q) = %q, want %q",
				tc.surface, tc.signature, got, tc.want)
		}
	}
}
