package httpapi

import "testing"

func TestValidateTokenAllowsStructuralTab(t *testing.T) {
	// Regression test: a real word sampled from the published dataset
	// ("бруковиця\tбруковиці") was rejected with "unsupported control
	// character" even though etl/src/ukstress/normalizer.py's
	// validate_characters explicitly allows tab as a multi-token
	// separator, alongside space and hyphen. Found via bench-lookup
	// against the live dataset, not a synthetic fixture.
	if err := validateToken("бруковиця\tбруковиці", 256); err != nil {
		t.Errorf("validateToken rejected a legitimate tab-separated multi-token key: %v", err)
	}
}

func TestValidateTokenRejectsOtherControlCharacters(t *testing.T) {
	for _, r := range []rune{'\x00', '\x01', '\n', '\r', '\x1b', '\x7f'} {
		token := "слово" + string(r)
		if err := validateToken(token, 256); err == nil {
			t.Errorf("validateToken(%q) = nil, want errTokenControlChar", token)
		}
	}
}

func TestValidateTokenBounds(t *testing.T) {
	if err := validateToken("", 256); err != errEmptyToken {
		t.Errorf("validateToken(\"\") = %v, want errEmptyToken", err)
	}
	long := make([]byte, 300)
	for i := range long {
		long[i] = 'a'
	}
	if err := validateToken(string(long), 256); err != errTokenTooLong {
		t.Errorf("validateToken(long) = %v, want errTokenTooLong", err)
	}
	if err := validateToken("мова", 256); err != nil {
		t.Errorf("validateToken(\"мова\") = %v, want nil", err)
	}
}
