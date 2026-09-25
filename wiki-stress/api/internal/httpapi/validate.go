package httpapi

import (
	"errors"
	"unicode"
)

var (
	errEmptyToken       = errors.New("token must not be empty")
	errTokenTooLong     = errors.New("token exceeds the maximum allowed length")
	errTokenControlChar = errors.New("token contains an unsupported control character")
	errInvalidOffset    = errors.New("offset must be a non-negative integer")
	errInvalidLimit     = errors.New("limit must be a positive integer")
)

// tab is a legitimate structural separator inside multi-token lookup keys
// (e.g. "бруковиця\tбруковиці"), not a control character in the linguistic
// sense: etl/src/ukstress/normalizer.py's validate_characters explicitly
// allows it alongside space and hyphen. Rejecting it here would make
// genuinely published multi-token entries permanently unlookupable through
// the API even though they exist in stress_lookup.
const tabCharacter = '\t'

// validateToken enforces the API-level bounds shared by every lookup
// input: presence, a byte-length cap, and no control characters other than
// the tab exception above. Deeper linguistic validation (Ukrainian
// character sets, stress placement) stays in Python; the Go service
// performs no morphology or linguistic rule authoring per the architecture
// boundary.
func validateToken(token string, maxBytes int) error {
	if token == "" {
		return errEmptyToken
	}
	if len(token) > maxBytes {
		return errTokenTooLong
	}
	for _, r := range token {
		if r != tabCharacter && unicode.IsControl(r) {
			return errTokenControlChar
		}
	}
	return nil
}
