package wirejson

import (
	"bytes"
	"encoding/json"
)

// The helpers below give raw JSON members one shared vocabulary. A member
// decoded into a json.RawMessage is empty when it was absent, and holds the
// literal null when it was sent as null.

// IsSet reports whether a member was sent, including as null.
func IsSet(raw json.RawMessage) bool {
	return len(bytes.TrimSpace(raw)) > 0
}

// IsNull reports whether a member was sent as JSON null, which decodes into
// Go values without an error.
func IsNull(raw json.RawMessage) bool {
	return bytes.Equal(bytes.TrimSpace(raw), []byte("null"))
}

// HasValue reports whether a member was sent with a value other than null.
func HasValue(raw json.RawMessage) bool {
	return IsSet(raw) && !IsNull(raw)
}
