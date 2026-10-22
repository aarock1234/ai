package codemode

import (
	"encoding/json"
	"fmt"
)

// encodePayload encodes a Go value for the program as JSON within a size
// limit. Nil encodes as the empty payload, which the program receives as
// undefined.
func encodePayload(value any, maxBytes int, label string) (string, error) {
	if value == nil {
		return "", nil
	}

	encoded, err := json.Marshal(value)
	if err != nil {
		return "", &Error{
			Code:    ErrorCodeSerialization,
			Message: fmt.Sprintf("%s is not JSON-serializable: %v", label, err),
			Cause:   err,
		}
	}

	if err := checkPayloadSize(string(encoded), maxBytes, label); err != nil {
		return "", err
	}

	return string(encoded), nil
}

// checkPayloadSize rejects a JSON payload larger than maxBytes.
func checkPayloadSize(encoded string, maxBytes int, label string) error {
	if len(encoded) <= maxBytes {
		return nil
	}

	return &Error{
		Code:    ErrorCodeSerialization,
		Message: fmt.Sprintf("%s exceeds the %d byte size limit", label, maxBytes),
		Limit:   int64(maxBytes),
		Size:    int64(len(encoded)),
	}
}
