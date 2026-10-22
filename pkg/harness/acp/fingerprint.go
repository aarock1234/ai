package acp

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

// stableFingerprint returns the SHA-256 digest of value's canonical JSON,
// in which object keys are sorted at every depth.
//
// Go adaptation: keys are sorted by byte order. Upstream sorts some
// identities with localeCompare, so digests can differ from upstream for
// keys that order differently; they are only compared with digests this
// package computed.
func stableFingerprint(value any) (string, error) {
	canonical, err := canonicalJSON(value)
	if err != nil {
		return "", err
	}

	return sha256Hex(canonical), nil
}

// plainFingerprint returns the SHA-256 digest of value's JSON encoding.
func plainFingerprint(value any) (string, error) {
	encoded, err := marshalJSON(value)
	if err != nil {
		return "", err
	}

	return sha256Hex(encoded), nil
}

// canonicalJSON encodes value with object keys sorted at every depth.
func canonicalJSON(value any) ([]byte, error) {
	encoded, err := marshalJSON(value)
	if err != nil {
		return nil, err
	}

	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.UseNumber()

	var generic any
	if err := decoder.Decode(&generic); err != nil {
		return nil, fmt.Errorf("decode canonical json: %w", err)
	}

	return marshalJSON(generic)
}

// marshalJSON encodes value as JavaScript's JSON.stringify does, without
// escaping <, > and &.
func marshalJSON(value any) ([]byte, error) {
	var buffer bytes.Buffer

	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)

	if err := encoder.Encode(value); err != nil {
		return nil, fmt.Errorf("encode json: %w", err)
	}

	return bytes.TrimSuffix(buffer.Bytes(), []byte("\n")), nil
}

// sha256Hex returns the hexadecimal SHA-256 digest of data.
func sha256Hex(data []byte) string {
	digest := sha256.Sum256(data)

	return hex.EncodeToString(digest[:])
}
