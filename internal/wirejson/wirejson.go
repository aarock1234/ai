// Package wirejson encodes pkg/model values in the JSON shape of upstream
// values, for integrations that report prompts and model output as JSON:
// OpenTelemetry span attributes, OPA policy inputs and DevTools recordings.
// It also owns the tagged file data and tool output content encoding that
// providers send when they accept tool results only as text. The model types
// have no JSON encoding of their own, so every type is encoded explicitly
// into a typed wire struct.
//
// Optional upstream properties are omitted when they are unset, as
// JSON.stringify omits undefined properties. Provider options and metadata
// are encoded with encoding/json.
package wirejson

import (
	"encoding/json"
	"time"
)

// FileDataEncoding selects how file contents are encoded.
type FileDataEncoding int

// File data encodings.
const (
	// FlattenedFileData encodes file contents as a base64 string, a URL,
	// text or a provider reference map, as upstream telemetry and policies
	// report prompts.
	FlattenedFileData FileDataEncoding = iota

	// TaggedFileData encodes file contents as the tagged union of the
	// provider specification, such as {"type":"data","data":"<base64>"}.
	TaggedFileData
)

// Encoder encodes model values. The zero Encoder flattens file contents.
type Encoder struct {
	FileData FileDataEncoding
}

// timestampLayout formats times like JavaScript's Date.toISOString, which
// JSON.stringify uses for dates.
const timestampLayout = "2006-01-02T15:04:05.000Z"

// Timestamp formats t like JavaScript's Date.toISOString.
func Timestamp(t time.Time) string {
	return t.UTC().Format(timestampLayout)
}

// RawValue returns raw as a JSON value: null when it is empty, raw when it
// is valid JSON, and its text as a JSON string otherwise.
func RawValue(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return json.RawMessage("null")
	}

	if !json.Valid(raw) {
		return encodeString(string(raw))
	}

	return raw
}

// encodeString encodes text as a JSON string.
func encodeString(text string) json.RawMessage {
	encoded, err := json.Marshal(text)
	if err != nil {
		return json.RawMessage("null")
	}

	return encoded
}
