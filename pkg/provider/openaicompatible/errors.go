package openaicompatible

import (
	"encoding/json"
)

// ErrorParser extracts the provider's error message from a failed response
// body. It returns an empty string when the body has no recognizable message.
type ErrorParser func(body []byte) string

// ErrorData is the error object of an OpenAI-compatible API. Providers differ
// in the optional fields, so they are decoded loosely.
type ErrorData struct {
	Message string `json:"message"`

	// Type, Param and Code are reported by some providers only.
	Type  string          `json:"type,omitempty"`
	Param json.RawMessage `json:"param,omitempty"`
	Code  json.RawMessage `json:"code,omitempty"`
}

// Error returns the provider's message.
func (e *ErrorData) Error() string {
	return e.Message
}

// errorEnvelope is the {"error": {...}} wrapper around ErrorData.
type errorEnvelope struct {
	Error *ErrorData `json:"error"`
}

// ParseError reads the message of an OpenAI-compatible error response. It is
// the default ErrorParser.
func ParseError(body []byte) string {
	var envelope errorEnvelope
	if err := json.Unmarshal(body, &envelope); err != nil {
		return ""
	}

	if envelope.Error == nil {
		return ""
	}

	return envelope.Error.Message
}
