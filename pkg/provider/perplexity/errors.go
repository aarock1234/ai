package perplexity

import (
	"bytes"
	"encoding/json"
	"strings"
)

// unknownErrorMessage is reported for error bodies without a message.
const unknownErrorMessage = "unknown error"

// errorBody is a Perplexity error response. Error is a string or an
// ErrorData object, and Detail is a string or a list of validation details.
type errorBody struct {
	Error   json.RawMessage `json:"error"`
	Detail  json.RawMessage `json:"detail"`
	Message *string         `json:"message"`
}

// ErrorData is a structured Perplexity error.
type ErrorData struct {
	// Code is a string or a number.
	Code    json.RawMessage `json:"code,omitempty"`
	Message *string         `json:"message"`
	Type    *string         `json:"type"`
}

type validationDetail struct {
	Msg string `json:"msg"`
}

// parseError reads the message of a Perplexity error response.
func parseError(body []byte) string {
	var decoded errorBody
	if err := json.Unmarshal(body, &decoded); err != nil {
		return ""
	}

	if message, ok := errorMessage(decoded.Error); ok {
		return message
	}

	if message, ok := detailMessage(decoded.Detail); ok {
		return message
	}

	if decoded.Message != nil {
		return *decoded.Message
	}

	return unknownErrorMessage
}

// errorMessage reads a string error or the message or type of an error
// object.
func errorMessage(raw json.RawMessage) (string, bool) {
	if isAbsent(raw) {
		return "", false
	}

	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text, true
	}

	var data ErrorData
	if err := json.Unmarshal(raw, &data); err != nil {
		return "", false
	}

	switch {
	case data.Message != nil:
		return *data.Message, true
	case data.Type != nil:
		return *data.Type, true
	default:
		return unknownErrorMessage, true
	}
}

// detailMessage reads a string detail or joins validation messages.
func detailMessage(raw json.RawMessage) (string, bool) {
	if isAbsent(raw) {
		return "", false
	}

	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text, true
	}

	var details []validationDetail
	if err := json.Unmarshal(raw, &details); err != nil {
		return "", false
	}

	messages := make([]string, 0, len(details))
	for _, detail := range details {
		messages = append(messages, detail.Msg)
	}

	return strings.Join(messages, ", "), true
}

func isAbsent(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)

	return len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null"))
}
