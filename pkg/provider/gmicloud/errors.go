package gmicloud

import (
	"encoding/json"
	"strings"
)

// ErrorData is a GMI Cloud error response. GMI Cloud's edge reports a generic
// banner in Message and nests the backend engine's diagnostic, as JSON text,
// in Details.
type ErrorData struct {
	Error ErrorDetail `json:"error"`
}

// ErrorDetail is the error object of a GMI Cloud error response.
type ErrorDetail struct {
	Message string `json:"message"`

	// Type, Param, Code and Details are optional. Code is a string or a
	// number.
	Type    *string         `json:"type,omitempty"`
	Param   json.RawMessage `json:"param,omitempty"`
	Code    json.RawMessage `json:"code,omitempty"`
	Details *string         `json:"details,omitempty"`
}

// detailsError is the engine error encoded in ErrorDetail.Details.
type detailsError struct {
	Error *struct {
		Message *string `json:"message"`
	} `json:"error"`
}

// parseError prefers the engine diagnostic nested in the details and falls
// back to the edge message. Plain-text bodies, such as GMI Cloud's 404
// responses, have no message and fall back to the status text.
func parseError(body []byte) string {
	var data ErrorData
	if err := json.Unmarshal(body, &data); err != nil {
		return ""
	}

	if message := detailsMessage(data.Error.Details); message != "" {
		return message
	}

	return data.Error.Message
}

// detailsMessage returns the non-blank engine message of the details, or an
// empty string when the details are absent, not JSON or have no message.
func detailsMessage(details *string) string {
	if details == nil || *details == "" {
		return ""
	}

	var parsed detailsError
	if err := json.Unmarshal([]byte(*details), &parsed); err != nil {
		return ""
	}

	if parsed.Error == nil || parsed.Error.Message == nil {
		return ""
	}

	message := *parsed.Error.Message
	if strings.TrimSpace(message) == "" {
		return ""
	}

	return message
}
