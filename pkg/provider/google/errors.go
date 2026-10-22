package google

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// Error is an error object reported by the Gemini API, either as a failed
// response body or inside a stream chunk.
type Error struct {
	// Code is the HTTP status code, when reported.
	Code *int `json:"code"`

	Message string            `json:"message"`
	Status  string            `json:"status"`
	Details []json.RawMessage `json:"details,omitempty"`
}

// Error returns the status and message.
func (e *Error) Error() string {
	if e.Status == "" {
		return e.Message
	}

	return fmt.Sprintf("%s: %s", e.Status, e.Message)
}

// IsRetryable reports whether the code indicates a transient failure.
func (e *Error) IsRetryable() bool {
	if e.Code == nil {
		return false
	}

	return model.RetryableStatus(*e.Code)
}

// StatusCode returns the reported code, or zero.
func (e *Error) StatusCode() int {
	if e.Code == nil {
		return 0
	}

	return *e.Code
}

type errorResponse struct {
	Error *Error `json:"error"`
}

// parseErrorMessage reads the message of a failed Gemini API response.
func parseErrorMessage(body []byte) string {
	var response errorResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return ""
	}

	if response.Error == nil {
		return ""
	}

	return response.Error.Message
}
