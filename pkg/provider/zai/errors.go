package zai

import "encoding/json"

// ErrorData is a Z.AI error, sent either bare or wrapped in an "error"
// member.
type ErrorData struct {
	// Code is a number or a string.
	Code json.RawMessage `json:"code,omitempty"`

	Message string `json:"message"`
}

// errorResponse accepts both error shapes.
type errorResponse struct {
	ErrorData

	Error *ErrorData `json:"error"`
}

// parseError reads the message of a Z.AI error response.
func parseError(body []byte) string {
	var response errorResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return ""
	}

	if response.Error != nil {
		return response.Error.Message
	}

	return response.Message
}
