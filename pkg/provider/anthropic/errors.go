package anthropic

import (
	"encoding/json"
	"net/http"

	"github.com/aarock1234/ai/pkg/model"
)

// errorResponse is the body of a failed API call.
type errorResponse struct {
	Type  string      `json:"type"`
	Error errorDetail `json:"error"`
}

type errorDetail struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// parseErrorMessage returns the message of a failed API call, or an empty
// string when the body is not an Anthropic error.
func parseErrorMessage(body []byte) string {
	var response errorResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return ""
	}

	if response.Type != "error" {
		return ""
	}

	return response.Error.Message
}

// streamErrorDetail is the error object of an in-band stream error. Status
// and retry hints are optional; they are inferred from Type when absent.
type streamErrorDetail struct {
	Type        string          `json:"type"`
	Message     string          `json:"message"`
	StatusCode  *int            `json:"statusCode"`
	IsRetryable *bool           `json:"isRetryable"`
	Data        json.RawMessage `json:"data"`
}

// errorTypeStatus is the HTTP status and retry behavior of an Anthropic
// error type.
type errorTypeStatus struct {
	statusCode int
	retryable  bool
}

var errorTypeStatuses = map[string]errorTypeStatus{
	"api_error": {
		statusCode: http.StatusInternalServerError,
		retryable:  true,
	},
	"overloaded_error": {
		statusCode: 529,
		retryable:  true,
	},
	"rate_limit_error": {
		statusCode: http.StatusTooManyRequests,
		retryable:  true,
	},
	"request_too_large": {
		statusCode: http.StatusRequestEntityTooLarge,
	},
	"authentication_error": {
		statusCode: http.StatusUnauthorized,
	},
	"permission_error": {
		statusCode: http.StatusForbidden,
	},
	"not_found_error": {
		statusCode: http.StatusNotFound,
	},
	"billing_error": {
		statusCode: http.StatusBadRequest,
	},
	"invalid_request_error": {
		statusCode: http.StatusBadRequest,
	},
}

// streamError converts an in-band stream error into an API call error.
// fallbackStatus is used when neither the event nor the error type implies a
// status.
func streamError(event *errorEvent, url string, headers http.Header, fallbackStatus int) *model.APICallError {
	detail := event.Detail
	inferred := errorTypeStatuses[detail.Type]

	statusCode := inferred.statusCode
	if detail.StatusCode != nil {
		statusCode = *detail.StatusCode
	}

	if statusCode == 0 {
		statusCode = fallbackStatus
	}

	retryable := inferred.retryable
	if detail.IsRetryable != nil {
		retryable = *detail.IsRetryable
	}

	body := event.Raw
	if detail.Data != nil {
		body = detail.Data
	}

	return &model.APICallError{
		Message:         detail.Message,
		URL:             url,
		StatusCode:      statusCode,
		ResponseHeaders: headers,
		ResponseBody:    string(body),
		IsRetryable:     retryable,
	}
}
