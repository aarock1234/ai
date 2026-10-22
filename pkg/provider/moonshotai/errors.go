package moonshotai

import (
	"encoding/json"
	"net/http"

	"github.com/aarock1234/ai/internal/optional"
	"github.com/aarock1234/ai/pkg/model"
)

// ErrorData is a Moonshot AI error.
type ErrorData struct {
	Message string  `json:"message"`
	Type    *string `json:"type"`
	Code    *string `json:"code"`
}

// Error returns the provider's message.
func (e *ErrorData) Error() string {
	return e.Message
}

// errorResponse is Moonshot AI's error body.
type errorResponse struct {
	Error *ErrorData `json:"error"`
}

// parseError reads the message of a Moonshot AI error response.
func parseError(body []byte) string {
	var response errorResponse
	if err := json.Unmarshal(body, &response); err != nil || response.Error == nil {
		return ""
	}

	return response.Error.Message
}

// streamErrorStatus is the HTTP status and retryability implied by a
// streamed error type.
type streamErrorStatus struct {
	statusCode  int
	isRetryable bool
}

// streamErrorStatuses maps streamed error types to their implied status.
var streamErrorStatuses = map[string]streamErrorStatus{
	"rate_limit_exceeded": {
		statusCode:  http.StatusTooManyRequests,
		isRetryable: true,
	},
	"rate_limit_error": {
		statusCode:  http.StatusTooManyRequests,
		isRetryable: true,
	},
	"server_error": {
		statusCode:  http.StatusInternalServerError,
		isRetryable: true,
	},
	"api_error": {
		statusCode:  http.StatusInternalServerError,
		isRetryable: true,
	},
	"internal_server_error": {
		statusCode:  http.StatusInternalServerError,
		isRetryable: true,
	},
	"overloaded_error": {
		statusCode:  http.StatusServiceUnavailable,
		isRetryable: true,
	},
	"service_unavailable": {
		statusCode:  http.StatusServiceUnavailable,
		isRetryable: true,
	},
	"timeout": {
		statusCode:  http.StatusGatewayTimeout,
		isRetryable: true,
	},
	"timeout_error": {
		statusCode:  http.StatusGatewayTimeout,
		isRetryable: true,
	},
	"authentication_error": {
		statusCode: http.StatusUnauthorized,
	},
	"invalid_api_key": {
		statusCode: http.StatusUnauthorized,
	},
	"permission_error": {
		statusCode: http.StatusForbidden,
	},
	"not_found_error": {
		statusCode: http.StatusNotFound,
	},
	"model_not_found": {
		statusCode: http.StatusNotFound,
	},
	"bad_request": {
		statusCode: http.StatusBadRequest,
	},
	"context_length_exceeded": {
		statusCode: http.StatusBadRequest,
	},
	"invalid_request_error": {
		statusCode: http.StatusBadRequest,
	},
}

// streamError converts an error chunk into an API call error whose status
// reflects the error type.
func streamError(data *ErrorData, body string) error {
	status := streamErrorStatuses[optional.Or(data.Type, "")]

	return &model.APICallError{
		Message:      data.Message,
		StatusCode:   status.statusCode,
		ResponseBody: body,
		IsRetryable:  status.isRetryable,
		Cause:        data,
	}
}
