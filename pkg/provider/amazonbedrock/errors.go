package amazonbedrock

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/amazonbedrock/internal/bedrock"
)

// errorResponse is the body of a failed Bedrock call.
type errorResponse struct {
	Message *string `json:"message"`
	Type    *string `json:"type"`
}

// parseError returns the message of a failed call, prefixed with the error
// type when there is one.
func parseError(body []byte) string {
	var response errorResponse
	if err := json.Unmarshal(body, &response); err != nil || response.Message == nil {
		return ""
	}

	if response.Type == nil {
		return *response.Message
	}

	return *response.Type + ": " + *response.Message
}

// parseConverseError returns the message of a failed non-streaming Converse
// call, which upstream reports without the error type.
func parseConverseError(body []byte) string {
	var response errorResponse
	if err := json.Unmarshal(body, &response); err != nil || response.Message == nil {
		return ""
	}

	return *response.Message
}

// streamExceptionMessage is the message field of an in-band stream
// exception.
type streamExceptionMessage struct {
	Message *string `json:"message"`
}

// streamError converts an in-band stream exception of errorType, whose
// payload is data, into an API call error.
func streamError(errorType string, data json.RawMessage, url string, headers http.Header) error {
	message := fmt.Sprintf("Amazon Bedrock stream failed with %s", errorType)

	var exception streamExceptionMessage
	if err := json.Unmarshal(data, &exception); err == nil && exception.Message != nil {
		message = *exception.Message
	}

	status, _ := bedrock.StreamErrorStatusOf(errorType)

	return &model.APICallError{
		Message:         message,
		URL:             url,
		StatusCode:      status.StatusCode,
		ResponseHeaders: headers,
		ResponseBody:    string(data),
		IsRetryable:     status.Retryable,
	}
}
