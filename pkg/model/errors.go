package model

import (
	"errors"
	"fmt"
	"net/http"
	"time"
)

// Sentinel errors. Providers wrap them with details, so test with errors.Is.
var (
	// ErrInvalidArgument reports an invalid call option or setting.
	ErrInvalidArgument = errors.New("invalid argument")

	// ErrInvalidPrompt reports a prompt the provider cannot represent.
	ErrInvalidPrompt = errors.New("invalid prompt")

	// ErrUnsupported reports functionality the provider or model lacks.
	ErrUnsupported = errors.New("unsupported functionality")

	// ErrInvalidResponse reports a provider response that does not match its
	// documented format.
	ErrInvalidResponse = errors.New("invalid response data")

	// ErrEmptyResponseBody reports a response without a body.
	ErrEmptyResponseBody = errors.New("empty response body")

	// ErrNoContentGenerated reports a successful call that produced nothing.
	ErrNoContentGenerated = errors.New("no content generated")

	// ErrNoSuchModel reports an unknown model identifier.
	ErrNoSuchModel = errors.New("no such model")

	// ErrLoadAPIKey reports a missing API key.
	ErrLoadAPIKey = errors.New("API key not configured")

	// ErrStreamConsumed reports a second iteration over a stream.
	ErrStreamConsumed = errors.New("stream already consumed")
)

// APICallError reports a failed HTTP call to a provider.
type APICallError struct {
	Message string
	URL     string

	// StatusCode is zero when no response was received.
	StatusCode int

	ResponseHeaders http.Header
	ResponseBody    string

	// IsRetryable reports whether repeating the call may succeed.
	IsRetryable bool

	// RetryAfter is the delay the provider requested, or zero.
	RetryAfter time.Duration

	Cause error
}

// Error returns the message with the status code.
func (e *APICallError) Error() string {
	if e.StatusCode == 0 {
		return e.Message
	}

	return fmt.Sprintf("%s (status %d)", e.Message, e.StatusCode)
}

// Unwrap returns the underlying cause.
func (e *APICallError) Unwrap() error {
	return e.Cause
}

// RetryableStatus reports whether an HTTP status indicates a transient
// failure: request timeout, conflict, rate limiting or a server error.
func RetryableStatus(status int) bool {
	switch status {
	case http.StatusRequestTimeout, http.StatusConflict, http.StatusTooManyRequests:
		return true
	default:
		return status >= http.StatusInternalServerError
	}
}
