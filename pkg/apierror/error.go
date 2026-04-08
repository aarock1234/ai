// Package apierror defines the APIError type used across provider
// implementations and the consumer-facing ai package. It lives in its
// own package so that low-level infrastructure (openaicompat) and
// high-level consumers (ai) can both reference it without a circular
// import.
package apierror

import "fmt"

// APIError represents an error response from a provider's HTTP API.
type APIError struct {
	StatusCode  int
	Message     string
	IsRetryable bool
}

// Error implements the error interface.
func (e *APIError) Error() string {
	return fmt.Sprintf("api error %d: %s", e.StatusCode, e.Message)
}

// New creates an APIError, automatically marking 429 and 5xx status
// codes as retryable.
func New(statusCode int, message string) *APIError {
	retryable := statusCode == 429 || statusCode >= 500

	return &APIError{
		StatusCode:  statusCode,
		Message:     message,
		IsRetryable: retryable,
	}
}
