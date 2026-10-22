package mcp

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"syscall"
	"time"

	"github.com/aarock1234/ai/internal/poll"
)

// Exponential backoff between tool call attempts.
const (
	initialRetryDelay  = 2 * time.Second
	retryBackoffFactor = 2
)

// retryableConnectionErrors are network failures after which a tool call
// may be repeated.
var retryableConnectionErrors = []error{
	syscall.ECONNREFUSED,
	syscall.ECONNRESET,
	syscall.ETIMEDOUT,
	syscall.EPIPE,
	io.ErrUnexpectedEOF,
}

// retryToolCall runs call and retries transient failures up to maxRetries
// times with exponential backoff. Cancellation is never retried.
func retryToolCall(
	ctx context.Context,
	maxRetries int,
	call func(ctx context.Context) (*CallToolResult, error),
) (*CallToolResult, error) {
	if maxRetries == 0 {
		return call(ctx)
	}

	var failures []error
	delay := initialRetryDelay

	for {
		result, err := call(ctx)
		if err == nil {
			return result, nil
		}

		if ctx.Err() != nil {
			return nil, err
		}

		failures = append(failures, err)
		attempts := len(failures)

		if attempts > maxRetries {
			return nil, &ClientError{
				Message: fmt.Sprintf("failed after %d attempts. last error: %v", attempts, err),
				Cause:   err,
			}
		}

		if !isRetryableToolCallError(err) {
			return nil, notRetryableError(failures)
		}

		if err := poll.Wait(ctx, delay); err != nil {
			return nil, err
		}

		delay *= retryBackoffFactor
	}
}

// notRetryableError returns the first failure unwrapped and later failures
// with the attempt count.
func notRetryableError(failures []error) error {
	last := failures[len(failures)-1]
	if len(failures) == 1 {
		return last
	}

	return &ClientError{
		Message: fmt.Sprintf("failed after %d attempts with non-retryable error: '%v'", len(failures), last),
		Cause:   last,
	}
}

// isRetryableToolCallError reports HTTP 408, 409, 429 and 5xx failures and
// connection failures as transient. JSON-RPC errors are not.
func isRetryableToolCallError(err error) bool {
	if clientErr, ok := errors.AsType[*ClientError](err); ok {
		if clientErr.StatusCode != 0 {
			return isRetryableStatus(clientErr.StatusCode)
		}

		if clientErr.Code != nil {
			return false
		}
	}

	for _, connectionErr := range retryableConnectionErrors {
		if errors.Is(err, connectionErr) {
			return true
		}
	}

	return false
}

func isRetryableStatus(statusCode int) bool {
	return statusCode == http.StatusRequestTimeout ||
		statusCode == http.StatusConflict ||
		statusCode == http.StatusTooManyRequests ||
		statusCode >= http.StatusInternalServerError
}
