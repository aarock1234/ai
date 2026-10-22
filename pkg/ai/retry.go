package ai

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

const (
	defaultMaxRetries        = 2
	defaultInitialRetryDelay = 2 * time.Second
	defaultRetryBackoff      = 2.0

	// maxHonoredRetryAfter is the longest provider-requested delay that is
	// used when it exceeds the exponential backoff delay.
	maxHonoredRetryAfter = 60 * time.Second
)

// RetryReason explains why retrying stopped.
type RetryReason string

// Retry reasons.
const (
	RetryMaxRetriesExceeded RetryReason = "maxRetriesExceeded"
	RetryErrorNotRetryable  RetryReason = "errorNotRetryable"
)

// RetryError reports a call that failed after at least one retry. Failures of
// the first attempt are returned unwrapped.
type RetryError struct {
	Reason RetryReason

	// Errors holds the failure of every attempt, in order.
	Errors []error
}

// Error returns the attempt count and the last failure.
func (e *RetryError) Error() string {
	if e.Reason == RetryErrorNotRetryable {
		return fmt.Sprintf("failed after %d attempts with non-retryable error: %v", len(e.Errors), e.LastError())
	}

	return fmt.Sprintf("failed after %d attempts: last error: %v", len(e.Errors), e.LastError())
}

// LastError returns the failure of the final attempt.
func (e *RetryError) LastError() error {
	if len(e.Errors) == 0 {
		return nil
	}

	return e.Errors[len(e.Errors)-1]
}

// Unwrap returns the failure of the final attempt.
func (e *RetryError) Unwrap() error {
	return e.LastError()
}

// retryPolicy repeats failed model calls with exponential backoff, honoring
// the delay a provider requests through Retry-After headers.
type retryPolicy struct {
	maxRetries    int
	initialDelay  time.Duration
	backoffFactor float64

	// isRetryable optionally marks additional errors as retryable, besides
	// retryable *model.APICallError values.
	isRetryable func(err error) bool
}

// prepareRetries validates a MaxRetries option and returns the retry policy
// for it. Nil selects the default of two retries.
func prepareRetries(maxRetries *int) (retryPolicy, error) {
	policy := retryPolicy{
		maxRetries:    defaultMaxRetries,
		initialDelay:  defaultInitialRetryDelay,
		backoffFactor: defaultRetryBackoff,
	}

	if maxRetries == nil {
		return policy, nil
	}

	if *maxRetries < 0 {
		return retryPolicy{}, &InvalidArgumentError{
			Parameter: "maxRetries",
			Value:     *maxRetries,
			Message:   "maxRetries must be >= 0",
		}
	}

	policy.maxRetries = *maxRetries

	return policy, nil
}

// retryCall runs call until it succeeds, fails with an error that is not
// retryable, exhausts the retries or ctx ends. Cancellation errors are
// returned as is, and so is any failure of the first attempt that is not
// retried.
func retryCall[T any](ctx context.Context, policy retryPolicy, call func(ctx context.Context) (T, error)) (T, error) {
	var zero T
	var failures []error

	delay := policy.initialDelay

	for {
		result, err := call(ctx)
		if err == nil {
			return result, nil
		}

		if ctx.Err() != nil || policy.maxRetries == 0 {
			return zero, err
		}

		failures = append(failures, err)
		attempts := len(failures)

		if attempts > policy.maxRetries {
			return zero, &RetryError{
				Reason: RetryMaxRetriesExceeded,
				Errors: failures,
			}
		}

		if !policy.shouldRetry(err) {
			if attempts == 1 {
				return zero, err
			}

			return zero, &RetryError{
				Reason: RetryErrorNotRetryable,
				Errors: failures,
			}
		}

		if err := sleep(ctx, retryDelay(err, delay)); err != nil {
			return zero, err
		}

		delay = time.Duration(float64(delay) * policy.backoffFactor)
	}
}

func (p retryPolicy) shouldRetry(err error) bool {
	var apiErr *model.APICallError
	if errors.As(err, &apiErr) && apiErr.IsRetryable {
		return true
	}

	return p.isRetryable != nil && p.isRetryable(err)
}

// retryDelay prefers the provider's requested delay when it is reasonable:
// under a minute, or shorter than the exponential backoff delay.
func retryDelay(err error, exponentialDelay time.Duration) time.Duration {
	var apiErr *model.APICallError
	if !errors.As(err, &apiErr) || apiErr.RetryAfter <= 0 {
		return exponentialDelay
	}

	requested := apiErr.RetryAfter
	if requested < maxHonoredRetryAfter || requested < exponentialDelay {
		return requested
	}

	return exponentialDelay
}

// sleep waits for delay or until ctx ends.
func sleep(ctx context.Context, delay time.Duration) error {
	if delay <= 0 {
		return context.Cause(ctx)
	}

	timer := time.NewTimer(delay)
	defer timer.Stop()

	select {
	case <-ctx.Done():
		return context.Cause(ctx)
	case <-timer.C:
		return nil
	}
}
