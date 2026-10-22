package vercel

import (
	"context"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Request retries follow @vercel/sandbox's withRetry: network errors, 429
// and 5xx responses are retried twice, with randomized exponential backoff
// from 400 milliseconds. A 429 response first waits for its Retry-After
// delay, and one that asks to wait more than 20 seconds is not retried.
const (
	defaultRequestRetries = 2
	retryMinTimeout       = 400 * time.Millisecond
	retryFactor           = 2
	maxRetryAfter         = 20 * time.Second
)

// do sends a request, retrying transient failures, and returns the
// response. A non-success status is an *APIError, and the body is closed.
// On success the caller owns the body.
func (c *apiClient) do(ctx context.Context, request apiRequest) (*http.Response, error) {
	retries := defaultRequestRetries
	if request.noRetry {
		retries = 0
	}

	for attempt := 1; ; attempt++ {
		hasRetriesRemaining := attempt <= retries

		httpRequest, err := c.prepare(ctx, request)
		if err != nil {
			return nil, err
		}

		response, err := c.httpClient.Do(httpRequest)
		if err != nil {
			if ctx.Err() != nil || !hasRetriesRemaining {
				return nil, fmt.Errorf("vercel sandbox api %s %s: %w", request.method, request.path, err)
			}

			if err := waitForRetry(ctx, retryBackoff(attempt)); err != nil {
				return nil, err
			}

			continue
		}

		if isSuccessStatus(response.StatusCode) {
			return response, nil
		}

		delay, retry := retryDelay(response, attempt, hasRetriesRemaining)
		if !retry {
			return nil, closeWithAPIError(response)
		}

		discardResponse(response)

		if err := waitForRetry(ctx, delay); err != nil {
			return nil, err
		}
	}
}

func isSuccessStatus(statusCode int) bool {
	return statusCode >= 200 && statusCode < 300
}

// retryDelay decides whether a failed response is retried and how long to
// wait first: the backoff, after the Retry-After delay of a 429 response.
func retryDelay(response *http.Response, attempt int, hasRetriesRemaining bool) (time.Duration, bool) {
	switch {
	case response.StatusCode == http.StatusTooManyRequests:
		retryAfter := retryAfterDelay(response.Header.Get("Retry-After"))
		if retryAfter > maxRetryAfter || !hasRetriesRemaining {
			return 0, false
		}

		return retryAfter + retryBackoff(attempt), true

	case response.StatusCode >= 500 && response.StatusCode < 600:
		return retryBackoff(attempt), hasRetriesRemaining

	default:
		return 0, false
	}
}

// retryAfterDelay parses a Retry-After header in seconds. Other forms, such
// as HTTP dates, are ignored as upstream ignores them.
func retryAfterDelay(header string) time.Duration {
	seconds, err := strconv.ParseFloat(strings.TrimSpace(header), 64)
	if err != nil || seconds <= 0 || math.IsInf(seconds, 0) {
		return 0
	}

	return time.Duration(seconds * float64(time.Second))
}

// retryBackoff is async-retry's randomized delay before retry attempt n:
// the minimum timeout times the factor to the power n - 1, times a random
// value between 1 and 2.
func retryBackoff(attempt int) time.Duration {
	base := float64(retryMinTimeout) * math.Pow(retryFactor, float64(attempt-1))

	return time.Duration(base * (1 + rand.Float64()))
}

func waitForRetry(ctx context.Context, delay time.Duration) error {
	timer := time.NewTimer(delay)
	defer timer.Stop()

	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return context.Cause(ctx)
	}
}

// discardResponse drains a little of a retried response so that its
// connection can be reused, and closes it.
func discardResponse(response *http.Response) {
	_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, maxErrorBodyBytes))
	_ = response.Body.Close()
}

func closeWithAPIError(response *http.Response) error {
	defer func() {
		_ = response.Body.Close()
	}()

	return apiErrorOf(response)
}
