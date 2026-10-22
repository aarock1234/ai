package apicall

import (
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

// maxErrorBodyBytes bounds how much of a failed response is read.
const maxErrorBodyBytes = 64 << 10

// statusError reads a failed response and closes its body.
func statusError(request Request, response *http.Response) error {
	defer closeBody(response.Body)

	body, _ := io.ReadAll(io.LimitReader(response.Body, maxErrorBodyBytes))

	return &model.APICallError{
		Message:         errorMessage(request.ParseError, response.StatusCode, body),
		URL:             request.URL,
		StatusCode:      response.StatusCode,
		ResponseHeaders: response.Header,
		ResponseBody:    string(body),
		IsRetryable:     model.RetryableStatus(response.StatusCode),
		RetryAfter:      retryAfter(response.Header, time.Now()),
	}
}

func errorMessage(parse ErrorParser, status int, body []byte) string {
	if parse != nil {
		if message := parse(body); message != "" {
			return message
		}
	}

	if text := http.StatusText(status); text != "" {
		return strings.ToLower(text)
	}

	return "provider request failed"
}

// retryAfter reads the delay requested by retry-after-ms or Retry-After,
// which may hold seconds or an HTTP date. It returns zero when absent or
// malformed.
func retryAfter(header http.Header, now time.Time) time.Duration {
	if value := header.Get("Retry-After-Ms"); value != "" {
		milliseconds, err := strconv.ParseFloat(value, 64)
		if err == nil && milliseconds > 0 {
			return time.Duration(milliseconds * float64(time.Millisecond))
		}
	}

	value := header.Get("Retry-After")
	if value == "" {
		return 0
	}

	if seconds, err := strconv.ParseFloat(value, 64); err == nil {
		if seconds <= 0 {
			return 0
		}

		return time.Duration(seconds * float64(time.Second))
	}

	if date, err := http.ParseTime(value); err == nil && date.After(now) {
		return date.Sub(now)
	}

	return 0
}
