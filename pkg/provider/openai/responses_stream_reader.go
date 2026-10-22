package openai

import (
	"encoding/json"
	"io"
	"iter"
	"net/http"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/internal/sse"
	"github.com/aarock1234/ai/pkg/model"
)

// responsesStreamChunk is one streaming event. Event is nil for events the
// stream ignores; Err reports a known event that failed to decode.
type responsesStreamChunk struct {
	data  []byte
	event responsesEvent
	err   error
}

// responsesEventReader reads decoded events from a response body. It can
// look ahead for early errors and replay what it read.
type responsesEventReader struct {
	body     io.ReadCloser
	next     func() (sse.Event, error, bool)
	stop     func()
	buffered []responsesStreamChunk
}

func newResponsesEventReader(body io.ReadCloser) *responsesEventReader {
	next, stop := iter.Pull2(sse.Read(body))

	return &responsesEventReader{
		body: body,
		next: next,
		stop: stop,
	}
}

// read returns the next chunk, replaying looked-ahead chunks first. It
// reports false at the end of the stream.
func (r *responsesEventReader) read() (responsesStreamChunk, bool, error) {
	if len(r.buffered) > 0 {
		chunk := r.buffered[0]
		r.buffered = r.buffered[1:]

		return chunk, true, nil
	}

	return r.pull()
}

// pull reads and decodes the next event from the body.
func (r *responsesEventReader) pull() (responsesStreamChunk, bool, error) {
	for {
		event, err, ok := r.next()
		if !ok {
			return responsesStreamChunk{}, false, nil
		}

		if err != nil {
			return responsesStreamChunk{}, false, err
		}

		if event.Data == "[DONE]" {
			continue
		}

		data := []byte(event.Data)
		decoded, decodeErr := decodeResponsesEvent(data)

		return responsesStreamChunk{
			data:  data,
			event: decoded,
			err:   decodeErr,
		}, true, nil
	}
}

// checkEarlyError reads ahead until the first output event and returns an
// error when the API reports a failure before it, such as an exhausted
// quota after a successful status. The events read are replayed.
func (r *responsesEventReader) checkEarlyError(failure responsesStreamFailure) error {
	for {
		chunk, ok, err := r.pull()
		if err != nil {
			return err
		}

		if !ok {
			return nil
		}

		r.buffered = append(r.buffered, chunk)

		if chunk.err != nil {
			return nil
		}

		if details, isError := responsesEarlyErrorDetails(chunk.event); isError {
			return failure.apiError(details, chunk.data)
		}

		if responsesIsOutputEvent(chunk.event) {
			return nil
		}
	}
}

// close stops reading and closes the body. It must not run concurrently
// with read.
func (r *responsesEventReader) close() error {
	r.stop()

	return r.body.Close()
}

// responsesIsOutputEvent reports whether an event carries output rather
// than response lifecycle or error information.
func responsesIsOutputEvent(event responsesEvent) bool {
	if event == nil {
		return false
	}

	switch event.responsesEventKind() {
	case responsesEventCreated, responsesEventInProgress, responsesEventFailed, responsesEventError:
		return false
	default:
		return true
	}
}

// responsesEarlyErrorDetails returns the details of error events and of
// failed responses that carry an error.
func responsesEarlyErrorDetails(event responsesEvent) (responsesErrorDetails, bool) {
	switch event.(type) {
	case responsesErrorEvent, responsesFailedEvent:
		return responsesErrorDetailsOf(event)
	default:
		return responsesErrorDetails{}, false
	}
}

// responsesStreamFailure is the request information that stream errors
// report.
type responsesStreamFailure struct {
	url     string
	headers http.Header
}

// responsesErrorDetails is the error reported by an error frame.
type responsesErrorDetails struct {
	message   string
	code      string
	errorType string
}

// responsesErrorDetailsOf extracts the error of an error event or a failed
// response. It reports false when the frame carries no error message.
func responsesErrorDetailsOf(event responsesEvent) (responsesErrorDetails, bool) {
	switch event := event.(type) {
	case responsesErrorEvent:
		if event.Error != nil {
			return responsesErrorDetails{
				message:   event.Error.Message,
				code:      string(event.Error.Code),
				errorType: event.Error.Type,
			}, event.Error.Message != ""
		}

		return responsesErrorDetails{
			message:   event.Message,
			code:      string(event.Code),
			errorType: string(responsesEventError),
		}, event.Message != ""

	case responsesFailedEvent:
		failure := event.Response.Error
		if failure == nil {
			return responsesErrorDetails{}, false
		}

		return responsesErrorDetails{
			message:   failure.Message,
			code:      string(failure.Code),
			errorType: string(responsesEventFailed),
		}, true

	default:
		return responsesErrorDetails{}, false
	}
}

// apiError converts error details into an API call error whose body is
// the error frame.
func (f responsesStreamFailure) apiError(details responsesErrorDetails, frame []byte) *model.APICallError {
	status := responsesErrorStatus(details)

	message := details.message
	if message == "" {
		message = "OpenAI stream failed"
	}

	isQuotaError := details.code == "insufficient_quota" || details.errorType == "insufficient_quota"

	return &model.APICallError{
		Message:         message,
		URL:             f.url,
		StatusCode:      status,
		ResponseHeaders: f.headers,
		ResponseBody:    string(frame),
		IsRetryable:     !isQuotaError && model.RetryableStatus(status),
	}
}

// chatCompletionsMismatch reports a Chat Completions chunk received from a
// Responses API request, which happens when a custom base URL points at a
// Chat Completions-compatible endpoint.
func (f responsesStreamFailure) chatCompletionsMismatch(frame []byte, cause error) *model.APICallError {
	return &model.APICallError{
		Message: "Received a Chat Completions stream while using the OpenAI Responses API. " +
			"The default OpenAI provider model uses the Responses API. If your custom base URL targets " +
			"a Chat Completions-compatible endpoint, use the provider's Chat model instead.",
		URL:             f.url,
		ResponseHeaders: f.headers,
		ResponseBody:    string(frame),
		Cause:           cause,
	}
}

// responsesIsChatCompletionChunk reports whether data looks like a Chat
// Completions chunk: an object with a choices array and no string type.
func responsesIsChatCompletionChunk(data []byte) bool {
	var chunk struct {
		Choices json.RawMessage `json:"choices"`
		Type    json.RawMessage `json:"type"`
	}

	if err := json.Unmarshal(data, &chunk); err != nil {
		return false
	}

	var text string
	hasStringType := json.Unmarshal(chunk.Type, &text) == nil

	return isJSONArray(chunk.Choices) && !hasStringType
}

// responsesErrorStatus derives an HTTP status from an error code or type.
func responsesErrorStatus(details responsesErrorDetails) int {
	if status, ok := responsesHTTPStatus(details.code); ok {
		return status
	}

	discriminator := strings.ToLower(details.code + " " + details.errorType)

	switch {
	case responsesContainsAny(discriminator, "insufficient_quota", "rate_limit"):
		return http.StatusTooManyRequests
	case strings.Contains(discriminator, "authentication"):
		return http.StatusUnauthorized
	case strings.Contains(discriminator, "permission"):
		return http.StatusForbidden
	case strings.Contains(discriminator, "not_found"):
		return http.StatusNotFound
	case responsesContainsAny(discriminator, "invalid", "bad_request", "context_length"):
		return http.StatusBadRequest
	case strings.Contains(discriminator, "overload"):
		return http.StatusServiceUnavailable
	case strings.Contains(discriminator, "timeout"):
		return http.StatusGatewayTimeout
	default:
		return http.StatusInternalServerError
	}
}

// responsesHTTPStatus parses a three-digit error status code.
func responsesHTTPStatus(code string) (int, bool) {
	if len(code) != 3 {
		return 0, false
	}

	status, err := strconv.Atoi(code)
	if err != nil {
		return 0, false
	}

	isErrorStatus := status >= http.StatusBadRequest && status <= 599

	return status, isErrorStatus
}

func responsesContainsAny(text string, terms ...string) bool {
	for _, term := range terms {
		if strings.Contains(text, term) {
			return true
		}
	}

	return false
}
