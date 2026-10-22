package openai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"regexp"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/internal/sse"
	"github.com/aarock1234/ai/pkg/model"
)

// chatDoneData is the data of the event that ends a Chat Completions or
// Completions stream.
const chatDoneData = "[DONE]"

// chatStatusCodePattern matches an HTTP status code given as an error code.
var chatStatusCodePattern = regexp.MustCompile(`^\d{3}$`)

// chatEvent is one server-sent event or the error that ended the stream.
type chatEvent struct {
	data string
	err  error
}

// chatEventSource reads the events of a Chat Completions or Completions
// stream. It can read ahead to detect a failure before any output, and then
// replays the events it read.
type chatEventSource struct {
	body io.ReadCloser
	next func() (sse.Event, error, bool)
	stop func()

	pending []chatEvent
}

func newChatEventSource(body io.ReadCloser) *chatEventSource {
	next, stop := iter.Pull2(sse.Read(body))

	return &chatEventSource{
		body: body,
		next: next,
		stop: stop,
	}
}

// chatPeekResult classifies an event read ahead of the consumer.
type chatPeekResult int

const (
	// chatPeekContinue keeps reading ahead.
	chatPeekContinue chatPeekResult = iota

	// chatPeekReady hands the stream to the consumer.
	chatPeekReady
)

// peek reads events until classify reports output, an undecodable event, or
// an error frame, which it returns as an error. The [DONE] event and the end
// of the stream also stop reading ahead. Read events are kept for events.
func (s *chatEventSource) peek(classify func(data string) (chatPeekResult, error)) error {
	for {
		event, ok := s.read()
		if !ok {
			return nil
		}

		s.pending = append(s.pending, event)

		if event.err != nil || event.data == chatDoneData {
			return nil
		}

		result, err := classify(event.data)
		if err != nil {
			return err
		}

		if result == chatPeekReady {
			return nil
		}
	}
}

// events yields the events read ahead, then the rest of the stream, without
// the [DONE] event. An error ends the sequence.
func (s *chatEventSource) events() iter.Seq[chatEvent] {
	return func(yield func(chatEvent) bool) {
		for {
			event, ok := s.nextEvent()
			if !ok {
				return
			}

			if event.data == chatDoneData && event.err == nil {
				continue
			}

			if !yield(event) || event.err != nil {
				return
			}
		}
	}
}

// close stops reading and closes the response body.
func (s *chatEventSource) close() error {
	s.stop()

	return s.body.Close()
}

func (s *chatEventSource) nextEvent() (chatEvent, bool) {
	if len(s.pending) > 0 {
		event := s.pending[0]
		s.pending = s.pending[1:]

		return event, true
	}

	return s.read()
}

func (s *chatEventSource) read() (chatEvent, bool) {
	event, err, ok := s.next()
	if !ok {
		return chatEvent{}, false
	}

	if err != nil {
		return chatEvent{
			err: err,
		}, true
	}

	return chatEvent{
		data: event.Data,
	}, true
}

// chatRaw returns the event data as a raw stream part, when it is JSON.
func chatRaw(data string) (model.Raw, bool) {
	if !json.Valid([]byte(data)) {
		return model.Raw{}, false
	}

	return model.Raw{
		Value: json.RawMessage(data),
	}, true
}

// chatErrorFrame is the error object of a stream error frame.
type chatErrorFrame struct {
	Message string          `json:"message"`
	Type    string          `json:"type"`
	Param   json.RawMessage `json:"param"`
	Code    json.RawMessage `json:"code"`
}

// chatStreamOrigin identifies the HTTP exchange of a stream, for errors.
type chatStreamOrigin struct {
	url     string
	headers http.Header
}

// apiError converts an error frame into an API call error with an inferred
// status code, so that retry decisions treat it like an HTTP failure.
func (o chatStreamOrigin) apiError(frame chatErrorFrame, data string) *model.APICallError {
	message := frame.Message
	if message == "" {
		message = "openai stream failed"
	}

	code := frame.code()
	statusCode := chatFrameStatusCode(code, frame.Type)

	return &model.APICallError{
		Message:         message,
		URL:             o.url,
		StatusCode:      statusCode,
		ResponseHeaders: o.headers,
		ResponseBody:    data,
		IsRetryable:     chatFrameRetryable(code, frame.Type, statusCode),
	}
}

// code returns the error code as text. Codes may be strings or numbers.
func (f chatErrorFrame) code() string {
	trimmed := bytes.TrimSpace(f.Code)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return ""
	}

	var text string
	if err := json.Unmarshal(trimmed, &text); err == nil {
		return text
	}

	var number json.Number
	if err := json.Unmarshal(trimmed, &number); err == nil {
		return number.String()
	}

	return ""
}

// chatFrameStatusCode uses an HTTP status given as the code, or infers one
// from words in the code and type.
func chatFrameStatusCode(code, errorType string) int {
	if chatStatusCodePattern.MatchString(code) {
		status, err := strconv.Atoi(code)
		if err == nil && status >= 400 && status <= 599 {
			return status
		}
	}

	discriminator := strings.ToLower(code + " " + errorType)

	for _, rule := range chatFrameStatusRules {
		for _, term := range rule.terms {
			if strings.Contains(discriminator, term) {
				return rule.status
			}
		}
	}

	return http.StatusInternalServerError
}

// chatFrameStatusRule maps words in an error code or type to a status code.
type chatFrameStatusRule struct {
	terms  []string
	status int
}

// chatFrameStatusRules are checked in order; the first matching rule wins.
var chatFrameStatusRules = []chatFrameStatusRule{
	{
		terms: []string{
			"insufficient_quota",
			"rate_limit",
		},
		status: http.StatusTooManyRequests,
	},
	{
		terms: []string{
			"authentication",
		},
		status: http.StatusUnauthorized,
	},
	{
		terms: []string{
			"permission",
		},
		status: http.StatusForbidden,
	},
	{
		terms: []string{
			"not_found",
		},
		status: http.StatusNotFound,
	},
	{
		terms: []string{
			"invalid",
			"bad_request",
			"context_length",
		},
		status: http.StatusBadRequest,
	},
	{
		terms: []string{
			"overload",
		},
		status: http.StatusServiceUnavailable,
	},
	{
		terms: []string{
			"timeout",
		},
		status: http.StatusGatewayTimeout,
	},
}

// chatFrameRetryable treats quota exhaustion as permanent and otherwise
// follows the status code.
func chatFrameRetryable(code, errorType string, statusCode int) bool {
	if code == "insufficient_quota" || errorType == "insufficient_quota" {
		return false
	}

	return model.RetryableStatus(statusCode)
}

// chatDecodeError wraps an undecodable stream event.
func chatDecodeError(err error) error {
	return fmt.Errorf("%w: decode stream event: %w", model.ErrInvalidResponse, err)
}
