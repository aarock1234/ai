package anthropic

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/internal/awseventstream"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/amazonbedrock/internal/bedrock"
)

const (
	eventStreamContentType = "application/vnd.amazon.eventstream"

	// maxErrorBodyBytes bounds the error bodies that are rewritten.
	maxErrorBodyBytes = 64 << 10
)

// eventStreamTransport makes Bedrock responses look like Anthropic API
// responses: error bodies take the Anthropic error shape, and AWS event
// streams become server-sent events carrying the Anthropic stream events.
type eventStreamTransport struct {
	base http.RoundTripper
}

// RoundTrip sends the request and adapts its response.
func (t *eventStreamTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	response, err := t.base.RoundTrip(request)
	if err != nil {
		return nil, err
	}

	if response.StatusCode < 200 || response.StatusCode > 299 {
		return anthropicErrorResponse(response)
	}

	if !strings.Contains(response.Header.Get("Content-Type"), eventStreamContentType) {
		return response, nil
	}

	response.Header = response.Header.Clone()
	response.Header.Set("Content-Type", "text/event-stream")
	response.Header.Del("Content-Length")
	response.ContentLength = -1
	response.Body = &sseBody{
		messages: awseventstream.NewReader(response.Body),
		source:   response.Body,
	}

	return response, nil
}

// bedrockError is the body of a failed Bedrock call.
type bedrockError struct {
	Message *string `json:"message"`
}

// anthropicError is the Anthropic API error body.
type anthropicError struct {
	Type  string               `json:"type"`
	Error anthropicErrorDetail `json:"error"`
}

type anthropicErrorDetail struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// anthropicErrorResponse rewrites a Bedrock error body into an Anthropic
// error body with the same message, so that its message is reported.
func anthropicErrorResponse(response *http.Response) (*http.Response, error) {
	defer func() {
		_ = response.Body.Close()
	}()

	text, err := io.ReadAll(io.LimitReader(response.Body, maxErrorBodyBytes))
	if err != nil {
		return nil, fmt.Errorf("read error response: %w", err)
	}

	message := string(text)

	var parsed bedrockError
	if err := json.Unmarshal(text, &parsed); err == nil && parsed.Message != nil && *parsed.Message != "" {
		message = *parsed.Message
	}

	body, err := json.Marshal(anthropicError{
		Type: "error",
		Error: anthropicErrorDetail{
			Type:    "error",
			Message: message,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("encode error response: %w", err)
	}

	response.Header = response.Header.Clone()
	response.Header.Set("Content-Length", strconv.Itoa(len(body)))
	response.ContentLength = int64(len(body))
	response.Body = io.NopCloser(bytes.NewReader(body))

	return response, nil
}

// sseBody converts event stream messages into server-sent events as they
// are read. Closing it closes the response body.
type sseBody struct {
	messages *awseventstream.Reader
	source   io.Closer
	pending  bytes.Buffer
	err      error
}

// Read returns converted events, reading messages as needed.
func (b *sseBody) Read(buffer []byte) (int, error) {
	for b.pending.Len() == 0 {
		if b.err != nil {
			return 0, b.err
		}

		message, err := b.messages.Next()
		if err != nil {
			b.err = err

			continue
		}

		event, err := sseEventOf(message)
		if err != nil {
			b.err = err

			continue
		}

		b.pending.Write(event)
	}

	return b.pending.Read(buffer)
}

// Close closes the response body.
func (b *sseBody) Close() error {
	return b.source.Close()
}

// chunkPayload is the payload of a chunk event: a base64 Anthropic event.
type chunkPayload struct {
	Bytes string `json:"bytes"`
}

// exceptionPayload holds the message of an exception.
type exceptionPayload struct {
	Message *string `json:"message"`
}

// streamErrorEvent is the Anthropic error event an exception becomes.
type streamErrorEvent struct {
	Type  string            `json:"type"`
	Error streamErrorDetail `json:"error"`
}

type streamErrorDetail struct {
	Type        string          `json:"type"`
	Message     string          `json:"message"`
	StatusCode  *int            `json:"statusCode,omitempty"`
	IsRetryable *bool           `json:"isRetryable,omitempty"`
	Data        json.RawMessage `json:"data"`
}

// sseEventOf converts a message: chunks into their Anthropic events,
// messageStop into [DONE] and exceptions into error events. Other messages
// produce nothing.
func sseEventOf(message awseventstream.Message) ([]byte, error) {
	switch message.MessageType() {
	case "event":
		return eventData(message)
	case "exception":
		return exceptionData(message)
	default:
		return nil, nil
	}
}

func eventData(message awseventstream.Message) ([]byte, error) {
	switch message.EventType() {
	case "chunk":
		var chunk chunkPayload
		if err := json.Unmarshal(message.Payload, &chunk); err != nil {
			return nil, fmt.Errorf("%w: decode Amazon Bedrock chunk: %w", model.ErrInvalidResponse, err)
		}

		if chunk.Bytes == "" {
			return sseData(message.Payload), nil
		}

		event, err := base64.StdEncoding.DecodeString(chunk.Bytes)
		if err != nil {
			return nil, fmt.Errorf("%w: decode Amazon Bedrock chunk bytes: %w", model.ErrInvalidResponse, err)
		}

		return sseData(event), nil

	case "messageStop":
		return sseData([]byte("[DONE]")), nil

	default:
		return nil, nil
	}
}

func exceptionData(message awseventstream.Message) ([]byte, error) {
	data := json.RawMessage(message.Payload)
	text := string(message.Payload)

	if !json.Valid(message.Payload) {
		encoded, err := json.Marshal(text)
		if err != nil {
			return nil, fmt.Errorf("encode exception payload: %w", err)
		}

		data = encoded
	}

	var payload exceptionPayload
	if err := json.Unmarshal(message.Payload, &payload); err == nil && payload.Message != nil {
		text = *payload.Message
	}

	errorType := message.EventType()
	if errorType == "" {
		errorType = message.ExceptionType()
	}

	if errorType == "" {
		errorType = "error"
	}

	detail := streamErrorDetail{
		Type:    errorType,
		Message: text,
		Data:    data,
	}

	if status, known := bedrock.StreamErrorStatusOf(errorType); known {
		detail.StatusCode = &status.StatusCode
		detail.IsRetryable = &status.Retryable
	}

	event, err := json.Marshal(streamErrorEvent{
		Type:  "error",
		Error: detail,
	})
	if err != nil {
		return nil, fmt.Errorf("encode exception event: %w", err)
	}

	return sseData(event), nil
}

func sseData(data []byte) []byte {
	return []byte("data: " + string(data) + "\n\n")
}
