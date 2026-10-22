package anthropic

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/sse"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// maxPeekBytes bounds how much of a stream is buffered while looking for
	// its first event.
	maxPeekBytes = 8 << 20

	// doneMarker is the data of a terminating event that some hosts, such as
	// Amazon Bedrock, append. It carries no content and is skipped.
	doneMarker = "[DONE]"
)

// Stream performs one streaming Messages API call. An error event that
// arrives before any content is returned as an *model.APICallError, since
// Anthropic reports some failures, such as overload, with a successful
// status.
func (m *MessagesModel) Stream(ctx context.Context, options model.CallOptions) (*model.StreamResult, error) {
	call, err := m.prepareCall(ctx, options, true)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSONStream(ctx, call.api, call.body)
	if err != nil {
		return nil, err
	}

	events, err := openEvents(response.Body)
	if err != nil {
		_ = response.Body.Close()

		return nil, fmt.Errorf("read stream: %w", err)
	}

	if events.firstError != nil {
		_ = response.Body.Close()

		return nil, streamError(events.firstError, call.api.URL, response.Headers, http.StatusInternalServerError)
	}

	session := &streamSession{
		events:     events.reader,
		includeRaw: options.IncludeRawChunks,
		warnings:   call.request.warnings,
		state:      newStreamState(call, response.Headers),
	}

	result := model.NewStreamResult(session.parts, response.Body.Close)
	result.Request = model.RequestInfo{
		Body: response.RequestBody,
	}
	result.Response = model.StreamResponseInfo{
		Headers: response.Headers,
	}

	return result, nil
}

// openedEvents is a stream whose beginning has been inspected. reader
// replays the inspected bytes before the rest of the body.
type openedEvents struct {
	reader     io.Reader
	firstError *errorEvent
}

// openEvents buffers the stream up to its first event that is not a ping
// and reports that event when it is an error.
func openEvents(body io.Reader) (openedEvents, error) {
	reader := bufio.NewReader(body)

	var peeked bytes.Buffer

	// eventStart is the offset of the event being read, so that each
	// completed event is parsed once.
	eventStart := 0

	for peeked.Len() < maxPeekBytes {
		line, readErr := reader.ReadBytes('\n')
		peeked.Write(line)

		if readErr != nil && !errors.Is(readErr, io.EOF) {
			return openedEvents{}, readErr
		}

		eventEnded := len(bytes.TrimRight(line, "\r\n")) == 0
		if eventEnded || readErr != nil {
			if first, found := firstContentEvent(peeked.Bytes()[eventStart:]); found {
				return openedEvents{
					reader:     io.MultiReader(bytes.NewReader(peeked.Bytes()), reader),
					firstError: first,
				}, nil
			}

			eventStart = peeked.Len()
		}

		if readErr != nil {
			break
		}
	}

	return openedEvents{
		reader: io.MultiReader(bytes.NewReader(peeked.Bytes()), reader),
	}, nil
}

// firstContentEvent finds the first complete event that is not a ping. It
// returns the event when it is an error event, and nil otherwise.
func firstContentEvent(data []byte) (*errorEvent, bool) {
	for event, err := range sse.Read(bytes.NewReader(data)) {
		if err != nil {
			return nil, false
		}

		if event.Data == doneMarker {
			continue
		}

		decoded, err := decodeStreamEvent([]byte(event.Data))
		if err != nil {
			return nil, true
		}

		if _, isPing := decoded.(*pingEvent); isPing {
			continue
		}

		failure, isError := decoded.(*errorEvent)
		if isError {
			return failure, true
		}

		return nil, true
	}

	return nil, false
}

// streamSession turns the events of one response into stream parts.
type streamSession struct {
	events     io.Reader
	includeRaw bool
	warnings   []model.Warning
	state      *streamState
}

func (s *streamSession) parts(yield func(model.StreamPart) bool) {
	start := model.StreamStart{
		Warnings: s.warnings,
	}
	if !yield(start) {
		return
	}

	for event, err := range sse.Read(s.events) {
		if err != nil {
			s.state.reportedError = true
			if !yield(model.StreamError{
				Err: err,
			}) {
				return
			}

			break
		}

		if !yieldAll(yield, s.eventParts(event)) {
			return
		}
	}

	yieldAll(yield, s.state.endParts())
}

// eventParts returns the parts of one server-sent event: its raw form when
// requested, then its decoded parts.
func (s *streamSession) eventParts(event sse.Event) []model.StreamPart {
	if s.state.invalidSequence || event.Data == doneMarker {
		return nil
	}

	var parts []model.StreamPart
	data := []byte(event.Data)

	if s.includeRaw {
		parts = append(parts, model.Raw{
			Value: rawEventValue(data),
		})
	}

	decoded, err := decodeStreamEvent(data)
	if err != nil {
		return append(parts, model.StreamError{
			Err: fmt.Errorf("%w: stream event: %w", model.ErrInvalidResponse, err),
		})
	}

	return append(parts, s.state.handle(decoded)...)
}

// rawEventValue returns event data as JSON, quoting data that is not JSON.
func rawEventValue(data []byte) json.RawMessage {
	if json.Valid(data) {
		return json.RawMessage(data)
	}

	return mustEncodeJSON(string(data))
}

func yieldAll(yield func(model.StreamPart) bool, parts []model.StreamPart) bool {
	for _, part := range parts {
		if !yield(part) {
			return false
		}
	}

	return true
}
