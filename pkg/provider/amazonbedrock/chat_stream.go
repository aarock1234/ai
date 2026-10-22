package amazonbedrock

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/awseventstream"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/amazonbedrock/internal/bedrock"
)

// Stream performs one ConverseStream call. The response is an AWS event
// stream; in-band exceptions become error parts.
func (m *ChatModel) Stream(ctx context.Context, options model.CallOptions) (*model.StreamResult, error) {
	prepared, err := m.prepareRequest(options)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(ctx, bedrock.Runtime, bedrock.ModelPath(m.modelID, "converse-stream"), options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSONStream(ctx, request, prepared.body)
	if err != nil {
		return nil, err
	}

	session := &streamSession{
		ctx:        ctx,
		reader:     awseventstream.NewReader(response.Body),
		includeRaw: options.IncludeRawChunks,
		warnings:   prepared.warnings,
		metadata:   m.responseMetadata(response.Headers),
		state:      newStreamState(m, prepared, request.URL, response.Headers),
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

// streamSession reads the event stream of one response.
type streamSession struct {
	ctx        context.Context
	reader     *awseventstream.Reader
	includeRaw bool
	warnings   []model.Warning
	metadata   model.ResponseMetadata
	state      *streamState
}

func (s *streamSession) parts(yield func(model.StreamPart) bool) {
	start := model.StreamStart{
		Warnings: s.warnings,
	}

	if !yield(start) || !yield(s.metadata) {
		return
	}

	for {
		message, err := s.reader.Next()
		if errors.Is(err, io.EOF) {
			break
		}

		if err != nil {
			if !yield(s.state.fail(s.readError(err))) {
				return
			}

			break
		}

		if !yieldAll(yield, s.messageParts(message)) {
			return
		}
	}

	yield(s.state.finish())
}

// messageParts returns the parts of one event stream message: its raw form
// when requested, then its decoded parts.
func (s *streamSession) messageParts(message awseventstream.Message) []model.StreamPart {
	payloadType := ""
	exception := false

	switch message.MessageType() {
	case "event":
		payloadType = message.EventType()
	case "exception":
		payloadType = message.ExceptionType()
		exception = true
	}

	if payloadType == "" {
		return nil
	}

	event, err := decodeStreamEvent(payloadType, message.Payload)
	if err != nil {
		return []model.StreamPart{
			s.state.fail(err),
		}
	}

	var parts []model.StreamPart
	if s.includeRaw {
		parts = append(parts, model.Raw{
			Value: event.raw,
		})
	}

	if exception || streamExceptionTypes[payloadType] {
		return append(parts, s.state.exception(payloadType, event))
	}

	eventParts, err := s.state.handle(event)
	if err != nil {
		return append(parts, s.state.fail(err))
	}

	return append(parts, eventParts...)
}

// readError reports the cause of the context when ctx ended the read.
func (s *streamSession) readError(err error) error {
	if s.ctx.Err() != nil {
		return context.Cause(s.ctx)
	}

	return fmt.Errorf("read amazon bedrock event stream: %w", err)
}

func yieldAll(yield func(model.StreamPart) bool, parts []model.StreamPart) bool {
	for _, part := range parts {
		if !yield(part) {
			return false
		}
	}

	return true
}

// newStreamState returns the state of a stream for the prepared request.
func newStreamState(m *ChatModel, prepared *preparedRequest, url string, headers http.Header) *streamState {
	state := &streamState{
		prepared: prepared,
		mistral:  isMistralModel(m.modelID),
		url:      url,
		headers:  headers,
		blocks:   make(map[int]*streamBlock),
		finishReason: model.FinishReason{
			Unified: model.FinishOther,
		},
	}

	if prepared.usesJSONInstruction {
		state.extractor = &jsonObjectExtractor{}
	}

	return state
}
