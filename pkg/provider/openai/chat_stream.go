package openai

import (
	"context"
	"encoding/json"
	"errors"
	"iter"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/toolcalls"
	"github.com/aarock1234/ai/pkg/model"
)

// chatTextID is the ID of the single text block of a streamed choice.
const chatTextID = "0"

// Stream performs one streaming Chat Completions call. An error frame that
// arrives before any output fails the call instead of the stream.
func (m *ChatModel) Stream(ctx context.Context, options model.CallOptions) (*model.StreamResult, error) {
	call, err := m.prepareCall(options)
	if err != nil {
		return nil, err
	}

	call.request.Stream = true
	call.request.StreamOptions = &chatStreamOptions{
		IncludeUsage: true,
	}

	request, err := m.config.request(ctx, "/chat/completions", options.Headers)
	if err != nil {
		return nil, err
	}

	stream, err := apicall.PostJSONStream(ctx, request, call.request)
	if err != nil {
		return nil, err
	}

	origin := chatStreamOrigin{
		url:     request.URL,
		headers: stream.Headers,
	}

	source := newChatEventSource(stream.Body)
	if err := source.peek(origin.classifyChatEvent); err != nil {
		return nil, errors.Join(err, source.close())
	}

	state := &chatStreamState{
		warnings:         call.warnings,
		includeRawChunks: options.IncludeRawChunks,
		origin:           origin,
		generateID:       m.config.generateID,
	}

	result := model.NewStreamResult(state.parts(source), source.close)
	result.Request = model.RequestInfo{
		Body: stream.RequestBody,
	}
	result.Response = model.StreamResponseInfo{
		Headers: stream.Headers,
	}

	return result, nil
}

// classifyChatEvent fails on an error frame and hands the stream over once a
// chunk carries output or cannot be decoded.
func (o chatStreamOrigin) classifyChatEvent(data string) (chatPeekResult, error) {
	var chunk chatChunk
	if err := json.Unmarshal([]byte(data), &chunk); err != nil {
		return chatPeekReady, nil
	}

	if chunk.isErrorFrame() {
		return chatPeekReady, o.apiError(*chunk.Error, data)
	}

	if chunk.hasOutput() {
		return chatPeekReady, nil
	}

	return chatPeekContinue, nil
}

// chatStreamState converts Chat Completions chunks into stream parts.
type chatStreamState struct {
	warnings         []model.Warning
	includeRawChunks bool
	origin           chatStreamOrigin
	generateID       func() string

	pending   []model.StreamPart
	toolCalls *toolcalls.Tracker

	finishReason    model.FinishReason
	usage           *chatUsage
	metadata        ChatMetadata
	metadataEmitted bool
	isTextActive    bool
}

func (s *chatStreamState) parts(source *chatEventSource) iter.Seq[model.StreamPart] {
	return func(yield func(model.StreamPart) bool) {
		s.toolCalls = toolcalls.NewTracker(toolcalls.Options{
			TypeValidation: toolcalls.ValidateTypeIfPresent,
			GenerateID:     s.generateID,
		})
		s.finishReason = model.FinishReason{
			Unified: model.FinishOther,
		}

		s.emit(model.StreamStart{
			Warnings: s.warnings,
		})

		for event := range source.events() {
			s.process(event)

			if !s.flushPending(yield) {
				return
			}
		}

		s.finish()
		s.flushPending(yield)
	}
}

func (s *chatStreamState) emit(parts ...model.StreamPart) {
	s.pending = append(s.pending, parts...)
}

func (s *chatStreamState) flushPending(yield func(model.StreamPart) bool) bool {
	for _, part := range s.pending {
		if !yield(part) {
			return false
		}
	}

	s.pending = s.pending[:0]

	return true
}

func (s *chatStreamState) fail(err error) {
	s.finishReason = model.FinishReason{
		Unified: model.FinishError,
	}

	s.emit(model.StreamError{
		Err: err,
	})
}

func (s *chatStreamState) process(event chatEvent) {
	if event.err != nil {
		s.fail(event.err)

		return
	}

	if s.includeRawChunks {
		if raw, ok := chatRaw(event.data); ok {
			s.emit(raw)
		}
	}

	var chunk chatChunk
	if err := json.Unmarshal([]byte(event.data), &chunk); err != nil {
		s.fail(chatDecodeError(err))

		return
	}

	if chunk.isErrorFrame() {
		s.fail(s.origin.apiError(*chunk.Error, event.data))

		return
	}

	if err := chunk.validate(); err != nil {
		s.fail(err)

		return
	}

	if err := s.processChunk(chunk); err != nil {
		s.fail(err)
	}
}

func (s *chatStreamState) processChunk(chunk chatChunk) error {
	s.emitMetadataOnce(chunk)

	if chunk.Usage != nil {
		s.usage = chunk.Usage
		s.recordPredictionTokens(chunk.Usage)
	}

	if len(chunk.Choices) == 0 {
		return nil
	}

	choice := chunk.Choices[0]

	if choice.FinishReason != "" {
		s.finishReason = model.FinishReason{
			Unified: chatFinishReason(choice.FinishReason),
			Raw:     choice.FinishReason,
		}
	}

	if choice.Logprobs != nil && choice.Logprobs.Content != nil {
		s.metadata.Logprobs = choice.Logprobs.Content
	}

	if choice.Delta == nil {
		return nil
	}

	return s.processDelta(*choice.Delta)
}

// emitMetadataOnce reports the response metadata of the first chunk that has
// any. Azure may first send a content filter chunk without metadata.
func (s *chatStreamState) emitMetadataOnce(chunk chatChunk) {
	if s.metadataEmitted {
		return
	}

	metadata := chatResponseMetadata(chunk.ID, chunk.Model, chunk.Created)
	if metadata == (model.ResponseMetadata{}) {
		return
	}

	s.metadataEmitted = true
	s.emit(metadata)
}

func (s *chatStreamState) recordPredictionTokens(usage *chatUsage) {
	details := usage.CompletionTokensDetails
	if details == nil {
		return
	}

	if details.AcceptedPredictionTokens != nil {
		s.metadata.AcceptedPredictionTokens = details.AcceptedPredictionTokens
	}

	if details.RejectedPredictionTokens != nil {
		s.metadata.RejectedPredictionTokens = details.RejectedPredictionTokens
	}
}

func (s *chatStreamState) processDelta(delta chatDelta) error {
	if delta.Content != nil {
		s.emitText(*delta.Content)
	}

	for _, toolCallDelta := range delta.ToolCalls {
		parts, err := s.toolCalls.ProcessDelta(chatTrackerDelta(toolCallDelta))
		s.emit(parts...)

		if err != nil {
			return err
		}
	}

	for _, annotation := range delta.Annotations {
		s.emit(chatCitationSource(annotation, s.generateID))
	}

	return nil
}

// chatTrackerDelta converts a streamed function call fragment for the tool
// call tracker.
func chatTrackerDelta(delta chatToolCallDelta) toolcalls.Delta {
	trackerDelta := toolcalls.Delta{
		Index: delta.Index,
		ID:    delta.ID,
		Type:  delta.Type,
	}

	if delta.Function != nil {
		trackerDelta.Name = delta.Function.Name
		trackerDelta.Arguments = delta.Function.Arguments
	}

	return trackerDelta
}

func (s *chatStreamState) emitText(text string) {
	if !s.isTextActive {
		s.isTextActive = true
		s.emit(model.TextStart{
			ID: chatTextID,
		})
	}

	s.emit(model.TextDelta{
		ID:    chatTextID,
		Delta: text,
	})
}

func (s *chatStreamState) finish() {
	if s.isTextActive {
		s.emit(model.TextEnd{
			ID: chatTextID,
		})
	}

	s.emit(s.toolCalls.Flush()...)

	s.emit(model.Finish{
		FinishReason: s.finishReason,
		Usage:        chatConvertUsage(s.usage),
		ProviderMetadata: model.ProviderMetadata{
			Namespace: s.metadata,
		},
	})
}
