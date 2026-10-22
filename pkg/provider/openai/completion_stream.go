package openai

import (
	"context"
	"encoding/json"
	"errors"
	"iter"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// Stream performs one streaming Completions call. An error frame that
// arrives before any output fails the call instead of the stream.
func (m *CompletionModel) Stream(ctx context.Context, options model.CallOptions) (*model.StreamResult, error) {
	call, err := m.prepareCall(options)
	if err != nil {
		return nil, err
	}

	call.request.Stream = true
	call.request.StreamOptions = &chatStreamOptions{
		IncludeUsage: true,
	}

	request, err := m.config.request(ctx, "/completions", options.Headers)
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
	if err := source.peek(origin.classifyCompletionEvent); err != nil {
		return nil, errors.Join(err, source.close())
	}

	state := &completionStreamState{
		warnings:         call.warnings,
		includeRawChunks: options.IncludeRawChunks,
		origin:           origin,
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

// classifyCompletionEvent fails on an error frame and hands the stream over
// once a chunk carries text or cannot be decoded.
func (o chatStreamOrigin) classifyCompletionEvent(data string) (chatPeekResult, error) {
	var chunk completionChunk
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

// completionStreamState converts Completions chunks into stream parts.
type completionStreamState struct {
	warnings         []model.Warning
	includeRawChunks bool
	origin           chatStreamOrigin

	finishReason model.FinishReason
	usage        *completionUsage
	metadata     CompletionMetadata
	started      bool
}

func (s *completionStreamState) parts(source *chatEventSource) iter.Seq[model.StreamPart] {
	return func(yield func(model.StreamPart) bool) {
		s.finishReason = model.FinishReason{
			Unified: model.FinishOther,
		}

		start := model.StreamStart{
			Warnings: s.warnings,
		}

		if !yield(start) {
			return
		}

		for event := range source.events() {
			for _, part := range s.process(event) {
				if !yield(part) {
					return
				}
			}
		}

		for _, part := range s.finish() {
			if !yield(part) {
				return
			}
		}
	}
}

func (s *completionStreamState) process(event chatEvent) []model.StreamPart {
	if event.err != nil {
		return s.fail(event.err)
	}

	var parts []model.StreamPart

	if s.includeRawChunks {
		if raw, ok := chatRaw(event.data); ok {
			parts = append(parts, raw)
		}
	}

	var chunk completionChunk
	if err := json.Unmarshal([]byte(event.data), &chunk); err != nil {
		return append(parts, s.fail(chatDecodeError(err))...)
	}

	if chunk.isErrorFrame() {
		return append(parts, s.fail(s.origin.apiError(*chunk.Error, event.data))...)
	}

	if err := chunk.validate(); err != nil {
		return append(parts, s.fail(err)...)
	}

	return append(parts, s.processChunk(chunk)...)
}

func (s *completionStreamState) processChunk(chunk completionChunk) []model.StreamPart {
	var parts []model.StreamPart

	if !s.started {
		s.started = true
		parts = append(parts,
			chatResponseMetadata(chunk.ID, chunk.Model, chunk.Created),
			model.TextStart{
				ID: chatTextID,
			},
		)
	}

	if chunk.Usage != nil {
		s.usage = chunk.Usage
	}

	if len(chunk.Choices) == 0 {
		return parts
	}

	choice := chunk.Choices[0]

	if choice.FinishReason != nil && *choice.FinishReason != "" {
		s.finishReason = model.FinishReason{
			Unified: chatFinishReason(*choice.FinishReason),
			Raw:     *choice.FinishReason,
		}
	}

	if choice.Logprobs != nil {
		s.metadata.Logprobs = choice.Logprobs
	}

	if *choice.Text != "" {
		parts = append(parts, model.TextDelta{
			ID:    chatTextID,
			Delta: *choice.Text,
		})
	}

	return parts
}

func (s *completionStreamState) fail(err error) []model.StreamPart {
	s.finishReason = model.FinishReason{
		Unified: model.FinishError,
	}

	return []model.StreamPart{
		model.StreamError{
			Err: err,
		},
	}
}

func (s *completionStreamState) finish() []model.StreamPart {
	var parts []model.StreamPart

	if s.started {
		parts = append(parts, model.TextEnd{
			ID: chatTextID,
		})
	}

	return append(parts, model.Finish{
		FinishReason: s.finishReason,
		Usage:        completionConvertUsage(s.usage),
		ProviderMetadata: model.ProviderMetadata{
			Namespace: s.metadata,
		},
	})
}
