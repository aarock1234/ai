package openaicompatible

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/chatsse"
	"github.com/aarock1234/ai/internal/sse"
	"github.com/aarock1234/ai/internal/toolcalls"
	"github.com/aarock1234/ai/pkg/model"
)

// Stream block IDs. A chat stream has at most one open text block and one
// open reasoning block at a time.
const (
	textBlockID      = "txt-0"
	reasoningBlockID = "reasoning-0"
)

// chatStreamSettings configures one chat stream.
type chatStreamSettings struct {
	metadataKey      string
	includeRawChunks bool
	convertUsage     UsageConverter
	extractor        StreamMetadataExtractor
	generateID       func() string
}

// chatStream converts chat completion chunks into stream parts.
type chatStream struct {
	settings chatStreamSettings
	tracker  *toolcalls.Tracker

	// pending buffers tool call deltas by index until their function name
	// arrives, because some providers send the name after the first delta.
	pending      map[int]*pendingToolCall
	pendingOrder []int
	forwarded    map[int]bool

	finishReason    *model.FinishReason
	usage           *ChatUsage
	sentMetadata    bool
	reasoningActive bool
	textActive      bool
}

type pendingToolCall struct {
	id           *string
	arguments    string
	extraContent *extraContent
}

func newChatStream(settings chatStreamSettings) *chatStream {
	return &chatStream{
		settings: settings,
		tracker: toolcalls.NewTracker(toolcalls.Options{
			GenerateID: settings.generateID,
		}),
		pending:   make(map[int]*pendingToolCall),
		forwarded: make(map[int]bool),
	}
}

// Process converts one server-sent event into stream parts.
func (s *chatStream) Process(event sse.Event) ([]model.StreamPart, error) {
	var parts []model.StreamPart
	if s.settings.includeRawChunks {
		if raw, ok := chatsse.Raw(event); ok {
			parts = append(parts, raw)
		}
	}

	var chunk chatChunk
	err := json.Unmarshal([]byte(event.Data), &chunk)
	if err == nil && chunk.Error == nil && chunk.Choices == nil {
		err = errMissingChoices
	}

	if err != nil {
		return append(parts, s.Fail(fmt.Errorf("%w: %w", model.ErrInvalidResponse, err))), nil
	}

	if s.settings.extractor != nil {
		s.settings.extractor.ProcessChunk(json.RawMessage(event.Data))
	}

	if chunk.Error != nil {
		return append(parts, s.Fail(chunkError(chunk.Error))), nil
	}

	parts = append(parts, s.metadataParts(chunk)...)

	if chunk.Usage != nil {
		s.usage = chunk.Usage
	}

	if len(chunk.Choices) == 0 {
		return parts, nil
	}

	choice := chunk.Choices[0]
	if choice.FinishReason != nil {
		finishReason := MapFinishReason(*choice.FinishReason)
		s.finishReason = &finishReason
	}

	if choice.Delta == nil {
		return parts, nil
	}

	deltaParts, err := s.deltaParts(*choice.Delta)

	return append(parts, deltaParts...), err
}

// Fail records an error finish reason and returns the error part.
func (s *chatStream) Fail(err error) model.StreamPart {
	s.finishReason = &model.FinishReason{
		Unified: model.FinishError,
	}

	return model.StreamError{
		Err: err,
	}
}

// chunkError converts the error member of an error chunk.
func chunkError(raw json.RawMessage) error {
	var data ErrorData
	if err := json.Unmarshal(raw, &data); err != nil {
		return fmt.Errorf("%w: provider error: %s", model.ErrInvalidResponse, raw)
	}

	return &data
}

// metadataParts emits the response metadata once. It usually arrives in the
// first chunk, but some providers send a placeholder chunk first.
func (s *chatStream) metadataParts(chunk chatChunk) []model.StreamPart {
	if s.sentMetadata {
		return nil
	}

	metadata := responseMetadata(chunk.ID, chunk.Model, chunk.Created)
	if isEmptyMetadata(metadata) {
		return nil
	}

	s.sentMetadata = true

	return []model.StreamPart{
		metadata,
	}
}

// deltaParts converts reasoning, then content, then tool calls.
func (s *chatStream) deltaParts(delta chatDelta) ([]model.StreamPart, error) {
	var parts []model.StreamPart

	reasoning := reasoningText(delta.ReasoningContent, delta.Reasoning)
	if reasoning != "" {
		parts = append(parts, s.reasoningDelta(reasoning)...)
	}

	for _, piece := range delta.Content.pieces() {
		if piece.Reasoning {
			parts = append(parts, s.reasoningDelta(piece.Text)...)
		} else {
			parts = append(parts, s.textDelta(piece.Text)...)
		}
	}

	if len(delta.ToolCalls) == 0 {
		return parts, nil
	}

	parts = append(parts, s.endReasoning()...)

	for _, toolCall := range delta.ToolCalls {
		toolParts, err := s.toolCallDelta(toolCall)
		parts = append(parts, toolParts...)

		if err != nil {
			return parts, err
		}
	}

	return parts, nil
}

func (s *chatStream) reasoningDelta(delta string) []model.StreamPart {
	parts := s.endText()

	if !s.reasoningActive {
		s.reasoningActive = true
		parts = append(parts, model.ReasoningStart{
			ID: reasoningBlockID,
		})
	}

	return append(parts, model.ReasoningDelta{
		ID:    reasoningBlockID,
		Delta: delta,
	})
}

func (s *chatStream) textDelta(delta string) []model.StreamPart {
	parts := s.endReasoning()

	if !s.textActive {
		s.textActive = true
		parts = append(parts, model.TextStart{
			ID: textBlockID,
		})
	}

	return append(parts, model.TextDelta{
		ID:    textBlockID,
		Delta: delta,
	})
}

func (s *chatStream) endReasoning() []model.StreamPart {
	if !s.reasoningActive {
		return nil
	}

	s.reasoningActive = false

	return []model.StreamPart{
		model.ReasoningEnd{
			ID: reasoningBlockID,
		},
	}
}

func (s *chatStream) endText() []model.StreamPart {
	if !s.textActive {
		return nil
	}

	s.textActive = false

	return []model.StreamPart{
		model.TextEnd{
			ID: textBlockID,
		},
	}
}

// toolCallDelta forwards a delta to the tracker once its call has a name,
// buffering earlier deltas of the same index.
func (s *chatStream) toolCallDelta(toolCall chatDeltaToolCall) ([]model.StreamPart, error) {
	if toolCall.Index == nil || s.forwarded[*toolCall.Index] {
		return s.tracker.ProcessDelta(s.trackerDelta(toolCall))
	}

	index := *toolCall.Index
	pending := s.bufferToolCall(index, toolCall)

	if toolCall.Function == nil || toolCall.Function.Name == nil {
		return nil, nil
	}

	delete(s.pending, index)
	s.forwarded[index] = true

	return s.tracker.ProcessDelta(toolcalls.Delta{
		Index:            &index,
		ID:               pending.id,
		Name:             toolCall.Function.Name,
		Arguments:        &pending.arguments,
		ProviderMetadata: toolCallMetadata(s.settings.metadataKey, pending.extraContent.thoughtSignature()),
	})
}

// bufferToolCall adds a delta to the pending call of its index.
func (s *chatStream) bufferToolCall(index int, toolCall chatDeltaToolCall) *pendingToolCall {
	pending, ok := s.pending[index]
	if !ok {
		pending = &pendingToolCall{}
		s.pending[index] = pending
		s.pendingOrder = append(s.pendingOrder, index)
	}

	if pending.id == nil {
		pending.id = toolCall.ID
	}

	if pending.extraContent == nil {
		pending.extraContent = toolCall.ExtraContent
	}

	if toolCall.Function != nil && toolCall.Function.Arguments != nil {
		pending.arguments += *toolCall.Function.Arguments
	}

	return pending
}

func (s *chatStream) trackerDelta(toolCall chatDeltaToolCall) toolcalls.Delta {
	delta := toolcalls.Delta{
		Index:            toolCall.Index,
		ID:               toolCall.ID,
		ProviderMetadata: toolCallMetadata(s.settings.metadataKey, toolCall.ExtraContent.thoughtSignature()),
	}

	if toolCall.Function != nil {
		delta.Name = toolCall.Function.Name
		delta.Arguments = toolCall.Function.Arguments
	}

	return delta
}

// Flush returns the parts that close the stream.
func (s *chatStream) Flush() ([]model.StreamPart, error) {
	parts := s.endReasoning()
	parts = append(parts, s.endText()...)

	// Deltas that never received a name are forwarded so that the tracker
	// rejects them as invalid.
	for _, index := range s.pendingOrder {
		pending, ok := s.pending[index]
		if !ok {
			continue
		}

		toolParts, err := s.tracker.ProcessDelta(toolcalls.Delta{
			Index:     &index,
			ID:        pending.id,
			Arguments: &pending.arguments,
		})
		parts = append(parts, toolParts...)

		if err != nil {
			return parts, err
		}
	}

	parts = append(parts, s.tracker.Flush()...)

	if s.finishReason == nil {
		parts = append(parts, s.Fail(fmt.Errorf("%w: response stream ended without a finish reason", model.ErrInvalidResponse)))
	}

	return append(parts, model.Finish{
		Usage:            s.settings.convertUsage(s.usage),
		FinishReason:     *s.finishReason,
		ProviderMetadata: s.providerMetadata(),
	}), nil
}

func (s *chatStream) providerMetadata() model.ProviderMetadata {
	var extracted model.ProviderMetadata
	if s.settings.extractor != nil {
		extracted = s.settings.extractor.BuildMetadata()
	}

	return chatMetadata(s.settings.metadataKey, s.usage, extracted)
}
