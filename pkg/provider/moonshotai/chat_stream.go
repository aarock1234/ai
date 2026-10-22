package moonshotai

import (
	"encoding/json"
	"fmt"
	"maps"
	"slices"

	"github.com/aarock1234/ai/internal/chatsse"
	"github.com/aarock1234/ai/internal/optional"
	"github.com/aarock1234/ai/internal/sse"
	"github.com/aarock1234/ai/internal/toolcalls"
	"github.com/aarock1234/ai/pkg/model"
)

// Stream block IDs.
const (
	textBlockID      = "txt-0"
	reasoningBlockID = "reasoning-0"
)

// chatStreamSettings configures one chat stream.
type chatStreamSettings struct {
	includeRawChunks bool
	generateID       func() string
}

// chatStream converts chat completion chunks into stream parts.
type chatStream struct {
	settings chatStreamSettings
	tracker  *toolcalls.Tracker

	finishReason model.FinishReason
	topLevel     *chatUsage
	choiceUsage  *chatUsage

	logprobs       []Logprob
	responseObject string
	choiceIndex    *int
	messageRole    string
	toolCallTypes  map[int]string

	sentMetadata    bool
	reasoningActive bool
	textActive      bool
}

func newChatStream(settings chatStreamSettings) *chatStream {
	return &chatStream{
		settings: settings,
		tracker: toolcalls.NewTracker(toolcalls.Options{
			GenerateID: settings.generateID,
		}),
		finishReason: model.FinishReason{
			Unified: model.FinishOther,
		},
		toolCallTypes: make(map[int]string),
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
	if err := json.Unmarshal([]byte(event.Data), &chunk); err != nil {
		return append(parts, s.Fail(fmt.Errorf("%w: %w", model.ErrInvalidResponse, err))), nil
	}

	if chunk.Error != nil {
		return append(parts, s.Fail(streamError(chunk.Error, event.Data))), nil
	}

	if !s.sentMetadata {
		s.sentMetadata = true
		parts = append(parts, responseMetadata(chunk.ID, chunk.Model, chunk.Created))
	}

	s.record(chunk)

	if len(chunk.Choices) == 0 || chunk.Choices[0].Delta == nil {
		return parts, nil
	}

	deltaParts, err := s.deltaParts(*chunk.Choices[0].Delta)

	return append(parts, deltaParts...), err
}

// record keeps the usage, finish reason and metadata of a chunk.
func (s *chatStream) record(chunk chatChunk) {
	if chunk.Usage != nil {
		s.topLevel = chunk.Usage
	}

	if chunk.Object != nil {
		s.responseObject = *chunk.Object
	}

	if len(chunk.Choices) == 0 {
		return
	}

	choice := chunk.Choices[0]

	if choice.Usage != nil {
		s.choiceUsage = choice.Usage
	}

	if choice.Index != nil {
		s.choiceIndex = choice.Index
	}

	if choice.FinishReason != nil {
		s.finishReason = mapFinishReason(choice.FinishReason)
	}

	if choice.Logprobs != nil {
		s.logprobs = append(s.logprobs, choice.Logprobs.Content...)
	}

	if choice.Delta != nil && choice.Delta.Role != nil {
		s.messageRole = *choice.Delta.Role
	}
}

// Fail records an error finish reason and returns the error part.
func (s *chatStream) Fail(err error) model.StreamPart {
	s.finishReason = model.FinishReason{
		Unified: model.FinishError,
	}

	return model.StreamError{
		Err: err,
	}
}

// deltaParts converts reasoning, then text, then tool calls.
func (s *chatStream) deltaParts(delta chunkDelta) ([]model.StreamPart, error) {
	var parts []model.StreamPart

	if reasoning := optional.Or(delta.ReasoningContent, ""); reasoning != "" {
		if !s.reasoningActive {
			s.reasoningActive = true
			parts = append(parts, model.ReasoningStart{
				ID: reasoningBlockID,
			})
		}

		parts = append(parts, model.ReasoningDelta{
			ID:    reasoningBlockID,
			Delta: reasoning,
		})
	}

	if text := optional.Or(delta.Content, ""); text != "" {
		if !s.textActive {
			s.textActive = true
			parts = append(parts, model.TextStart{
				ID: textBlockID,
			})
		}

		parts = append(parts, s.endReasoning()...)
		parts = append(parts, model.TextDelta{
			ID:    textBlockID,
			Delta: text,
		})
	}

	if len(delta.ToolCalls) == 0 {
		return parts, nil
	}

	parts = append(parts, s.endReasoning()...)

	for position, toolCall := range delta.ToolCalls {
		index := optional.Or(toolCall.Index, position)

		if toolCall.Type != nil {
			s.toolCallTypes[index] = *toolCall.Type
		}

		toolParts, err := s.tracker.ProcessDelta(toolcalls.Delta{
			Index:     &index,
			ID:        toolCall.ID,
			Type:      toolCall.Type,
			Name:      toolCall.Function.Name,
			Arguments: toolCall.Function.Arguments,
		})
		parts = append(parts, toolParts...)

		if err != nil {
			return parts, err
		}
	}

	return parts, nil
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

// Flush closes the open blocks and tool calls and reports the finish.
func (s *chatStream) Flush() ([]model.StreamPart, error) {
	parts := s.endReasoning()

	if s.textActive {
		s.textActive = false
		parts = append(parts, model.TextEnd{
			ID: textBlockID,
		})
	}

	parts = append(parts, s.tracker.Flush()...)

	usage := s.topLevel
	if usage == nil {
		usage = s.choiceUsage
	}

	parts = append(parts, model.Finish{
		Usage:        convertUsage(usage),
		FinishReason: s.finishReason,
		ProviderMetadata: model.ProviderMetadata{
			Namespace: s.metadata(),
		},
	})

	return parts, nil
}

// metadata describes the streamed choice.
func (s *chatStream) metadata() ChatMetadata {
	metadata := ChatMetadata{
		ResponseObject: s.responseObject,
		ChoiceIndex:    s.choiceIndex,
		MessageRole:    s.messageRole,
	}

	if len(s.logprobs) > 0 {
		metadata.Logprobs = &Logprobs{
			Content: s.logprobs,
		}
	}

	for _, index := range slices.Sorted(maps.Keys(s.toolCallTypes)) {
		metadata.ToolCallTypes = append(metadata.ToolCallTypes, s.toolCallTypes[index])
	}

	return metadata
}
