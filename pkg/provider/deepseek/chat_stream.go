package deepseek

import (
	"encoding/json"
	"fmt"
	"maps"
	"slices"

	"github.com/aarock1234/ai/internal/chatsse"
	"github.com/aarock1234/ai/internal/sse"
	"github.com/aarock1234/ai/internal/toolcalls"
	"github.com/aarock1234/ai/pkg/model"
)

// Stream block IDs. A chat stream has at most one text block and one open
// reasoning block.
const (
	textBlockID      = "txt-0"
	reasoningBlockID = "reasoning-0"
)

// chatStreamSettings configures one chat stream.
type chatStreamSettings struct {
	namespace        string
	includeRawChunks bool
	generateID       func() string
	origin           streamOrigin
}

// chatStream converts chat completion chunks into stream parts and collects
// the response details reported in the finish metadata.
type chatStream struct {
	settings chatStreamSettings
	tracker  *toolcalls.Tracker

	finishReason    model.FinishReason
	usage           *chatUsage
	started         bool
	textActive      bool
	reasoningActive bool

	responseObject    string
	systemFingerprint string
	choiceIndex       *int
	messageRole       string
	toolCallTypes     map[int]string
	contentLogprobs   []Logprob
	reasoningLogprobs []Logprob
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

// Process converts one server-sent event into stream parts. Malformed and
// error chunks are reported without ending the stream.
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
		return append(parts, s.Fail(s.settings.origin.apiError(chunk.Error, event.Data))), nil
	}

	if err := chunk.validate(); err != nil {
		return append(parts, s.Fail(err)), nil
	}

	if !s.started {
		s.started = true
		parts = append(parts, responseMetadata(chunk.ID, chunk.Model, chunk.Created))
	}

	s.recordChunk(chunk)

	if len(chunk.Choices) == 0 {
		return parts, nil
	}

	choice := chunk.Choices[0]
	s.recordChoice(choice)

	if choice.Delta == nil {
		return parts, nil
	}

	deltaParts, err := s.deltaParts(*choice.Delta)

	return append(parts, deltaParts...), err
}

// recordChunk keeps the usage, object and latest system fingerprint.
func (s *chatStream) recordChunk(chunk chatChunk) {
	if chunk.Usage != nil {
		s.usage = chunk.Usage
	}

	if chunk.Object != nil {
		s.responseObject = *chunk.Object
	}

	if chunk.SystemFingerprint != nil {
		s.systemFingerprint = *chunk.SystemFingerprint
	}
}

// recordChoice keeps the index, finish reason and log probabilities.
func (s *chatStream) recordChoice(choice chunkChoice) {
	if choice.Index != nil {
		s.choiceIndex = choice.Index
	}

	if choice.FinishReason != nil {
		s.finishReason = mapFinishReason(choice.FinishReason)
	}

	if choice.Logprobs != nil {
		s.contentLogprobs = append(s.contentLogprobs, choice.Logprobs.Content...)
		s.reasoningLogprobs = append(s.reasoningLogprobs, choice.Logprobs.ReasoningContent...)
	}
}

// deltaParts streams reasoning, text and tool calls. Text and tool calls
// close an open reasoning block.
func (s *chatStream) deltaParts(delta chunkDelta) ([]model.StreamPart, error) {
	if delta.Role != nil {
		s.messageRole = *delta.Role
	}

	var parts []model.StreamPart

	if reasoning := valueOrEmpty(delta.ReasoningContent); reasoning != "" {
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

	if text := valueOrEmpty(delta.Content); text != "" {
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

	for _, toolCall := range delta.ToolCalls {
		if toolCall.Type != nil {
			s.toolCallTypes[*toolCall.Index] = *toolCall.Type
		}

		toolParts, err := s.tracker.ProcessDelta(toolcalls.Delta{
			Index:     toolCall.Index,
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

// Fail records an error finish reason and returns the error part.
func (s *chatStream) Fail(err error) model.StreamPart {
	s.finishReason = model.FinishReason{
		Unified: model.FinishError,
	}

	return model.StreamError{
		Err: err,
	}
}

// Flush returns the parts that close the stream.
func (s *chatStream) Flush() ([]model.StreamPart, error) {
	parts := s.endReasoning()

	if s.textActive {
		parts = append(parts, model.TextEnd{
			ID: textBlockID,
		})
	}

	parts = append(parts, s.tracker.Flush()...)

	return append(parts, model.Finish{
		Usage:        convertUsage(s.usage),
		FinishReason: s.finishReason,
		ProviderMetadata: model.ProviderMetadata{
			s.settings.namespace: s.metadata(),
		},
	}), nil
}

// metadata reports cache usage and the collected response details.
func (s *chatStream) metadata() ChatMetadata {
	hit, miss := cacheTokens(s.usage)

	metadata := ChatMetadata{
		PromptCacheHitTokens:  hit,
		PromptCacheMissTokens: miss,
		ResponseObject:        s.responseObject,
		ChoiceIndex:           s.choiceIndex,
		MessageRole:           s.messageRole,
		SystemFingerprint:     s.systemFingerprint,
	}

	for _, index := range slices.Sorted(maps.Keys(s.toolCallTypes)) {
		metadata.ToolCallTypes = append(metadata.ToolCallTypes, s.toolCallTypes[index])
	}

	if len(s.contentLogprobs) > 0 || len(s.reasoningLogprobs) > 0 {
		metadata.Logprobs = &Logprobs{
			Content:          s.contentLogprobs,
			ReasoningContent: s.reasoningLogprobs,
		}
	}

	return metadata
}
