package groq

import (
	"encoding/json"
	"errors"
	"fmt"

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

// errMissingChoices reports a chunk that is neither a delta nor an error.
var errMissingChoices = errors.New("chunk has no choices")

// chatStreamSettings configures one chat stream.
type chatStreamSettings struct {
	includeRawChunks bool
	generateID       func() string
	origin           streamOrigin
}

// chatStream converts chat completion chunks into stream parts.
type chatStream struct {
	settings chatStreamSettings
	tracker  *toolcalls.Tracker

	finishReason    model.FinishReason
	usage           *chatUsage
	started         bool
	textActive      bool
	reasoningActive bool
}

func newChatStream(settings chatStreamSettings) *chatStream {
	return &chatStream{
		settings: settings,
		tracker: toolcalls.NewTracker(toolcalls.Options{
			GenerateID:     settings.generateID,
			TypeValidation: toolcalls.ValidateTypeRequired,
		}),
		finishReason: model.FinishReason{
			Unified: model.FinishOther,
		},
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
	err := json.Unmarshal([]byte(event.Data), &chunk)
	if err == nil && chunk.Error == nil && chunk.Choices == nil {
		err = errMissingChoices
	}

	if err != nil {
		return append(parts, s.Fail(fmt.Errorf("%w: %w", model.ErrInvalidResponse, err))), nil
	}

	if chunk.Error != nil {
		return append(parts, s.Fail(s.settings.origin.apiError(chunk.Error, event.Data))), nil
	}

	if !s.started {
		s.started = true
		parts = append(parts, responseMetadata(chunk.ID, chunk.Model, chunk.Created))
	}

	if chunk.XGroq != nil && chunk.XGroq.Usage != nil {
		s.usage = chunk.XGroq.Usage
	}

	if len(chunk.Choices) == 0 {
		return parts, nil
	}

	choice := chunk.Choices[0]
	if choice.FinishReason != nil {
		s.finishReason = mapFinishReason(choice.FinishReason)
	}

	if choice.Delta == nil {
		return parts, nil
	}

	deltaParts, err := s.deltaParts(*choice.Delta)

	return append(parts, deltaParts...), err
}

// deltaParts streams reasoning, text and tool calls. Text and tool calls
// close an open reasoning block.
func (s *chatStream) deltaParts(delta chunkDelta) ([]model.StreamPart, error) {
	var parts []model.StreamPart

	if reasoning := valueOrEmpty(delta.Reasoning); reasoning != "" {
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
		parts = append(parts, s.endReasoning()...)

		if !s.textActive {
			s.textActive = true
			parts = append(parts, model.TextStart{
				ID: textBlockID,
			})
		}

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
	}), nil
}
