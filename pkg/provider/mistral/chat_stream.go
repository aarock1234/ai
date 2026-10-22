package mistral

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/chatsse"
	"github.com/aarock1234/ai/internal/sse"
	"github.com/aarock1234/ai/internal/toolcalls"
	"github.com/aarock1234/ai/pkg/model"
)

// chatTextID is the ID of the text block of a chat stream.
const chatTextID = "0"

// chatStream converts chat completion chunks into stream parts.
type chatStream struct {
	includeRawChunks bool
	generateID       func() string
	tracker          *toolcalls.Tracker

	finishReason model.FinishReason
	usage        *Usage
	started      bool
	textActive   bool

	// reasoningID is the ID of the open reasoning block, if any.
	reasoningID string
}

func newChatStream(includeRawChunks bool, generateID func() string) *chatStream {
	return &chatStream{
		includeRawChunks: includeRawChunks,
		generateID:       generateID,
		tracker: toolcalls.NewTracker(toolcalls.Options{
			GenerateID: generateID,
		}),
		finishReason: model.FinishReason{
			Unified: model.FinishOther,
		},
	}
}

// Process converts one server-sent event into stream parts.
func (s *chatStream) Process(event sse.Event) ([]model.StreamPart, error) {
	var parts []model.StreamPart
	if s.includeRawChunks {
		if raw, ok := chatsse.Raw(event); ok {
			parts = append(parts, raw)
		}
	}

	var chunk chatChunk
	if err := json.Unmarshal([]byte(event.Data), &chunk); err != nil {
		return append(parts, s.Fail(fmt.Errorf("%w: %w", model.ErrInvalidResponse, err))), nil
	}

	if !s.started {
		s.started = true
		parts = append(parts, responseMetadata(chunk.ID, chunk.Model, chunk.Created))
	}

	if chunk.Usage != nil {
		s.usage = chunk.Usage
	}

	if len(chunk.Choices) == 0 {
		return parts, nil
	}

	choice := chunk.Choices[0]

	parts = append(parts, s.reasoningParts(choice.Delta.Content)...)
	parts = append(parts, s.textParts(choice.Delta.Content.text())...)

	for _, toolCall := range choice.Delta.ToolCalls {
		toolParts, err := s.tracker.ProcessDelta(toolcalls.Delta{
			Index:     toolCall.Index,
			ID:        toolCall.ID,
			Name:      toolCall.Function.Name,
			Arguments: toolCall.Function.Arguments,
		})
		parts = append(parts, toolParts...)

		if err != nil {
			return parts, err
		}
	}

	if choice.FinishReason != nil {
		s.finishReason = mapFinishReason(*choice.FinishReason)
	}

	return parts, nil
}

// reasoningParts streams thinking parts, opening a reasoning block and
// closing any open text block.
func (s *chatStream) reasoningParts(content responseContent) []model.StreamPart {
	var parts []model.StreamPart

	for _, part := range content.Parts {
		if part.Type != contentThinking {
			continue
		}

		reasoning := joinThinking(part.Thinking)
		if reasoning == "" {
			continue
		}

		if s.reasoningID == "" {
			if s.textActive {
				s.textActive = false
				parts = append(parts, model.TextEnd{
					ID: chatTextID,
				})
			}

			s.reasoningID = s.generateID()
			parts = append(parts, model.ReasoningStart{
				ID: s.reasoningID,
			})
		}

		parts = append(parts, model.ReasoningDelta{
			ID:    s.reasoningID,
			Delta: reasoning,
		})
	}

	return parts
}

// textParts streams text, opening a text block and closing any open
// reasoning block.
func (s *chatStream) textParts(text string) []model.StreamPart {
	if text == "" {
		return nil
	}

	var parts []model.StreamPart
	if !s.textActive {
		parts = append(parts, s.endReasoning()...)

		s.textActive = true
		parts = append(parts, model.TextStart{
			ID: chatTextID,
		})
	}

	return append(parts, model.TextDelta{
		ID:    chatTextID,
		Delta: text,
	})
}

func (s *chatStream) endReasoning() []model.StreamPart {
	if s.reasoningID == "" {
		return nil
	}

	parts := []model.StreamPart{
		model.ReasoningEnd{
			ID: s.reasoningID,
		},
	}
	s.reasoningID = ""

	return parts
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
			ID: chatTextID,
		})
	}

	parts = append(parts, s.tracker.Flush()...)

	return append(parts, model.Finish{
		Usage:        convertUsage(s.usage),
		FinishReason: s.finishReason,
	}), nil
}
