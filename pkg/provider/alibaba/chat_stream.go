package alibaba

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/chatsse"
	"github.com/aarock1234/ai/internal/sse"
	"github.com/aarock1234/ai/internal/toolcalls"
	"github.com/aarock1234/ai/pkg/model"
)

// textBlockID is the ID of the text block of a chat stream.
const textBlockID = "0"

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

// Process converts one server-sent event into stream parts. Malformed
// chunks are reported without ending the stream or changing the finish
// reason.
func (s *chatStream) Process(event sse.Event) ([]model.StreamPart, error) {
	var parts []model.StreamPart
	if s.includeRawChunks {
		if raw, ok := chatsse.Raw(event); ok {
			parts = append(parts, raw)
		}
	}

	var chunk chatChunk
	err := json.Unmarshal([]byte(event.Data), &chunk)
	if err != nil {
		err = fmt.Errorf("%w: %w", model.ErrInvalidResponse, err)
	} else {
		err = chunk.validate()
	}

	if err != nil {
		return append(parts, model.StreamError{
			Err: err,
		}), nil
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

	deltaParts, err := s.deltaParts(*choice.Delta)
	parts = append(parts, deltaParts...)

	if err != nil {
		return parts, err
	}

	if choice.FinishReason != nil {
		s.finishReason = mapFinishReason(choice.FinishReason)
	}

	return parts, nil
}

// deltaParts streams reasoning, text and tool calls, closing the other open
// blocks first.
func (s *chatStream) deltaParts(delta chunkDelta) ([]model.StreamPart, error) {
	var parts []model.StreamPart

	if reasoning := valueOrEmpty(delta.ReasoningContent); reasoning != "" {
		if s.reasoningID == "" {
			parts = append(parts, s.endText()...)

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
	parts = append(parts, s.endText()...)

	for _, toolCall := range delta.ToolCalls {
		toolParts, err := s.tracker.ProcessDelta(toolCallDelta(toolCall))
		parts = append(parts, toolParts...)

		if err != nil {
			return parts, err
		}
	}

	return parts, nil
}

func toolCallDelta(toolCall chunkToolCall) toolcalls.Delta {
	delta := toolcalls.Delta{
		Index: toolCall.Index,
		ID:    toolCall.ID,
		Type:  toolCall.Type,
	}

	if toolCall.Function != nil {
		delta.Name = toolCall.Function.Name
		delta.Arguments = toolCall.Function.Arguments
	}

	return delta
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
	parts = append(parts, s.endText()...)
	parts = append(parts, s.tracker.Flush()...)

	return append(parts, model.Finish{
		Usage:        convertUsage(s.usage),
		FinishReason: s.finishReason,
	}), nil
}
