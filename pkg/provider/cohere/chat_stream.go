package cohere

import (
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/internal/chatsse"
	"github.com/aarock1234/ai/internal/sse"
	"github.com/aarock1234/ai/internal/toolresult"
	"github.com/aarock1234/ai/pkg/model"
)

// chatStream converts Cohere chat events into stream parts.
type chatStream struct {
	includeRawChunks bool

	finishReason    model.FinishReason
	usage           *Usage
	reasoningActive bool

	// toolCall is the tool call whose arguments are streaming, if any.
	toolCall *pendingToolCall
}

type pendingToolCall struct {
	id        string
	name      string
	arguments strings.Builder
}

// Process converts one server-sent event into stream parts.
func (s *chatStream) Process(event sse.Event) ([]model.StreamPart, error) {
	var parts []model.StreamPart
	if s.includeRawChunks {
		if raw, ok := chatsse.Raw(event); ok {
			parts = append(parts, raw)
		}
	}

	var decoded chatEvent
	if err := json.Unmarshal([]byte(event.Data), &decoded); err != nil {
		return append(parts, s.Fail(fmt.Errorf("%w: %w", model.ErrInvalidResponse, err))), nil
	}

	converted, err := s.event(decoded)
	if _, invalid := errors.AsType[*invalidEventError](err); invalid {
		return append(parts, s.Fail(err)), nil
	}

	if err != nil {
		return parts, err
	}

	return append(parts, converted...), nil
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

// event converts one event by its type. Events without the members of their
// type are invalid.
func (s *chatStream) event(event chatEvent) ([]model.StreamPart, error) {
	blockID := strconv.Itoa(event.Index)

	switch event.Type {
	case eventContentStart:
		content, err := eventContent(event)
		if err != nil {
			return nil, err
		}

		return s.contentStart(blockID, content), nil

	case eventContentDelta:
		content, err := eventContent(event)
		if err != nil {
			return nil, err
		}

		return s.contentDelta(blockID, content), nil

	case eventContentEnd:
		return s.contentEnd(blockID), nil

	case eventToolCallStart:
		toolCall, err := eventToolCall(event)
		if err != nil {
			return nil, err
		}

		return s.toolCallStart(toolCall), nil

	case eventToolCallDelta:
		toolCall, err := eventToolCall(event)
		if err != nil {
			return nil, err
		}

		return s.toolCallDelta(toolCall), nil

	case eventToolCallEnd:
		return s.toolCallEnd()

	case eventMessageStart:
		metadata := model.ResponseMetadata{}
		if event.ID != nil {
			metadata.ID = *event.ID
		}

		return []model.StreamPart{
			metadata,
		}, nil

	case eventMessageEnd:
		if event.Delta == nil {
			return nil, invalidEvent(event.Type)
		}

		s.finishReason = mapFinishReason(event.Delta.FinishReason)
		s.usage = event.Delta.Usage

		return nil, nil

	case eventCitationStart, eventCitationEnd, eventToolPlanDelta:
		return nil, nil

	default:
		return nil, &invalidEventError{
			message: fmt.Sprintf("unsupported event type %q", event.Type),
		}
	}
}

func (s *chatStream) contentStart(blockID string, content deltaContent) []model.StreamPart {
	if content.Type == "thinking" {
		s.reasoningActive = true

		return []model.StreamPart{
			model.ReasoningStart{
				ID: blockID,
			},
		}
	}

	return []model.StreamPart{
		model.TextStart{
			ID: blockID,
		},
	}
}

func (s *chatStream) contentDelta(blockID string, content deltaContent) []model.StreamPart {
	if content.Thinking != nil {
		return []model.StreamPart{
			model.ReasoningDelta{
				ID:    blockID,
				Delta: *content.Thinking,
			},
		}
	}

	return []model.StreamPart{
		model.TextDelta{
			ID:    blockID,
			Delta: content.Text,
		},
	}
}

func (s *chatStream) contentEnd(blockID string) []model.StreamPart {
	if s.reasoningActive {
		s.reasoningActive = false

		return []model.StreamPart{
			model.ReasoningEnd{
				ID: blockID,
			},
		}
	}

	return []model.StreamPart{
		model.TextEnd{
			ID: blockID,
		},
	}
}

func (s *chatStream) toolCallStart(toolCall deltaToolCall) []model.StreamPart {
	s.toolCall = &pendingToolCall{
		id:   toolCall.ID,
		name: toolCall.Function.Name,
	}
	s.toolCall.arguments.WriteString(toolCall.Function.Arguments)

	parts := []model.StreamPart{
		model.ToolInputStart{
			ID:       toolCall.ID,
			ToolName: toolCall.Function.Name,
		},
	}

	if toolCall.Function.Arguments == "" {
		return parts
	}

	return append(parts, model.ToolInputDelta{
		ID:    toolCall.ID,
		Delta: toolCall.Function.Arguments,
	})
}

func (s *chatStream) toolCallDelta(toolCall deltaToolCall) []model.StreamPart {
	if s.toolCall == nil {
		return nil
	}

	s.toolCall.arguments.WriteString(toolCall.Function.Arguments)

	return []model.StreamPart{
		model.ToolInputDelta{
			ID:    s.toolCall.id,
			Delta: toolCall.Function.Arguments,
		},
	}
}

// toolCallEnd finishes the streaming tool call. Its arguments must be valid
// JSON; empty arguments are an empty object.
func (s *chatStream) toolCallEnd() ([]model.StreamPart, error) {
	if s.toolCall == nil {
		return nil, nil
	}

	toolCall := s.toolCall
	s.toolCall = nil

	arguments := strings.TrimSpace(toolCall.arguments.String())
	if arguments == "" {
		arguments = "{}"
	}

	input, err := toolresult.CompactJSON(json.RawMessage(arguments))
	if err != nil {
		return nil, fmt.Errorf("%w: tool call arguments: %w", model.ErrInvalidResponse, err)
	}

	return []model.StreamPart{
		model.ToolInputEnd{
			ID: toolCall.id,
		},
		model.ToolCall{
			ToolCallID: toolCall.id,
			ToolName:   toolCall.name,
			Input:      input,
		},
	}, nil
}

// Flush returns the parts that close the stream.
func (s *chatStream) Flush() ([]model.StreamPart, error) {
	return []model.StreamPart{
		model.Finish{
			Usage:        convertUsage(s.usage),
			FinishReason: s.finishReason,
		},
	}, nil
}

// invalidEventError reports an event that does not match its type's schema.
// The stream reports it and continues.
type invalidEventError struct {
	message string
}

func (e *invalidEventError) Error() string {
	return e.message
}

func (e *invalidEventError) Unwrap() error {
	return model.ErrInvalidResponse
}

func invalidEvent(eventType string) error {
	return &invalidEventError{
		message: fmt.Sprintf("%q event is missing required members", eventType),
	}
}

func eventContent(event chatEvent) (deltaContent, error) {
	if event.Delta == nil || event.Delta.Message == nil || event.Delta.Message.Content == nil {
		return deltaContent{}, invalidEvent(event.Type)
	}

	return *event.Delta.Message.Content, nil
}

func eventToolCall(event chatEvent) (deltaToolCall, error) {
	if event.Delta == nil || event.Delta.Message == nil || event.Delta.Message.ToolCalls == nil {
		return deltaToolCall{}, invalidEvent(event.Type)
	}

	return *event.Delta.Message.ToolCalls, nil
}
