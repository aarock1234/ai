package huggingface

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/chatsse"
	"github.com/aarock1234/ai/internal/sse"
	"github.com/aarock1234/ai/pkg/model"
)

// Stream event types.
const (
	eventCreated        = "response.created"
	eventItemAdded      = "response.output_item.added"
	eventItemDone       = "response.output_item.done"
	eventCompleted      = "response.completed"
	eventFailed         = "response.failed"
	eventReasoningDelta = "response.reasoning_text.delta"
	eventReasoningDone  = "response.reasoning_text.done"
	eventTextDelta      = "response.output_text.delta"
	eventError          = "error"
)

// responsesStream converts Responses API events into stream parts.
type responsesStream struct {
	includeRawChunks bool

	finishReason model.FinishReason
	responseID   *string
	usage        *responsesUsage
}

// Process converts one server-sent event into stream parts.
func (s *responsesStream) Process(event sse.Event) ([]model.StreamPart, error) {
	var parts []model.StreamPart
	if s.includeRawChunks {
		if raw, ok := chatsse.Raw(event); ok {
			parts = append(parts, raw)
		}
	}

	var chunk streamChunk
	if err := json.Unmarshal([]byte(event.Data), &chunk); err != nil {
		return append(parts, s.Fail(fmt.Errorf("%w: %w", model.ErrInvalidResponse, err))), nil
	}

	if failure, ok := chunkFailure(chunk, event.Data); ok {
		s.finishReason = model.FinishReason{
			Unified: model.FinishError,
			Raw:     failure.raw,
		}

		return append(parts, model.StreamError{
			Err: failure.err,
		}), nil
	}

	return append(parts, s.chunkParts(chunk)...), nil
}

func (s *responsesStream) chunkParts(chunk streamChunk) []model.StreamPart {
	switch chunk.Type {
	case eventCreated:
		if chunk.Response == nil {
			return nil
		}

		s.responseID = new(chunk.Response.ID)

		return []model.StreamPart{
			model.ResponseMetadata{
				ID:        chunk.Response.ID,
				Timestamp: unixTime(chunk.Response.CreatedAt),
				ModelID:   chunk.Response.Model,
			},
		}

	case eventItemAdded:
		return itemAddedParts(chunk.Item)

	case eventItemDone:
		return itemDoneParts(chunk.Item)

	case eventCompleted:
		s.complete(chunk.Response)

		return nil

	case eventReasoningDelta:
		return []model.StreamPart{
			model.ReasoningDelta{
				ID:    chunk.ItemID,
				Delta: chunk.Delta,
			},
		}

	case eventReasoningDone:
		return []model.StreamPart{
			model.ReasoningEnd{
				ID: chunk.ItemID,
			},
		}

	case eventTextDelta:
		return []model.StreamPart{
			model.TextDelta{
				ID:    chunk.ItemID,
				Delta: chunk.Delta,
			},
		}

	default:
		return nil
	}
}

// complete records the finish reason and usage of the completed response.
func (s *responsesStream) complete(response *responsesResponse) {
	if response == nil {
		return
	}

	s.responseID = new(response.ID)
	s.finishReason = finishReason(response.IncompleteDetails)

	if response.Usage != nil {
		s.usage = response.Usage
	}
}

// itemAddedParts opens assistant messages, function call inputs and
// reasoning.
func itemAddedParts(item *outputItem) []model.StreamPart {
	switch {
	case item == nil:
		return nil

	case item.Message != nil && item.Message.Role == roleAssistant:
		return []model.StreamPart{
			model.TextStart{
				ID:               item.Message.ID,
				ProviderMetadata: itemMetadata(item.Message.ID),
			},
		}

	case item.FunctionCall != nil:
		return []model.StreamPart{
			model.ToolInputStart{
				ID:       item.FunctionCall.CallID,
				ToolName: item.FunctionCall.Name,
			},
		}

	case item.Reasoning != nil:
		return []model.StreamPart{
			model.ReasoningStart{
				ID:               item.Reasoning.ID,
				ProviderMetadata: itemMetadata(item.Reasoning.ID),
			},
		}

	default:
		return nil
	}
}

// itemDoneParts closes assistant messages and finishes function calls.
func itemDoneParts(item *outputItem) []model.StreamPart {
	switch {
	case item == nil:
		return nil

	case item.Message != nil && item.Message.Role == roleAssistant:
		return []model.StreamPart{
			model.TextEnd{
				ID: item.Message.ID,
			},
		}

	case item.FunctionCall != nil:
		return functionCallDoneParts(*item.FunctionCall)

	default:
		return nil
	}
}

func functionCallDoneParts(call functionCallItem) []model.StreamPart {
	parts := []model.StreamPart{
		model.ToolInputEnd{
			ID: call.CallID,
		},
	}

	content, err := toolContent(toolCallSpec{
		toolCallID: call.CallID,
		toolName:   call.Name,
		input:      call.Arguments,
		output:     call.Output,
	})
	if err != nil {
		return append(parts, model.StreamError{
			Err: err,
		})
	}

	for _, item := range content {
		switch item := item.(type) {
		case model.ToolCall:
			parts = append(parts, item)
		case model.ToolResult:
			parts = append(parts, item)
		}
	}

	return parts
}

// Fail records an error finish reason and returns the error part.
func (s *responsesStream) Fail(err error) model.StreamPart {
	s.finishReason = model.FinishReason{
		Unified: model.FinishError,
	}

	return model.StreamError{
		Err: err,
	}
}

// Flush reports the finish.
func (s *responsesStream) Flush() ([]model.StreamPart, error) {
	return []model.StreamPart{
		model.Finish{
			Usage:        convertUsage(s.usage),
			FinishReason: s.finishReason,
			ProviderMetadata: model.ProviderMetadata{
				Namespace: ResponsesMetadata{
					ResponseID: s.responseID,
				},
			},
		},
	}, nil
}

// streamFailure is a streamed error with its raw finish reason.
type streamFailure struct {
	err error
	raw string
}

// chunkFailure reports "error" and "response.failed" events that carry an
// error message.
func chunkFailure(chunk streamChunk, body string) (streamFailure, bool) {
	var message *string
	var code json.RawMessage

	switch chunk.Type {
	case eventFailed:
		if chunk.Response == nil || chunk.Response.Error == nil {
			return streamFailure{}, false
		}

		message = &chunk.Response.Error.Message
		code = chunk.Response.Error.Code

	case eventError:
		if chunk.Error != nil {
			message = &chunk.Error.Message
			code = chunk.Error.Code
		} else {
			message = chunk.Message
			code = chunk.Code
		}

	default:
		return streamFailure{}, false
	}

	if message == nil {
		return streamFailure{}, false
	}

	raw := chunk.Type
	if codeText, ok := errorCode(code); ok {
		raw = codeText
	}

	return streamFailure{
		err: &model.APICallError{
			Message:      *message,
			ResponseBody: body,
		},
		raw: raw,
	}, true
}

// errorCode returns a string or number error code as text.
func errorCode(code json.RawMessage) (string, bool) {
	trimmed := bytes.TrimSpace(code)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return "", false
	}

	var text string
	if err := json.Unmarshal(trimmed, &text); err == nil {
		return text, true
	}

	var number json.Number
	if err := json.Unmarshal(trimmed, &number); err == nil {
		return number.String(), true
	}

	return "", false
}
