package openresponses

import (
	"cmp"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/chatsse"
	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/internal/optional"
	"github.com/aarock1234/ai/internal/sse"
	"github.com/aarock1234/ai/pkg/model"
)

// Stream event types.
const (
	eventItemAdded             = "response.output_item.added"
	eventItemDone              = "response.output_item.done"
	eventArgumentsDelta        = "response.function_call_arguments.delta"
	eventArgumentsDone         = "response.function_call_arguments.done"
	eventCustomInputDelta      = "response.custom_tool_call_input.delta"
	eventCustomInputDone       = "response.custom_tool_call_input.done"
	eventReasoningSummaryDelta = "response.reasoning_summary_text.delta"
	eventReasoningTextDelta    = "response.reasoning_text.delta"
	eventTextDelta             = "response.output_text.delta"
	eventCompleted             = "response.completed"
	eventIncomplete            = "response.incomplete"
	eventFailed                = "response.failed"
	eventError                 = "error"
)

// responsesStreamSettings configures one stream.
type responsesStreamSettings struct {
	namespace        string
	includeRawChunks bool
	classify         ResponseErrorClassifier
}

// trackedCall accumulates a streamed function or custom tool call. Fields
// are empty until their event arrives.
type trackedCall struct {
	toolName   string
	toolCallID string
	input      string
}

// responsesStream converts Open Responses events into stream parts.
type responsesStream struct {
	settings responsesStreamSettings

	usage           model.Usage
	finishReason    model.FinishReason
	hasToolCalls    bool
	activeReasoning string

	functionCalls map[string]*trackedCall
	customCalls   map[string]*trackedCall
}

func newResponsesStream(settings responsesStreamSettings) *responsesStream {
	return &responsesStream{
		settings: settings,
		finishReason: model.FinishReason{
			Unified: model.FinishOther,
		},
		functionCalls: make(map[string]*trackedCall),
		customCalls:   make(map[string]*trackedCall),
	}
}

// Process converts one server-sent event into stream parts. Malformed
// events are reported without ending the stream.
func (s *responsesStream) Process(event sse.Event) ([]model.StreamPart, error) {
	var parts []model.StreamPart
	if s.settings.includeRawChunks {
		if raw, ok := chatsse.Raw(event); ok {
			parts = append(parts, raw)
		}
	}

	var chunk streamChunk
	if err := json.Unmarshal([]byte(event.Data), &chunk); err != nil {
		return append(parts, model.StreamError{
			Err: fmt.Errorf("%w: %w", model.ErrInvalidResponse, err),
		}), nil
	}

	return append(parts, s.chunkParts(chunk, event.Data)...), nil
}

func (s *responsesStream) chunkParts(chunk streamChunk, body string) []model.StreamPart {
	switch chunk.Type {
	case eventItemAdded:
		return s.itemAdded(chunk.Item)

	case eventItemDone:
		return s.itemDone(chunk.Item)

	case eventArgumentsDelta:
		call := trackCall(s.functionCalls, chunk.ItemID)
		call.input += chunk.Delta

		return nil

	case eventArgumentsDone:
		trackCall(s.functionCalls, chunk.ItemID).input = chunk.Arguments

		return nil

	case eventCustomInputDelta:
		call := trackCall(s.customCalls, chunk.ItemID)
		call.input += chunk.Delta

		return []model.StreamPart{
			model.ToolInputDelta{
				ID:    cmp.Or(call.toolCallID, chunk.ItemID),
				Delta: chunk.Delta,
			},
		}

	case eventCustomInputDone:
		trackCall(s.customCalls, chunk.ItemID).input = chunk.Input

		return nil

	case eventReasoningSummaryDelta, eventReasoningTextDelta:
		return []model.StreamPart{
			model.ReasoningDelta{
				ID:    chunk.ItemID,
				Delta: chunk.Delta,
			},
		}

	case eventTextDelta:
		return []model.StreamPart{
			model.TextDelta{
				ID:    chunk.ItemID,
				Delta: chunk.Delta,
			},
		}

	case eventCompleted, eventIncomplete:
		s.complete(chunk.Response)

		return nil

	case eventFailed:
		return s.failed(chunk.Response, body)

	case eventError:
		return s.streamedError(chunk.Error, body)

	default:
		return nil
	}
}

// trackCall returns the tracked call of an item, creating it when needed.
func trackCall(calls map[string]*trackedCall, itemID string) *trackedCall {
	call, ok := calls[itemID]
	if !ok {
		call = &trackedCall{}
		calls[itemID] = call
	}

	return call
}

func (s *responsesStream) itemAdded(item *outputItem) []model.StreamPart {
	switch {
	case item == nil:
		return nil

	case item.FunctionCall != nil:
		s.functionCalls[item.FunctionCall.ID] = &trackedCall{
			toolName:   item.FunctionCall.Name,
			toolCallID: item.FunctionCall.CallID,
			input:      item.FunctionCall.Arguments,
		}

		return nil

	case item.CustomToolCall != nil:
		s.customCalls[item.CustomToolCall.ID] = &trackedCall{
			toolName:   item.CustomToolCall.Name,
			toolCallID: item.CustomToolCall.CallID,
			input:      item.CustomToolCall.Input,
		}

		return []model.StreamPart{
			model.ToolInputStart{
				ID:       item.CustomToolCall.CallID,
				ToolName: item.CustomToolCall.Name,
			},
		}

	case item.Reasoning != nil:
		s.activeReasoning = item.Reasoning.ID

		return []model.StreamPart{
			model.ReasoningStart{
				ID: item.Reasoning.ID,
			},
		}

	case item.Message != nil:
		return []model.StreamPart{
			model.TextStart{
				ID: item.Message.ID,
			},
		}

	default:
		return nil
	}
}

func (s *responsesStream) itemDone(item *outputItem) []model.StreamPart {
	switch {
	case item == nil:
		return nil

	case item.FunctionCall != nil:
		return s.functionCallDone(*item.FunctionCall)

	case item.CustomToolCall != nil:
		return s.customToolCallDone(*item.CustomToolCall)

	case item.Reasoning != nil:
		if s.activeReasoning == item.Reasoning.ID {
			s.activeReasoning = ""
		}

		return []model.StreamPart{
			model.ReasoningEnd{
				ID:               item.Reasoning.ID,
				ProviderMetadata: reasoningMetadata(s.settings.namespace, *item.Reasoning, item.Reasoning.Content),
			},
		}

	case item.Message != nil:
		var annotations []URLCitation
		for _, part := range item.Message.Content {
			annotations = append(annotations, outputAnnotations(part.Annotations)...)
		}

		return []model.StreamPart{
			model.TextEnd{
				ID:               item.Message.ID,
				ProviderMetadata: textMetadata(s.settings.namespace, item.Message.ID, annotations),
			},
		}

	default:
		return nil
	}
}

// functionCallDone emits the finished function call, preferring the item's
// own arguments over the streamed ones.
func (s *responsesStream) functionCallDone(item functionCallOutput) []model.StreamPart {
	call := s.functionCalls[item.ID]
	delete(s.functionCalls, item.ID)

	if call == nil {
		call = &trackedCall{}
	}

	input := item.Arguments
	if input == "" {
		input = call.input
	}

	s.hasToolCalls = true

	return []model.StreamPart{
		model.ToolCall{
			ToolCallID:       cmp.Or(call.toolCallID, item.CallID),
			ToolName:         cmp.Or(call.toolName, item.Name),
			Input:            input,
			ProviderMetadata: itemMetadata(s.settings.namespace, item.ID),
		},
	}
}

// customToolCallDone closes the custom tool input and emits the call with
// its raw input as a JSON string.
func (s *responsesStream) customToolCallDone(item customToolCallOutput) []model.StreamPart {
	call := s.customCalls[item.ID]
	delete(s.customCalls, item.ID)

	if call == nil {
		call = &trackedCall{}
	}

	input := item.Input
	if input == "" {
		input = call.input
	}

	toolCallID := cmp.Or(call.toolCallID, item.CallID)
	s.hasToolCalls = true

	return []model.StreamPart{
		model.ToolInputEnd{
			ID: toolCallID,
		},
		model.ToolCall{
			ToolCallID:       toolCallID,
			ToolName:         cmp.Or(call.toolName, item.Name),
			Input:            string(jsonobject.EncodeString(input)),
			ProviderMetadata: itemMetadata(s.settings.namespace, item.ID),
		},
	}
}

// complete records the finish reason and usage of a finished response.
func (s *responsesStream) complete(response *responsesResponse) {
	if response == nil {
		return
	}

	s.finishReason = finishReason(response.IncompleteDetails, s.hasToolCalls)
	s.updateUsage(response.Usage)
}

// failed records an error finish reason and reports the response error.
func (s *responsesStream) failed(response *responsesResponse, body string) []model.StreamPart {
	if response == nil {
		return nil
	}

	raw := response.Status
	if response.Error != nil && response.Error.Code != "" {
		raw = response.Error.Code
	}

	s.finishReason = model.FinishReason{
		Unified: model.FinishError,
		Raw:     raw,
	}

	s.updateUsage(response.Usage)

	if response.Error == nil {
		return nil
	}

	return []model.StreamPart{
		model.StreamError{
			Err: s.streamError(*response.Error, body),
		},
	}
}

// streamedError records an error finish reason and reports the error.
func (s *responsesStream) streamedError(responseErr *ResponseError, body string) []model.StreamPart {
	var reported ResponseError
	if responseErr != nil {
		reported = *responseErr
	}

	s.finishReason = model.FinishReason{
		Unified: model.FinishError,
		Raw:     reported.Code,
	}

	return []model.StreamPart{
		model.StreamError{
			Err: s.streamError(reported, body),
		},
	}
}

// streamError converts a response error, applying the endpoint's status
// classification.
func (s *responsesStream) streamError(responseErr ResponseError, body string) error {
	metadata := s.settings.classify(responseErr)
	statusRetryable := metadata.StatusCode != 0 && model.RetryableStatus(metadata.StatusCode)

	return &model.APICallError{
		Message:      responseErr.Message,
		StatusCode:   metadata.StatusCode,
		ResponseBody: body,
		IsRetryable:  optional.Or(metadata.IsRetryable, statusRetryable),
		Cause:        &responseErr,
	}
}

// updateUsage replaces the usage with a reported one.
func (s *responsesStream) updateUsage(usage *responsesUsage) {
	if usage == nil {
		return
	}

	s.usage = convertUsage(usage)
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

// Flush closes open reasoning and reports the finish.
func (s *responsesStream) Flush() ([]model.StreamPart, error) {
	var parts []model.StreamPart

	if s.activeReasoning != "" {
		parts = append(parts, model.ReasoningEnd{
			ID: s.activeReasoning,
		})
	}

	return append(parts, model.Finish{
		Usage:        s.usage,
		FinishReason: s.finishReason,
	}), nil
}
