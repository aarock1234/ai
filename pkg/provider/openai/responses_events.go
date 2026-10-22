package openai

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// responsesEventType is the type of a streaming event.
type responsesEventType string

// Streaming event types handled by the stream.
const (
	responsesEventTextDelta              responsesEventType = "response.output_text.delta"
	responsesEventCompleted              responsesEventType = "response.completed"
	responsesEventIncomplete             responsesEventType = "response.incomplete"
	responsesEventFailed                 responsesEventType = "response.failed"
	responsesEventCreated                responsesEventType = "response.created"
	responsesEventInProgress             responsesEventType = "response.in_progress"
	responsesEventItemAdded              responsesEventType = "response.output_item.added"
	responsesEventItemDone               responsesEventType = "response.output_item.done"
	responsesEventFunctionArgumentsDelta responsesEventType = "response.function_call_arguments.delta"
	responsesEventCustomToolInputDelta   responsesEventType = "response.custom_tool_call_input.delta"
	responsesEventPartialImage           responsesEventType = "response.image_generation_call.partial_image"
	responsesEventCodeDelta              responsesEventType = "response.code_interpreter_call_code.delta"
	responsesEventCodeDone               responsesEventType = "response.code_interpreter_call_code.done"
	responsesEventAnnotationAdded        responsesEventType = "response.output_text.annotation.added"
	responsesEventSummaryPartAdded       responsesEventType = "response.reasoning_summary_part.added"
	responsesEventSummaryTextDelta       responsesEventType = "response.reasoning_summary_text.delta"
	responsesEventSummaryPartDone        responsesEventType = "response.reasoning_summary_part.done"
	responsesEventPatchDiffDelta         responsesEventType = "response.apply_patch_call_operation_diff.delta"
	responsesEventPatchDiffDone          responsesEventType = "response.apply_patch_call_operation_diff.done"
	responsesEventError                  responsesEventType = "error"
)

// responsesEvent is one decoded streaming event.
type responsesEvent interface {
	responsesEventKind() responsesEventType
}

// responsesTextDeltaEvent appends text to a message.
type responsesTextDeltaEvent struct {
	ItemID      string             `json:"item_id"`
	OutputIndex *int               `json:"output_index"`
	Delta       string             `json:"delta"`
	Logprobs    []ResponsesLogprob `json:"logprobs"`
}

// responsesFinishedEvent is response.completed or response.incomplete.
type responsesFinishedEvent struct {
	Type     responsesEventType        `json:"type"`
	Response responsesFinishedResponse `json:"response"`
}

// responsesFinishedResponse is the final state of a streamed response.
type responsesFinishedResponse struct {
	IncompleteDetails *responsesIncompleteDetails `json:"incomplete_details"`
	Usage             *responsesUsage             `json:"usage"`
	Reasoning         *responsesReasoningState    `json:"reasoning"`
	ServiceTier       string                      `json:"service_tier"`
}

// responsesFailedEvent reports a failed response.
type responsesFailedEvent struct {
	SequenceNumber int                     `json:"sequence_number"`
	Response       responsesFailedResponse `json:"response"`
}

// responsesFailedResponse is the final state of a failed response.
type responsesFailedResponse struct {
	Error             *responsesFailure           `json:"error"`
	IncompleteDetails *responsesIncompleteDetails `json:"incomplete_details"`
	Usage             *responsesUsage             `json:"usage"`
	Reasoning         *responsesReasoningState    `json:"reasoning"`
	ServiceTier       string                      `json:"service_tier"`
}

// responsesFailure is the error of a failed response.
type responsesFailure struct {
	Code    responsesErrorCode `json:"code"`
	Message string             `json:"message"`
}

// responsesStartedEvent is response.created or response.in_progress.
type responsesStartedEvent struct {
	Type     responsesEventType       `json:"type"`
	Response responsesStartedResponse `json:"response"`
}

// responsesStartedResponse identifies a streamed response.
type responsesStartedResponse struct {
	ID          string  `json:"id"`
	CreatedAt   float64 `json:"created_at"`
	Model       string  `json:"model"`
	ServiceTier string  `json:"service_tier"`
}

// responsesItemEvent is response.output_item.added or done.
type responsesItemEvent struct {
	Type        responsesEventType
	OutputIndex int
	Item        responsesOutputItem
}

// responsesArgumentsDeltaEvent appends to a function or custom tool input.
type responsesArgumentsDeltaEvent struct {
	Type        responsesEventType `json:"type"`
	ItemID      string             `json:"item_id"`
	OutputIndex int                `json:"output_index"`
	Delta       string             `json:"delta"`
}

// responsesPartialImageEvent carries a partial generated image.
type responsesPartialImageEvent struct {
	ItemID          string `json:"item_id"`
	OutputIndex     int    `json:"output_index"`
	PartialImageB64 string `json:"partial_image_b64"`
}

// responsesCodeDeltaEvent appends to code interpreter code.
type responsesCodeDeltaEvent struct {
	ItemID      string `json:"item_id"`
	OutputIndex int    `json:"output_index"`
	Delta       string `json:"delta"`
}

// responsesCodeDoneEvent completes code interpreter code.
type responsesCodeDoneEvent struct {
	ItemID      string `json:"item_id"`
	OutputIndex int    `json:"output_index"`
	Code        string `json:"code"`
}

// responsesAnnotationEvent adds an annotation to the current message.
type responsesAnnotationEvent struct {
	Annotation ResponsesAnnotation `json:"annotation"`
}

// responsesSummaryEvent is a reasoning summary part or text event.
type responsesSummaryEvent struct {
	Type         responsesEventType `json:"type"`
	ItemID       string             `json:"item_id"`
	OutputIndex  *int               `json:"output_index"`
	SummaryIndex int                `json:"summary_index"`
	Delta        string             `json:"delta"`
}

// responsesPatchDiffEvent is an apply patch diff delta or completion.
type responsesPatchDiffEvent struct {
	Type        responsesEventType `json:"type"`
	ItemID      string             `json:"item_id"`
	OutputIndex int                `json:"output_index"`
	Delta       string             `json:"delta"`
	Diff        string             `json:"diff"`
}

// responsesErrorEvent is an error event. OpenAI sends the details either at
// the top level or nested under "error".
type responsesErrorEvent struct {
	SequenceNumber int                  `json:"sequence_number"`
	Error          *responsesErrorFrame `json:"error"`
	Code           responsesErrorCode   `json:"code"`
	Message        string               `json:"message"`
	Param          string               `json:"param"`
}

// responsesErrorFrame is the nested error of an error event.
type responsesErrorFrame struct {
	Type    string             `json:"type"`
	Code    responsesErrorCode `json:"code"`
	Message string             `json:"message"`
	Param   string             `json:"param"`
}

// responsesErrorCode is an error code sent as a string or a number.
type responsesErrorCode string

// UnmarshalJSON accepts a string, a number or null.
func (c *responsesErrorCode) UnmarshalJSON(data []byte) error {
	var text string
	if err := json.Unmarshal(data, &text); err == nil {
		*c = responsesErrorCode(text)

		return nil
	}

	var number json.Number
	if err := json.Unmarshal(data, &number); err != nil {
		return fmt.Errorf("error code must be a string or a number: %w", err)
	}

	*c = responsesErrorCode(number)

	return nil
}

func (responsesTextDeltaEvent) responsesEventKind() responsesEventType {
	return responsesEventTextDelta
}

func (e responsesFinishedEvent) responsesEventKind() responsesEventType {
	return e.Type
}

func (responsesFailedEvent) responsesEventKind() responsesEventType {
	return responsesEventFailed
}

func (e responsesStartedEvent) responsesEventKind() responsesEventType {
	return e.Type
}

func (e responsesItemEvent) responsesEventKind() responsesEventType {
	return e.Type
}

func (e responsesArgumentsDeltaEvent) responsesEventKind() responsesEventType {
	return e.Type
}

func (responsesPartialImageEvent) responsesEventKind() responsesEventType {
	return responsesEventPartialImage
}

func (responsesCodeDeltaEvent) responsesEventKind() responsesEventType {
	return responsesEventCodeDelta
}

func (responsesCodeDoneEvent) responsesEventKind() responsesEventType {
	return responsesEventCodeDone
}

func (responsesAnnotationEvent) responsesEventKind() responsesEventType {
	return responsesEventAnnotationAdded
}

func (e responsesSummaryEvent) responsesEventKind() responsesEventType {
	return e.Type
}

func (e responsesPatchDiffEvent) responsesEventKind() responsesEventType {
	return e.Type
}

func (responsesErrorEvent) responsesEventKind() responsesEventType {
	return responsesEventError
}

// errResponsesEventType reports an event without a string type.
var errResponsesEventType = errors.New("event has no type")

// decodeResponsesEvent decodes one streaming event. It returns nil for
// unknown events, including output item events of unknown item types, and
// for events the stream does not use.
func decodeResponsesEvent(data []byte) (responsesEvent, error) {
	var header struct {
		Type *string `json:"type"`
	}
	if err := json.Unmarshal(data, &header); err != nil {
		return nil, fmt.Errorf("%w: decode event: %w", model.ErrInvalidResponse, err)
	}

	if header.Type == nil {
		return nil, fmt.Errorf("%w: %w", model.ErrInvalidResponse, errResponsesEventType)
	}

	eventType := responsesEventType(*header.Type)

	switch eventType {
	case responsesEventTextDelta:
		return decodeResponsesEventAs[responsesTextDeltaEvent](data, eventType)
	case responsesEventCompleted, responsesEventIncomplete:
		return decodeResponsesEventAs[responsesFinishedEvent](data, eventType)
	case responsesEventFailed:
		return decodeResponsesEventAs[responsesFailedEvent](data, eventType)
	case responsesEventCreated, responsesEventInProgress:
		return decodeResponsesEventAs[responsesStartedEvent](data, eventType)
	case responsesEventItemAdded, responsesEventItemDone:
		return decodeResponsesItemEvent(data, eventType)
	case responsesEventFunctionArgumentsDelta, responsesEventCustomToolInputDelta:
		return decodeResponsesEventAs[responsesArgumentsDeltaEvent](data, eventType)
	case responsesEventPartialImage:
		return decodeResponsesEventAs[responsesPartialImageEvent](data, eventType)
	case responsesEventCodeDelta:
		return decodeResponsesEventAs[responsesCodeDeltaEvent](data, eventType)
	case responsesEventCodeDone:
		return decodeResponsesEventAs[responsesCodeDoneEvent](data, eventType)
	case responsesEventAnnotationAdded:
		return decodeResponsesEventAs[responsesAnnotationEvent](data, eventType)
	case responsesEventSummaryPartAdded, responsesEventSummaryTextDelta, responsesEventSummaryPartDone:
		return decodeResponsesEventAs[responsesSummaryEvent](data, eventType)
	case responsesEventPatchDiffDelta, responsesEventPatchDiffDone:
		return decodeResponsesEventAs[responsesPatchDiffEvent](data, eventType)
	case responsesEventError:
		return decodeResponsesEventAs[responsesErrorEvent](data, eventType)
	default:
		return nil, nil
	}
}

// decodeResponsesEventAs decodes data into the concrete event type T.
func decodeResponsesEventAs[T responsesEvent](data []byte, eventType responsesEventType) (responsesEvent, error) {
	var event T
	if err := json.Unmarshal(data, &event); err != nil {
		return nil, fmt.Errorf("%w: decode %s event: %w", model.ErrInvalidResponse, eventType, err)
	}

	return event, nil
}

// decodeResponsesItemEvent decodes an output item event. It returns nil when
// the item type is unknown.
func decodeResponsesItemEvent(data []byte, eventType responsesEventType) (responsesEvent, error) {
	var envelope struct {
		OutputIndex *int            `json:"output_index"`
		Item        json.RawMessage `json:"item"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil {
		return nil, fmt.Errorf("%w: decode %s event: %w", model.ErrInvalidResponse, eventType, err)
	}

	if envelope.OutputIndex == nil || len(bytes.TrimSpace(envelope.Item)) == 0 {
		return nil, fmt.Errorf("%w: %s event without output index or item", model.ErrInvalidResponse, eventType)
	}

	item, err := decodeResponsesOutputItem(envelope.Item)
	if err != nil {
		return nil, err
	}

	if item == nil {
		return nil, nil
	}

	return responsesItemEvent{
		Type:        eventType,
		OutputIndex: *envelope.OutputIndex,
		Item:        item,
	}, nil
}
