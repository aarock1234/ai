package anthropic

import (
	"encoding/json"
	"fmt"
)

// Stream event types.
const (
	eventMessageStart      = "message_start"
	eventContentBlockStart = "content_block_start"
	eventContentBlockDelta = "content_block_delta"
	eventContentBlockStop  = "content_block_stop"
	eventMessageDelta      = "message_delta"
	eventMessageStop       = "message_stop"
	eventPing              = "ping"
	eventError             = "error"
)

// Content block delta types.
const (
	deltaInputJSON  = "input_json_delta"
	deltaText       = "text_delta"
	deltaThinking   = "thinking_delta"
	deltaSignature  = "signature_delta"
	deltaCompaction = "compaction_delta"
	deltaCitations  = "citations_delta"
)

// streamEvent is one decoded server-sent event of a streaming response.
type streamEvent interface {
	isStreamEvent()
}

type messageStartEvent struct {
	Message streamMessage `json:"message"`
}

type streamMessage struct {
	ID                   *string               `json:"id"`
	Model                *string               `json:"model"`
	Usage                json.RawMessage       `json:"usage"`
	Content              []startedToolUse      `json:"content"`
	StopReason           *string               `json:"stop_reason"`
	InputTransformations []InputTransformation `json:"input_transformations"`
	Container            *wireContainer        `json:"container"`
}

// startedToolUse is one item of message_start content. Upstream accepts only
// complete tool_use blocks there, which programmatic tool calling delivers.
type startedToolUse struct {
	Type string `json:"type"`

	responseToolUse
}

type contentBlockStartEvent struct {
	Index        int             `json:"index"`
	RawBlock     json.RawMessage `json:"content_block"`
	ContentBlock responseBlock   `json:"-"`
}

type contentBlockDeltaEvent struct {
	Index int        `json:"index"`
	Delta blockDelta `json:"delta"`
}

// blockDelta is one of the content block delta variants, discriminated by
// Type.
type blockDelta struct {
	Type        string    `json:"type"`
	PartialJSON string    `json:"partial_json"`
	Text        string    `json:"text"`
	Thinking    string    `json:"thinking"`
	Signature   string    `json:"signature"`
	Content     *string   `json:"content"`
	Citation    *Citation `json:"citation"`
}

type contentBlockStopEvent struct {
	Index int `json:"index"`
}

type messageDeltaEvent struct {
	Delta                messageDelta           `json:"delta"`
	Usage                json.RawMessage        `json:"usage"`
	InputTransformations []InputTransformation  `json:"input_transformations"`
	ContextManagement    *wireContextManagement `json:"context_management"`
}

type messageDelta struct {
	StopReason       *string           `json:"stop_reason"`
	StopSequence     *string           `json:"stop_sequence"`
	StopDetails      *wireStopDetails  `json:"stop_details"`
	SafeguardResults []SafeguardResult `json:"safeguard_results"`
	Container        *wireContainer    `json:"container"`
}

// deltaUsage is the cumulative usage reported by message_delta.
type deltaUsage struct {
	InputTokens              *int                 `json:"input_tokens"`
	OutputTokens             int                  `json:"output_tokens"`
	OutputTokensDetails      *outputTokensDetails `json:"output_tokens_details"`
	CacheCreationInputTokens *int                 `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     *int                 `json:"cache_read_input_tokens"`
	Iterations               []wireUsageIteration `json:"iterations"`
}

// startUsage is the usage reported by message_start.
type startUsage struct {
	InputTokens              int  `json:"input_tokens"`
	CacheCreationInputTokens *int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     *int `json:"cache_read_input_tokens"`
}

type messageStopEvent struct{}

type pingEvent struct{}

// errorEvent is an in-band error. Raw is the error object as received.
type errorEvent struct {
	Detail streamErrorDetail
	Raw    json.RawMessage
}

type wireErrorEvent struct {
	Error json.RawMessage `json:"error"`
}

// decodeStreamEvent decodes the data of one server-sent event.
func decodeStreamEvent(data []byte) (streamEvent, error) {
	var header blockHeader
	if err := json.Unmarshal(data, &header); err != nil {
		return nil, err
	}

	switch header.Type {
	case eventMessageStart:
		return decodeEvent[messageStartEvent](data)
	case eventContentBlockStart:
		return decodeContentBlockStart(data)
	case eventContentBlockDelta:
		return decodeContentBlockDelta(data)
	case eventContentBlockStop:
		return decodeEvent[contentBlockStopEvent](data)
	case eventMessageDelta:
		return decodeEvent[messageDeltaEvent](data)
	case eventMessageStop:
		return &messageStopEvent{}, nil
	case eventPing:
		return &pingEvent{}, nil
	case eventError:
		return decodeErrorEvent(data)
	default:
		return nil, fmt.Errorf("unsupported stream event type %q", header.Type)
	}
}

func decodeEvent[Event any, Pointer interface {
	*Event
	streamEvent
}](data []byte) (streamEvent, error) {
	var event Event
	if err := json.Unmarshal(data, &event); err != nil {
		return nil, err
	}

	return Pointer(&event), nil
}

func decodeContentBlockStart(data []byte) (streamEvent, error) {
	var event contentBlockStartEvent
	if err := json.Unmarshal(data, &event); err != nil {
		return nil, err
	}

	block, err := decodeResponseBlock(event.RawBlock)
	if err != nil {
		return nil, err
	}

	event.ContentBlock = block

	return &event, nil
}

func decodeContentBlockDelta(data []byte) (streamEvent, error) {
	var event contentBlockDeltaEvent
	if err := json.Unmarshal(data, &event); err != nil {
		return nil, err
	}

	err := checkRequiredEnum(
		"delta type",
		event.Delta.Type,
		deltaInputJSON,
		deltaText,
		deltaThinking,
		deltaSignature,
		deltaCompaction,
		deltaCitations,
	)
	if err != nil {
		return nil, err
	}

	if event.Delta.Type == deltaCitations && event.Delta.Citation == nil {
		return nil, fmt.Errorf("citations delta without citation")
	}

	return &event, nil
}

func decodeErrorEvent(data []byte) (streamEvent, error) {
	var wire wireErrorEvent
	if err := json.Unmarshal(data, &wire); err != nil {
		return nil, err
	}

	var detail streamErrorDetail
	if err := json.Unmarshal(wire.Error, &detail); err != nil {
		return nil, err
	}

	return &errorEvent{
		Detail: detail,
		Raw:    wire.Error,
	}, nil
}

func (*messageStartEvent) isStreamEvent()      {}
func (*contentBlockStartEvent) isStreamEvent() {}
func (*contentBlockDeltaEvent) isStreamEvent() {}
func (*contentBlockStopEvent) isStreamEvent()  {}
func (*messageDeltaEvent) isStreamEvent()      {}
func (*messageStopEvent) isStreamEvent()       {}
func (*pingEvent) isStreamEvent()              {}
func (*errorEvent) isStreamEvent()             {}
