package amazonbedrock

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// paddingKey is the random padding member Bedrock adds to stream payloads.
const paddingKey = "p"

// streamEvent is one decoded stream payload.
type streamEvent struct {
	eventType string

	// payload is the event object without its padding.
	payload json.RawMessage

	// raw is the payload wrapped under its event type, the raw chunk form.
	raw json.RawMessage
}

func decodeStreamEvent(eventType string, payload []byte) (streamEvent, error) {
	object, err := jsonobject.Parse(payload)
	if err != nil {
		return streamEvent{}, fmt.Errorf("%w: %s stream event: %w", model.ErrInvalidResponse, eventType, err)
	}

	encoded, err := object.Without(paddingKey).Encode()
	if err != nil {
		return streamEvent{}, fmt.Errorf("%w: %s stream event: %w", model.ErrInvalidResponse, eventType, err)
	}

	wrapped := jsonobject.Object{
		{
			Key:   eventType,
			Value: encoded,
		},
	}

	raw, err := wrapped.Encode()
	if err != nil {
		return streamEvent{}, fmt.Errorf("%w: %s stream event: %w", model.ErrInvalidResponse, eventType, err)
	}

	return streamEvent{
		eventType: eventType,
		payload:   encoded,
		raw:       raw,
	}, nil
}

// blockKind is the kind of an open content block.
type blockKind int

const (
	blockText blockKind = iota
	blockReasoning
	blockToolCall
)

// streamBlock is an open content block.
type streamBlock struct {
	kind blockKind

	toolCallID string
	toolName   string
	jsonText   strings.Builder

	// jsonTool marks a call of the JSON response tool, which streams as
	// text when it ends.
	jsonTool bool

	// redactedContent accumulates encrypted reasoning, attached once when
	// the block ends.
	redactedContent *strings.Builder
}

// streamState turns the events of one response into stream parts.
type streamState struct {
	prepared  *preparedRequest
	mistral   bool
	extractor *jsonObjectExtractor
	url       string
	headers   http.Header

	blocks map[int]*streamBlock

	finishReason model.FinishReason
	usage        *converseUsage
	rawUsage     json.RawMessage
	metadata     *ChatMetadata
	jsonFromTool bool
	stopSequence *string
}

// fail records a failed event and returns its error part.
func (s *streamState) fail(err error) model.StreamPart {
	s.finishReason = model.FinishReason{
		Unified: model.FinishError,
	}

	return model.StreamError{
		Err: err,
	}
}

// exception converts an in-band exception into an error part.
func (s *streamState) exception(errorType string, event streamEvent) model.StreamPart {
	return s.fail(streamError(errorType, event.payload, s.url, s.headers))
}

// handle converts one event.
func (s *streamState) handle(event streamEvent) ([]model.StreamPart, error) {
	switch event.eventType {
	case eventMessageStop:
		return nil, s.messageStop(event.payload)

	case eventMetadata:
		return nil, s.metadataEvent(event.payload)

	case eventContentBlockStart:
		return s.blockStart(event.payload)

	case eventContentBlockDelta:
		return s.blockDelta(event.payload)

	case eventContentBlockStop:
		return s.blockStop(event.payload)

	default:
		return nil, nil
	}
}

func (s *streamState) messageStop(payload json.RawMessage) error {
	var event messageStopEvent
	if err := decodeEvent(payload, eventMessageStop, &event); err != nil {
		return err
	}

	if event.StopReason == nil {
		return fmt.Errorf("%w: messageStop event without stopReason", model.ErrInvalidResponse)
	}

	s.finishReason = finishReason(*event.StopReason, s.jsonFromTool)
	s.stopSequence = event.AdditionalModelResponseFields.stopSequence()

	return nil
}

// metadataEvent records usage and the metadata that the finish part
// reports.
func (s *streamState) metadataEvent(payload json.RawMessage) error {
	var event metadataEvent
	if err := decodeEvent(payload, eventMetadata, &event); err != nil {
		return err
	}

	var usage *converseUsage
	if len(event.Usage) > 0 && string(event.Usage) != "null" {
		decoded, err := decodeUsage(event.Usage, false)
		if err != nil {
			return err
		}

		usage = decoded
		s.usage = decoded
		s.rawUsage = event.Usage
	}

	metadata := ChatMetadata{
		Trace:             traceOf(event.Trace),
		PerformanceConfig: event.PerformanceConfig,
		ServiceTier:       event.ServiceTier,
		Usage:             cacheUsageOf(usage),
	}

	hasTrace := len(event.Trace) > 0 && string(event.Trace) != "null"
	if metadata.Usage != nil || hasTrace || metadata.PerformanceConfig != nil || metadata.ServiceTier != nil {
		s.metadata = &metadata
	}

	return nil
}

func (s *streamState) blockStart(payload json.RawMessage) ([]model.StreamPart, error) {
	var event contentBlockStartEvent
	if err := decodeEvent(payload, eventContentBlockStart, &event); err != nil {
		return nil, err
	}

	if event.ContentBlockIndex == nil {
		return nil, fmt.Errorf("%w: contentBlockStart event without contentBlockIndex", model.ErrInvalidResponse)
	}

	index := *event.ContentBlockIndex

	if event.Start == nil || event.Start.ToolUse == nil {
		s.blocks[index] = &streamBlock{
			kind: blockText,
		}

		return []model.StreamPart{
			model.TextStart{
				ID: strconv.Itoa(index),
			},
		}, nil
	}

	toolUse := event.Start.ToolUse
	if toolUse.ToolUseID == "" || toolUse.Name == "" {
		return nil, fmt.Errorf("%w: tool use start without toolUseId or name", model.ErrInvalidResponse)
	}

	block := &streamBlock{
		kind:       blockToolCall,
		toolCallID: normalizeToolCallID(toolUse.ToolUseID, s.mistral),
		toolName:   toolUse.Name,
		jsonTool:   s.prepared.usesJSONResponseTool && toolUse.Name == jsonToolName,
	}

	s.blocks[index] = block

	// The JSON response tool streams no tool events.
	if block.jsonTool {
		return nil, nil
	}

	return []model.StreamPart{
		model.ToolInputStart{
			ID:       block.toolCallID,
			ToolName: block.toolName,
		},
	}, nil
}

func (s *streamState) blockDelta(payload json.RawMessage) ([]model.StreamPart, error) {
	var event contentBlockDeltaEvent
	if err := decodeEvent(payload, eventContentBlockDelta, &event); err != nil {
		return nil, err
	}

	if event.ContentBlockIndex == nil {
		return nil, fmt.Errorf("%w: contentBlockDelta event without contentBlockIndex", model.ErrInvalidResponse)
	}

	delta := event.Delta
	if delta == nil {
		return nil, nil
	}

	index := *event.ContentBlockIndex

	switch {
	case delta.Text != nil:
		return s.textDelta(index, *delta.Text), nil

	case delta.Citation != nil:
		return nil, nil

	case delta.ToolUse != nil:
		return s.toolInputDelta(index, delta.ToolUse), nil

	case delta.ReasoningContent != nil:
		return s.reasoningDelta(index, *delta.ReasoningContent)

	default:
		return nil, fmt.Errorf("%w: unknown contentBlockDelta delta", model.ErrInvalidResponse)
	}
}

func (s *streamState) textDelta(index int, text string) []model.StreamPart {
	if text == "" {
		return nil
	}

	id := strconv.Itoa(index)

	var parts []model.StreamPart
	if s.blocks[index] == nil {
		s.blocks[index] = &streamBlock{
			kind: blockText,
		}

		parts = append(parts, model.TextStart{
			ID: id,
		})
	}

	if s.extractor != nil {
		text = s.extractor.process(text)
	}

	if text == "" {
		return parts
	}

	return append(parts, model.TextDelta{
		ID:    id,
		Delta: text,
	})
}

func (s *streamState) toolInputDelta(index int, delta *toolInputDelta) []model.StreamPart {
	block := s.blocks[index]
	if block == nil || block.kind != blockToolCall {
		return nil
	}

	input := valueOrZero(delta.Input)
	block.jsonText.WriteString(input)

	if block.jsonTool {
		return nil
	}

	return []model.StreamPart{
		model.ToolInputDelta{
			ID:    block.toolCallID,
			Delta: input,
		},
	}
}

// reasoningDelta converts reasoning text, a signature, redacted data or
// encrypted content. The first present member selects the delta.
func (s *streamState) reasoningDelta(index int, delta reasoningContentDelta) ([]model.StreamPart, error) {
	switch {
	case delta.Text != nil:
		if *delta.Text == "" {
			return nil, nil
		}

		return s.reasoningPart(index, model.ReasoningDelta{
			ID:    strconv.Itoa(index),
			Delta: *delta.Text,
		}), nil

	case delta.Signature != nil:
		if *delta.Signature == "" {
			return nil, nil
		}

		return s.reasoningPart(index, model.ReasoningDelta{
			ID: strconv.Itoa(index),
			ProviderMetadata: providerMetadataOf(ReasoningMetadata{
				Signature: *delta.Signature,
			}),
		}), nil

	case delta.Data != nil:
		if *delta.Data == "" {
			return nil, nil
		}

		return s.reasoningPart(index, model.ReasoningDelta{
			ID: strconv.Itoa(index),
			ProviderMetadata: providerMetadataOf(ReasoningMetadata{
				RedactedData: *delta.Data,
			}),
		}), nil

	case delta.RedactedContent != nil:
		if *delta.RedactedContent == "" {
			return nil, nil
		}

		parts := s.openReasoning(index)

		// Encrypted content accumulates and is attached once at the end:
		// reasoning metadata is last-write-wins, so per-delta metadata
		// would drop earlier chunks.
		if block := s.blocks[index]; block.kind == blockReasoning {
			if block.redactedContent == nil {
				block.redactedContent = &strings.Builder{}
			}

			block.redactedContent.WriteString(*delta.RedactedContent)
		}

		return parts, nil

	default:
		return nil, fmt.Errorf("%w: unknown reasoningContent delta", model.ErrInvalidResponse)
	}
}

// reasoningPart returns delta, preceded by a reasoning start when the block
// is new.
func (s *streamState) reasoningPart(index int, delta model.ReasoningDelta) []model.StreamPart {
	return append(s.openReasoning(index), delta)
}

// openReasoning starts a reasoning block at index when none is open.
func (s *streamState) openReasoning(index int) []model.StreamPart {
	if s.blocks[index] != nil {
		return nil
	}

	s.blocks[index] = &streamBlock{
		kind: blockReasoning,
	}

	return []model.StreamPart{
		model.ReasoningStart{
			ID: strconv.Itoa(index),
		},
	}
}

// blockStop ends the block at an index. A JSON response tool call streams
// its input as text.
func (s *streamState) blockStop(payload json.RawMessage) ([]model.StreamPart, error) {
	var event contentBlockStopEvent
	if err := decodeEvent(payload, eventContentBlockStop, &event); err != nil {
		return nil, err
	}

	if event.ContentBlockIndex == nil {
		return nil, fmt.Errorf("%w: contentBlockStop event without contentBlockIndex", model.ErrInvalidResponse)
	}

	index := *event.ContentBlockIndex
	block := s.blocks[index]

	if block == nil {
		return nil, nil
	}

	delete(s.blocks, index)

	id := strconv.Itoa(index)

	switch block.kind {
	case blockReasoning:
		end := model.ReasoningEnd{
			ID: id,
		}

		if block.redactedContent != nil {
			end.ProviderMetadata = providerMetadataOf(ReasoningMetadata{
				RedactedContent: block.redactedContent.String(),
			})
		}

		return []model.StreamPart{
			end,
		}, nil

	case blockToolCall:
		return s.toolCallEnd(id, block), nil

	default:
		return []model.StreamPart{
			model.TextEnd{
				ID: id,
			},
		}, nil
	}
}

func (s *streamState) toolCallEnd(id string, block *streamBlock) []model.StreamPart {
	input := block.jsonText.String()

	if block.jsonTool {
		s.jsonFromTool = true

		return []model.StreamPart{
			model.TextStart{
				ID: id,
			},
			model.TextDelta{
				ID:    id,
				Delta: input,
			},
			model.TextEnd{
				ID: id,
			},
		}
	}

	if input == "" {
		input = "{}"
	}

	return []model.StreamPart{
		model.ToolInputEnd{
			ID: block.toolCallID,
		},
		model.ToolCall{
			ToolCallID: block.toolCallID,
			ToolName:   block.toolName,
			Input:      input,
		},
	}
}

// finish returns the Finish part with the usage and metadata of the
// stream.
func (s *streamState) finish() model.StreamPart {
	metadata := s.metadata

	if s.jsonFromTool || s.stopSequence != nil {
		updated := ChatMetadata{}
		if metadata != nil {
			updated = *metadata
		}

		updated.IsJSONResponseFromTool = s.jsonFromTool
		updated.StopSequence = s.stopSequence
		metadata = &updated
	}

	finish := model.Finish{
		Usage:        convertUsage(s.usage, s.rawUsage),
		FinishReason: s.finishReason,
	}

	if metadata != nil {
		finish.ProviderMetadata = providerMetadataOf(*metadata)
	}

	return finish
}

func decodeEvent(payload json.RawMessage, eventType string, target any) error {
	if err := json.Unmarshal(payload, target); err != nil {
		return fmt.Errorf("%w: %s stream event: %w", model.ErrInvalidResponse, eventType, err)
	}

	return nil
}
