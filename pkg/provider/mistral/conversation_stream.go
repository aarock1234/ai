package mistral

import (
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"github.com/aarock1234/ai/internal/chatsse"
	"github.com/aarock1234/ai/internal/sse"
	"github.com/aarock1234/ai/internal/toolcalls"
	"github.com/aarock1234/ai/pkg/model"
)

// conversationStreamSettings configures one conversation stream.
type conversationStreamSettings struct {
	includeRawChunks bool
	converter        *contentConverter
	generateID       func() string
}

// conversationStream converts conversation events into stream parts.
type conversationStream struct {
	settings conversationStreamSettings
	tracker  *toolcalls.Tracker

	// executions holds built-in tool runs that have not finished.
	executions map[string]*pendingExecution

	// active is the open text or reasoning block, if any.
	active *activeBlock

	usage            *Usage
	hasFunctionCalls bool
	hasError         bool
	isDone           bool
	sentModelID      bool
}

type pendingExecution struct {
	name      string
	arguments string
	function  *string
}

// activeBlock is an open text or reasoning block.
type activeBlock struct {
	id        string
	reasoning bool
}

func newConversationStream(settings conversationStreamSettings) *conversationStream {
	return &conversationStream{
		settings: settings,
		tracker: toolcalls.NewTracker(toolcalls.Options{
			GenerateID: settings.generateID,
		}),
		executions: make(map[string]*pendingExecution),
	}
}

// Process converts one server-sent event into stream parts.
func (s *conversationStream) Process(event sse.Event) ([]model.StreamPart, error) {
	var parts []model.StreamPart
	if s.settings.includeRawChunks {
		if raw, ok := chatsse.Raw(event); ok {
			parts = append(parts, raw)
		}
	}

	converted, err := s.event([]byte(event.Data))
	if err != nil {
		return append(parts, s.Fail(err)), nil
	}

	return append(parts, converted...), nil
}

// event converts one event by its type.
func (s *conversationStream) event(data []byte) ([]model.StreamPart, error) {
	var header typeHeader
	if err := json.Unmarshal(data, &header); err != nil {
		return nil, fmt.Errorf("%w: %w", model.ErrInvalidResponse, err)
	}

	switch header.Type {
	case eventResponseStarted:
		return decodeEvent(data, s.responseStarted)

	case eventResponseDone:
		return decodeEvent(data, s.responseDone)

	case eventResponseError:
		return decodeEvent(data, s.responseError)

	case eventMessageOutputDelta:
		return decodeEvent(data, s.messageOutputDelta)

	case eventFunctionCallDelta:
		return decodeEvent(data, s.functionCallDelta)

	case eventToolExecutionStarted, eventToolExecutionDelta:
		return decodeEvent(data, s.toolExecutionDelta)

	case eventToolExecutionDone:
		return decodeEvent(data, s.toolExecutionDone)

	default:
		return nil, fmt.Errorf("%w: unsupported conversation event type %q", model.ErrInvalidResponse, header.Type)
	}
}

// decodeEvent decodes an event into E and converts it with handle.
func decodeEvent[E any](data []byte, handle func(event E) ([]model.StreamPart, error)) ([]model.StreamPart, error) {
	var event E
	if err := json.Unmarshal(data, &event); err != nil {
		return nil, fmt.Errorf("%w: %w", model.ErrInvalidResponse, err)
	}

	return handle(event)
}

// responseStarted reports the conversation ID and creation time. A creation
// time that is not RFC 3339 is left out.
func (s *conversationStream) responseStarted(event responseStartedEvent) ([]model.StreamPart, error) {
	metadata := model.ResponseMetadata{
		ID: event.ConversationID,
	}

	if event.CreatedAt != nil {
		if timestamp, err := time.Parse(time.RFC3339Nano, *event.CreatedAt); err == nil {
			metadata.Timestamp = timestamp
		}
	}

	return []model.StreamPart{
		metadata,
	}, nil
}

func (s *conversationStream) responseDone(event responseDoneEvent) ([]model.StreamPart, error) {
	s.usage = event.Usage
	s.isDone = true

	return nil, nil
}

func (s *conversationStream) responseError(event ConversationError) ([]model.StreamPart, error) {
	return nil, &event
}

func (s *conversationStream) messageOutputDelta(event messageOutputDeltaEvent) ([]model.StreamPart, error) {
	var parts []model.StreamPart

	if !s.sentModelID && event.Model != nil {
		s.sentModelID = true
		parts = append(parts, model.ResponseMetadata{
			ModelID: *event.Model,
		})
	}

	content, ok := s.settings.converter.part(event.Content.Part)
	if !ok {
		return parts, nil
	}

	contentIndex := 0
	if event.ContentIndex != nil {
		contentIndex = *event.ContentIndex
	}

	blockID := event.ID + "-" + strconv.Itoa(contentIndex)

	switch content := content.(type) {
	case model.Text:
		parts = append(parts, s.blockDelta(blockID, false, content.Text)...)

	case model.Reasoning:
		parts = append(parts, s.blockDelta(blockID, true, content.Text)...)

	case model.Source:
		parts = append(parts, content)
	}

	return parts, nil
}

// blockDelta appends to the block with the given ID and kind, closing any
// other open block first.
func (s *conversationStream) blockDelta(id string, reasoning bool, delta string) []model.StreamPart {
	var parts []model.StreamPart

	if s.active == nil || s.active.id != id || s.active.reasoning != reasoning {
		parts = append(parts, s.closeBlock()...)

		s.active = &activeBlock{
			id:        id,
			reasoning: reasoning,
		}

		if reasoning {
			parts = append(parts, model.ReasoningStart{
				ID: id,
			})
		} else {
			parts = append(parts, model.TextStart{
				ID: id,
			})
		}
	}

	if reasoning {
		return append(parts, model.ReasoningDelta{
			ID:    id,
			Delta: delta,
		})
	}

	return append(parts, model.TextDelta{
		ID:    id,
		Delta: delta,
	})
}

func (s *conversationStream) closeBlock() []model.StreamPart {
	if s.active == nil {
		return nil
	}

	block := s.active
	s.active = nil

	if block.reasoning {
		return []model.StreamPart{
			model.ReasoningEnd{
				ID: block.id,
			},
		}
	}

	return []model.StreamPart{
		model.TextEnd{
			ID: block.id,
		},
	}
}

func (s *conversationStream) functionCallDelta(event functionCallDeltaEvent) ([]model.StreamPart, error) {
	parts := s.closeBlock()
	s.hasFunctionCalls = true

	toolParts, err := s.tracker.ProcessDelta(toolcalls.Delta{
		Index:     event.OutputIndex,
		ID:        &event.ToolCallID,
		Name:      &event.Name,
		Arguments: &event.Arguments,
	})

	return append(parts, toolParts...), err
}

// toolExecutionDelta starts a built-in tool run or appends to its arguments.
func (s *conversationStream) toolExecutionDelta(event toolExecutionDeltaEvent) ([]model.StreamPart, error) {
	parts := s.closeBlock()

	execution, ok := s.executions[event.ID]
	if ok {
		execution.arguments += event.Arguments

		return parts, nil
	}

	s.executions[event.ID] = &pendingExecution{
		name:      event.Name,
		arguments: event.Arguments,
		function:  event.Function,
	}

	return append(parts, model.ToolInputStart{
		ID:               event.ID,
		ToolName:         s.settings.converter.mapping.ToApplication(event.Name),
		ProviderExecuted: true,
	}), nil
}

// toolExecutionDone emits the run's input, call and result.
func (s *conversationStream) toolExecutionDone(event toolExecutionDoneEvent) ([]model.StreamPart, error) {
	execution, ok := s.executions[event.ID]
	if !ok {
		return nil, fmt.Errorf("%w: missing tool execution start for %s", model.ErrInvalidResponse, event.ID)
	}

	delete(s.executions, event.ID)

	function := event.Function
	if function == nil {
		function = execution.function
	}

	call, result, err := s.settings.converter.toolExecution(toolExecution{
		ID:        event.ID,
		Name:      execution.name,
		Arguments: execution.arguments,
		Function:  function,
		Info:      event.Info,
	})
	if err != nil {
		return nil, err
	}

	return []model.StreamPart{
		model.ToolInputDelta{
			ID:    event.ID,
			Delta: call.Input,
		},
		model.ToolInputEnd{
			ID: event.ID,
		},
		call,
		result,
	}, nil
}

// Fail records that the stream failed and returns the error part.
func (s *conversationStream) Fail(err error) model.StreamPart {
	s.hasError = true

	return model.StreamError{
		Err: err,
	}
}

// Flush closes the open block and finishes. The stream failed when it
// reported an error, did not finish, or left a tool run unfinished.
func (s *conversationStream) Flush() ([]model.StreamPart, error) {
	parts := s.closeBlock()
	parts = append(parts, s.tracker.Flush()...)

	finishReason := model.FinishReason{
		Unified: model.FinishStop,
	}

	switch {
	case s.hasError || !s.isDone || len(s.executions) > 0:
		finishReason.Unified = model.FinishError
	case s.hasFunctionCalls:
		finishReason.Unified = model.FinishToolCalls
	}

	return append(parts, model.Finish{
		Usage:        convertUsage(s.usage),
		FinishReason: finishReason,
	}), nil
}
