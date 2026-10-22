package google

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"slices"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/internal/sse"
	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/model"
)

// InteractionError is an error event of an interaction stream.
type InteractionError struct {
	Code    string
	Message string

	// Raw is the error event.
	Raw json.RawMessage
}

// Error returns the message with the code, if any.
func (e *InteractionError) Error() string {
	if e.Code == "" {
		return e.Message
	}

	return fmt.Sprintf("%s (%s)", e.Message, e.Code)
}

// openBlockKind is the kind of an open step of a stream.
type openBlockKind int

const (
	blockUnknown openBlockKind = iota

	// blockPendingModelOutput is a model output step whose content kind
	// the first delta reveals.
	blockPendingModelOutput

	blockText
	blockReasoning
	blockImage
	blockFunctionCall
	blockBuiltinCall
	blockBuiltinResult
	blockCustom
)

// interactionsOpenBlock is the state of one open step, keyed by its index.
type interactionsOpenBlock struct {
	kind openBlockKind
	id   string

	// signature belongs to reasoning and function calls.
	signature *string

	// data, mimeType and uri hold an image that arrived on step.start.
	data     string
	mimeType string
	uri      string

	// stepType is the step type of a built-in tool call or result.
	stepType   string
	toolCallID string
	toolName   string

	// arguments accumulates the partial JSON of a function call.
	arguments strings.Builder

	// builtinArguments are the arguments object of a built-in tool call.
	builtinArguments json.RawMessage

	result json.RawMessage

	// customKind and custom describe a processing step.
	customKind string
	custom     InteractionsPartMetadata
}

// interactionsStreamSetup configures the stream state of one call.
type interactionsStreamSetup struct {
	warnings         []model.Warning
	generateID       func() string
	includeRawChunks bool

	// serviceTier is the tier of the response header, which the completed
	// interaction overrides.
	serviceTier string
}

// interactionsStreamState converts the events of one interaction stream.
// Concurrent steps are framed by index, so each open step is tracked
// separately. Sources are emitted once per stream.
type interactionsStreamState struct {
	interactionsStreamSetup

	interactionID   string
	finishStatus    string
	hasFunctionCall bool
	usage           json.RawMessage

	openBlocks  map[int]*interactionsOpenBlock
	seenSources map[string]bool

	pending []model.StreamPart
}

func newInteractionsStreamState(setup interactionsStreamSetup) *interactionsStreamState {
	return &interactionsStreamState{
		interactionsStreamSetup: setup,
		openBlocks:              map[int]*interactionsOpenBlock{},
		seenSources:             map[string]bool{},
	}
}

// bodyEvents reads the events of a response body. A read failure caused by
// the cancellation of ctx reports the context error.
func bodyEvents(ctx context.Context, body io.Reader) iter.Seq2[sse.Event, error] {
	return func(yield func(sse.Event, error) bool) {
		for event, err := range sse.Read(body) {
			if err != nil {
				if ctx.Err() != nil {
					err = context.Cause(ctx)
				} else {
					err = fmt.Errorf("read stream: %w", err)
				}
			}

			if !yield(event, err) || err != nil {
				return
			}
		}
	}
}

// parts yields the stream start, the parts of every event and the finish
// part. A read failure is reported before the finish part.
func (s *interactionsStreamState) parts(events iter.Seq2[sse.Event, error]) iter.Seq[model.StreamPart] {
	return func(yield func(model.StreamPart) bool) {
		start := model.StreamStart{
			Warnings: s.warnings,
		}

		if !yield(start) {
			return
		}

		for event, err := range events {
			if err != nil {
				s.finishStatus = interactionFailed
				s.emit(model.StreamError{
					Err: err,
				})

				break
			}

			s.handleEvent(event)

			if !s.flush(yield) {
				return
			}
		}

		s.finish()
		s.flush(yield)
	}
}

func (s *interactionsStreamState) emit(part model.StreamPart) {
	s.pending = append(s.pending, part)
}

func (s *interactionsStreamState) flush(yield func(model.StreamPart) bool) bool {
	for len(s.pending) > 0 {
		part := s.pending[0]
		s.pending = s.pending[1:]

		if !yield(part) {
			return false
		}
	}

	return true
}

func (s *interactionsStreamState) handleEvent(event sse.Event) {
	if event.Data == "[DONE]" {
		return
	}

	if s.includeRawChunks && json.Valid([]byte(event.Data)) {
		s.emit(model.Raw{
			Value: json.RawMessage(event.Data),
		})
	}

	decoded, err := decodeInteractionsEvent([]byte(event.Data))
	if err != nil {
		s.finishStatus = interactionFailed
		s.emit(model.StreamError{
			Err: err,
		})

		return
	}

	s.process(decoded, []byte(event.Data))
}

// decodeInteractionsEvent decodes an event. Step events require an index.
func decodeInteractionsEvent(data []byte) (interactionsEvent, error) {
	var event interactionsEvent
	if err := json.Unmarshal(data, &event); err != nil {
		return interactionsEvent{}, fmt.Errorf("%w: decode interaction event: %w", model.ErrInvalidResponse, err)
	}

	isStepEvent := strings.HasPrefix(event.EventType, "step.")
	if isStepEvent && event.Index == nil {
		return interactionsEvent{}, fmt.Errorf("%w: %s event has no index", model.ErrInvalidResponse, event.EventType)
	}

	if event.EventType == "step.start" && event.Step == nil {
		return interactionsEvent{}, fmt.Errorf("%w: step.start event has no step", model.ErrInvalidResponse)
	}

	if event.EventType == "step.delta" && event.Delta == nil {
		return interactionsEvent{}, fmt.Errorf("%w: step.delta event has no delta", model.ErrInvalidResponse)
	}

	isInteractionEvent := event.EventType == "interaction.created" || event.EventType == "interaction.completed"
	if isInteractionEvent && event.Interaction == nil {
		return interactionsEvent{}, fmt.Errorf("%w: %s event has no interaction", model.ErrInvalidResponse, event.EventType)
	}

	return event, nil
}

func (s *interactionsStreamState) process(event interactionsEvent, raw []byte) {
	switch event.EventType {
	case "interaction.created":
		s.interactionCreated(event.Interaction)

	case "step.start":
		s.stepStart(*event.Index, *event.Step)

	case "step.delta":
		s.stepDelta(*event.Index, *event.Delta)

	case "step.stop":
		s.stepStop(*event.Index)

	case "interaction.status_update", "interaction.in_progress", "interaction.requires_action":
		s.statusUpdate(event)

	case "interaction.completed":
		s.interactionCompleted(event.Interaction)

	case "error":
		s.errorEvent(event.Error, raw)
	}
}

func (s *interactionsStreamState) interactionCreated(interaction *interactionsEventInteraction) {
	if interaction == nil {
		return
	}

	s.interactionID = nonEmptyID(interaction.ID)

	s.emit(model.ResponseMetadata{
		ID:        s.interactionID,
		Timestamp: interactionsTimestamp(interaction.Created),
		ModelID:   valueOrEmpty(interaction.Model),
	})
}

func (s *interactionsStreamState) blockID(index int) string {
	prefix := s.interactionID
	if prefix == "" {
		prefix = "interaction"
	}

	return prefix + ":" + strconv.Itoa(index)
}

func (s *interactionsStreamState) stepStart(index int, step interactionsStep) {
	block := &interactionsOpenBlock{
		id: s.blockID(index),
	}

	s.openBlocks[index] = block

	switch {
	case step.Type == "model_output":
		s.startModelOutput(block, step.Content)

	case step.Type == "thought":
		block.kind = blockReasoning
		block.signature = step.Signature

		s.emit(model.ReasoningStart{
			ID: block.id,
		})

		for _, item := range step.Summary {
			if item.Type == "text" && item.Text != nil {
				s.emit(model.ReasoningDelta{
					ID:    block.id,
					Delta: *item.Text,
				})
			}
		}

	case step.Type == "processing_call" || step.Type == "processing_result":
		s.startProcessing(block, step)

	case step.Type == "function_call":
		block.kind = blockFunctionCall
		block.toolCallID = firstNonEmpty(step.ID, block.id)
		block.toolName = firstNonEmpty(step.Name, "unknown")
		block.signature = step.Signature
		s.hasFunctionCall = true

		s.emit(model.ToolInputStart{
			ID:       block.toolCallID,
			ToolName: block.toolName,
		})

	case slices.Contains(builtinToolCallSteps, step.Type):
		block.kind = blockBuiltinCall
		block.stepType = step.Type
		block.toolCallID = firstNonEmpty(step.ID, block.id)
		block.toolName = builtinToolName(step.Type, step.Name)
		block.builtinArguments = step.Arguments

	case slices.Contains(builtinToolResultSteps, step.Type):
		block.kind = blockBuiltinResult
		block.stepType = step.Type
		block.toolCallID = firstNonEmpty(step.CallID, block.id)
		block.toolName = builtinToolName(step.Type, step.Name)
		block.result = step.Result

	default:
		block.kind = blockUnknown
	}
}

// startModelOutput opens a model output step. Its content usually arrives
// on deltas; an initial text or image block opens the step eagerly.
func (s *interactionsStreamState) startModelOutput(block *interactionsOpenBlock, content []interactionsContent) {
	block.kind = blockPendingModelOutput

	if len(content) == 0 {
		return
	}

	initial := content[0]

	switch initial.Type {
	case "text":
		block.kind = blockText

		s.emit(model.TextStart{
			ID: block.id,
		})
		s.emitSources(annotationSources(initial.Annotations, s.generateID))

	case "image":
		block.kind = blockImage
		block.data = initial.Data
		block.mimeType = initial.MimeType
		block.uri = initial.URI
	}
}

func (s *interactionsStreamState) startProcessing(block *interactionsOpenBlock, step interactionsStep) {
	block.kind = blockCustom
	block.customKind = "google." + step.Type
	block.custom = InteractionsPartMetadata{
		Signature:     valueOrEmpty(step.Signature),
		InteractionID: s.interactionID,
	}

	if step.Type == "processing_call" {
		block.custom.ProcessingID = firstNonEmpty(step.ID, block.id)
	} else {
		block.custom.ProcessingCallID = firstNonEmpty(step.CallID, block.id)
	}
}

func (s *interactionsStreamState) stepDelta(index int, delta interactionsDelta) {
	block, ok := s.openBlocks[index]
	if !ok {
		return
	}

	isTextDelta := delta.Type == "text" || delta.Type == "text_annotation" || delta.Type == "text_annotation_delta"
	if block.kind == blockPendingModelOutput && isTextDelta {
		block.kind = blockText

		s.emit(model.TextStart{
			ID: block.id,
		})
	}

	// Image and video deltas carry the whole payload, so the file is
	// emitted at once, whether or not a text block is open.
	isOutputBlock := block.kind == blockPendingModelOutput || block.kind == blockText
	if delta.Type == "image" && (isOutputBlock || block.kind == blockImage) {
		s.emitDeltaFile("image", delta)

		block.data = ""
		block.uri = ""

		return
	}

	if delta.Type == "video" && isOutputBlock {
		s.emitDeltaFile("video", delta)

		return
	}

	switch block.kind {
	case blockCustom:
		s.processingDelta(block, delta)

	case blockText:
		s.textDelta(block, delta)

	case blockReasoning:
		s.reasoningDelta(block, delta)

	case blockFunctionCall:
		s.argumentsDelta(block, delta)

	case blockBuiltinCall:
		s.builtinCallDelta(block, delta)

	case blockBuiltinResult:
		s.builtinResultDelta(block, delta)
	}
}

func (s *interactionsStreamState) emitDeltaFile(kind string, delta interactionsDelta) {
	file, ok, err := mediaFile(kind, valueOrEmpty(delta.Data), valueOrEmpty(delta.URI), valueOrEmpty(delta.MimeType), s.interactionID)
	if err != nil {
		s.emit(model.StreamError{
			Err: err,
		})

		return
	}

	if ok {
		s.emit(file)
	}
}

func (s *interactionsStreamState) processingDelta(block *interactionsOpenBlock, delta interactionsDelta) {
	if delta.Type != "processing_call" && delta.Type != "processing_result" {
		return
	}

	if delta.Signature != nil {
		block.custom.Signature = *delta.Signature
	}

	if delta.Type == "processing_call" && valueOrEmpty(delta.ID) != "" {
		block.custom.ProcessingID = *delta.ID
	}

	if delta.Type == "processing_result" && valueOrEmpty(delta.CallID) != "" {
		block.custom.ProcessingCallID = *delta.CallID
	}
}

func (s *interactionsStreamState) textDelta(block *interactionsOpenBlock, delta interactionsDelta) {
	switch delta.Type {
	case "text":
		if text := valueOrEmpty(delta.Text); text != "" {
			s.emit(model.TextDelta{
				ID:    block.id,
				Delta: text,
			})
		}

	case "text_annotation", "text_annotation_delta":
		s.emitSources(annotationSources(delta.Annotations, s.generateID))
	}
}

func (s *interactionsStreamState) reasoningDelta(block *interactionsOpenBlock, delta interactionsDelta) {
	switch delta.Type {
	case "thought_summary":
		item := delta.Content
		if item != nil && item.Type == "text" && item.Text != nil {
			s.emit(model.ReasoningDelta{
				ID:    block.id,
				Delta: *item.Text,
			})
		}

	case "thought_signature":
		if delta.Signature != nil {
			block.signature = delta.Signature
		}
	}
}

// argumentsDelta appends partial JSON arguments. The call ID of a delta
// replaces the provisional one after the delta was emitted.
func (s *interactionsStreamState) argumentsDelta(block *interactionsOpenBlock, delta interactionsDelta) {
	if delta.Type != "arguments_delta" {
		return
	}

	var slice string
	if err := json.Unmarshal(delta.Arguments, &slice); err == nil && slice != "" {
		block.arguments.WriteString(slice)

		s.emit(model.ToolInputDelta{
			ID:    block.toolCallID,
			Delta: slice,
		})
	}

	if id := valueOrEmpty(delta.ID); id != "" {
		block.toolCallID = id
	}

	if delta.Signature != nil {
		block.signature = delta.Signature
	}

	s.hasFunctionCall = true
}

func (s *interactionsStreamState) builtinCallDelta(block *interactionsOpenBlock, delta interactionsDelta) {
	if delta.Type != block.stepType {
		return
	}

	if id := valueOrEmpty(delta.ID); id != "" {
		block.toolCallID = id
	}

	if jsonobject.IsObject(delta.Arguments) {
		block.builtinArguments = delta.Arguments
	}

	if delta.Name != nil && block.stepType == mcpServerToolCallStep {
		block.toolName = *delta.Name
	}
}

func (s *interactionsStreamState) builtinResultDelta(block *interactionsOpenBlock, delta interactionsDelta) {
	if delta.Type != block.stepType {
		return
	}

	if callID := valueOrEmpty(delta.CallID); callID != "" {
		block.toolCallID = callID
	}

	if delta.Result != nil {
		block.result = delta.Result
	}

	if delta.Name != nil && block.stepType == mcpServerToolResultStep {
		block.toolName = *delta.Name
	}
}

func (s *interactionsStreamState) stepStop(index int) {
	block, ok := s.openBlocks[index]
	if !ok {
		return
	}

	delete(s.openBlocks, index)

	partMetadata := InteractionsPartMetadata{
		InteractionID: s.interactionID,
	}

	switch block.kind {
	case blockText:
		s.emit(model.TextEnd{
			ID:               block.id,
			ProviderMetadata: partMetadata.metadata(),
		})

	case blockReasoning:
		partMetadata.Signature = valueOrEmpty(block.signature)

		s.emit(model.ReasoningEnd{
			ID:               block.id,
			ProviderMetadata: partMetadata.metadata(),
		})

	case blockImage:
		file, ok, err := mediaFile("image", block.data, block.uri, block.mimeType, s.interactionID)
		if err != nil {
			s.emit(model.StreamError{
				Err: err,
			})
		} else if ok {
			s.emit(file)
		}

	case blockFunctionCall:
		s.endFunctionCall(block, partMetadata)

	case blockCustom:
		s.emit(model.Custom{
			Kind:             block.customKind,
			ProviderMetadata: block.custom.metadata(),
		})

	case blockBuiltinCall:
		s.emit(model.ToolCall{
			ToolCallID:       block.toolCallID,
			ToolName:         block.toolName,
			Input:            compactArguments(block.builtinArguments),
			ProviderExecuted: true,
		})

	case blockBuiltinResult:
		s.emit(model.ToolResult{
			ToolCallID: block.toolCallID,
			ToolName:   block.toolName,
			Result:     resultOrNull(block.result),
		})

		s.emitSources(builtinResultSources(block.stepType, block.result, s.generateID))
	}
}

func (s *interactionsStreamState) endFunctionCall(block *interactionsOpenBlock, partMetadata InteractionsPartMetadata) {
	input := block.arguments.String()
	if input == "" {
		input = "{}"
	}

	partMetadata.Signature = valueOrEmpty(block.signature)

	s.emit(model.ToolInputEnd{
		ID: block.toolCallID,
	})

	s.emit(model.ToolCall{
		ToolCallID:       block.toolCallID,
		ToolName:         block.toolName,
		Input:            input,
		ProviderMetadata: partMetadata.metadata(),
	})
}

// emitSources emits the sources not yet emitted in this stream.
func (s *interactionsStreamState) emitSources(sources []model.Source) {
	for _, source := range sources {
		key := sourceKey(source)
		if s.seenSources[key] {
			continue
		}

		s.seenSources[key] = true
		s.emit(source)
	}
}

func (s *interactionsStreamState) statusUpdate(event interactionsEvent) {
	switch {
	case event.Status != nil:
		s.finishStatus = *event.Status

	case event.EventType == "interaction.requires_action":
		s.finishStatus = interactionRequiresAction

	default:
		s.finishStatus = interactionInProgress
	}
}

// interactionCompleted records the final status, usage and service tier,
// which the body reports more reliably than the header.
func (s *interactionsStreamState) interactionCompleted(interaction *interactionsEventInteraction) {
	if interaction == nil {
		return
	}

	if id := nonEmptyID(interaction.ID); id != "" {
		s.interactionID = id
	}

	if interaction.Status != nil {
		s.finishStatus = *interaction.Status
	}

	if wirejson.HasValue(interaction.Usage) {
		s.usage = interaction.Usage
	}

	if interaction.ServiceTier != nil {
		s.serviceTier = *interaction.ServiceTier
	}
}

func (s *interactionsStreamState) errorEvent(eventError *interactionsEventError, raw []byte) {
	s.finishStatus = interactionFailed

	streamErr := &InteractionError{
		Message: "Unknown interaction error",
		Raw:     json.RawMessage(raw),
	}

	if eventError != nil {
		streamErr.Code = valueOrEmpty(eventError.Code)

		if eventError.Message != nil {
			streamErr.Message = *eventError.Message
		}
	}

	s.emit(model.StreamError{
		Err: streamErr,
	})
}

// finish emits the finish part.
func (s *interactionsStreamState) finish() {
	usage, decodedUsage, err := convertInteractionsUsage(s.usage)
	if err != nil {
		s.emit(model.StreamError{
			Err: err,
		})
	}

	s.emit(model.Finish{
		Usage:        usage,
		FinishReason: interactionsFinishReason(s.finishStatus, s.hasFunctionCall),
		ProviderMetadata: model.ProviderMetadata{
			Namespace: InteractionsMetadata{
				InteractionID:          s.interactionID,
				ServiceTier:            s.serviceTier,
				OutputTokensByModality: outputTokensByModality(decodedUsage),
			},
		},
	})
}
