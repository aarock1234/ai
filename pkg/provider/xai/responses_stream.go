package xai

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/internal/chatsse"
	"github.com/aarock1234/ai/internal/optional"
	"github.com/aarock1234/ai/internal/sse"
	"github.com/aarock1234/ai/pkg/model"
)

// Stream event types.
const (
	eventCreated               = "response.created"
	eventInProgress            = "response.in_progress"
	eventSummaryPartAdded      = "response.reasoning_summary_part.added"
	eventSummaryTextDelta      = "response.reasoning_summary_text.delta"
	eventReasoningTextDelta    = "response.reasoning_text.delta"
	eventTextDelta             = "response.output_text.delta"
	eventTextDone              = "response.output_text.done"
	eventAnnotationAdded       = "response.output_text.annotation.added"
	eventDone                  = "response.done"
	eventCompleted             = "response.completed"
	eventIncomplete            = "response.incomplete"
	eventFailed                = "response.failed"
	eventError                 = "error"
	eventArgumentsDelta        = "response.function_call_arguments.delta"
	eventItemAdded             = "response.output_item.added"
	eventItemDone              = "response.output_item.done"
	imageGenerationEventPrefix = "response.image_generation_call."
	reasoningBlockPrefix       = "reasoning-"
	textBlockPrefix            = "text-"
)

// responsesStreamSettings configures one stream.
type responsesStreamSettings struct {
	names            toolNames
	includeRawChunks bool
	generateID       func() string
}

// ongoingCall is a function call whose arguments are streaming.
type ongoingCall struct {
	toolName   string
	toolCallID string
}

// responsesStream converts Responses API events into stream parts.
type responsesStream struct {
	settings responsesStreamSettings

	finishReason    model.FinishReason
	hasFunctionCall bool
	usage           *model.Usage
	metadata        ResponsesMetadata
	sentMetadata    bool

	textBlocks      []string
	openTextBlocks  map[string]bool
	seenToolCalls   map[string]bool
	ongoingCalls    map[int]ongoingCall
	activeReasoning map[string]bool
}

func newResponsesStream(settings responsesStreamSettings) *responsesStream {
	return &responsesStream{
		settings: settings,
		finishReason: model.FinishReason{
			Unified: model.FinishOther,
		},
		openTextBlocks:  make(map[string]bool),
		seenToolCalls:   make(map[string]bool),
		ongoingCalls:    make(map[int]ongoingCall),
		activeReasoning: make(map[string]bool),
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

	chunkParts, err := s.chunkParts(chunk, event.Data)

	return append(parts, chunkParts...), err
}

func (s *responsesStream) chunkParts(chunk streamChunk, body string) ([]model.StreamPart, error) {
	switch chunk.Type {
	case eventCreated, eventInProgress:
		return s.metadataParts(chunk.Response), nil

	case eventSummaryPartAdded:
		return s.startReasoning(chunk.ItemID), nil

	case eventSummaryTextDelta:
		return []model.StreamPart{
			s.reasoningDelta(chunk.ItemID, chunk.Delta),
		}, nil

	case eventReasoningTextDelta:
		return append(s.startReasoning(chunk.ItemID), s.reasoningDelta(chunk.ItemID, chunk.Delta)), nil

	case eventTextDelta:
		return s.textDelta(textBlockPrefix+chunk.ItemID, chunk.Delta), nil

	case eventTextDone:
		return sourceParts(citationSources(chunk.Annotations, s.settings.generateID)), nil

	case eventAnnotationAdded:
		if chunk.Annotation == nil {
			return nil, nil
		}

		return sourceParts(citationSources([]annotation{*chunk.Annotation}, s.settings.generateID)), nil

	case eventDone, eventCompleted, eventIncomplete:
		s.complete(chunk.Type, chunk.Response)

		return nil, nil

	case eventFailed:
		return s.failed(chunk, body), nil

	case eventError:
		return []model.StreamPart{
			model.StreamError{
				Err: streamError(chunk.Message, chunk.Code, body),
			},
		}, nil

	case eventArgumentsDelta:
		call, ok := s.ongoingCalls[chunk.OutputIndex]
		if !ok {
			return nil, nil
		}

		return []model.StreamPart{
			model.ToolInputDelta{
				ID:    call.toolCallID,
				Delta: chunk.Delta,
			},
		}, nil

	case eventItemAdded, eventItemDone:
		if chunk.Item == nil {
			return nil, nil
		}

		return s.itemParts(*chunk.Item, chunk.Type == eventItemDone, chunk.OutputIndex)

	default:
		if strings.HasPrefix(chunk.Type, imageGenerationEventPrefix) {
			return s.toolCallOnce(chunk.ItemID, s.settings.names.imageGeneration, emptyObjectInput), nil
		}

		return nil, nil
	}
}

// metadataParts emits the response metadata once.
func (s *responsesStream) metadataParts(response *responsesResponse) []model.StreamPart {
	if s.sentMetadata || response == nil {
		return nil
	}

	s.sentMetadata = true

	return []model.StreamPart{
		responseMetadata(*response),
	}
}

func (s *responsesStream) startReasoning(itemID string) []model.StreamPart {
	if s.activeReasoning[itemID] {
		return nil
	}

	s.activeReasoning[itemID] = true

	return []model.StreamPart{
		model.ReasoningStart{
			ID:               reasoningBlockPrefix + itemID,
			ProviderMetadata: reasoningMetadata(itemID, ""),
		},
	}
}

func (s *responsesStream) reasoningDelta(itemID string, delta string) model.StreamPart {
	return model.ReasoningDelta{
		ID:               reasoningBlockPrefix + itemID,
		Delta:            delta,
		ProviderMetadata: reasoningMetadata(itemID, ""),
	}
}

func (s *responsesStream) textDelta(blockID string, delta string) []model.StreamPart {
	var parts []model.StreamPart

	if !s.openTextBlocks[blockID] {
		s.openTextBlocks[blockID] = true
		s.textBlocks = append(s.textBlocks, blockID)
		parts = append(parts, model.TextStart{
			ID: blockID,
		})
	}

	return append(parts, model.TextDelta{
		ID:    blockID,
		Delta: delta,
	})
}

func sourceParts(sources []model.Source) []model.StreamPart {
	parts := make([]model.StreamPart, 0, len(sources))
	for _, source := range sources {
		parts = append(parts, source)
	}

	return parts
}

// complete records the usage, metadata and finish reason of a finished
// response.
func (s *responsesStream) complete(eventType string, response *responsesResponse) {
	if response == nil {
		return
	}

	if response.Usage != nil {
		s.usage = new(convertUsage(response.Usage))
		s.metadata.CostInUSDTicks = response.Usage.CostInUSDTicks
	}

	s.metadata.ServiceTier = optional.Or(response.ServiceTier, "")
	s.metadata.PromptCacheKey = optional.Or(response.PromptCacheKey, "")
	s.metadata.SafetyIdentifier = optional.Or(response.SafetyIdentifier, "")

	if eventType == eventIncomplete {
		s.finishReason = model.FinishReason{
			Unified: model.FinishOther,
			Raw:     "incomplete",
		}

		if response.IncompleteDetails != nil && response.IncompleteDetails.Reason != "" {
			s.finishReason = model.FinishReason{
				Unified: unifiedFinishReason(response.IncompleteDetails.Reason),
				Raw:     response.IncompleteDetails.Reason,
			}
		}

		return
	}

	if response.Status != nil && *response.Status != "" {
		s.finishReason = statusFinishReason(response.Status, s.hasFunctionCall)
	}
}

// failed records the finish reason and usage of a failed response and
// reports its error.
func (s *responsesStream) failed(chunk streamChunk, body string) []model.StreamPart {
	response := chunk.Response
	if response == nil {
		return nil
	}

	s.finishReason = model.FinishReason{
		Unified: model.FinishError,
		Raw:     "error",
	}

	if response.IncompleteDetails != nil && response.IncompleteDetails.Reason != "" {
		s.finishReason = model.FinishReason{
			Unified: unifiedFinishReason(response.IncompleteDetails.Reason),
			Raw:     response.IncompleteDetails.Reason,
		}
	}

	if response.Usage != nil {
		s.usage = new(convertUsage(response.Usage))
	}

	if response.Error == nil {
		return nil
	}

	return []model.StreamPart{
		model.StreamError{
			Err: streamError(response.Error.Message, response.Error.Code, body),
		},
	}
}

// itemParts converts an added or finished output item.
func (s *responsesStream) itemParts(item outputItem, done bool, outputIndex int) ([]model.StreamPart, error) {
	switch {
	case item.Type == itemReasoning:
		if !done {
			return nil, nil
		}

		return s.reasoningDone(item), nil

	case item.Type == itemFileSearchCall:
		return s.fileSearchItem(item, done)

	case item.Type == itemImageGenerationCall:
		return s.imageGenerationItem(item, done)

	case serverToolCallTypes[item.Type]:
		return s.serverToolItem(item, done)

	case item.Type == itemMessage:
		return s.messageItem(item), nil

	case item.Type == itemFunctionCall:
		return s.functionCallItem(item, done, outputIndex), nil

	default:
		return nil, nil
	}
}

// reasoningDone closes reasoning, opening it first when no summary part
// arrived, as with encrypted reasoning.
func (s *responsesStream) reasoningDone(item outputItem) []model.StreamPart {
	parts := s.startReasoning(item.ID)
	delete(s.activeReasoning, item.ID)

	return append(parts, model.ReasoningEnd{
		ID:               reasoningBlockPrefix + item.ID,
		ProviderMetadata: reasoningMetadata(item.ID, optional.Or(item.EncryptedContent, "")),
	})
}

func (s *responsesStream) fileSearchItem(item outputItem, done bool) ([]model.StreamPart, error) {
	toolName := s.settings.names.fileSearch
	parts := s.toolCallOnce(item.ID, toolName, "")

	if !done {
		return parts, nil
	}

	result, err := encodeResult(fileSearchResult(item))
	if err != nil {
		return nil, err
	}

	return append(parts, model.ToolResult{
		ToolCallID: item.ID,
		ToolName:   toolName,
		Result:     result,
	}), nil
}

func (s *responsesStream) imageGenerationItem(item outputItem, done bool) ([]model.StreamPart, error) {
	toolName := s.settings.names.imageGeneration
	parts := s.toolCallOnce(item.ID, toolName, emptyObjectInput)

	if !done {
		return parts, nil
	}

	result, err := imageGenerationResult(item, toolName)
	if err != nil {
		return nil, err
	}

	return append(parts, result), nil
}

// serverToolItem emits a server-side tool call once, when the item is added
// or, for custom tools, when it is done, and its result when it is done.
func (s *responsesStream) serverToolItem(item outputItem, done bool) ([]model.StreamPart, error) {
	toolName := s.settings.names.toolName(item)

	var parts []model.StreamPart
	if item.Type != itemCustomToolCall || done {
		parts = s.toolCallOnce(item.ID, toolName, toolInput(item))
	}

	if !done {
		return parts, nil
	}

	if item.Type != itemWebSearchCall {
		return append(parts, model.ToolResult{
			ToolCallID: item.ID,
			ToolName:   toolName,
			Result:     json.RawMessage(emptyObjectInput),
		}), nil
	}

	resultParts, err := serverToolParts(item, toolName, false)
	if err != nil {
		return nil, err
	}

	return append(parts, resultParts...), nil
}

// toolCallOnce emits the input and call of a provider-executed tool the
// first time its item appears.
func (s *responsesStream) toolCallOnce(toolCallID string, toolName string, input string) []model.StreamPart {
	if s.seenToolCalls[toolCallID] {
		return nil
	}

	s.seenToolCalls[toolCallID] = true

	return []model.StreamPart{
		model.ToolInputStart{
			ID:       toolCallID,
			ToolName: toolName,
		},
		model.ToolInputDelta{
			ID:    toolCallID,
			Delta: input,
		},
		model.ToolInputEnd{
			ID: toolCallID,
		},
		providerToolCall(toolCallID, toolName, input),
	}
}

// messageItem emits message text that did not stream as deltas, and the
// message's citations.
func (s *responsesStream) messageItem(item outputItem) []model.StreamPart {
	var parts []model.StreamPart
	blockID := textBlockPrefix + item.ID

	for _, part := range item.Content {
		if part.Text != "" && !s.openTextBlocks[blockID] {
			parts = append(parts, s.textDelta(blockID, part.Text)...)
		}

		parts = append(parts, sourceParts(citationSources(part.Annotations, s.settings.generateID))...)
	}

	return parts
}

// functionCallItem opens the input of an added function call and emits the
// finished call.
func (s *responsesStream) functionCallItem(item outputItem, done bool, outputIndex int) []model.StreamPart {
	toolCallID := optional.Or(item.CallID, "")
	toolName := optional.Or(item.Name, "")

	if !done {
		s.ongoingCalls[outputIndex] = ongoingCall{
			toolName:   toolName,
			toolCallID: toolCallID,
		}

		return []model.StreamPart{
			model.ToolInputStart{
				ID:       toolCallID,
				ToolName: toolName,
			},
		}
	}

	s.hasFunctionCall = true
	delete(s.ongoingCalls, outputIndex)

	return []model.StreamPart{
		model.ToolInputEnd{
			ID: toolCallID,
		},
		model.ToolCall{
			ToolCallID: toolCallID,
			ToolName:   toolName,
			Input:      optional.Or(item.Arguments, ""),
		},
	}
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

// Flush closes the text blocks and reports the finish.
func (s *responsesStream) Flush() ([]model.StreamPart, error) {
	parts := make([]model.StreamPart, 0, len(s.textBlocks)+1)
	for _, blockID := range s.textBlocks {
		parts = append(parts, model.TextEnd{
			ID: blockID,
		})
	}

	usage := convertUsage(nil)
	if s.usage != nil {
		usage = *s.usage
	}

	return append(parts, model.Finish{
		Usage:            usage,
		FinishReason:     s.finishReason,
		ProviderMetadata: s.metadata.providerMetadata(),
	}), nil
}

// streamErrorStatus is the HTTP status and retryability implied by a
// streamed error code.
type streamErrorStatus struct {
	statusCode  int
	isRetryable bool
}

// streamErrorStatuses maps streamed error codes to their implied status.
var streamErrorStatuses = map[string]streamErrorStatus{
	"rate_limit_exceeded": {
		statusCode:  http.StatusTooManyRequests,
		isRetryable: true,
	},
	"rate_limit_error": {
		statusCode:  http.StatusTooManyRequests,
		isRetryable: true,
	},
	"insufficient_quota": {
		statusCode: http.StatusTooManyRequests,
	},
	"api_error": {
		statusCode:  http.StatusInternalServerError,
		isRetryable: true,
	},
	"internal_server_error": {
		statusCode:  http.StatusInternalServerError,
		isRetryable: true,
	},
	"server_error": {
		statusCode:  http.StatusInternalServerError,
		isRetryable: true,
	},
	"overloaded_error": {
		statusCode:  http.StatusServiceUnavailable,
		isRetryable: true,
	},
	"service_unavailable": {
		statusCode:  http.StatusServiceUnavailable,
		isRetryable: true,
	},
	"timeout": {
		statusCode:  http.StatusGatewayTimeout,
		isRetryable: true,
	},
	"timeout_error": {
		statusCode:  http.StatusGatewayTimeout,
		isRetryable: true,
	},
	"authentication_error": {
		statusCode: http.StatusUnauthorized,
	},
	"invalid_api_key": {
		statusCode: http.StatusUnauthorized,
	},
	"permission_error": {
		statusCode: http.StatusForbidden,
	},
	"not_found_error": {
		statusCode: http.StatusNotFound,
	},
	"model_not_found": {
		statusCode: http.StatusNotFound,
	},
	"bad_request": {
		statusCode: http.StatusBadRequest,
	},
	"context_length_exceeded": {
		statusCode: http.StatusBadRequest,
	},
	"invalid_request_error": {
		statusCode: http.StatusBadRequest,
	},
}

// streamError converts a streamed error. A three-digit client or server
// error code is used as the status; other codes map through
// streamErrorStatuses.
func streamError(message string, code *string, body string) error {
	codeText := optional.Or(code, "")

	status := streamErrorStatuses[codeText]
	if statusCode, ok := httpErrorStatus(codeText); ok {
		status = streamErrorStatus{
			statusCode:  statusCode,
			isRetryable: model.RetryableStatus(statusCode),
		}
	}

	return &model.APICallError{
		Message:      message,
		StatusCode:   status.statusCode,
		ResponseBody: body,
		IsRetryable:  status.isRetryable,
	}
}

// httpErrorStatus parses code as a three-digit client or server error
// status, such as "429".
func httpErrorStatus(code string) (int, bool) {
	if len(code) != 3 {
		return 0, false
	}

	statusCode, err := strconv.Atoi(code)
	if err != nil {
		return 0, false
	}

	isErrorStatus := statusCode >= 400 && statusCode <= 599

	return statusCode, isErrorStatus
}
