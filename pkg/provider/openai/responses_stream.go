package openai

import (
	"encoding/json"
	"iter"
	"maps"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// responsesOngoingToolCall is a tool call whose input is being streamed,
// keyed by output index.
type responsesOngoingToolCall struct {
	toolName   string
	toolCallID string

	// containerID is the code interpreter container.
	containerID string

	// applyPatch tracks the streamed diff of an apply patch call.
	applyPatch *responsesPatchProgress

	toolSearchExecution string

	// suppressInputStreaming buffers the input of a parallel wrapper call
	// until it is known whether the call expands.
	suppressInputStreaming bool
	bufferedInputDeltas    []string

	async *bool
}

// responsesPatchProgress tracks the streamed input of an apply patch call.
type responsesPatchProgress struct {
	hasDiff    bool
	endEmitted bool
}

// responsesStreamState converts streaming events into stream parts.
type responsesStreamState struct {
	context          responsesCallContext
	failure          responsesStreamFailure
	warnings         []model.Warning
	includeRawChunks bool

	pending []model.StreamPart

	finishReason     model.FinishReason
	usage            *responsesUsage
	logprobs         [][]ResponsesLogprob
	responseID       string
	serviceTier      string
	reasoningContext string

	hasFunctionCall        bool
	encounteredStreamError bool

	ongoingToolCalls        map[int]*responsesOngoingToolCall
	activeReasoning         map[string]*responsesActiveReasoning
	activeMessagePhase      string
	hostedToolSearchCallIDs []string
	approvalToolCallIDs     map[string]string

	// activeOutputItemIDs keeps the first item ID of each output index,
	// because compatible providers can rotate item IDs between events.
	activeOutputItemIDs map[int]string

	ongoingAnnotations        []ResponsesAnnotation
	fallbackAnnotations       []ResponsesAnnotation
	hasWebSearchActionSources bool
	hasFileSearchResults      bool
	retrievedFileIDs          map[string]bool
}

func newResponsesStreamState(call *responsesCall, options model.CallOptions, failure responsesStreamFailure) *responsesStreamState {
	return &responsesStreamState{
		context:          call.context,
		failure:          failure,
		warnings:         call.warnings,
		includeRawChunks: options.IncludeRawChunks,
		finishReason: model.FinishReason{
			Unified: model.FinishOther,
		},
		ongoingToolCalls:    make(map[int]*responsesOngoingToolCall),
		activeReasoning:     make(map[string]*responsesActiveReasoning),
		approvalToolCallIDs: make(map[string]string),
		activeOutputItemIDs: make(map[int]string),
		retrievedFileIDs:    make(map[string]bool),
	}
}

// parts returns the stream parts produced from events.
func (s *responsesStreamState) parts(events *responsesEventReader) iter.Seq[model.StreamPart] {
	return func(yield func(model.StreamPart) bool) {
		s.emit(model.StreamStart{
			Warnings: s.warnings,
		})

		for {
			chunk, ok, err := events.read()
			if err != nil {
				s.fail(err)

				break
			}

			if !ok {
				break
			}

			s.process(chunk)

			if !s.flush(yield) {
				return
			}
		}

		s.finish()
		s.flush(yield)
	}
}

// process converts one chunk.
func (s *responsesStreamState) process(chunk responsesStreamChunk) {
	if s.includeRawChunks {
		s.emit(model.Raw{
			Value: json.RawMessage(chunk.data),
		})
	}

	if chunk.err != nil {
		s.invalidChunk(chunk)

		return
	}

	switch event := chunk.event.(type) {
	case responsesItemEvent:
		if event.Type == responsesEventItemAdded {
			s.itemAdded(event.OutputIndex, event.Item)
		} else {
			s.itemDone(event.OutputIndex, event.Item)
		}

	case responsesArgumentsDeltaEvent:
		s.argumentsDelta(event)

	case responsesPatchDiffEvent:
		s.patchDiff(event)

	case responsesPartialImageEvent:
		s.emit(s.context.imageGenerationResult(event.ItemID, event.PartialImageB64, true))

	case responsesCodeDeltaEvent:
		s.codeDelta(event)

	case responsesCodeDoneEvent:
		s.codeDone(event)

	case responsesStartedEvent:
		s.started(event)

	case responsesTextDeltaEvent:
		s.textDelta(event)

	case responsesSummaryEvent:
		s.summary(event)

	case responsesFinishedEvent:
		s.finished(event)

	case responsesFailedEvent:
		s.failed(event, chunk.data)

	case responsesAnnotationEvent:
		s.annotation(event.Annotation)

	case responsesErrorEvent:
		s.errorEvent(event, chunk.data)
	}
}

// invalidChunk reports an event that failed to decode.
func (s *responsesStreamState) invalidChunk(chunk responsesStreamChunk) {
	err := chunk.err
	if responsesIsChatCompletionChunk(chunk.data) {
		err = s.failure.chatCompletionsMismatch(chunk.data, chunk.err)
	}

	s.encounteredStreamError = true
	s.finishReason = model.FinishReason{
		Unified: model.FinishError,
	}

	s.emit(model.StreamError{
		Err: err,
	})
}

// fail reports a failure to read the stream.
func (s *responsesStreamState) fail(err error) {
	s.encounteredStreamError = true
	s.finishReason = model.FinishReason{
		Unified: model.FinishError,
	}

	s.emit(model.StreamError{
		Err: err,
	})
}

// itemAdded starts streaming an output item.
func (s *responsesStreamState) itemAdded(index int, item responsesOutputItem) {
	c := s.context

	switch item := item.(type) {
	case responsesFunctionCallItem:
		suppress := c.isUndeclaredParallelCall(item.Name)
		s.ongoingToolCalls[index] = &responsesOngoingToolCall{
			toolName:               item.Name,
			toolCallID:             item.CallID,
			suppressInputStreaming: suppress,
			async:                  item.Async,
		}

		if !suppress {
			s.emit(model.ToolInputStart{
				ID:       item.CallID,
				ToolName: item.Name,
			})
		}

	case responsesCustomToolCallItem:
		toolName := c.toolName(item.Name)
		s.ongoingToolCalls[index] = &responsesOngoingToolCall{
			toolName:   toolName,
			toolCallID: item.CallID,
			async:      item.Async,
		}

		s.emit(model.ToolInputStart{
			ID:            item.CallID,
			ToolName:      toolName,
			InputEncoding: model.ToolInputText,
		})

	case responsesWebSearchCallItem:
		call := c.webSearchCall(item.ID)
		s.ongoingToolCalls[index] = &responsesOngoingToolCall{
			toolName:   call.ToolName,
			toolCallID: item.ID,
		}

		s.emit(model.ToolInputStart{
			ID:               item.ID,
			ToolName:         call.ToolName,
			ProviderExecuted: true,
		})
		s.emit(model.ToolInputEnd{
			ID: item.ID,
		})
		s.emit(call)

	case responsesComputerCallItem:
		toolCallID := item.ID
		if item.CallID != nil {
			toolCallID = *item.CallID
		}

		s.ongoingToolCalls[index] = &responsesOngoingToolCall{
			toolName:   c.toolName("computer"),
			toolCallID: toolCallID,
		}

		s.emit(model.ToolInputStart{
			ID:       toolCallID,
			ToolName: c.toolName("computer"),
		})

	case responsesCodeInterpreterCallItem:
		s.codeInterpreterAdded(index, item)

	case responsesFileSearchCallItem:
		s.emit(c.fileSearchCall(item.ID))

	case responsesImageGenerationCallItem:
		s.emit(c.imageGenerationCall(item.ID))

	case responsesToolSearchCallItem:
		s.toolSearchAdded(index, item)

	case responsesApplyPatchCallItem:
		s.applyPatchAdded(index, item)

	case responsesShellCallItem:
		s.ongoingToolCalls[index] = &responsesOngoingToolCall{
			toolName:   c.toolName("shell"),
			toolCallID: item.CallID,
		}

	case responsesMessageItem:
		s.activeOutputItemIDs[index] = item.ID
		s.ongoingAnnotations = nil
		s.activeMessagePhase = item.Phase

		s.emit(model.TextStart{
			ID:               item.ID,
			ProviderMetadata: c.textMetadata(item.ID, item.Phase, nil),
		})

	case responsesReasoningOutput:
		s.activeOutputItemIDs[index] = item.ID
		s.activeReasoning[item.ID] = &responsesActiveReasoning{
			encryptedContent: item.EncryptedContent,
			summaryParts: map[int]responsesSummaryStatus{
				0: responsesSummaryActive,
			},
		}

		s.emit(model.ReasoningStart{
			ID:               responsesSummaryID(item.ID, 0),
			ProviderMetadata: c.reasoningMetadata(item.ID, item.EncryptedContent),
		})
	}

	// MCP items are handled when they are done, so that calls can reuse the
	// tool call IDs of their approval requests. Tool search outputs and shell
	// outputs are also handled then.
}

// itemDone completes an output item.
func (s *responsesStreamState) itemDone(index int, item responsesOutputItem) {
	c := s.context

	switch item := item.(type) {
	case responsesMessageItem:
		s.messageDone(index, item)

	case responsesFunctionCallItem:
		s.functionCallDone(index, item)

	case responsesProgramItem:
		s.emit(c.programCall(item))

	case responsesProgramOutputItem:
		s.emit(c.programResult(item))

	case responsesCustomToolCallItem:
		ongoing := s.takeToolCall(index)
		s.hasFunctionCall = true

		s.emit(model.ToolInputEnd{
			ID: item.CallID,
		})
		s.emit(c.customToolCall(item, responsesAsync(item.Async, ongoing)))

	case responsesWebSearchCallItem:
		s.takeToolCall(index)
		s.emit(c.webSearchResult(item))

		for _, source := range c.webSearchSources(item) {
			s.hasWebSearchActionSources = true
			s.emit(source)
		}

	case responsesComputerCallItem:
		s.computerCallDone(index, item)

	case responsesFileSearchCallItem:
		s.takeToolCall(index)
		s.emit(c.fileSearchResult(item))

		if len(item.Results) > 0 {
			s.hasFileSearchResults = true
		}

		for _, source := range c.fileSearchSources(item, s.retrievedFileIDs) {
			s.emit(source)
		}

	case responsesCodeInterpreterCallItem:
		s.takeToolCall(index)
		s.emit(c.codeInterpreterResult(item))

	case responsesImageGenerationCallItem:
		s.emit(c.imageGenerationResult(item.ID, item.Result, false))

	case responsesToolSearchCallItem:
		s.toolSearchDone(index, item)

	case responsesToolSearchOutputItem:
		toolCallID := responsesToolSearchOutputCallID(item, &s.hostedToolSearchCallIDs)
		s.emit(c.toolSearchResult(item, toolCallID))

	case responsesMCPCallItem:
		s.takeToolCall(index)

		toolCallID := c.mcpToolCallID(item, s.approvalToolCallIDs)
		s.emit(c.mcpCall(toolCallID, item.Name, item.Arguments))
		s.emit(c.mcpResult(item, toolCallID))

	case responsesMCPListToolsItem:
		s.takeToolCall(index)

	case responsesApplyPatchCallItem:
		s.applyPatchDone(index, item)

	case responsesMCPApprovalRequestItem:
		s.takeToolCall(index)

		call, request := c.mcpApproval(item)
		s.approvalToolCallIDs[request.ApprovalID] = call.ToolCallID

		s.emit(call)
		s.emit(request)

	case responsesLocalShellCallItem:
		s.takeToolCall(index)
		s.emit(c.localShellCall(item))

	case responsesShellCallItem:
		s.takeToolCall(index)
		s.emit(c.shellCall(item))

	case responsesShellCallOutputItem:
		s.emit(c.shellResult(item))

	case responsesReasoningOutput:
		s.reasoningDone(index, item)

	case responsesCompactionItem:
		s.emit(c.compaction(item))
	}
}

// messageDone ends a text block with the annotations collected for it.
func (s *responsesStreamState) messageDone(index int, item responsesMessageItem) {
	c := s.context
	itemID := s.outputItemID(item.ID, &index)

	phase := item.Phase
	if phase == "" {
		phase = s.activeMessagePhase
	}

	s.activeMessagePhase = ""

	s.emit(model.TextEnd{
		ID:               itemID,
		Citations:        c.citations(s.ongoingAnnotations),
		ProviderMetadata: c.textMetadata(itemID, phase, s.ongoingAnnotations),
	})

	delete(s.activeOutputItemIDs, index)
}

// started reports response metadata when the response is created.
func (s *responsesStreamState) started(event responsesStartedEvent) {
	if event.Type != responsesEventCreated {
		return
	}

	s.responseID = event.Response.ID

	s.emit(model.ResponseMetadata{
		ID:        event.Response.ID,
		Timestamp: responsesTimestamp(&event.Response.CreatedAt),
		ModelID:   event.Response.Model,
	})
}

// textDelta streams message text.
func (s *responsesStreamState) textDelta(event responsesTextDeltaEvent) {
	s.emit(model.TextDelta{
		ID:    s.outputItemID(event.ItemID, event.OutputIndex),
		Delta: event.Delta,
	})

	if s.context.collectLogprobs && event.Logprobs != nil {
		s.logprobs = append(s.logprobs, event.Logprobs)
	}
}

// finished records the final state of a completed or incomplete response.
func (s *responsesStreamState) finished(event responsesFinishedEvent) {
	response := event.Response
	reason := ""
	if response.IncompleteDetails != nil {
		reason = response.IncompleteDetails.Reason
	}

	if !s.encounteredStreamError {
		s.finishReason = model.FinishReason{
			Unified: responsesFinishReason(reason, s.hasFunctionCall),
			Raw:     reason,
		}
	}

	s.usage = response.Usage

	if response.ServiceTier != "" {
		s.serviceTier = response.ServiceTier
	}

	if response.Reasoning != nil && response.Reasoning.Context != "" {
		s.reasoningContext = response.Reasoning.Context
	}
}

// failed records a failed response and reports its error once.
func (s *responsesStreamState) failed(event responsesFailedEvent, frame []byte) {
	response := event.Response

	reason := ""
	if response.IncompleteDetails != nil {
		reason = response.IncompleteDetails.Reason
	}

	s.finishReason = model.FinishReason{
		Unified: model.FinishError,
		Raw:     "error",
	}

	if reason != "" {
		s.finishReason = model.FinishReason{
			Unified: responsesFinishReason(reason, s.hasFunctionCall),
			Raw:     reason,
		}
	}

	s.usage = response.Usage

	if response.Reasoning != nil && response.Reasoning.Context != "" {
		s.reasoningContext = response.Reasoning.Context
	}

	if s.encounteredStreamError || response.Error == nil {
		return
	}

	s.encounteredStreamError = true

	details, _ := responsesErrorDetailsOf(event)
	s.emit(model.StreamError{
		Err: s.failure.apiError(details, frame),
	})
}

// annotation collects an annotation of the current message. Citation
// sources wait until the end of the stream, because the search items that
// may already provide them can arrive later.
func (s *responsesStreamState) annotation(annotation ResponsesAnnotation) {
	s.ongoingAnnotations = append(s.ongoingAnnotations, annotation)

	isCitation := annotation.Type == ResponsesURLCitation || annotation.Type == ResponsesFileCitation
	if isCitation {
		s.fallbackAnnotations = append(s.fallbackAnnotations, annotation)

		return
	}

	s.emit(s.context.annotationSource(annotation, s.context.generateID()))
}

// errorEvent reports an error event.
func (s *responsesStreamState) errorEvent(event responsesErrorEvent, frame []byte) {
	s.encounteredStreamError = true
	s.finishReason = model.FinishReason{
		Unified: model.FinishError,
		Raw:     "error",
	}

	details, _ := responsesErrorDetailsOf(event)
	s.emit(model.StreamError{
		Err: s.failure.apiError(details, frame),
	})
}

// finish emits the remaining sources and suppressed inputs, then the
// finish part.
func (s *responsesStreamState) finish() {
	c := s.context

	for _, annotation := range s.fallbackAnnotations {
		providedByRetrieval := (annotation.Type == ResponsesURLCitation && s.hasWebSearchActionSources) ||
			(annotation.Type == ResponsesFileCitation && s.hasFileSearchResults)
		if providedByRetrieval {
			continue
		}

		s.emit(c.annotationSource(annotation, c.generateID()))
	}

	for _, index := range slices.Sorted(maps.Keys(s.ongoingToolCalls)) {
		ongoing := s.ongoingToolCalls[index]
		if !ongoing.suppressInputStreaming {
			continue
		}

		s.emit(model.ToolInputStart{
			ID:       ongoing.toolCallID,
			ToolName: ongoing.toolName,
		})

		for _, delta := range ongoing.bufferedInputDeltas {
			s.emit(model.ToolInputDelta{
				ID:    ongoing.toolCallID,
				Delta: delta,
			})
		}
	}

	s.emit(model.Finish{
		Usage:        responsesConvertUsage(s.usage),
		FinishReason: s.finishReason,
		ProviderMetadata: c.metadata(ResponsesMetadata{
			ResponseID:       s.responseID,
			Logprobs:         s.logprobs,
			ServiceTier:      s.serviceTier,
			ReasoningContext: s.reasoningContext,
		}),
	})
}

// outputItemID returns the item ID first seen at an output index, or
// itemID when there is none.
func (s *responsesStreamState) outputItemID(itemID string, outputIndex *int) string {
	if outputIndex == nil {
		return itemID
	}

	if active, ok := s.activeOutputItemIDs[*outputIndex]; ok {
		return active
	}

	return itemID
}

// takeToolCall removes and returns the ongoing tool call at an output
// index.
func (s *responsesStreamState) takeToolCall(index int) *responsesOngoingToolCall {
	ongoing := s.ongoingToolCalls[index]
	delete(s.ongoingToolCalls, index)

	return ongoing
}

func (s *responsesStreamState) emit(part model.StreamPart) {
	s.pending = append(s.pending, part)
}

// flush yields the pending parts. It reports false when the consumer
// stopped.
func (s *responsesStreamState) flush(yield func(model.StreamPart) bool) bool {
	pending := s.pending
	s.pending = nil

	for _, part := range pending {
		if !yield(part) {
			return false
		}
	}

	return true
}
