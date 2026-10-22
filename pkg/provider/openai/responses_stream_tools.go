package openai

import (
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// codeInterpreterAdded starts the streamed JSON input of a code interpreter
// call. The code deltas fill the open "code" string.
func (s *responsesStreamState) codeInterpreterAdded(index int, item responsesCodeInterpreterCallItem) {
	toolName := s.context.toolName("code_interpreter")

	s.ongoingToolCalls[index] = &responsesOngoingToolCall{
		toolName:    toolName,
		toolCallID:  item.ID,
		containerID: item.ContainerID,
	}

	s.emit(model.ToolInputStart{
		ID:               item.ID,
		ToolName:         toolName,
		ProviderExecuted: true,
	})
	s.emit(model.ToolInputDelta{
		ID:    item.ID,
		Delta: fmt.Sprintf(`{"containerId":"%s","code":"`, responsesEscapeJSONString(item.ContainerID)),
	})
}

// toolSearchAdded starts a tool search. Hosted searches start their input
// now; client searches start it when the call is done.
func (s *responsesStreamState) toolSearchAdded(index int, item responsesToolSearchCallItem) {
	toolName := s.context.toolName("tool_search")

	execution := item.Execution
	if execution == "" {
		execution = "server"
	}

	s.ongoingToolCalls[index] = &responsesOngoingToolCall{
		toolName:            toolName,
		toolCallID:          item.ID,
		toolSearchExecution: execution,
	}

	if item.Execution == "server" {
		s.emit(model.ToolInputStart{
			ID:               item.ID,
			ToolName:         toolName,
			ProviderExecuted: true,
		})
	}
}

// applyPatchAdded starts the streamed JSON input of an apply patch call.
// Delete operations have no diff and are complete at once; other
// operations leave the "diff" string open for the diff deltas.
func (s *responsesStreamState) applyPatchAdded(index int, item responsesApplyPatchCallItem) {
	toolName := s.context.toolName("apply_patch")
	isDelete := item.Operation.Type == ApplyPatchDeleteFile

	s.ongoingToolCalls[index] = &responsesOngoingToolCall{
		toolName:   toolName,
		toolCallID: item.CallID,
		applyPatch: &responsesPatchProgress{
			hasDiff:    isDelete,
			endEmitted: isDelete,
		},
	}

	s.emit(model.ToolInputStart{
		ID:       item.CallID,
		ToolName: toolName,
	})

	if isDelete {
		input := responsesEncodeJSON(ApplyPatchInput{
			CallID:    item.CallID,
			Operation: item.Operation,
		})

		s.emit(model.ToolInputDelta{
			ID:    item.CallID,
			Delta: string(input),
		})
		s.emit(model.ToolInputEnd{
			ID: item.CallID,
		})

		return
	}

	s.emit(model.ToolInputDelta{
		ID: item.CallID,
		Delta: fmt.Sprintf(
			`{"callId":"%s","operation":{"type":"%s","path":"%s","diff":"`,
			responsesEscapeJSONString(item.CallID),
			responsesEscapeJSONString(string(item.Operation.Type)),
			responsesEscapeJSONString(item.Operation.Path),
		),
	})
}

// functionCallDone completes a function call. A parallel wrapper call is
// expanded into its wrapped calls when possible.
func (s *responsesStreamState) functionCallDone(index int, item responsesFunctionCallItem) {
	ongoing := s.takeToolCall(index)
	s.hasFunctionCall = true

	suppressed := s.context.isUndeclaredParallelCall(item.Name)
	if ongoing != nil {
		suppressed = ongoing.suppressInputStreaming
	}

	if !suppressed {
		s.emitFunctionCall(item, ongoing, false)

		return
	}

	calls, ok := s.context.expandParallelCall(item)
	if !ok {
		s.emitFunctionCall(item, ongoing, true)

		return
	}

	for _, call := range calls {
		s.emit(model.ToolInputStart{
			ID:       call.ToolCallID,
			ToolName: call.ToolName,
		})
		s.emit(model.ToolInputDelta{
			ID:    call.ToolCallID,
			Delta: call.Input,
		})
		s.emit(model.ToolInputEnd{
			ID: call.ToolCallID,
		})
		s.emit(call)
	}
}

// emitFunctionCall ends a function call's input and emits the call. A
// suppressed call first replays its buffered input.
func (s *responsesStreamState) emitFunctionCall(item responsesFunctionCallItem, ongoing *responsesOngoingToolCall, suppressed bool) {
	c := s.context

	if suppressed {
		s.emit(model.ToolInputStart{
			ID:       item.CallID,
			ToolName: item.Name,
		})

		s.replayInput(item, ongoing)
	}

	inputEnd := model.ToolInputEnd{
		ID: item.CallID,
	}

	if item.Namespace != "" {
		inputEnd.ProviderMetadata = c.metadata(ResponsesToolCallMetadata{
			Namespace: item.Namespace,
		})
	}

	s.emit(inputEnd)
	s.emit(c.functionCall(item, responsesAsync(item.Async, ongoing)))
}

// replayInput emits the buffered input deltas, or the whole arguments when
// none were buffered.
func (s *responsesStreamState) replayInput(item responsesFunctionCallItem, ongoing *responsesOngoingToolCall) {
	if ongoing != nil && len(ongoing.bufferedInputDeltas) > 0 {
		for _, delta := range ongoing.bufferedInputDeltas {
			s.emit(model.ToolInputDelta{
				ID:    item.CallID,
				Delta: delta,
			})
		}

		return
	}

	if item.Arguments != "" {
		s.emit(model.ToolInputDelta{
			ID:    item.CallID,
			Delta: item.Arguments,
		})
	}
}

// computerCallDone completes a computer call for the application, or a
// legacy computer use call executed by OpenAI.
func (s *responsesStreamState) computerCallDone(index int, item responsesComputerCallItem) {
	c := s.context
	s.takeToolCall(index)

	if item.CallID == nil {
		s.emit(model.ToolInputEnd{
			ID: item.ID,
		})
		s.emit(c.computerUseCall(item.ID))
		s.emit(c.computerUseResult(item))

		return
	}

	s.hasFunctionCall = true
	call := c.computerCall(item, *item.CallID)

	s.emit(model.ToolInputDelta{
		ID:    call.ToolCallID,
		Delta: call.Input,
	})
	s.emit(model.ToolInputEnd{
		ID: call.ToolCallID,
	})
	s.emit(call)
}

// toolSearchDone completes a tool search call. Hosted calls keep the item
// ID and are remembered for pairing with their outputs; client calls use
// their call ID.
func (s *responsesStreamState) toolSearchDone(index int, item responsesToolSearchCallItem) {
	ongoing := s.takeToolCall(index)
	if ongoing == nil {
		return
	}

	isHosted := item.Execution == "server"

	toolCallID := ongoing.toolCallID
	var inputCallID *string

	if !isHosted {
		toolCallID = item.ID
		if item.CallID != nil {
			toolCallID = *item.CallID
		}

		inputCallID = &toolCallID

		s.emit(model.ToolInputStart{
			ID:       toolCallID,
			ToolName: ongoing.toolName,
		})
	} else {
		s.hostedToolSearchCallIDs = append(s.hostedToolSearchCallIDs, toolCallID)
	}

	s.emit(model.ToolInputEnd{
		ID: toolCallID,
	})
	s.emit(s.context.toolSearchCall(item, toolCallID, inputCallID))
}

// applyPatchDone closes the streamed input when the diff events did not,
// and emits the call once the patch is complete.
func (s *responsesStreamState) applyPatchDone(index int, item responsesApplyPatchCallItem) {
	ongoing := s.takeToolCall(index)
	if ongoing == nil {
		return
	}

	progress := ongoing.applyPatch
	needsInputEnd := progress != nil && !progress.endEmitted && item.Operation.Type != ApplyPatchDeleteFile

	if needsInputEnd {
		if !progress.hasDiff {
			s.emit(model.ToolInputDelta{
				ID:    ongoing.toolCallID,
				Delta: responsesEscapeJSONString(item.Operation.Diff),
			})
		}

		s.closePatchInput(ongoing)
	}

	if item.Status != "completed" {
		return
	}

	s.hasFunctionCall = true
	s.emit(s.context.applyPatchCall(item, ongoing.toolCallID))
}

// closePatchInput closes the diff string and the input object.
func (s *responsesStreamState) closePatchInput(ongoing *responsesOngoingToolCall) {
	s.emit(model.ToolInputDelta{
		ID:    ongoing.toolCallID,
		Delta: `"}}`,
	})
	s.emit(model.ToolInputEnd{
		ID: ongoing.toolCallID,
	})

	ongoing.applyPatch.endEmitted = true
}

// argumentsDelta streams function and custom tool input.
func (s *responsesStreamState) argumentsDelta(event responsesArgumentsDeltaEvent) {
	ongoing := s.ongoingToolCalls[event.OutputIndex]
	if ongoing == nil {
		return
	}

	if event.Type == responsesEventFunctionArgumentsDelta && ongoing.suppressInputStreaming {
		ongoing.bufferedInputDeltas = append(ongoing.bufferedInputDeltas, event.Delta)

		return
	}

	s.emit(model.ToolInputDelta{
		ID:    ongoing.toolCallID,
		Delta: event.Delta,
	})
}

// patchDiff streams the diff of an apply patch call into its open JSON
// string, closing the input when the diff is done.
func (s *responsesStreamState) patchDiff(event responsesPatchDiffEvent) {
	ongoing := s.ongoingToolCalls[event.OutputIndex]
	if ongoing == nil || ongoing.applyPatch == nil {
		return
	}

	progress := ongoing.applyPatch

	if event.Type == responsesEventPatchDiffDelta {
		s.emit(model.ToolInputDelta{
			ID:    ongoing.toolCallID,
			Delta: responsesEscapeJSONString(event.Delta),
		})

		progress.hasDiff = true

		return
	}

	if progress.endEmitted {
		return
	}

	if !progress.hasDiff {
		s.emit(model.ToolInputDelta{
			ID:    ongoing.toolCallID,
			Delta: responsesEscapeJSONString(event.Diff),
		})

		progress.hasDiff = true
	}

	s.closePatchInput(ongoing)
}

// codeDelta streams code into the open "code" string.
func (s *responsesStreamState) codeDelta(event responsesCodeDeltaEvent) {
	ongoing := s.ongoingToolCalls[event.OutputIndex]
	if ongoing == nil {
		return
	}

	s.emit(model.ToolInputDelta{
		ID:    ongoing.toolCallID,
		Delta: responsesEscapeJSONString(event.Delta),
	})
}

// codeDone closes the code interpreter input and emits the call right away.
func (s *responsesStreamState) codeDone(event responsesCodeDoneEvent) {
	ongoing := s.ongoingToolCalls[event.OutputIndex]
	if ongoing == nil {
		return
	}

	s.emit(model.ToolInputDelta{
		ID:    ongoing.toolCallID,
		Delta: `"}`,
	})
	s.emit(model.ToolInputEnd{
		ID: ongoing.toolCallID,
	})
	s.emit(s.context.codeInterpreterCall(ongoing.toolCallID, new(event.Code), ongoing.containerID))
}

// responsesAsync returns the async flag of a call item, falling back to
// the flag seen when the call started.
func responsesAsync(async *bool, ongoing *responsesOngoingToolCall) *bool {
	if async == nil && ongoing != nil {
		return ongoing.async
	}

	return async
}
