package google

import (
	"slices"

	"github.com/aarock1234/ai/internal/optional"
	"github.com/aarock1234/ai/internal/wirejson"

	"github.com/aarock1234/ai/pkg/model"
)

// processFunctionCall handles one function call part of a stream chunk. A
// call either arrives complete, opens a call whose arguments stream as
// partial arguments, continues the open call, or ends it with an empty part.
func (s *streamState) processFunctionCall(call functionCall, metadata model.ProviderMetadata) {
	switch {
	case isStreamingCallChunk(call):
		s.processStreamingCallChunk(call, metadata)

	case isTerminalCallChunk(call):
		if len(s.streamingCalls) > 0 {
			s.finishStreamingCall()
		}

	case isCompleteCall(call):
		s.emitCompleteCall(call, functionCallArgumentsText(call.Args), metadata)

	case isNoArgumentsCall(call):
		s.emitCompleteCall(call, "{}", metadata)
	}
}

func isCompleteCall(call functionCall) bool {
	return call.Name != "" && wirejson.HasValue(call.Args) && call.PartialArgs == nil
}

func isStreamingCallChunk(call functionCall) bool {
	if call.PartialArgs != nil {
		return true
	}

	return call.Name != "" && optional.Or(call.WillContinue, false)
}

func isTerminalCallChunk(call functionCall) bool {
	return call.Name == "" && !wirejson.HasValue(call.Args) && call.PartialArgs == nil && call.WillContinue == nil
}

// isNoArgumentsCall reports a single-chunk call without arguments.
func isNoArgumentsCall(call functionCall) bool {
	willContinue := optional.Or(call.WillContinue, false)

	return call.Name != "" && !wirejson.HasValue(call.Args) && call.PartialArgs == nil && !willContinue
}

func (s *streamState) processStreamingCallChunk(call functionCall, metadata model.ProviderMetadata) {
	if call.Name != "" {
		s.openStreamingCall(call, metadata)
	}

	if call.PartialArgs == nil || len(s.streamingCalls) == 0 {
		return
	}

	active := s.streamingCalls[len(s.streamingCalls)-1]

	delta := active.accumulator.process(call.PartialArgs)
	if delta != "" {
		s.emit(model.ToolInputDelta{
			ID:               active.toolCallID,
			Delta:            delta,
			ProviderMetadata: metadata,
		})
	}

	if callContinues(call) {
		return
	}

	s.finishStreamingCall()
}

func (s *streamState) openStreamingCall(call functionCall, metadata model.ProviderMetadata) {
	callID := call.ID
	if callID == "" {
		callID = s.generateID()
	}

	s.streamingCalls = append(s.streamingCalls, streamingCall{
		toolCallID:  callID,
		toolName:    call.Name,
		accumulator: newArgumentAccumulator(),
		metadata:    metadata,
	})

	s.emit(model.ToolInputStart{
		ID:               callID,
		ToolName:         call.Name,
		ProviderMetadata: metadata,
	})
}

// callContinues reports whether the call or any of its partial arguments
// continues in a later chunk.
func callContinues(call functionCall) bool {
	if optional.Or(call.WillContinue, false) {
		return true
	}

	return slices.ContainsFunc(call.PartialArgs, partialArgument.continues)
}

// finishStreamingCall closes the innermost streaming call and emits it.
func (s *streamState) finishStreamingCall() {
	active := s.streamingCalls[len(s.streamingCalls)-1]
	s.streamingCalls = s.streamingCalls[:len(s.streamingCalls)-1]

	input, closingDelta := active.accumulator.finalize()

	if closingDelta != "" {
		s.emit(model.ToolInputDelta{
			ID:               active.toolCallID,
			Delta:            closingDelta,
			ProviderMetadata: active.metadata,
		})
	}

	s.emit(model.ToolInputEnd{
		ID:               active.toolCallID,
		ProviderMetadata: active.metadata,
	})

	s.emit(model.ToolCall{
		ToolCallID:       active.toolCallID,
		ToolName:         active.toolName,
		Input:            input,
		ProviderMetadata: active.metadata,
	})

	s.hasToolCalls = true
}

// emitCompleteCall emits a call that arrived in one chunk as a full input
// sequence. Calls without arguments have no input delta.
func (s *streamState) emitCompleteCall(call functionCall, input string, metadata model.ProviderMetadata) {
	callID := call.ID
	if callID == "" {
		callID = s.generateID()
	}

	s.emit(model.ToolInputStart{
		ID:               callID,
		ToolName:         call.Name,
		ProviderMetadata: metadata,
	})

	if wirejson.HasValue(call.Args) {
		s.emit(model.ToolInputDelta{
			ID:               callID,
			Delta:            input,
			ProviderMetadata: metadata,
		})
	}

	s.emit(model.ToolInputEnd{
		ID:               callID,
		ProviderMetadata: metadata,
	})

	s.emit(model.ToolCall{
		ToolCallID:       callID,
		ToolName:         call.Name,
		Input:            input,
		ProviderMetadata: metadata,
	})

	s.hasToolCalls = true
}
