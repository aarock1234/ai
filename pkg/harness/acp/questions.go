package acp

import (
	"encoding/json"
	"fmt"
	"maps"
	"slices"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/bridge"
	"github.com/aarock1234/ai/pkg/model"
)

// Question request classifications.
const (
	questionHandled   = "handled"
	questionUnhandled = "unhandled"
)

// bufferedQuestionResult is an answer submitted before the implementation
// re-sent its question request, as happens after a recovery.
type bufferedQuestionResult struct {
	toolCallID string
	isError    bool
	toolResult model.ToolResultPart
}

// handleQuestionRequest turns a native question request into an
// askUserQuestions tool call, or answers it at once with a buffered answer.
func (t *turn) handleQuestionRequest(data json.RawMessage) bool {
	request, err := decodeQuestionRequest(data)
	if err != nil {
		return t.fail(err)
	}

	questions := t.session.config.harness.settings.AskUserQuestions
	if questions == nil {
		return t.classifyQuestion(request.RequestID, questionClassification{
			Type: questionUnhandled,
		})
	}

	nativeToolCall, err := questions.FromNativeRequest(NativeQuestionRequest{
		NativeRequest:  request.NativeRequest,
		NativeToolCall: request.NativeToolCall,
	})
	if err != nil {
		return t.fail(err)
	}

	if nativeToolCall == nil {
		return t.classifyQuestion(request.RequestID, questionClassification{
			Type: questionUnhandled,
		})
	}

	return t.acceptQuestion(request, *nativeToolCall)
}

// acceptQuestion registers a handled question request and emits its tool
// call, or submits a buffered answer.
func (t *turn) acceptQuestion(request questionRequest, nativeToolCall harness.ToolCallPart) bool {
	harnessID := t.session.harnessID()

	if nativeToolCall.ToolName != string(harness.ToolAskUserQuestions) || nativeToolCall.ProviderExecuted {
		return t.fail(fmt.Errorf("%s acp askUserQuestions FromNativeRequest must return a client-executed askUserQuestions tool call", harnessID))
	}

	toolCall, err := withNativeQuestionRequest(harnessID, request.NativeRequest, nativeToolCall)
	if err != nil {
		return t.fail(err)
	}

	buffered, hasBuffered := t.session.takeBufferedQuestionResult(toolCall.ToolCallID, request.NativeRequest)

	t.mu.Lock()
	t.activeQuestions[toolCall.ToolCallID] = activeQuestion{
		requestID:     request.RequestID,
		nativeRequest: request.NativeRequest,
	}
	t.questionToolCalls[request.RequestID] = toolCall.ToolCallID
	t.mu.Unlock()

	ended := t.classifyQuestion(request.RequestID, questionClassification{
		Type:       questionHandled,
		ToolCallID: toolCall.ToolCallID,
	})
	if ended {
		return true
	}

	if !hasBuffered {
		t.forward(toolCall)

		return false
	}

	err = t.sendQuestionAnswer(questionAnswer{
		toolCallID:    toolCall.ToolCallID,
		nativeRequest: request.NativeRequest,
		isError:       buffered.isError,
		toolResult:    buffered.toolResult,
	})
	if err != nil {
		return t.fail(err)
	}

	return false
}

// classifyQuestion tells the bridge whether the host handles a question
// request. A failed send ends the turn.
func (t *turn) classifyQuestion(requestID string, classification questionClassification) bool {
	command, err := classificationCommand(requestID, classification)
	if err != nil {
		return t.fail(err)
	}

	if err := t.session.channel.Send(command); err != nil {
		return t.fail(err)
	}

	return false
}

// isQuestionAnswer reports a tool result that answers a native question
// request.
func (t *turn) isQuestionAnswer(submission harness.ToolResultSubmission) bool {
	return t.session.config.harness.settings.AskUserQuestions != nil &&
		submission.ToolResult != nil &&
		submission.ToolResult.ToolName == string(harness.ToolAskUserQuestions)
}

// submitQuestionAnswer answers the active question request of the tool
// call. Without one, it answers the request recorded in the result, or
// buffers the answer until the implementation re-sends its request.
func (t *turn) submitQuestionAnswer(submission harness.ToolResultSubmission) error {
	t.mu.Lock()
	active, ok := t.activeQuestions[submission.ToolCallID]
	t.mu.Unlock()

	answer := questionAnswer{
		toolCallID: submission.ToolCallID,
		isError:    submission.IsError,
		toolResult: *submission.ToolResult,
	}

	if ok {
		answer.nativeRequest = active.nativeRequest

		return t.sendQuestionAnswer(answer)
	}

	previous, hasPrevious := nativeRequestOf(*submission.ToolResult, t.session.harnessID())
	if hasPrevious && !t.session.config.launch.lossyRerun {
		answer.nativeRequest = previous

		return t.sendQuestionAnswer(answer)
	}

	t.session.bufferQuestionResult(bufferedQuestionResult{
		toolCallID: submission.ToolCallID,
		isError:    submission.IsError,
		toolResult: *submission.ToolResult,
	})

	return nil
}

// questionAnswer is the answer to one native question request.
type questionAnswer struct {
	toolCallID    string
	nativeRequest json.RawMessage
	isError       bool
	toolResult    model.ToolResultPart
}

// sendQuestionAnswer converts an answer into the native response and sends
// it.
func (t *turn) sendQuestionAnswer(answer questionAnswer) error {
	questions := t.session.config.harness.settings.AskUserQuestions

	output, err := questions.ToNativeResponse(NativeQuestionResponse{
		NativeRequest: answer.nativeRequest,
		ToolResult:    answer.toolResult,
	})
	if err != nil {
		return fmt.Errorf("acp askUserQuestions native response: %w", err)
	}

	toolResult := answer.toolResult

	command, err := bridge.NewToolResultCommand(harness.ToolResultSubmission{
		ToolCallID: answer.toolCallID,
		Output:     output,
		IsError:    answer.isError,
		ToolResult: &toolResult,
	})
	if err != nil {
		return err
	}

	return t.session.channel.Send(command)
}

// bufferQuestionResult keeps an answer for a re-sent question request.
func (s *session) bufferQuestionResult(result bufferedQuestionResult) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.bufferedQuestionResults = slices.DeleteFunc(s.bufferedQuestionResults, func(buffered bufferedQuestionResult) bool {
		return buffered.toolCallID == result.toolCallID
	})
	s.bufferedQuestionResults = append(s.bufferedQuestionResults, result)
}

// takeBufferedQuestionResult removes and returns the buffered answer of a
// tool call, or the first one whose recorded request matches the re-sent
// request.
func (s *session) takeBufferedQuestionResult(toolCallID string, nativeRequest json.RawMessage) (bufferedQuestionResult, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	index := slices.IndexFunc(s.bufferedQuestionResults, func(buffered bufferedQuestionResult) bool {
		return buffered.toolCallID == toolCallID
	})

	if index < 0 {
		index = s.matchingQuestionResultLocked(nativeRequest)
	}

	if index < 0 {
		return bufferedQuestionResult{}, false
	}

	buffered := s.bufferedQuestionResults[index]
	s.bufferedQuestionResults = slices.Delete(s.bufferedQuestionResults, index, index+1)

	return buffered, true
}

// matchingQuestionResultLocked returns the index of the first buffered
// answer whose recorded request matches, or -1. The caller holds mu.
func (s *session) matchingQuestionResultLocked(nativeRequest json.RawMessage) int {
	questions := s.config.harness.settings.AskUserQuestions
	if questions == nil || questions.MatchesNativeRequest == nil {
		return -1
	}

	return slices.IndexFunc(s.bufferedQuestionResults, func(buffered bufferedQuestionResult) bool {
		previous, ok := nativeRequestOf(buffered.toolResult, s.harnessID())

		return ok && questions.MatchesNativeRequest(NativeQuestionMatch{
			PreviousNativeRequest: previous,
			NativeRequest:         nativeRequest,
		})
	})
}

// nativeQuestionMetadata is the adapter's metadata of a question tool call.
type nativeQuestionMetadata struct {
	NativeRequest json.RawMessage `json:"nativeRequest"`
}

// nativeRequestOf reads the native request recorded in a tool result's
// provider options.
func nativeRequestOf(toolResult model.ToolResultPart, harnessID string) (json.RawMessage, bool) {
	metadata, ok, err := model.ProviderValue[nativeQuestionMetadata](toolResult.ProviderOptions, harnessID)
	if err != nil || !ok || len(metadata.NativeRequest) == 0 {
		return nil, false
	}

	return metadata.NativeRequest, true
}

// withNativeQuestionRequest records the native request in the tool call's
// provider metadata, so that its answer can be delivered after a recovery.
func withNativeQuestionRequest(harnessID string, nativeRequest json.RawMessage, toolCall harness.ToolCallPart) (harness.ToolCallPart, error) {
	existing, _, err := model.ProviderValue[map[string]any](toolCall.ProviderMetadata, harnessID)
	if err != nil {
		return harness.ToolCallPart{}, fmt.Errorf("read %s question tool call metadata: %w", harnessID, err)
	}

	harnessMetadata := maps.Clone(existing)
	if harnessMetadata == nil {
		harnessMetadata = make(map[string]any)
	}

	harnessMetadata["nativeRequest"] = nativeRequest

	providerMetadata := maps.Clone(toolCall.ProviderMetadata)
	if providerMetadata == nil {
		providerMetadata = make(model.ProviderMetadata)
	}

	providerMetadata[harnessID] = harnessMetadata
	toolCall.ProviderMetadata = providerMetadata

	return toolCall, nil
}
