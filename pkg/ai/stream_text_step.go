package ai

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

// streamStepOutcome is the state of a streamed step that the loop needs.
type streamStepOutcome struct {
	prepared        preparedStep
	finishReason    model.UnifiedFinishReason
	rawFinishReason string
	progress        loopProgress
}

// streamStep performs one streaming model call: it maps the parts of the
// model call into stream parts, resolves tool approvals, executes the tool
// calls once the model call ends and closes the step with a FinishStepPart.
// It reports false when the stream must end, because the consumer stopped
// or the step failed.
func (r *streamTextRun) streamStep(ctx context.Context, input stepInput, yield func(StreamPart) bool) (streamStepOutcome, bool) {
	timers := newStreamTimers(r.timeout, r.abort)
	timers.startStep()
	defer timers.stop()

	prepared, err := r.prepareStepCall(ctx, input)
	if err != nil {
		r.fail(ctx, err, yield)

		return streamStepOutcome{}, false
	}

	r.recorder.stepModel = StepModel{
		Provider: prepared.model.Provider(),
		ModelID:  prepared.model.ModelID(),
	}

	r.notifyStepStart(ctx, r.stepStartEvent(prepared, input.steps), r.options.OnStepStart)

	prepared.callOptions.IncludeRawChunks = r.options.Include.RawChunks
	started := time.Now()

	attempts := r.newStreamAttempts(prepared, timers)

	if err := attempts.start(ctx); err != nil {
		r.fail(ctx, err, yield)

		return streamStepOutcome{}, false
	}

	// Ranging over the parts releases each attempt's stream; this covers
	// early returns. A release failure cannot change the outcome.
	defer attempts.close()

	step := newStreamStep(streamStepSetup{
		run:      r,
		prepared: prepared,
		attempts: attempts,
		started:  started,
	})

	for part := range attempts.parts(ctx) {
		if !step.handle(ctx, part, yield) {
			return streamStepOutcome{}, false
		}
	}

	return step.finish(yield)
}

// streamStepSetup holds what a streamed step starts from.
type streamStepSetup struct {
	run      *streamTextRun
	prepared preparedStep
	attempts *streamAttempts
	started  time.Time
}

// streamStep is the state of one streamed step.
type streamStep struct {
	streamStepSetup

	warnings    []model.Warning
	stepStarted bool

	// textIDs and reasoningIDs map provider part IDs to stream part IDs.
	textIDs      map[string]string
	reasoningIDs map[string]string

	// inputToolNames maps the IDs of tool inputs being streamed to their
	// tool names.
	inputToolNames map[string]string

	calls             []ToolCall
	callsToExecute    []ToolCall
	outputs           []ToolOutput
	approvalResponses []ToolApprovalResponse
	executionTimes    map[string]time.Duration

	finishReason     model.UnifiedFinishReason
	rawFinishReason  string
	usage            LanguageModelUsage
	providerMetadata model.ProviderMetadata
	response         model.ResponseMetadata
	performance      LanguageModelCallPerformance

	// hasTerminal reports a model call end or error part; hasOutput reports
	// any generated output. A stream with neither produced nothing.
	hasTerminal bool
	hasOutput   bool
}

func newStreamStep(setup streamStepSetup) *streamStep {
	return &streamStep{
		streamStepSetup: setup,
		textIDs:         make(map[string]string),
		reasoningIDs:    make(map[string]string),
		inputToolNames:  make(map[string]string),
		executionTimes:  make(map[string]time.Duration),
		finishReason:    model.FinishOther,
		response:        responseMetadata(model.ResponseMetadata{}, setup.prepared.model),
	}
}

// handle processes one part of the model call. It reports false when the
// stream must end.
func (s *streamStep) handle(ctx context.Context, part LanguageModelStreamPart, yield func(StreamPart) bool) bool {
	switch typed := part.(type) {
	case streamRetryBoundary:
		return s.restartAttempt(typed, yield)

	case ModelCallStartPart:
		s.warnings = typed.Warnings

		return true
	}

	if !s.stepStarted {
		s.stepStarted = true

		start := StartStepPart{
			Request:  s.stepRequest(),
			Warnings: s.warnings,
		}

		if !yield(start) {
			return false
		}
	}

	if isLanguageModelOutputPart(part) {
		s.hasOutput = true
	}

	switch typed := part.(type) {
	case TextStartPart, TextDeltaPart, TextEndPart:
		return s.handleTextPart(typed, yield)

	case ReasoningStartPart, ReasoningDeltaPart, ReasoningEndPart:
		return s.handleReasoningPart(typed, yield)

	case ToolInputStartPart:
		return s.handleToolInputStart(ctx, typed, yield)

	case ToolInputDeltaPart:
		return s.handleToolInputDelta(ctx, typed, yield)

	case ToolCall:
		return s.handleToolCall(ctx, typed, yield)

	case ToolResult:
		return s.handleToolOutput(typed, yield)

	case ToolError:
		return s.handleToolOutput(typed, yield)

	case ModelCallResponseMetadataPart:
		s.updateResponse(typed.ResponseMetadata)

		return true

	case ModelCallEndPart:
		return s.handleModelCallEnd(ctx, typed, yield)

	case ErrorPart:
		s.hasTerminal = true
		s.finishReason = model.FinishError

		return yield(typed)

	case RawPart:
		if !s.run.options.Include.RawChunks {
			return true
		}

		return yield(typed)

	case StreamPart:
		return yield(typed)

	default:
		return true
	}
}

// restartAttempt resets the state of the failed attempt and passes the
// boundary on, so that the recorder and the transforms restart too.
func (s *streamStep) restartAttempt(boundary streamRetryBoundary, yield func(StreamPart) bool) bool {
	s.warnings = boundary.warnings
	s.finishReason = model.FinishOther
	s.rawFinishReason = ""
	s.usage = LanguageModelUsage{}
	s.providerMetadata = nil
	s.performance = LanguageModelCallPerformance{}
	s.response = responseMetadata(model.ResponseMetadata{}, s.prepared.model)
	s.hasTerminal = false
	s.hasOutput = false

	clear(s.textIDs)
	clear(s.reasoningIDs)

	return yield(streamRetryBoundaryPart{
		request:  s.stepRequest(),
		warnings: s.warnings,
	})
}

// handleTextPart maps the provider's text IDs to the stream's text IDs and
// drops empty deltas without metadata.
func (s *streamStep) handleTextPart(part LanguageModelStreamPart, yield func(StreamPart) bool) bool {
	switch typed := part.(type) {
	case TextStartPart:
		typed.ID = reservePartID(s.run.textIDs, s.textIDs, typed.ID)

		return yield(typed)

	case TextDeltaPart:
		if typed.Text == "" && typed.ProviderMetadata == nil {
			return true
		}

		typed.ID = mappedPartID(s.textIDs, typed.ID)

		return yield(typed)

	case TextEndPart:
		providerID := typed.ID
		typed.ID = mappedPartID(s.textIDs, providerID)
		delete(s.textIDs, providerID)

		return yield(typed)

	default:
		return true
	}
}

// handleReasoningPart maps the provider's reasoning IDs to the stream's
// reasoning IDs.
func (s *streamStep) handleReasoningPart(part LanguageModelStreamPart, yield func(StreamPart) bool) bool {
	switch typed := part.(type) {
	case ReasoningStartPart:
		typed.ID = reservePartID(s.run.reasoningIDs, s.reasoningIDs, typed.ID)

		return yield(typed)

	case ReasoningDeltaPart:
		typed.ID = mappedPartID(s.reasoningIDs, typed.ID)

		return yield(typed)

	case ReasoningEndPart:
		providerID := typed.ID
		typed.ID = mappedPartID(s.reasoningIDs, providerID)
		delete(s.reasoningIDs, providerID)

		return yield(typed)

	default:
		return true
	}
}

// reservePartID reserves a stream-unique ID for a provider part ID and
// records the mapping for the step.
func reservePartID(reserver *partIDReserver, ids map[string]string, providerID string) string {
	streamID := reserver.reserve(providerID)
	ids[providerID] = streamID

	return streamID
}

func (s *streamStep) handleToolInputStart(ctx context.Context, part ToolInputStartPart, yield func(StreamPart) bool) bool {
	if !yield(part) {
		return false
	}

	s.inputToolNames[part.ID] = part.ToolName

	tool, found := findTool(s.prepared.executionTools, part.ToolName)
	if found && tool.OnInputStart != nil {
		tool.OnInputStart(ctx, s.toolExecutionOptions(part.ID))
	}

	return true
}

func (s *streamStep) handleToolInputDelta(ctx context.Context, part ToolInputDeltaPart, yield func(StreamPart) bool) bool {
	if !yield(part) {
		return false
	}

	tool, found := findTool(s.prepared.executionTools, s.inputToolNames[part.ID])
	if found && tool.OnInputDelta != nil {
		tool.OnInputDelta(ctx, ToolInputDeltaEvent{
			ToolExecutionOptions: s.toolExecutionOptions(part.ID),
			InputTextDelta:       part.Delta,
		})
	}

	return true
}

// handleToolCall records a parsed tool call, notifies the tool and resolves
// the call's approval. Invalid client calls are reported by the ToolError
// part that follows them.
func (s *streamStep) handleToolCall(ctx context.Context, call ToolCall, yield func(StreamPart) bool) bool {
	s.calls = append(s.calls, call)
	delete(s.inputToolNames, call.ToolCallID)

	if !yield(call) {
		return false
	}

	if call.Invalid {
		return true
	}

	tool, found := findTool(s.prepared.executionTools, call.ToolName)
	if !found {
		return true
	}

	if tool.OnInputAvailable != nil {
		tool.OnInputAvailable(ctx, ToolInputEvent[json.RawMessage]{
			ToolExecutionOptions: s.toolExecutionOptions(call.ToolCallID),
			Input:                call.Input,
		})
	}

	return s.resolveApproval(ctx, tool, call, yield)
}

// resolveApproval streams the approval request and automatic response of a
// call and schedules the call for execution when it may run.
func (s *streamStep) resolveApproval(ctx context.Context, tool Tool, call ToolCall, yield func(StreamPart) bool) bool {
	approval, err := s.run.resolveCallApproval(ctx, call, s.prepared.executionTools, s.prepared.messages)
	if err != nil {
		yield(ErrorPart{
			Err: err,
		})

		return false
	}

	if approval.request != nil {
		if !yield(*approval.request) {
			return false
		}
	}

	if approval.response != nil {
		if !s.yieldApprovalResponse(call, *approval.response, yield) {
			return false
		}
	}

	runsLocally := tool.Execute != nil && !call.ProviderExecuted
	if runsLocally && !approval.blocked {
		s.callsToExecute = append(s.callsToExecute, call)
	}

	return true
}

// yieldApprovalResponse records and streams an automatic approval response,
// followed by the denial of a denied call.
func (s *streamStep) yieldApprovalResponse(call ToolCall, response ToolApprovalResponse, yield func(StreamPart) bool) bool {
	s.approvalResponses = append(s.approvalResponses, response)

	if !yield(response) {
		return false
	}

	if response.Approved {
		return true
	}

	denial := ToolOutputDeniedPart{
		ToolCallID:       call.ToolCallID,
		ToolName:         call.ToolName,
		ProviderExecuted: call.ProviderExecuted,
	}

	return yield(denial)
}

// handleToolOutput records a provider-executed result or the error of an
// invalid call. Preliminary results are streamed but not recorded.
func (s *streamStep) handleToolOutput(output ToolOutput, yield func(StreamPart) bool) bool {
	result, isResult := output.(ToolResult)
	isPreliminary := isResult && result.Preliminary

	if !isPreliminary {
		s.outputs = append(s.outputs, output)
	}

	return yield(output)
}

// handleModelCallEnd records the usage and finish reason and executes the
// scheduled tool calls.
func (s *streamStep) handleModelCallEnd(ctx context.Context, part ModelCallEndPart, yield func(StreamPart) bool) bool {
	s.hasTerminal = true
	s.usage = part.Usage
	s.finishReason = part.FinishReason
	s.rawFinishReason = part.RawFinishReason
	s.providerMetadata = part.ProviderMetadata
	s.performance = part.Performance

	if !isToolExecutionAllowed(s.finishReason) {
		return true
	}

	executed, ok := s.run.streamToolExecutions(ctx, streamedToolExecution{
		execution: s.run.toolExecution(s.prepared.executionTools, s.prepared.messages),
		calls:     s.callsToExecute,
		yield:     yield,
	})

	s.outputs = append(s.outputs, executed.outputs...)
	for toolCallID, executionTime := range executed.executionTimes {
		s.executionTimes[toolCallID] = executionTime
	}

	return ok
}

// finish closes the step. A stream that ended without a model call end and
// without output fails the step.
func (s *streamStep) finish(yield func(StreamPart) bool) (streamStepOutcome, bool) {
	if !s.hasTerminal && !s.hasOutput {
		yield(ErrorPart{
			Err: &NoOutputGeneratedError{
				Cause: errors.New("the model stream ended without a finish part"),
			},
		})

		return streamStepOutcome{}, false
	}

	outputCallIDs := make([]string, 0, len(s.outputs))
	for _, output := range s.outputs {
		outputCallIDs = append(outputCallIDs, toolOutputCallID(output))
	}

	s.run.deferred.track(s.calls, s.prepared.executionTools, outputCallIDs)

	finishStep := FinishStepPart{
		Response: model.ResponseInfo{
			ResponseMetadata: s.response,
			Headers:          s.attempts.current.Response.Headers,
		},
		Usage: s.usage,
		Performance: StepPerformance{
			LanguageModelCallPerformance: s.performance,
			StepTime:                     time.Since(s.started),
			ToolExecution:                s.executionTimes,
		},
		FinishReason:     s.finishReason,
		RawFinishReason:  s.rawFinishReason,
		ProviderMetadata: s.providerMetadata,
	}

	if !yield(finishStep) {
		return streamStepOutcome{}, false
	}

	return streamStepOutcome{
		prepared:        s.prepared,
		finishReason:    s.finishReason,
		rawFinishReason: s.rawFinishReason,
		progress:        s.progress(),
	}, true
}

// progress counts the client tool calls, their outputs and denials.
func (s *streamStep) progress() loopProgress {
	var progress loopProgress

	for _, call := range s.calls {
		if !call.ProviderExecuted {
			progress.clientToolCalls++
		}
	}

	for _, output := range s.outputs {
		if !toolOutputIsProviderExecuted(output) {
			progress.clientToolOutputs++
		}
	}

	progress.deniedResponses = countDenied(s.approvalResponses)

	return progress
}

func (s *streamStep) updateResponse(metadata model.ResponseMetadata) {
	if metadata.ID != "" {
		s.response.ID = metadata.ID
	}

	if !metadata.Timestamp.IsZero() {
		s.response.Timestamp = metadata.Timestamp
	}

	if metadata.ModelID != "" {
		s.response.ModelID = metadata.ModelID
	}
}

// stepRequest retains the request payloads of the current attempt that
// Include selects.
func (s *streamStep) stepRequest() StepRequest {
	return newStepRequest(retainedStepRequest{
		body:            s.attempts.current.Request.Body,
		messages:        s.prepared.messages,
		includeBody:     s.run.options.Include.RequestBody,
		includeMessages: s.run.options.Include.RequestMessages,
	})
}

func (s *streamStep) toolExecutionOptions(toolCallID string) ToolExecutionOptions {
	return ToolExecutionOptions{
		ToolCallID: toolCallID,
		Messages:   s.prepared.messages,
	}
}

func mappedPartID(ids map[string]string, providerID string) string {
	if streamID, ok := ids[providerID]; ok {
		return streamID
	}

	return providerID
}

// isOutputStreamPart reports whether a provider part carries generated
// output, for timing.
func isOutputStreamPart(part model.StreamPart) bool {
	switch typed := part.(type) {
	case model.TextDelta:
		return typed.Delta != ""

	case model.ReasoningDelta:
		return typed.Delta != ""

	case model.ToolInputDelta:
		return typed.Delta != ""

	case model.File, model.ReasoningFile, model.ToolCall:
		return true

	default:
		return false
	}
}
