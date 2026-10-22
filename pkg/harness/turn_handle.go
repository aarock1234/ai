package harness

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// handleEvent processes one adapter event. It reports true when the turn
// settled or paused and the driver must stop reading events.
func (d *turnDriver) handleEvent(ctx context.Context, value StreamPart) (bool, error) {
	if raw, ok := value.(RawPart); ok {
		d.result.enqueueRaw(raw.Value)

		return false, nil
	}

	if result, ok := value.(ToolResultPart); ok {
		delete(d.hostToolCallsAwaitingEcho, result.ToolCallID)
	}

	if d.stopPending {
		stopped, err := d.checkStopConditions(ctx, value)
		if err != nil || stopped {
			return stopped, err
		}
	}

	d.observeTurnProgress(ctx, value)

	if isToolInputPart(value) {
		d.forwardToolInput(value)

		return false, nil
	}

	display := stripWorkDir(value, d.setup.sessionWorkDir)
	if d.isSettledReplay(display) {
		return false, nil
	}

	if handled, settled, err := d.handleBoundaries(ctx, value, display); handled || err != nil {
		return settled, err
	}

	if approval, ok := display.(ToolApprovalRequestPart); ok {
		denied, err := d.denyFilteredBuiltinApproval(ctx, approval)
		if err != nil || denied {
			return false, err
		}
	}

	if failure, ok := value.(ErrorPart); ok {
		return true, d.handleErrorPart(ctx, failure, display.(ErrorPart))
	}

	return d.handleContent(ctx, value, display)
}

// checkStopConditions evaluates the stop conditions at the step boundary
// pinned by the previous finish-step part. When one is met the turn is
// suspended there and the result finishes.
func (d *turnDriver) checkStopConditions(ctx context.Context, value StreamPart) (bool, error) {
	if _, isFinish := value.(FinishPart); isFinish {
		d.releaseStopBoundary()

		return false, nil
	}

	if !d.stopConditionMet(ctx) {
		d.releaseStopBoundary()

		return false, nil
	}

	d.mu.Unlock()
	err := d.setup.hooks.stopConditionMet(ctx)
	d.mu.Lock()

	if err != nil {
		return false, err
	}

	d.releaseStopBoundary()
	d.lifecycle.end(ctx, d.completedSteps, d.completedUsage())
	d.result.finish(ctx, nil)

	return true, nil
}

// stopConditionMet reports whether any stop condition holds for the
// completed steps.
func (d *turnDriver) stopConditionMet(ctx context.Context) bool {
	event := ai.StopConditionEvent{
		Steps: d.completedSteps,
	}

	for _, condition := range d.setup.stopConditions {
		if condition(ctx, event) {
			return true
		}
	}

	return false
}

// observeTurnProgress starts the operation on stream start and opens a step
// before the first content of each step.
func (d *turnDriver) observeTurnProgress(ctx context.Context, value StreamPart) {
	if start, ok := value.(StreamStartPart); ok {
		modelID := start.ModelID
		if modelID == "" {
			modelID = d.setup.settings.Model
		}

		if modelID != "" {
			d.result.setModelID(modelID)
		}

		d.result.addWarnings(start.Warnings)
		d.lifecycle.start(ctx, modelID)
	}

	switch value.(type) {
	case StreamStartPart, FinishStepPart, FinishPart, ErrorPart:
		return
	default:
		d.lifecycle.ensureStepOpen(ctx)
	}
}

// forwardToolInput forwards streamed tool input with the working directory
// stripped, unless it replays a settled call.
func (d *turnDriver) forwardToolInput(value StreamPart) {
	toolCallID := toolInputID(value)
	if d.settledHostToolCalls[toolCallID] || d.settledBuiltinApprovals[toolCallID] {
		return
	}

	for _, stripped := range d.stripper.strip(value) {
		for _, part := range translateStreamPart(stripped, d.lookupToolCall) {
			d.result.enqueue(part)
		}
	}
}

// isSettledReplay reports a replayed event of a call that a continuation
// already settled.
func (d *turnDriver) isSettledReplay(display StreamPart) bool {
	switch typed := display.(type) {
	case ToolCallPart:
		return d.settledHostToolCalls[typed.ToolCallID] || d.settledBuiltinApprovals[typed.ToolCallID]
	case ToolApprovalRequestPart:
		return d.settledHostToolCalls[typed.ToolCallID] || d.settledBuiltinApprovals[typed.ToolCallID]
	case ToolResultPart:
		return d.settledHostToolCalls[typed.ToolCallID]
	default:
		return false
	}
}

// handleBoundaries applies the pause and resume rules of step and turn
// boundaries. It reports whether the event was fully handled.
func (d *turnDriver) handleBoundaries(ctx context.Context, value StreamPart, display StreamPart) (bool, bool, error) {
	switch typed := value.(type) {
	case FinishStepPart:
		if d.hostInputPauseRequested {
			if err := d.joinHostToolsReleasingLock(); err != nil {
				return true, false, err
			}

			if len(d.hostToolCallsAwaitingEcho) == 0 {
				return true, true, d.pauseForHostInput(ctx, &typed)
			}

			d.deferredPauseFinishStep = &typed

			return true, false, nil
		}

		if len(d.hostToolCallsAwaitingEcho) > 0 {
			d.ignoredFinishStep = &typed

			return true, false, nil
		}

		d.ignoredFinishStep = nil
	case FinishPart:
		if d.hostInputPauseRequested {
			return true, true, d.pauseForHostInput(ctx, d.deferredPauseFinishStep)
		}

		if err := d.closeIgnoredFinishStep(ctx); err != nil {
			return true, false, err
		}
	}

	if _, isStepEnd := display.(FinishStepPart); isStepEnd && d.closingResumedStep {
		d.closeResumedStep(ctx)

		return true, false, nil
	}

	return false, false, nil
}

// closeIgnoredFinishStep closes a step whose finish-step part arrived while
// host tools were still running, when the turn ends without another one.
func (d *turnDriver) closeIgnoredFinishStep(ctx context.Context) error {
	if d.ignoredFinishStep == nil {
		return nil
	}

	finishStep := *d.ignoredFinishStep
	d.ignoredFinishStep = nil

	if err := d.joinHostToolsReleasingLock(); err != nil {
		return err
	}

	if d.closingResumedStep {
		d.closeResumedStep(ctx)

		return nil
	}

	return d.completeStep(ctx, stepBoundary{
		finishReason: finishStep.FinishReason,
		usage:        finishStep.Usage,
		metadata:     providerMetadata(finishStep.HarnessMetadata),
	})
}

// closeResumedStep drops the replayed content of a step that a previous
// slice already recorded.
func (d *turnDriver) closeResumedStep(ctx context.Context) {
	d.closingResumedStep = false
	d.publishToolExecutions(ctx)
	d.resetStepContent()
	d.result.discardCurrentStepContent()
}

// denyFilteredBuiltinApproval denies the approval request of an inactive
// built-in tool. The decision is answered here, so the runtime's step
// boundary must wait for the denied call's result.
func (d *turnDriver) denyFilteredBuiltinApproval(ctx context.Context, approval ToolApprovalRequestPart) (bool, error) {
	call, known := d.toolCalls[approval.ToolCallID]
	if !known {
		return false, d.unknownApprovalError(approval)
	}

	toolName := call.ToolName
	if raw, ok := d.rawToolCalls[approval.ToolCallID]; ok {
		toolName = raw.ToolName
	}

	if IsBuiltinToolIncluded(toolName, d.setup.builtinToolFiltering) {
		return false, nil
	}

	approver, err := d.toolApprover()
	if err != nil {
		return false, err
	}

	d.hostToolCallsAwaitingEcho[approval.ToolCallID] = true

	err = approver.SubmitToolApproval(ctx, ToolApprovalSubmission{
		ApprovalID: approval.ApprovalID,
		Approved:   false,
		Reason:     BuiltinToolFilteringDenialReason(toolName),
	})
	if err != nil {
		return false, fmt.Errorf("deny inactive built-in tool %q: %w", toolName, err)
	}

	return true, nil
}

// handleErrorPart settles the turn with an adapter error. The consumer sees
// the stripped error through the settled result only, never also as a
// forwarded error part.
func (d *turnDriver) handleErrorPart(ctx context.Context, failure ErrorPart, display ErrorPart) error {
	_ = d.joinHostToolsReleasingLock()

	d.settleLifecycle(ctx, failure.Err)
	d.logError(ctx, "harness stream error", failure.Err)
	d.settleFailure(ctx, display.Err)

	return errTurnSettled
}

// handleContent forwards content, records it in the step, drives step
// boundaries and runs host tools.
func (d *turnDriver) handleContent(ctx context.Context, value StreamPart, display StreamPart) (bool, error) {
	translated := translateStreamPart(display, d.lookupToolCall)

	if _, isResult := value.(ToolResultPart); !isResult {
		for _, part := range translated {
			d.result.enqueue(part)
		}
	}

	var validatedHostCall *ai.ToolCall
	if call, ok := display.(ToolCallPart); ok {
		validatedHostCall = d.recordToolCall(value.(ToolCallPart), call)
	}

	switch typed := value.(type) {
	case TextDeltaPart:
		d.stepText.WriteString(typed.Delta)
	case ReasoningDeltaPart:
		d.stepReasoning.WriteString(typed.Delta)
	case ToolCallPart:
		d.trackToolCall(ctx, typed)
	case ToolResultPart:
		d.recordToolResult(ctx, typed, translated)
	case ToolApprovalRequestPart:
		return d.handleApprovalRequest(ctx, typed)
	case FinishStepPart:
		return false, d.handleFinishStep(ctx, typed)
	case FinishPart:
		return false, d.handleFinish(ctx, typed)
	}

	if call, ok := value.(ToolCallPart); ok && !call.ProviderExecuted {
		return d.handleHostToolCall(ctx, call, validatedHostCall)
	}

	return false, nil
}

// recordToolCall validates a tool call, records it and forwards the parsed
// call. Host tools are validated against the raw input, which keeps the
// absolute paths they need; consumers see the stripped input. It returns the
// validated call of a host tool.
func (d *turnDriver) recordToolCall(value ToolCallPart, display ToolCallPart) *ai.ToolCall {
	isHostTool := !value.ProviderExecuted &&
		hasTool(d.setup.activeTools, value.ToolName) &&
		!d.isClientExecutedBuiltin(value.ToolName)

	// An MCP tool can share its name with a built-in tool that is inactive
	// in this session. Its input must not be validated against the built-in
	// schema unless an active host tool owns the name.
	isFilteredBuiltinDynamicCall := display.ProviderExecuted &&
		display.Dynamic &&
		isBuiltinTool(d.setup.harness, display.ToolName) &&
		!IsBuiltinToolIncluded(display.ToolName, d.setup.builtinToolFiltering) &&
		!hasTool(d.setup.activeTools, display.ToolName)

	source := display
	if isHostTool {
		source = value
	}

	parsed := d.parser.parse(source.ToolCall, d.setup.tools, isFilteredBuiltinDynamicCall)

	displayed := parsed
	if isHostTool {
		displayed = displayHostToolCall(parsed, d.setup.sessionWorkDir)
	}

	d.rawToolCalls[display.ToolCallID] = display
	d.toolCalls[display.ToolCallID] = displayed
	d.result.enqueue(displayed)

	if !isHostTool {
		return nil
	}

	return &parsed
}

// trackToolCall adds a tool call to the step and starts tracking its
// execution. Runtime-executed calls report their execution start right away.
func (d *turnDriver) trackToolCall(ctx context.Context, value ToolCallPart) {
	call, ok := d.toolCalls[value.ToolCallID]
	if !ok {
		return
	}

	d.stepToolCalls = append(d.stepToolCalls, call)

	execution := d.findToolExecution(value.ToolCallID)
	if execution == nil {
		execution = &toolExecution{}
		d.toolExecutions = append(d.toolExecutions, execution)
	}

	*execution = toolExecution{
		call: call,
	}

	if value.ProviderExecuted {
		d.publishToolExecutionStart(ctx, execution)
	}
}

// recordToolResult records the outcome of a tool call and forwards its
// translated parts.
func (d *turnDriver) recordToolResult(ctx context.Context, value ToolResultPart, translated []ai.StreamPart) {
	execution := d.findToolExecution(value.ToolCallID)

	if execution != nil && execution.output == nil {
		execution.output = echoedToolOutput(execution.call, value)
	}

	if execution != nil && execution.duration == 0 && !execution.startedAt.IsZero() {
		execution.duration = time.Since(execution.startedAt)
	}

	raw, known := d.rawToolCalls[value.ToolCallID]
	if known && raw.ProviderExecuted && execution != nil && execution.output != nil {
		d.stepProviderToolResults = append(d.stepProviderToolResults, execution.output)
	}

	if execution != nil {
		d.publishToolExecutionEnd(ctx, execution)
	}

	for _, part := range translated {
		d.result.enqueue(part)
	}
}

// handleApprovalRequest records an approval request of a built-in tool. A
// matching continuation answers it right away; otherwise the turn pauses for
// a decision at the next step boundary.
func (d *turnDriver) handleApprovalRequest(ctx context.Context, request ToolApprovalRequestPart) (bool, error) {
	call, known := d.toolCalls[request.ToolCallID]
	if !known {
		return false, d.unknownApprovalError(request)
	}

	pending, exists := d.pendingApprovalsByApprovalID[request.ApprovalID]
	if !exists {
		pending = d.builtinPendingApproval(request, call)
	}

	d.pendingApprovalsByApprovalID[pending.ApprovalID] = pending
	d.pendingApprovalsByToolCallID[pending.ToolCallID] = pending

	if continuation, ok := d.approvalContinuations[pending.ApprovalID]; ok {
		paused, err := d.processApprovalContinuation(ctx, pending, continuation)
		if err != nil || paused {
			return paused, err
		}

		d.closingResumedStep = true

		return false, nil
	}

	d.setup.hooks.pendingToolApproval(pending)
	d.enqueueApprovalRequest(pending.ApprovalID, call, false)
	d.hostInputPauseRequested = true

	return false, nil
}

// builtinPendingApproval records an approval request of a built-in tool.
func (d *turnDriver) builtinPendingApproval(request ToolApprovalRequestPart, call ai.ToolCall) PendingToolApproval {
	pending := PendingToolApproval{
		ApprovalID:       request.ApprovalID,
		ToolCallID:       request.ToolCallID,
		ToolName:         call.ToolName,
		Input:            string(call.Input),
		Kind:             ApprovalBuiltin,
		ProviderExecuted: true,
	}

	if raw, ok := d.rawToolCalls[request.ToolCallID]; ok {
		pending.Input = raw.Input
		pending.ProviderExecuted = raw.ProviderExecuted
		pending.NativeName = raw.NativeName
	}

	return pending
}

// handleFinishStep completes the step and pins a stop condition boundary.
func (d *turnDriver) handleFinishStep(ctx context.Context, finishStep FinishStepPart) error {
	if err := d.joinHostToolsReleasingLock(); err != nil {
		return err
	}

	err := d.completeStep(ctx, stepBoundary{
		finishReason: finishStep.FinishReason,
		usage:        finishStep.Usage,
		metadata:     providerMetadata(finishStep.HarnessMetadata),
	})
	if err != nil {
		return err
	}

	if len(d.setup.stopConditions) == 0 {
		return nil
	}

	d.stopPending = true
	if finishStep.Checkpoint != nil {
		d.releaseStopPin = finishStep.Checkpoint.Pin()
	}

	return nil
}

// handleFinish records the terminal finish and reports the operation end.
func (d *turnDriver) handleFinish(ctx context.Context, finish FinishPart) error {
	if err := d.joinHostToolsReleasingLock(); err != nil {
		return err
	}

	d.finalFinish = &finish
	d.lifecycle.end(ctx, d.completedSteps, languageModelUsage(finish.TotalUsage))

	return nil
}

// completeStep closes the current model step: it reports the model call
// end, publishes the step's tool executions, records the step result and
// reports the step end.
func (d *turnDriver) completeStep(ctx context.Context, boundary stepBoundary) error {
	d.lifecycle.languageModelCallEnd(ctx, modelCallEnd{
		finishReason: boundary.finishReason.Unified,
		usage:        languageModelUsage(boundary.usage),
		content:      d.modelCallContent(),
		metadata:     boundary.metadata,
	})

	d.publishToolExecutions(ctx)

	step, err := d.result.finishStep(ctx, boundary)
	if err != nil {
		return err
	}

	d.completedSteps = append(d.completedSteps, step)
	d.lifecycle.stepEnd(ctx, step)
	d.resetStepContent()

	return nil
}

// modelCallContent returns the content the model produced in the current
// step, without host tool outputs.
func (d *turnDriver) modelCallContent() []ai.ContentPart {
	var content []ai.ContentPart

	if d.stepText.Len() > 0 {
		content = append(content, ai.TextContent{
			Text: d.stepText.String(),
		})
	}

	if d.stepReasoning.Len() > 0 {
		content = append(content, ai.ReasoningContent{
			Text: d.stepReasoning.String(),
		})
	}

	content = append(content, d.stepToolCalls...)
	content = append(content, d.stepApprovalRequests...)
	content = append(content, d.stepProviderToolResults...)

	return content
}

// resetStepContent clears the accumulated content of the current step.
func (d *turnDriver) resetStepContent() {
	d.stepText.Reset()
	d.stepReasoning.Reset()
	d.stepToolCalls = nil
	d.stepProviderToolResults = nil
	d.stepApprovalRequests = nil
	d.bufferedToolOutcomes = nil
	d.toolExecutions = nil
	d.ignoredFinishStep = nil
}

// pauseForHostInput ends the result at a step that waits for an approval or
// a client tool result. The runtime's turn stays open.
func (d *turnDriver) pauseForHostInput(ctx context.Context, finishStep *FinishStepPart) error {
	if err := d.joinHostToolsReleasingLock(); err != nil {
		return err
	}

	d.closingResumedStep = false

	boundary := stepBoundary{
		finishReason: model.FinishReason{
			Unified: model.FinishToolCalls,
		},
	}

	if finishStep != nil {
		boundary.usage = finishStep.Usage
		boundary.metadata = providerMetadata(finishStep.HarnessMetadata)
	}

	if err := d.completeStep(ctx, boundary); err != nil {
		return err
	}

	if err := d.finishForHostInputPause(ctx); err != nil {
		return err
	}

	return errTurnSettled
}

// finishForHostInputPause finishes the result while the turn waits for host
// input.
func (d *turnDriver) finishForHostInputPause(ctx context.Context) error {
	if err := d.joinHostToolsReleasingLock(); err != nil {
		return err
	}

	d.publishToolExecutions(ctx)
	d.lifecycle.end(ctx, d.completedSteps, ai.LanguageModelUsage{})
	d.result.finish(ctx, nil)

	return nil
}

// publishToolExecutionStart reports a tool execution start once.
func (d *turnDriver) publishToolExecutionStart(ctx context.Context, execution *toolExecution) {
	if execution.startNotified {
		return
	}

	execution.startNotified = true
	d.lifecycle.toolExecutionStart(ctx, execution.call)
}

// publishToolExecutionEnd reports a finished tool execution once.
func (d *turnDriver) publishToolExecutionEnd(ctx context.Context, execution *toolExecution) {
	if execution.output == nil || execution.endNotified {
		return
	}

	d.publishToolExecutionStart(ctx, execution)

	execution.endNotified = true
	d.lifecycle.toolExecutionEnd(ctx, execution.call, execution.output, execution.duration)
}

// publishToolExecutions reports every finished execution of the step and
// forwards the buffered tool outcomes.
func (d *turnDriver) publishToolExecutions(ctx context.Context) {
	for _, execution := range d.toolExecutions {
		d.publishToolExecutionEnd(ctx, execution)
	}

	for _, publish := range d.bufferedToolOutcomes {
		publish()
	}

	d.toolExecutions = nil
	d.bufferedToolOutcomes = nil
}

// enqueueApprovalRequest forwards an approval request and records it in the
// step.
func (d *turnDriver) enqueueApprovalRequest(approvalID string, call ai.ToolCall, automatic bool) {
	request := ai.ToolApprovalRequest{
		ApprovalID:  approvalID,
		ToolCall:    call,
		IsAutomatic: automatic,
	}

	d.result.enqueue(request)
	d.stepApprovalRequests = append(d.stepApprovalRequests, request)
}

// findToolExecution returns the tracked execution of a tool call.
func (d *turnDriver) findToolExecution(toolCallID string) *toolExecution {
	for _, execution := range d.toolExecutions {
		if execution.call.ToolCallID == toolCallID {
			return execution
		}
	}

	return nil
}

// removeToolExecution stops tracking a tool call.
func (d *turnDriver) removeToolExecution(toolCallID string) {
	for index, execution := range d.toolExecutions {
		if execution.call.ToolCallID == toolCallID {
			d.toolExecutions = append(d.toolExecutions[:index], d.toolExecutions[index+1:]...)

			return
		}
	}
}

// lookupToolCall returns the parsed call of a tool call ID.
func (d *turnDriver) lookupToolCall(toolCallID string) (ai.ToolCall, bool) {
	call, ok := d.toolCalls[toolCallID]

	return call, ok
}

// completedUsage sums the usage of the completed steps.
func (d *turnDriver) completedUsage() ai.LanguageModelUsage {
	var total ai.LanguageModelUsage

	for _, step := range d.completedSteps {
		total = addUsage(total, step.Usage)
	}

	return total
}

// isClientExecutedBuiltin reports the built-in questions tool, which the
// application answers through a tool result.
func (d *turnDriver) isClientExecutedBuiltin(toolName string) bool {
	return toolName == string(ToolAskUserQuestions) && isBuiltinTool(d.setup.harness, toolName)
}

// toolApprover returns the control's approval submitter.
func (d *turnDriver) toolApprover() (ToolApprovalSubmitter, error) {
	approver, ok := d.control.(ToolApprovalSubmitter)
	if !ok {
		return nil, fmt.Errorf(
			"harness %q emitted a built-in tool approval request but does not support approval responses",
			d.setup.harness.HarnessID(),
		)
	}

	return approver, nil
}

// unknownApprovalError reports an approval request for an unknown call.
func (d *turnDriver) unknownApprovalError(request ToolApprovalRequestPart) error {
	return fmt.Errorf(
		"harness %q emitted approval request %q for unknown tool call %q",
		d.setup.harness.HarnessID(),
		request.ApprovalID,
		request.ToolCallID,
	)
}

// echoedToolOutput converts a runtime tool result into the outcome of the
// tracked call.
func echoedToolOutput(call ai.ToolCall, result ToolResultPart) ai.ToolOutput {
	if result.IsError {
		return ai.ToolError{
			ToolCallID: call.ToolCallID,
			ToolName:   call.ToolName,
			Input:      call.Input,
			Error: &ai.ProviderToolError{
				Result: result.Result,
			},
			ProviderExecuted: call.ProviderExecuted,
			Dynamic:          call.Dynamic,
			Title:            call.Title,
			ProviderMetadata: call.ProviderMetadata,
			ToolMetadata:     call.ToolMetadata,
		}
	}

	return ai.ToolResult{
		ToolCallID:       call.ToolCallID,
		ToolName:         call.ToolName,
		Input:            call.Input,
		Output:           json.RawMessage(result.Result),
		ProviderExecuted: call.ProviderExecuted,
		Dynamic:          call.Dynamic,
		Title:            call.Title,
		ProviderMetadata: call.ProviderMetadata,
		ToolMetadata:     call.ToolMetadata,
	}
}

// isToolInputPart reports a streamed tool input part.
func isToolInputPart(part StreamPart) bool {
	switch part.(type) {
	case ToolInputStartPart, ToolInputDeltaPart, ToolInputEndPart:
		return true
	default:
		return false
	}
}

// toolInputID returns the tool call ID of a tool input part.
func toolInputID(part StreamPart) string {
	switch typed := part.(type) {
	case ToolInputStartPart:
		return typed.ID
	case ToolInputDeltaPart:
		return typed.ID
	case ToolInputEndPart:
		return typed.ID
	default:
		return ""
	}
}

// isBuiltinTool reports whether the harness declares a built-in tool.
func isBuiltinTool(harness Harness, toolName string) bool {
	for _, tool := range harness.BuiltinTools() {
		if tool.Name == toolName {
			return true
		}
	}

	return false
}

// displayHostToolCall strips the working directory from a host tool call
// for consumers and marks an invalid call with the validation message.
func displayHostToolCall(call ai.ToolCall, workDir string) ai.ToolCall {
	call.Input = stripWorkDirJSON(call.Input, workDir)

	if call.Invalid {
		call.Error = fmt.Errorf("%w: %w", errInvalidToolInput, call.Error)
	}

	return call
}
