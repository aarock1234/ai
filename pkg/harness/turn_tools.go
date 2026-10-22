package harness

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/sandbox"
)

// toolSubmitter submits a tool result to the runtime.
type toolSubmitter func(ctx context.Context, submission ToolResultSubmission) error

// handleHostToolCall handles a call of an application tool: it denies
// inactive tools, pauses for client-executed tools and approvals, and
// otherwise starts the tool in its own goroutine.
func (d *turnDriver) handleHostToolCall(ctx context.Context, value ToolCallPart, validated *ai.ToolCall) (bool, error) {
	parsed, known := d.toolCalls[value.ToolCallID]
	if !known {
		return false, fmt.Errorf("harness %q could not find parsed tool call %q", d.setup.harness.HarnessID(), value.ToolCallID)
	}

	isClientExecuted := d.isClientExecutedBuiltin(value.ToolName)

	if !isClientExecuted && !hasTool(d.setup.activeTools, value.ToolName) {
		return false, d.denyInactiveTool(ctx, value, parsed)
	}

	if isClientExecuted {
		d.recordPendingToolResult(value)
		d.hostInputPauseRequested = true

		return false, nil
	}

	if validated == nil {
		return false, fmt.Errorf("harness %q could not validate host tool %q", d.setup.harness.HarnessID(), value.ToolName)
	}

	if validated.Invalid {
		return false, d.rejectInvalidToolCall(ctx, value, parsed)
	}

	decision := resolveCustomToolApproval(value.ToolName, d.setup.toolApproval)
	if decision.Type == ai.ToolApprovalDenied {
		return false, d.denyByPolicy(ctx, value, parsed, decision.Reason)
	}

	pending, needsApproval := d.pendingApprovalsByToolCallID[value.ToolCallID]
	if !needsApproval && decision.Type == ai.ToolApprovalUserApproval {
		pending = d.customPendingApproval(value)
		needsApproval = true
	}

	if needsApproval {
		return d.requestApproval(ctx, pending)
	}

	tool, _ := findTool(d.setup.activeTools, value.ToolName)
	if tool.Execute == nil {
		d.recordPendingToolResult(value)
		d.hostInputPauseRequested = true

		return false, nil
	}

	d.hostToolCallsAwaitingEcho[value.ToolCallID] = true
	d.startHostToolExecution(ctx, hostToolCall{
		part:      value,
		validated: *validated,
		reported:  parsed,
	})

	return false, nil
}

// hostToolCall is an application tool call the host executes.
type hostToolCall struct {
	// part is the runtime's tool call.
	part ToolCallPart

	// validated is the call the tool executes, with its input validated
	// against the tool's schema.
	validated ai.ToolCall

	// reported is the call recorded in the tool's output and telemetry.
	reported ai.ToolCall
}

// denyInactiveTool denies a call of an application tool that is inactive.
func (d *turnDriver) denyInactiveTool(ctx context.Context, value ToolCallPart, parsed ai.ToolCall) error {
	output := executionDeniedOutput{
		Type:   "execution-denied",
		Reason: BuiltinToolFilteringDenialReason(value.ToolName),
	}

	err := d.submitToolResultLocked(ctx, ToolResultSubmission{
		ToolCallID: value.ToolCallID,
		Output:     output,
	})
	if err != nil {
		return err
	}

	if execution := d.findToolExecution(value.ToolCallID); execution != nil {
		execution.output = toolOutputOf(parsed, hostToolOutcome{
			output: output,
		})
	}

	return nil
}

// rejectInvalidToolCall reports a host tool call with invalid input to the
// runtime without running the tool.
func (d *turnDriver) rejectInvalidToolCall(ctx context.Context, value ToolCallPart, parsed ai.ToolCall) error {
	d.settledHostToolCalls[value.ToolCallID] = true
	d.removeToolExecution(value.ToolCallID)

	d.bufferedToolOutcomes = append(d.bufferedToolOutcomes, func() {
		d.result.enqueue(ai.ToolError{
			ToolCallID: value.ToolCallID,
			ToolName:   value.ToolName,
			Input:      parsed.Input,
			Error:      errInvalidToolInput,
			Dynamic:    true,
		})
	})

	return d.submitToolResultLocked(ctx, ToolResultSubmission{
		ToolCallID: value.ToolCallID,
		Output: toolFailureOutput{
			Error: invalidToolInputMessage,
		},
		IsError: true,
	})
}

// denyByPolicy denies a call the tool approval configuration denies,
// reporting an automatic approval request and response.
func (d *turnDriver) denyByPolicy(ctx context.Context, value ToolCallPart, parsed ai.ToolCall, reason string) error {
	approvalID := id.New("")

	d.enqueueApprovalRequest(approvalID, parsed, true)
	d.result.enqueue(ai.ToolApprovalResponse{
		ApprovalID: approvalID,
		ToolCall:   parsed,
		Approved:   false,
		Reason:     reason,
	})

	output := executionDeniedOutput{
		Type:   "execution-denied",
		Reason: reason,
	}

	execution := d.findToolExecution(value.ToolCallID)
	if execution != nil {
		d.publishToolExecutionStart(ctx, execution)
	}

	err := d.submitToolResultLocked(ctx, ToolResultSubmission{
		ToolCallID: value.ToolCallID,
		Output:     output,
	})
	if err != nil {
		return err
	}

	if execution != nil {
		execution.output = toolOutputOf(parsed, hostToolOutcome{
			output: output,
		})
		d.publishToolExecutionEnd(ctx, execution)
	}

	return nil
}

// customPendingApproval records an approval request of an application tool.
func (d *turnDriver) customPendingApproval(value ToolCallPart) PendingToolApproval {
	return PendingToolApproval{
		ApprovalID: id.New(""),
		ToolCallID: value.ToolCallID,
		ToolName:   value.ToolName,
		Input:      value.Input,
		Kind:       ApprovalCustom,
		NativeName: value.NativeName,
	}
}

// requestApproval answers an approval from a continuation, or forwards the
// request and pauses the turn for a decision.
func (d *turnDriver) requestApproval(ctx context.Context, pending PendingToolApproval) (bool, error) {
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

	call, known := d.toolCalls[pending.ToolCallID]
	if !known {
		return false, fmt.Errorf(
			"harness %q could not find parsed tool call %q for approval request %q",
			d.setup.harness.HarnessID(),
			pending.ToolCallID,
			pending.ApprovalID,
		)
	}

	d.setup.hooks.pendingToolApproval(pending)
	d.enqueueApprovalRequest(pending.ApprovalID, call, false)
	d.hostInputPauseRequested = true

	return false, nil
}

// processApprovalContinuation applies a caller's decision on a pending
// approval. Built-in tools receive the decision through the control;
// approved application tools run on the host. It reports true when the
// turn paused again for a client tool result.
func (d *turnDriver) processApprovalContinuation(ctx context.Context, approval PendingToolApproval, continuation model.ToolApprovalResponsePart) (bool, error) {
	raw := d.approvalToolCall(approval)

	var validated *ai.ToolCall
	var call ai.ToolCall

	if approval.Kind == ApprovalCustom && continuation.Approved {
		parsed := d.parser.parse(raw.ToolCall, d.setup.tools, false)
		validated = &parsed
		call = displayHostToolCall(parsed, d.setup.sessionWorkDir)
	} else {
		call = ai.ToolCall{
			ToolCallID:       raw.ToolCallID,
			ToolName:         raw.ToolName,
			Input:            invalidToolInput(raw.Input),
			ProviderExecuted: raw.ProviderExecuted,
		}
	}

	d.result.enqueueContinuation(ai.ToolApprovalResponse{
		ApprovalID:       approval.ApprovalID,
		ToolCall:         call,
		Approved:         continuation.Approved,
		Reason:           continuation.Reason,
		ProviderExecuted: approval.ProviderExecuted,
	})

	d.setup.hooks.toolApprovalSettled(approval.ApprovalID)
	delete(d.pendingApprovalsByApprovalID, approval.ApprovalID)
	delete(d.pendingApprovalsByToolCallID, approval.ToolCallID)

	if approval.Kind == ApprovalBuiltin {
		return false, d.answerBuiltinApproval(ctx, approval, continuation)
	}

	d.settledHostToolCalls[approval.ToolCallID] = true

	if !continuation.Approved {
		return false, d.submitToolResultLocked(ctx, ToolResultSubmission{
			ToolCallID: approval.ToolCallID,
			Output: executionDeniedOutput{
				Type:   "execution-denied",
				Reason: continuation.Reason,
			},
		})
	}

	if validated == nil {
		return false, fmt.Errorf("harness %q could not validate approved host tool %q", d.setup.harness.HarnessID(), approval.ToolName)
	}

	if validated.Invalid {
		return false, d.reportInvalidApprovedCall(ctx, approval, call)
	}

	return d.runApprovedTool(ctx, hostToolCall{
		part:      raw,
		validated: *validated,
		reported:  call,
	})
}

// approvalToolCall returns the runtime tool call of a pending approval.
func (d *turnDriver) approvalToolCall(approval PendingToolApproval) ToolCallPart {
	if approval.Kind == ApprovalBuiltin {
		if raw, ok := d.rawToolCalls[approval.ToolCallID]; ok {
			return raw
		}
	}

	part := ToolCallPart{
		NativeName: approval.NativeName,
	}
	part.ToolCallID = approval.ToolCallID
	part.ToolName = approval.ToolName
	part.Input = approval.Input
	part.ProviderExecuted = approval.ProviderExecuted

	return part
}

// answerBuiltinApproval sends a decision on a built-in tool to the runtime.
func (d *turnDriver) answerBuiltinApproval(ctx context.Context, approval PendingToolApproval, continuation model.ToolApprovalResponsePart) error {
	d.settledBuiltinApprovals[approval.ToolCallID] = true

	approver, err := d.toolApprover()
	if err != nil {
		return err
	}

	err = approver.SubmitToolApproval(ctx, ToolApprovalSubmission{
		ApprovalID: approval.ApprovalID,
		Approved:   continuation.Approved,
		Reason:     continuation.Reason,
	})
	if err != nil {
		return fmt.Errorf("submit tool approval %q: %w", approval.ApprovalID, err)
	}

	return nil
}

// reportInvalidApprovedCall reports an approved call whose input failed
// validation.
func (d *turnDriver) reportInvalidApprovedCall(ctx context.Context, approval PendingToolApproval, call ai.ToolCall) error {
	err := d.submitToolResultLocked(ctx, ToolResultSubmission{
		ToolCallID: approval.ToolCallID,
		Output: toolFailureOutput{
			Error: invalidToolInputMessage,
		},
		IsError: true,
	})
	if err != nil {
		return err
	}

	d.bufferedToolOutcomes = append(d.bufferedToolOutcomes, func() {
		d.enqueueHostToolOutcome(call, hostToolOutcome{
			err: errInvalidToolInput,
		})
	})
	d.publishToolExecutions(ctx)

	return nil
}

// runApprovedTool runs an approved application tool on the driver
// goroutine. It reports true when the tool has no Execute function and the
// turn paused for a client result.
func (d *turnDriver) runApprovedTool(ctx context.Context, call hostToolCall) (bool, error) {
	d.lifecycle.start(ctx, d.setup.settings.Model)

	execution := &toolExecution{
		call: call.reported,
	}
	d.toolExecutions = append(d.toolExecutions, execution)
	d.publishToolExecutionStart(ctx, execution)

	startedAt := time.Now()

	outcome, executed, err := d.executeHostTool(ctx, call.validated, d.submitToolResultLocked)
	if err != nil {
		return false, err
	}

	if !executed {
		d.recordPendingToolResult(call.part)

		return true, d.finishForHostInputPause(ctx)
	}

	execution.output = toolOutputOf(call.reported, outcome)
	execution.duration = time.Since(startedAt)
	d.publishToolExecutionEnd(ctx, execution)
	d.enqueueHostToolOutcome(call.reported, outcome)

	return false, nil
}

// processResultContinuation submits a caller-provided result, or a result
// completed while suspending, for a pending client tool call.
func (d *turnDriver) processResultContinuation(ctx context.Context, pending PendingToolResult, continuation *model.ToolResultPart) error {
	submission, err := continuationSubmission(pending, continuation)
	if err != nil {
		return err
	}

	d.settledHostToolCalls[pending.ToolCallID] = true
	d.setup.hooks.toolResultSettled(pending.ToolCallID)

	if err := d.submitToolResultLocked(ctx, submission); err != nil {
		return err
	}

	delete(d.pendingResults, pending.ToolCallID)

	return nil
}

// continuationSubmission returns the submission for a pending tool result.
func continuationSubmission(pending PendingToolResult, continuation *model.ToolResultPart) (ToolResultSubmission, error) {
	if continuation == nil {
		completed := pending.CompletedResult

		return ToolResultSubmission{
			ToolCallID: pending.ToolCallID,
			Output:     completed.Output,
			IsError:    completed.IsError,
			ToolResult: completed.ToolResult,
		}, nil
	}

	output, isError, err := unwrapToolResultOutput(*continuation)
	if err != nil {
		return ToolResultSubmission{}, err
	}

	toolResult := *continuation
	toolResult.ToolName = pending.ToolName

	if toolResult.ProviderOptions == nil && pending.ProviderOptions != nil {
		toolResult.ProviderOptions = pending.ProviderOptions
	}

	return ToolResultSubmission{
		ToolCallID: pending.ToolCallID,
		Output:     output,
		IsError:    isError,
		ToolResult: &toolResult,
	}, nil
}

// startHostToolExecution runs a host tool in its own goroutine. The driver
// joins it at the next step boundary.
func (d *turnDriver) startHostToolExecution(ctx context.Context, call hostToolCall) {
	execution := &hostExecution{
		done: make(chan struct{}),
	}
	d.outstanding = append(d.outstanding, execution)

	go func() {
		defer close(execution.done)

		execution.err = d.runHostTool(ctx, call)
	}()
}

// runHostTool executes a host tool and reports its execution. It runs on
// its own goroutine and takes mu for the shared state.
func (d *turnDriver) runHostTool(ctx context.Context, call hostToolCall) error {
	part := call.part

	d.mu.Lock()
	tracked := d.findToolExecution(part.ToolCallID)
	if tracked == nil {
		d.mu.Unlock()

		return fmt.Errorf("harness %q could not track host tool %q", d.setup.harness.HarnessID(), part.ToolName)
	}

	d.publishToolExecutionStart(ctx, tracked)
	startedAt := time.Now()
	tracked.startedAt = startedAt
	d.mu.Unlock()

	outcome, executed, err := d.executeHostTool(ctx, call.validated, d.submitToolResult)
	if err != nil {
		return err
	}

	if !executed {
		return fmt.Errorf("harness %q could not execute host tool %q", d.setup.harness.HarnessID(), part.ToolName)
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	tracked.output = toolOutputOf(call.reported, outcome)
	tracked.duration = time.Since(startedAt)
	d.publishToolExecutionEnd(ctx, tracked)

	return nil
}

// joinHostToolsReleasingLock waits for the running host tools and returns
// the first failure. The caller holds mu. While waiting it releases mu, so
// that the tool goroutines can take it to report their executions, and it
// holds mu again on return; state read under mu before the call may have
// changed.
func (d *turnDriver) joinHostToolsReleasingLock() error {
	executions := d.outstanding
	d.outstanding = nil

	if len(executions) == 0 {
		return nil
	}

	d.mu.Unlock()
	defer d.mu.Lock()

	var first error

	for _, execution := range executions {
		<-execution.done

		if execution.err != nil && first == nil {
			first = execution.err
		}
	}

	return first
}

// executeHostTool runs an application tool and submits its result. It
// reports false when the tool has no Execute function. Tool failures are
// submitted to the runtime and returned in the outcome; the error result
// is reserved for failed submissions.
func (d *turnDriver) executeHostTool(ctx context.Context, call ai.ToolCall, submit toolSubmitter) (hostToolOutcome, bool, error) {
	tool, found := findTool(d.setup.activeTools, call.ToolName)
	if !found || tool.Execute == nil {
		return hostToolOutcome{}, false, nil
	}

	if call.Invalid {
		failure := errInvalidToolInput

		return d.submitToolFailure(ctx, call, failure, submit)
	}

	output, err := d.invokeTool(ctx, *tool, call)
	if err != nil {
		return d.submitToolFailure(ctx, call, err, submit)
	}

	submission := ToolResultSubmission{
		ToolCallID: call.ToolCallID,
		Output:     output,
	}

	if tool.ToModelOutput != nil {
		modelOutput, err := toolModelOutput(ctx, toolModelOutputInput{
			tool:       tool,
			toolCallID: call.ToolCallID,
			input:      call.Input,
			output:     output,
		})
		if err != nil {
			return d.submitToolFailure(ctx, call, err, submit)
		}

		submission.ToolResult = &model.ToolResultPart{
			ToolCallID: call.ToolCallID,
			ToolName:   call.ToolName,
			Output:     modelOutput,
		}
	}

	if err := submit(ctx, submission); err != nil {
		return hostToolOutcome{}, true, err
	}

	return hostToolOutcome{
		output: output,
	}, true, nil
}

// invokeTool runs a tool's Execute function inside the telemetry wrappers.
// The restricted sandbox is attached to the context, so tools can retrieve
// it with sandbox.SessionFromContext. Preliminary outputs are streamed as
// preliminary tool results; only the final output reaches the runtime.
func (d *turnDriver) invokeTool(ctx context.Context, tool ai.Tool, call ai.ToolCall) (any, error) {
	var output any

	err := d.lifecycle.executeTool(ctx, call, func(ctx context.Context) error {
		if d.setup.sandbox != nil {
			ctx = sandbox.WithSession(ctx, d.setup.sandbox)
		}

		result, err := ai.ExecuteTool(ctx, ai.ExecuteToolOptions{
			Tool:       tool,
			Input:      call.Input,
			ToolCallID: call.ToolCallID,
			OnPreliminary: func(preliminary any) {
				d.enqueuePreliminaryResult(call, preliminary)
			},
		})
		output = result

		return err
	})

	return output, err
}

// enqueuePreliminaryResult streams a preliminary output of a host tool with
// the working directory stripped, as the final output is. Preliminary
// results are not recorded in the step content, so they never reach the
// response messages.
func (d *turnDriver) enqueuePreliminaryResult(call ai.ToolCall, output any) {
	d.result.enqueueContinuation(ai.ToolResult{
		ToolCallID:  call.ToolCallID,
		ToolName:    call.ToolName,
		Output:      stripWorkDirOutput(output, d.setup.sessionWorkDir),
		Preliminary: true,
	})
}

// submitToolFailure submits a tool failure to the runtime.
func (d *turnDriver) submitToolFailure(ctx context.Context, call ai.ToolCall, failure error, submit toolSubmitter) (hostToolOutcome, bool, error) {
	err := submit(ctx, ToolResultSubmission{
		ToolCallID: call.ToolCallID,
		Output: toolFailureOutput{
			Error: failure.Error(),
		},
		IsError: true,
	})
	if err != nil {
		return hostToolOutcome{}, true, err
	}

	return hostToolOutcome{
		err: failure,
	}, true, nil
}

// submitToolResult submits a result from a host tool goroutine.
func (d *turnDriver) submitToolResult(ctx context.Context, submission ToolResultSubmission) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	return d.submitToolResultLocked(ctx, submission)
}

// submitToolResultLocked normalizes the model-facing output and submits a
// result to the runtime. While the session suspends the turn, the result is
// recorded for the continuation instead. The caller holds mu.
func (d *turnDriver) submitToolResultLocked(ctx context.Context, submission ToolResultSubmission) error {
	if submission.ToolResult != nil {
		normalized, err := d.normalizeSubmittedOutput(submission)
		if err != nil {
			return err
		}

		toolResult := *submission.ToolResult
		toolResult.Output = normalized
		submission.ToolResult = &toolResult

		d.result.setToolModelOutput(submission.ToolCallID, normalized)
	}

	if d.setup.hooks.turnSuspending() {
		return d.recordSuspendedResult(submission)
	}

	if !d.settledHostToolCalls[submission.ToolCallID] {
		d.hostToolCallsAwaitingEcho[submission.ToolCallID] = true
	}

	if err := d.control.SubmitToolResult(ctx, submission); err != nil {
		return fmt.Errorf("submit tool result %q: %w", submission.ToolCallID, err)
	}

	return nil
}

// normalizeSubmittedOutput checks the model-facing output of a submission.
// A caller-provided result the runtime cannot represent falls back to the
// JSON text of the raw output.
func (d *turnDriver) normalizeSubmittedOutput(submission ToolResultSubmission) (model.ToolResultOutput, error) {
	normalized, err := NormalizeToolModelOutput(submission.ToolResult.Output)
	if err == nil {
		return normalized, nil
	}

	_, fromContinuation := d.resultContinuations[submission.ToolCallID]
	_, unsupported := errors.AsType[*CapabilityUnsupportedError](err)

	if !fromContinuation || !unsupported {
		return nil, err
	}

	encoded, encodeErr := json.Marshal(submission.Output)
	if encodeErr != nil {
		encoded = []byte("null")
	}

	return model.TextOutput{
		Value: string(encoded),
	}, nil
}

// recordSuspendedResult records a result submitted while the turn is being
// suspended, so that the continuation submits it without running the tool
// again. The caller holds mu.
func (d *turnDriver) recordSuspendedResult(submission ToolResultSubmission) error {
	toolName, input, found := d.suspendedToolCall(submission.ToolCallID)
	if !found {
		return fmt.Errorf("unknown suspended tool call %q", submission.ToolCallID)
	}

	d.setup.hooks.pendingToolResult(PendingToolResult{
		ToolCallID: submission.ToolCallID,
		ToolName:   toolName,
		Input:      input,
		CompletedResult: &CompletedToolResult{
			Output:     submission.Output,
			IsError:    submission.IsError,
			ToolResult: submission.ToolResult,
		},
	})

	parsed, known := d.toolCalls[submission.ToolCallID]
	if !known || d.settledHostToolCalls[submission.ToolCallID] {
		return nil
	}

	outcome := hostToolOutcome{
		output: submission.Output,
	}

	if submission.IsError {
		outcome = hostToolOutcome{
			err: outputError(submission.Output),
		}
	}

	d.bufferedToolOutcomes = append(d.bufferedToolOutcomes, func() {
		d.enqueueHostToolOutcome(parsed, outcome)
	})

	return nil
}

// suspendedToolCall returns the name and input of a tool call the turn knows
// about.
func (d *turnDriver) suspendedToolCall(toolCallID string) (string, string, bool) {
	if raw, ok := d.rawToolCalls[toolCallID]; ok {
		return raw.ToolName, raw.Input, true
	}

	if pending, ok := d.pendingResults[toolCallID]; ok {
		return pending.ToolName, pending.Input, true
	}

	for _, approval := range d.setup.pendingToolApprovals {
		if approval.ToolCallID == toolCallID {
			return approval.ToolName, approval.Input, true
		}
	}

	return "", "", false
}

// recordPendingToolResult records a client tool call that waits for a
// caller-provided result.
func (d *turnDriver) recordPendingToolResult(call ToolCallPart) {
	pending, exists := d.pendingResults[call.ToolCallID]
	if !exists {
		pending = PendingToolResult{
			ToolCallID:      call.ToolCallID,
			ToolName:        call.ToolName,
			Input:           call.Input,
			ProviderOptions: call.ProviderMetadata.Options(),
		}
	}

	d.pendingResults[pending.ToolCallID] = pending
	d.setup.hooks.pendingToolResult(pending)
}

// enqueueHostToolOutcome forwards the outcome of a host tool, with the
// working directory stripped from its output.
func (d *turnDriver) enqueueHostToolOutcome(call ai.ToolCall, outcome hostToolOutcome) {
	if outcome.err != nil {
		d.result.enqueueContinuation(ai.ToolError{
			ToolCallID: call.ToolCallID,
			ToolName:   call.ToolName,
			Input:      call.Input,
			Error:      outcome.err,
		})

		return
	}

	d.result.enqueueContinuation(ai.ToolResult{
		ToolCallID: call.ToolCallID,
		ToolName:   call.ToolName,
		Input:      call.Input,
		Output:     stripWorkDirOutput(outcome.output, d.setup.sessionWorkDir),
	})
}

// outputError describes a failed tool output as an error: text as is and
// other values as JSON.
func outputError(output any) error {
	if text, ok := output.(string); ok {
		return errors.New(text)
	}

	encoded, err := json.Marshal(output)
	if err != nil {
		return fmt.Errorf("tool failed: %v", output)
	}

	return errors.New(string(encoded))
}

// toolOutputOf converts a host tool outcome into the output of its call.
func toolOutputOf(call ai.ToolCall, outcome hostToolOutcome) ai.ToolOutput {
	if outcome.err != nil {
		return ai.ToolError{
			ToolCallID:       call.ToolCallID,
			ToolName:         call.ToolName,
			Input:            call.Input,
			Error:            outcome.err,
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
		Output:           outcome.output,
		ProviderExecuted: call.ProviderExecuted,
		Dynamic:          call.Dynamic,
		Title:            call.Title,
		ProviderMetadata: call.ProviderMetadata,
		ToolMetadata:     call.ToolMetadata,
	}
}

// resolveCustomToolApproval returns the configured approval decision of an
// application tool. Unlisted tools need no approval.
func resolveCustomToolApproval(toolName string, approvals map[string]ai.ToolApprovalStatus) ai.ToolApprovalStatus {
	status, ok := approvals[toolName]
	if !ok || status.Type == "" {
		return ai.ToolApprovalStatus{
			Type:   ai.ToolApprovalNotApplicable,
			Reason: status.Reason,
		}
	}

	return status
}
