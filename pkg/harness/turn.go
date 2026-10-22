package harness

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// errTurnSettled reports that the driver already settled the result.
var errTurnSettled = errors.New("harness turn settled")

// turnDriver drives one turn end to end, the Go form of upstream runPrompt:
// it starts the adapter's turn, translates its events into ai stream parts,
// runs host tools when the runtime calls them and submits their results,
// handles approvals and continuations, and settles the result.
//
// The driver goroutine holds mu while it processes an event and releases it
// while it waits for events or host tools. Host tool goroutines take mu to
// update the shared state below.
type turnDriver struct {
	mu sync.Mutex

	setup     turnSetup
	result    *turnResult
	lifecycle *turnLifecycle
	parser    toolCallParser
	stripper  *toolInputStripper
	control   PromptControl

	toolCalls    map[string]ai.ToolCall
	rawToolCalls map[string]ToolCallPart

	pendingApprovalsByApprovalID map[string]PendingToolApproval
	pendingApprovalsByToolCallID map[string]PendingToolApproval
	approvalContinuations        map[string]model.ToolApprovalResponsePart
	pendingResults               map[string]PendingToolResult
	resultContinuations          map[string]model.ToolResultPart

	settledHostToolCalls      map[string]bool
	settledBuiltinApprovals   map[string]bool
	hostToolCallsAwaitingEcho map[string]bool

	closingResumedStep bool
	releaseStopPin     func()
	stopPending        bool
	finalFinish        *FinishPart
	completedSteps     []ai.StepResult
	outstanding        []*hostExecution

	stepText                strings.Builder
	stepReasoning           strings.Builder
	stepToolCalls           []ai.ContentPart
	stepProviderToolResults []ai.ContentPart
	stepApprovalRequests    []ai.ContentPart
	bufferedToolOutcomes    []func()
	toolExecutions          []*toolExecution

	hostInputPauseRequested bool
	deferredPauseFinishStep *FinishStepPart
	ignoredFinishStep       *FinishStepPart
}

// toolExecution tracks the lifecycle notifications of one tool call in the
// current step.
type toolExecution struct {
	call      ai.ToolCall
	output    ai.ToolOutput
	duration  time.Duration
	startedAt time.Time

	startNotified bool
	endNotified   bool
}

// hostExecution is a host tool running in its own goroutine.
type hostExecution struct {
	done chan struct{}
	err  error
}

// hostToolOutcome is the outcome of a host tool execution.
type hostToolOutcome struct {
	output any
	err    error
}

// startTurn starts a turn in a goroutine owned by the returned result.
func startTurn(ctx context.Context, setup turnSetup) *StreamResult {
	turnCtx, cancel := context.WithCancelCause(ctx)
	callID := id.New("")

	driver := newTurnDriver(setup, callID)
	stream := &StreamResult{
		result: driver.result,
		tools:  setup.tools,
		cancel: cancel,
	}

	stream.driver.Go(func() {
		defer cancel(nil)

		driver.run(turnCtx)
	})

	return stream
}

// newTurnDriver prepares the driver state of a turn.
func newTurnDriver(setup turnSetup, callID string) *turnDriver {
	driver := &turnDriver{
		setup: setup,
		result: newTurnResult(turnResultSetup{
			tools:     setup.tools,
			output:    setup.output,
			harnessID: setup.harness.HarnessID(),
			callID:    callID,
			modelID:   setup.settings.Model,
		}),
		lifecycle: newTurnLifecycle(turnLifecycleSetup{
			callID:          callID,
			harnessID:       setup.harness.HarnessID(),
			modelID:         setup.settings.Model,
			callbacks:       setup.callbacks,
			telemetry:       setup.telemetry,
			instructions:    setup.settings.Instructions,
			messages:        promptMessages(setup),
			tools:           setup.tools,
			activeToolNames: activeToolNames(setup),
			toolSpecs:       setup.settings.Tools,
			output:          setup.output,
		}),
		stripper:                     newToolInputStripper(setup.sessionWorkDir),
		toolCalls:                    make(map[string]ai.ToolCall),
		rawToolCalls:                 make(map[string]ToolCallPart),
		pendingApprovalsByApprovalID: make(map[string]PendingToolApproval),
		pendingApprovalsByToolCallID: make(map[string]PendingToolApproval),
		approvalContinuations:        make(map[string]model.ToolApprovalResponsePart),
		pendingResults:               make(map[string]PendingToolResult),
		resultContinuations:          make(map[string]model.ToolResultPart),
		settledHostToolCalls:         make(map[string]bool),
		settledBuiltinApprovals:      make(map[string]bool),
		hostToolCallsAwaitingEcho:    make(map[string]bool),
	}

	for _, approval := range setup.pendingToolApprovals {
		driver.pendingApprovalsByApprovalID[approval.ApprovalID] = approval
		driver.pendingApprovalsByToolCallID[approval.ToolCallID] = approval
	}

	for _, continuation := range setup.approvalContinuations {
		driver.approvalContinuations[continuation.ApprovalID] = continuation
	}

	for _, pending := range setup.pendingToolResults {
		driver.pendingResults[pending.ToolCallID] = pending
	}

	for _, continuation := range setup.resultContinuations {
		driver.resultContinuations[continuation.ToolCallID] = continuation
	}

	return driver
}

// run drives the turn and settles its result.
func (d *turnDriver) run(ctx context.Context) {
	d.mu.Lock()
	defer d.mu.Unlock()

	defer func() {
		_ = d.joinHostToolsReleasingLock()
	}()

	events, err := d.startAdapterTurn(ctx)
	if err != nil {
		d.settleLifecycle(ctx, err)
		d.logError(ctx, "failed to start harness turn", err)
		d.settleFailure(ctx, err)

		return
	}

	defer events.release()
	defer d.releaseStopBoundary()

	err = d.drive(ctx, events)
	if err == nil || errors.Is(err, errTurnSettled) {
		return
	}

	_ = d.joinHostToolsReleasingLock()

	d.settleLifecycle(ctx, err)
	d.logError(ctx, "harness turn failed", err)
	d.settleFailure(ctx, err)
}

// startAdapterTurn calls the adapter's prompt or continue entry point and
// connects its events.
func (d *turnDriver) startAdapterTurn(ctx context.Context) (*turnEvents, error) {
	events := newTurnEvents()

	control, err := d.invokeAdapter(ctx, events)
	if err != nil {
		return nil, err
	}

	if control == nil {
		return nil, fmt.Errorf("harness %q returned no prompt control", d.setup.harness.HarnessID())
	}

	d.control = control
	events.watch(control)
	d.setup.hooks.promptControlAvailable(control)

	return events, nil
}

// invokeAdapter calls PromptTurn or ContinueTurn.
func (d *turnDriver) invokeAdapter(ctx context.Context, events *turnEvents) (PromptControl, error) {
	if d.setup.mode == turnContinue {
		return d.setup.session.ContinueTurn(ctx, ContinueTurnOptions{
			TurnSettings:   d.setup.settings,
			ResponseFormat: d.setup.responseFormat,
			Emit:           events.emit,
		})
	}

	return d.setup.session.PromptTurn(ctx, PromptTurnOptions{
		TurnSettings:   d.setup.settings,
		Prompt:         d.setup.prompt,
		ResponseFormat: d.setup.responseFormat,
		Emit:           events.emit,
	})
}

// drive processes the continuations of a resumed turn and then the turn's
// events. It returns errTurnSettled when it settled the result itself.
func (d *turnDriver) drive(ctx context.Context, events *turnEvents) error {
	settled, err := d.processContinuations(ctx)
	if err != nil || settled {
		return settledOr(err, settled)
	}

	for {
		if d.deferredPauseFinishStep != nil && len(d.hostToolCallsAwaitingEcho) == 0 {
			return d.pauseForHostInput(ctx, d.deferredPauseFinishStep)
		}

		part, ok, err := d.nextEvent(ctx, events)
		if err != nil {
			return err
		}

		if !ok {
			d.releaseStopBoundary()

			break
		}

		settled, err := d.handleEvent(ctx, part)
		if err != nil || settled {
			return settledOr(err, settled)
		}
	}

	return d.finishTurn(ctx)
}

// processContinuations submits the decisions and results that continue a
// suspended turn. It reports true when the turn paused again.
func (d *turnDriver) processContinuations(ctx context.Context) (bool, error) {
	for _, approval := range d.setup.pendingToolApprovals {
		continuation, ok := d.approvalContinuations[approval.ApprovalID]
		if !ok {
			continue
		}

		paused, err := d.processApprovalContinuation(ctx, approval, continuation)
		if err != nil || paused {
			return paused, err
		}

		d.closingResumedStep = true
	}

	for _, pending := range d.setup.pendingToolResults {
		continuation, hasContinuation := d.resultContinuations[pending.ToolCallID]
		if !hasContinuation && pending.CompletedResult == nil {
			continue
		}

		var supplied *model.ToolResultPart
		if hasContinuation {
			supplied = &continuation
		}

		if err := d.processResultContinuation(ctx, pending, supplied); err != nil {
			return false, err
		}

		if pending.CompletedResult == nil {
			d.closingResumedStep = true
		}
	}

	return false, nil
}

// nextEvent waits for the next adapter event without holding mu.
func (d *turnDriver) nextEvent(ctx context.Context, events *turnEvents) (StreamPart, bool, error) {
	d.mu.Unlock()
	defer d.mu.Lock()

	return events.next(ctx)
}

// finishTurn settles the result after the adapter's stream ended.
func (d *turnDriver) finishTurn(ctx context.Context) error {
	suspending := d.setup.hooks.turnSuspending()

	if d.hostInputPauseRequested && !suspending {
		return d.pauseForHostInput(ctx, d.deferredPauseFinishStep)
	}

	if err := d.joinHostToolsReleasingLock(); err != nil {
		return err
	}

	switch {
	case suspending:
		d.publishToolExecutions(ctx)

		if d.finalFinish == nil {
			d.result.discardCurrentStepContent()
		}
	case d.finalFinish != nil:
		d.setup.hooks.turnFinished()
	default:
		d.setup.hooks.turnFailed()
	}

	d.result.finish(ctx, d.terminalFinish())

	return nil
}

// terminalFinish returns the adapter's terminal finish, or nil.
func (d *turnDriver) terminalFinish() *turnFinish {
	if d.finalFinish == nil {
		return nil
	}

	return &turnFinish{
		finishReason: d.finalFinish.FinishReason,
		totalUsage:   d.finalFinish.TotalUsage,
		metadata:     providerMetadata(d.finalFinish.HarnessMetadata),
	}
}

// settleFailure settles a failed turn. A turn whose context was cancelled
// ends with an ai.AbortPart, as StreamText does, instead of an error part.
// Both notify the session so that it returns to idle, unless the turn is
// being suspended.
func (d *turnDriver) settleFailure(ctx context.Context, err error) {
	if !d.setup.hooks.turnSuspending() {
		d.setup.hooks.turnFailed()
	}

	if ctx.Err() != nil {
		d.result.abort(context.Cause(ctx))

		return
	}

	d.result.fail(err)
}

// settleLifecycle reports a failed or cancelled turn to the lifecycle.
func (d *turnDriver) settleLifecycle(ctx context.Context, err error) {
	notifyCtx := context.WithoutCancel(ctx)

	if ctx.Err() != nil {
		d.lifecycle.abort(notifyCtx, context.Cause(ctx))

		return
	}

	d.lifecycle.fail(notifyCtx, err)
}

// logError logs a turn failure, except for the caller's own cancellation.
func (d *turnDriver) logError(ctx context.Context, message string, err error) {
	if ctx.Err() != nil {
		return
	}

	slog.Default().ErrorContext(
		ctx,
		message,
		slog.String("harness", d.setup.harness.HarnessID()),
		slog.String("session", d.setup.session.SessionID()),
		slog.Any("error", err),
	)
}

// releaseStopBoundary releases a pinned stop condition checkpoint.
func (d *turnDriver) releaseStopBoundary() {
	if d.releaseStopPin != nil {
		d.releaseStopPin()
	}

	d.releaseStopPin = nil
	d.stopPending = false
}

// settledOr returns err, or errTurnSettled when the turn settled.
func settledOr(err error, settled bool) error {
	if err != nil {
		return err
	}

	if settled {
		return errTurnSettled
	}

	return nil
}

// promptMessages returns the turn's prompt as messages for telemetry.
func promptMessages(setup turnSetup) []model.Message {
	if setup.mode == turnContinue {
		return nil
	}

	return []model.Message{
		setup.prompt,
	}
}

// activeToolNames lists the tools the runtime may call in the turn.
func activeToolNames(setup turnSetup) []string {
	var names []string

	for _, tool := range setup.tools {
		isActiveApplicationTool := hasTool(setup.activeTools, tool.Name)
		isIncludedBuiltin := isBuiltinTool(setup.harness, tool.Name) &&
			IsBuiltinToolIncluded(tool.Name, setup.builtinToolFiltering)

		if isActiveApplicationTool || isIncludedBuiltin {
			names = append(names, tool.Name)
		}
	}

	return names
}
