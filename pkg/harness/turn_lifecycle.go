package harness

import (
	"context"
	"strconv"
	"sync"
	"time"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// harnessOperationID identifies harness turns in telemetry.
const harnessOperationID = "ai.harness"

// lifecycleCallbacks are the callbacks of a turn: the agent's settings
// first, then the call's.
type lifecycleCallbacks struct {
	onStart                  []func(ctx context.Context, event ai.GenerateTextStartEvent)
	onStepStart              []func(ctx context.Context, event ai.GenerateTextStepStartEvent)
	onLanguageModelCallStart []func(ctx context.Context, event ai.LanguageModelCallStartEvent)
	onLanguageModelCallEnd   []func(ctx context.Context, event ai.LanguageModelCallEndEvent)
	onToolExecutionStart     []func(ctx context.Context, event ai.ToolExecutionStartEvent)
	onToolExecutionEnd       []func(ctx context.Context, event ai.ToolExecutionEndEvent)
	onStepEnd                []func(ctx context.Context, step ai.StepResult)
	onEnd                    []func(ctx context.Context, event ai.GenerateTextEndEvent)
}

// turnLifecycle drives the lifecycle callbacks and telemetry of one turn
// with the event shapes of StreamText: one operation, a span per step with
// one model call, and a span per tool execution. It is safe for concurrent
// use; host tools report their executions from their own goroutines.
type turnLifecycle struct {
	mu sync.Mutex

	callID   string
	provider string
	modelID  string

	callbacks lifecycleCallbacks
	telemetry telemetryDispatcher

	instructions    string
	messages        []model.Message
	tools           []ai.Tool
	activeToolNames []string
	modelTools      []model.Tool
	output          ai.Output

	started            bool
	stepOpen           bool
	ended              bool
	stepNumber         int
	modelCallStartedAt time.Time
	completedSteps     []ai.StepResult
}

// turnLifecycleSetup configures a turn lifecycle.
type turnLifecycleSetup struct {
	callID          string
	harnessID       string
	modelID         string
	callbacks       lifecycleCallbacks
	telemetry       ai.TelemetryOptions
	instructions    string
	messages        []model.Message
	tools           []ai.Tool
	activeToolNames []string
	toolSpecs       []ToolSpec
	output          ai.Output
}

// newTurnLifecycle returns the lifecycle of a turn.
func newTurnLifecycle(setup turnLifecycleSetup) *turnLifecycle {
	return &turnLifecycle{
		callID:          setup.callID,
		provider:        "harness:" + setup.harnessID,
		modelID:         setup.modelID,
		callbacks:       setup.callbacks,
		telemetry:       newTelemetryDispatcher(setup.telemetry),
		instructions:    setup.instructions,
		messages:        setup.messages,
		tools:           setup.tools,
		activeToolNames: setup.activeToolNames,
		modelTools:      modelToolsOf(setup.toolSpecs),
		output:          setup.output,
	}
}

// start reports the start of the operation once, using modelID when the
// runtime resolved one.
func (l *turnLifecycle) start(ctx context.Context, modelID string) {
	l.mu.Lock()
	if l.started {
		l.mu.Unlock()

		return
	}

	if modelID != "" {
		l.modelID = modelID
	}

	l.started = true
	event := ai.GenerateTextStartEvent{
		CallID:      l.callID,
		OperationID: harnessOperationID,
		Provider:    l.provider,
		ModelID:     l.modelID,
		System:      l.instructions,
		Messages:    l.messages,
		Tools:       l.tools,
		ActiveTools: l.activeToolNames,
		Output:      l.output,
	}
	l.mu.Unlock()

	for _, callback := range l.callbacks.onStart {
		callback(ctx, event)
	}

	l.telemetry.onStart(ctx, event)
}

// ensureStepOpen opens a step lazily before its first content, reporting the
// step start and its model call start.
func (l *turnLifecycle) ensureStepOpen(ctx context.Context) {
	l.start(ctx, "")

	l.mu.Lock()
	if l.stepOpen || l.ended {
		l.mu.Unlock()

		return
	}

	l.stepOpen = true
	l.modelCallStartedAt = time.Now()
	stepEvent := ai.GenerateTextStepStartEvent{
		CallID:      l.callID,
		Provider:    l.provider,
		ModelID:     l.modelID,
		StepNumber:  l.stepNumber,
		System:      l.instructions,
		Messages:    l.messages,
		Tools:       l.tools,
		ActiveTools: l.activeToolNames,
		Steps:       append([]ai.StepResult(nil), l.completedSteps...),
		Output:      l.output,
		StepTools:   l.modelTools,
	}
	modelEvent := ai.LanguageModelCallStartEvent{
		CallID:   l.callID,
		Provider: l.provider,
		ModelID:  l.modelID,
		System:   l.instructions,
		Messages: l.messages,
		Tools:    l.modelTools,
	}
	l.mu.Unlock()

	for _, callback := range l.callbacks.onStepStart {
		callback(ctx, stepEvent)
	}

	l.telemetry.onStepStart(ctx, stepEvent)

	for _, callback := range l.callbacks.onLanguageModelCallStart {
		callback(ctx, modelEvent)
	}

	l.telemetry.onLanguageModelCallStart(ctx, modelEvent)
}

// modelCallEnd is the outcome of a step's model call.
type modelCallEnd struct {
	finishReason model.UnifiedFinishReason
	usage        ai.LanguageModelUsage
	content      []ai.ContentPart
	metadata     model.ProviderMetadata
}

// languageModelCallEnd reports the end of the current step's model call.
func (l *turnLifecycle) languageModelCallEnd(ctx context.Context, end modelCallEnd) {
	l.ensureStepOpen(ctx)

	l.mu.Lock()
	event := ai.LanguageModelCallEndEvent{
		CallID:           l.callID,
		Provider:         l.provider,
		ModelID:          l.modelID,
		FinishReason:     end.finishReason,
		Usage:            end.usage,
		Content:          end.content,
		ResponseID:       stepResponseID(l.callID, l.stepNumber),
		ProviderMetadata: end.metadata,
		Performance: ai.LanguageModelCallPerformance{
			ResponseTime: time.Since(l.modelCallStartedAt),
		},
	}
	l.mu.Unlock()

	for _, callback := range l.callbacks.onLanguageModelCallEnd {
		callback(ctx, event)
	}

	l.telemetry.onLanguageModelCallEnd(ctx, event)
}

// toolExecutionStart reports a tool execution about to be reported.
func (l *turnLifecycle) toolExecutionStart(ctx context.Context, call ai.ToolCall) {
	event := ai.ToolExecutionStartEvent{
		CallID:   l.callID,
		ToolCall: call,
		Messages: l.messages,
	}

	for _, callback := range l.callbacks.onToolExecutionStart {
		callback(ctx, event)
	}

	l.telemetry.onToolExecutionStart(ctx, event)
}

// toolExecutionEnd reports a finished tool execution.
func (l *turnLifecycle) toolExecutionEnd(ctx context.Context, call ai.ToolCall, output ai.ToolOutput, duration time.Duration) {
	event := ai.ToolExecutionEndEvent{
		CallID:        l.callID,
		ToolCall:      call,
		Messages:      l.messages,
		Output:        output,
		ExecutionTime: duration,
	}

	for _, callback := range l.callbacks.onToolExecutionEnd {
		callback(ctx, event)
	}

	l.telemetry.onToolExecutionEnd(ctx, event)
}

// stepEnd reports a completed step and closes it.
func (l *turnLifecycle) stepEnd(ctx context.Context, step ai.StepResult) {
	l.mu.Lock()
	if !l.stepOpen {
		l.mu.Unlock()

		return
	}

	l.completedSteps = append(l.completedSteps, step)
	l.stepOpen = false
	l.stepNumber++
	l.mu.Unlock()

	for _, callback := range l.callbacks.onStepEnd {
		callback(ctx, step)
	}

	l.telemetry.onStepEnd(ctx, step)
}

// end reports the completed operation once. It does nothing without steps.
func (l *turnLifecycle) end(ctx context.Context, steps []ai.StepResult, usage ai.LanguageModelUsage) {
	l.start(ctx, "")

	l.mu.Lock()
	if l.ended || len(steps) == 0 {
		l.mu.Unlock()

		return
	}

	l.ended = true
	l.mu.Unlock()

	event := ai.GenerateTextEndEvent{
		CallID:             l.callID,
		GenerateTextResult: *summarizeSteps(steps, steps[len(steps)-1], usage),
	}

	for _, callback := range l.callbacks.onEnd {
		callback(ctx, event)
	}

	l.telemetry.onEnd(ctx, event)
}

// abort reports a cancelled operation once.
func (l *turnLifecycle) abort(ctx context.Context, reason error) {
	if !l.markEnded(ctx) {
		return
	}

	l.mu.Lock()
	event := ai.StreamTextAbortEvent{
		CallID: l.callID,
		Steps:  append([]ai.StepResult(nil), l.completedSteps...),
		Reason: reason,
	}
	l.mu.Unlock()

	l.telemetry.onAbort(ctx, event)
}

// fail reports a failed operation once.
func (l *turnLifecycle) fail(ctx context.Context, err error) {
	if !l.markEnded(ctx) {
		return
	}

	l.telemetry.onError(ctx, ai.TelemetryErrorEvent{
		CallID: l.callID,
		Error:  err,
	})
}

// markEnded starts the operation if needed and marks it ended. It reports
// false when it had already ended.
func (l *turnLifecycle) markEnded(ctx context.Context) bool {
	l.start(ctx, "")

	l.mu.Lock()
	defer l.mu.Unlock()

	if l.ended {
		return false
	}

	l.ended = true

	return true
}

// executeTool runs a tool execution inside the telemetry integrations'
// ExecuteTool wrappers.
func (l *turnLifecycle) executeTool(ctx context.Context, call ai.ToolCall, execute func(ctx context.Context) error) error {
	return l.telemetry.executeTool(ctx, ai.ToolExecutionStartEvent{
		CallID:   l.callID,
		ToolCall: call,
		Messages: l.messages,
	}, execute)
}

// stepResponseID returns the response ID of a step.
func stepResponseID(callID string, stepNumber int) string {
	return callID + "-" + strconv.Itoa(stepNumber)
}

// modelToolsOf declares tool specs as function tools for model call events.
func modelToolsOf(specs []ToolSpec) []model.Tool {
	if len(specs) == 0 {
		return nil
	}

	tools := make([]model.Tool, 0, len(specs))
	for _, spec := range specs {
		tools = append(tools, model.FunctionTool{
			Name:        spec.Name,
			Description: spec.Description,
			InputSchema: spec.InputSchema,
		})
	}

	return tools
}
