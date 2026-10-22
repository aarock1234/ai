package harness

import (
	"context"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/sandbox"
)

// turnMode selects the session method that runs a turn.
type turnMode int

const (
	// turnPrompt starts a fresh turn with Session.PromptTurn.
	turnPrompt turnMode = iota

	// turnContinue continues the in-flight turn with Session.ContinueTurn.
	turnContinue
)

// turnSetup is everything one turn needs. It is the input of the upstream
// runPrompt driver.
type turnSetup struct {
	harness Harness
	session Session
	mode    turnMode

	// prompt is the user input of a fresh turn.
	prompt model.UserMessage

	// settings are the persisted settings of the turn.
	settings TurnSettings

	// tools are the merged built-in and application tools, for validation.
	tools []ai.Tool

	// activeTools are the application tools the runtime may call.
	activeTools []ai.Tool

	builtinToolFiltering *BuiltinToolFiltering

	// sandbox is the restricted sandbox view handed to host tools.
	sandbox        sandbox.Session
	sessionWorkDir string

	responseFormat *model.ResponseFormat
	output         ai.Output
	telemetry      ai.TelemetryOptions
	callbacks      lifecycleCallbacks
	stopConditions []ai.StopCondition
	toolApproval   map[string]ai.ToolApprovalStatus

	pendingToolApprovals  []PendingToolApproval
	pendingToolResults    []PendingToolResult
	approvalContinuations []model.ToolApprovalResponsePart
	resultContinuations   []model.ToolResultPart

	hooks turnHooks
}

// turnHooks connect a turn to the session that tracks it. Every hook is
// optional.
type turnHooks struct {
	onPendingToolApproval    func(approval PendingToolApproval)
	onToolApprovalSettled    func(approvalID string)
	onPendingToolResult      func(pending PendingToolResult)
	onToolResultSettled      func(toolCallID string)
	onTurnFinished           func()
	onTurnFailed             func()
	onPromptControlAvailable func(control PromptControl)

	// isTurnSuspending reports that the adapter's stream ended because the
	// session suspended the still-running turn.
	isTurnSuspending func() bool

	// onStopConditionMet suspends the turn at a stop condition boundary.
	onStopConditionMet func(ctx context.Context) error
}

// pendingToolApproval calls the hook if set.
func (h turnHooks) pendingToolApproval(approval PendingToolApproval) {
	if h.onPendingToolApproval != nil {
		h.onPendingToolApproval(approval)
	}
}

// toolApprovalSettled calls the hook if set.
func (h turnHooks) toolApprovalSettled(approvalID string) {
	if h.onToolApprovalSettled != nil {
		h.onToolApprovalSettled(approvalID)
	}
}

// pendingToolResult calls the hook if set.
func (h turnHooks) pendingToolResult(pending PendingToolResult) {
	if h.onPendingToolResult != nil {
		h.onPendingToolResult(pending)
	}
}

// toolResultSettled calls the hook if set.
func (h turnHooks) toolResultSettled(toolCallID string) {
	if h.onToolResultSettled != nil {
		h.onToolResultSettled(toolCallID)
	}
}

// turnFinished calls the hook if set.
func (h turnHooks) turnFinished() {
	if h.onTurnFinished != nil {
		h.onTurnFinished()
	}
}

// turnFailed calls the hook if set.
func (h turnHooks) turnFailed() {
	if h.onTurnFailed != nil {
		h.onTurnFailed()
	}
}

// promptControlAvailable calls the hook if set.
func (h turnHooks) promptControlAvailable(control PromptControl) {
	if h.onPromptControlAvailable != nil {
		h.onPromptControlAvailable(control)
	}
}

// turnSuspending reports whether the session is suspending the turn.
func (h turnHooks) turnSuspending() bool {
	return h.isTurnSuspending != nil && h.isTurnSuspending()
}

// stopConditionMet calls the hook if set.
func (h turnHooks) stopConditionMet(ctx context.Context) error {
	if h.onStopConditionMet == nil {
		return nil
	}

	return h.onStopConditionMet(ctx)
}
