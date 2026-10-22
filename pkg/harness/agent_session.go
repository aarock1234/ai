package harness

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/sandbox"
)

// sessionState is the lifecycle state of an AgentSession.
type sessionState int

const (
	sessionActive sessionState = iota
	sessionDetached
	sessionStopped
	sessionDestroyed
)

// turnState is the state of a session's current turn.
type turnState int

const (
	turnIdle turnState = iota
	turnRunning
	turnAwaitingApproval
	turnAwaitingToolResult
	turnSuspended
)

// AgentSession is a live harness session created by Agent.CreateSession. It
// owns the adapter's session and holds its sandbox session. Pass it to every
// Generate and Stream call, and end it with Detach, Stop or Destroy; after
// that every turn on it fails. It is safe for concurrent use, but it runs one
// turn at a time.
type AgentSession struct {
	sessionID      string
	isResume       bool
	harness        Harness
	sessionWorkDir string
	toolApproval   map[string]ai.ToolApprovalStatus

	mu         sync.Mutex
	underlying Session
	sandbox    sandbox.Session
	state      sessionState
	turn       turnState

	turnSequence       int
	activeTurnSequence int
	activeTurn         *StreamResult
	activeControl      *promptControlWaiter
	suspension         *turnSuspension

	activeSettings    *activeTurnSettings
	persistedSettings *TurnSettings

	pendingApprovals orderedPending[PendingToolApproval]
	pendingResults   orderedPending[PendingToolResult]
}

// promptControlWaiter delivers the control of a turn once the adapter
// started it, or nil when the turn ended first.
type promptControlWaiter struct {
	turnID  int
	ready   chan struct{}
	control PromptControl
	settled bool
}

// turnSuspension is the shared outcome of suspending the current turn.
type turnSuspension struct {
	done  chan struct{}
	state ContinueTurnState
	err   error
}

// activeTurnSettings are the frozen settings of an unfinished turn.
type activeTurnSettings struct {
	persisted            TurnSettings
	tools                []ai.Tool
	activeTools          []ai.Tool
	builtinToolFiltering *BuiltinToolFiltering
}

// agentSessionSetup configures a new AgentSession.
type agentSessionSetup struct {
	sessionID      string
	harness        Harness
	underlying     Session
	sandbox        sandbox.Session
	sessionWorkDir string
	toolApproval   map[string]ai.ToolApprovalStatus

	// continueFrom is the unfinished turn the session resumes, if any.
	continueFrom *ContinueTurnState
}

// newAgentSession returns an active session.
func newAgentSession(setup agentSessionSetup) *AgentSession {
	session := &AgentSession{
		sessionID:      setup.sessionID,
		isResume:       setup.underlying.IsResume(),
		harness:        setup.harness,
		sessionWorkDir: setup.sessionWorkDir,
		toolApproval:   setup.toolApproval,
		underlying:     setup.underlying,
		sandbox:        setup.sandbox,
	}

	if setup.continueFrom == nil {
		return session
	}

	for _, approval := range setup.continueFrom.PendingToolApprovals {
		session.pendingApprovals.set(approval.ApprovalID, approval)
	}

	for _, pending := range setup.continueFrom.PendingToolResults {
		session.pendingResults.set(pending.ToolCallID, pending)
	}

	session.persistedSettings = setup.continueFrom.TurnSettings

	switch {
	case session.pendingApprovals.len() > 0:
		session.turn = turnAwaitingApproval
	case session.pendingResults.len() > 0:
		session.turn = turnAwaitingToolResult
	default:
		session.turn = turnSuspended
	}

	return session
}

// SessionID is the session identifier the adapter saw in Harness.Start.
// Persist it to resume the session in another process.
func (s *AgentSession) SessionID() string {
	return s.sessionID
}

// IsResume reports whether the session was created from ResumeFrom or
// ContinueFrom.
func (s *AgentSession) IsResume() bool {
	return s.isResume
}

// SessionWorkDir is the absolute directory the agent runs in.
func (s *AgentSession) SessionWorkDir() string {
	return s.sessionWorkDir
}

// HasUnfinishedTurn reports whether a turn is running, paused for host
// input or suspended.
func (s *AgentSession) HasUnfinishedTurn() bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.turn != turnIdle
}

// Compact asks the runtime to compact its context. The runtime reports the
// compaction as a compaction tool call on the active or next turn. Harnesses
// that cannot compact on request return *CapabilityUnsupportedError.
func (s *AgentSession) Compact(ctx context.Context, customInstructions string) error {
	underlying, err := s.reusableSession()
	if err != nil {
		return err
	}

	return underlying.Compact(ctx, customInstructions)
}

// SteerTurn submits another user message to the running turn. It returns once
// the runtime accepted the message for its next safe input boundary; output
// caused by it stays in the running turn's stream. This API is experimental.
func (s *AgentSession) SteerTurn(ctx context.Context, text string) error {
	waiter, err := s.steerableTurn()
	if err != nil {
		return err
	}

	select {
	case <-waiter.ready:
	case <-ctx.Done():
		return context.Cause(ctx)
	}

	s.mu.Lock()
	stillTargeted := s.state == sessionActive &&
		s.turn == turnRunning &&
		s.suspension == nil &&
		s.activeControl == waiter &&
		s.activeTurnSequence == waiter.turnID
	s.mu.Unlock()

	if waiter.control == nil || !stillTargeted {
		return fmt.Errorf("harness session %s no longer has the running turn targeted for steering", s.sessionID)
	}

	submitter, ok := waiter.control.(UserMessageSubmitter)
	if !ok {
		return &CapabilityUnsupportedError{
			Message:   fmt.Sprintf("Harness '%s' does not support steering active turns.", s.harness.HarnessID()),
			HarnessID: s.harness.HarnessID(),
		}
	}

	return submitter.SubmitUserMessage(ctx, text)
}

// steerableTurn returns the control waiter of the running turn.
func (s *AgentSession) steerableTurn() (*promptControlWaiter, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.state != sessionActive || s.underlying == nil {
		return nil, s.endedError()
	}

	if s.turn != turnRunning || s.suspension != nil || s.activeControl == nil {
		return nil, fmt.Errorf("harness session %s has no running turn to steer", s.sessionID)
	}

	return s.activeControl, nil
}

// ReadHistory reads the conversation history the runtime persisted,
// including exchanges outside this process. Pass the Cursor of a previous
// read as Since to read only what follows. Harnesses without history reads
// return *CapabilityUnsupportedError; harnesses that cannot reach their
// store return *HistoryUnavailableError.
func (s *AgentSession) ReadHistory(ctx context.Context, options ReadHistoryOptions) (ReadHistoryResult, error) {
	s.mu.Lock()
	underlying := s.underlying
	active := s.state == sessionActive && underlying != nil
	s.mu.Unlock()

	if !active {
		return ReadHistoryResult{}, fmt.Errorf("harness session %s is not active and cannot read history", s.sessionID)
	}

	reader, ok := underlying.(HistoryReader)
	if !ok {
		return ReadHistoryResult{}, &CapabilityUnsupportedError{
			Message:   fmt.Sprintf("Harness '%s' does not support reading the runtime's conversation history.", s.harness.HarnessID()),
			HarnessID: s.harness.HarnessID(),
		}
	}

	return reader.ReadHistory(ctx, options)
}

// Detach parks the session and returns the state to pass to
// Agent.CreateSession with the same session ID to reconnect. The runtime and
// sandbox keep running; an unfinished turn is suspended and stored in the
// state's ContinueFrom.
func (s *AgentSession) Detach(ctx context.Context) (ResumeSessionState, error) {
	underlying, unfinished, err := s.lifecycleTarget("detached")
	if err != nil {
		return ResumeSessionState{}, err
	}

	var state ResumeSessionState

	if unfinished {
		continueFrom, err := s.finalizeSuspension(ctx, underlying)
		if err != nil {
			return ResumeSessionState{}, err
		}

		state = resumeStateWithContinuation(continueFrom)
	} else {
		detached, err := underlying.Detach(ctx)
		if err != nil {
			return ResumeSessionState{}, err
		}

		state, err = validateResumeSessionState(s.harness, detached)
		if err != nil {
			return ResumeSessionState{}, err
		}
	}

	s.endLocalHandle(sessionDetached)

	return state, nil
}

// Stop persists the state to resume later, then stops the runtime. An
// unfinished turn is suspended instead and stored in the state's
// ContinueFrom. The sandbox is not stopped; it belongs to the caller.
func (s *AgentSession) Stop(ctx context.Context) (ResumeSessionState, error) {
	underlying, unfinished, err := s.lifecycleTarget("stopped")
	if err != nil {
		return ResumeSessionState{}, err
	}

	defer s.endLocalHandle(sessionStopped)

	if unfinished {
		continueFrom, err := s.finalizeSuspension(ctx, underlying)
		if err != nil {
			return ResumeSessionState{}, err
		}

		return resumeStateWithContinuation(continueFrom), nil
	}

	stopped, err := underlying.Stop(ctx)
	if err != nil {
		return ResumeSessionState{}, err
	}

	return validateResumeSessionState(s.harness, stopped)
}

// Destroy stops the runtime and discards resumability. It does nothing when
// the session already ended. The sandbox belongs to the caller.
func (s *AgentSession) Destroy(ctx context.Context) error {
	s.mu.Lock()
	if s.state != sessionActive {
		s.mu.Unlock()

		return nil
	}

	underlying := s.underlying
	s.endLocalHandleLocked(sessionDestroyed)
	s.mu.Unlock()

	if underlying == nil {
		return nil
	}

	if err := underlying.Destroy(ctx); err != nil {
		return fmt.Errorf("destroy harness session %s: %w", s.sessionID, err)
	}

	return nil
}

// SuspendTurn freezes the active turn at a precise cursor and returns the
// state to continue it, leaving the runtime and sandbox running. It returns
// once the running stream wound down. Afterwards this handle no longer
// drives turns: create a session from the returned state to continue.
func (s *AgentSession) SuspendTurn(ctx context.Context) (ContinueTurnState, error) {
	underlying, unfinished, err := s.lifecycleTarget("suspended")
	if err != nil {
		return ContinueTurnState{}, err
	}

	if !unfinished {
		return ContinueTurnState{}, fmt.Errorf("harness session %s has no unfinished turn to suspend", s.sessionID)
	}

	defer s.endLocalHandle(sessionDetached)

	return s.finalizeSuspension(ctx, underlying)
}

// lifecycleTarget returns the adapter session of an active session and
// whether it has an unfinished turn.
func (s *AgentSession) lifecycleTarget(action string) (Session, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.state != sessionActive || s.underlying == nil {
		return nil, false, fmt.Errorf("harness session %s is not active and cannot be %s", s.sessionID, action)
	}

	return s.underlying, s.turn != turnIdle, nil
}

// finalizeSuspension suspends the current turn, waits until the running
// stream delivered every dispatched event, and records the pending host
// work in the state.
func (s *AgentSession) finalizeSuspension(ctx context.Context, underlying Session) (ContinueTurnState, error) {
	state, err := s.suspendCurrentTurn(ctx, underlying)
	if err != nil {
		return ContinueTurnState{}, err
	}

	s.mu.Lock()
	active := s.activeTurn
	s.mu.Unlock()

	if active != nil {
		active.driver.Wait()
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	return s.withPendingToolState(state), nil
}

// suspendCurrentTurn suspends the adapter's turn once; concurrent callers
// share the outcome.
func (s *AgentSession) suspendCurrentTurn(ctx context.Context, underlying Session) (ContinueTurnState, error) {
	s.mu.Lock()
	s.clearActivePromptControlLocked(0)

	suspension := s.suspension
	owner := suspension == nil

	if owner {
		suspension = &turnSuspension{
			done: make(chan struct{}),
		}
		s.suspension = suspension
	}
	s.mu.Unlock()

	if owner {
		suspension.state, suspension.err = s.suspendAdapterTurn(ctx, underlying)
		close(suspension.done)
	}

	select {
	case <-suspension.done:
	case <-ctx.Done():
		return ContinueTurnState{}, context.Cause(ctx)
	}

	if suspension.err != nil {
		return ContinueTurnState{}, suspension.err
	}

	s.mu.Lock()
	s.turn = turnSuspended
	s.mu.Unlock()

	return suspension.state, nil
}

// suspendAdapterTurn suspends the adapter's turn and validates its state.
func (s *AgentSession) suspendAdapterTurn(ctx context.Context, underlying Session) (ContinueTurnState, error) {
	suspended, err := underlying.SuspendTurn(ctx)
	if err != nil {
		return ContinueTurnState{}, err
	}

	validated, err := validateContinueTurnState(s.harness, suspended)
	if err != nil {
		return ContinueTurnState{}, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	return s.withPendingToolState(validated), nil
}

// captureStopConditionBoundary suspends the turn at a stop condition, unless
// the session already moved on.
func (s *AgentSession) captureStopConditionBoundary(ctx context.Context, turnID int) error {
	s.mu.Lock()
	underlying := s.underlying
	current := s.state == sessionActive && s.activeTurnSequence == turnID
	s.mu.Unlock()

	if !current || underlying == nil {
		return nil
	}

	_, err := s.suspendCurrentTurn(ctx, underlying)

	return err
}

// withPendingToolState adds the agent-owned pending work and turn settings
// to a continuation state. The caller holds mu.
func (s *AgentSession) withPendingToolState(state ContinueTurnState) ContinueTurnState {
	state.TurnSettings = s.persistedSettings
	state.PendingToolApprovals = s.pendingApprovals.values()
	state.PendingToolResults = s.pendingResults.values()

	return state
}

// resumeStateWithContinuation wraps a continuation in a resume state.
func resumeStateWithContinuation(continueFrom ContinueTurnState) ResumeSessionState {
	return ResumeSessionState{
		Type:                 StateResumeSession,
		HarnessID:            continueFrom.HarnessID,
		SpecificationVersion: continueFrom.SpecificationVersion,
		Data:                 continueFrom.Data,
		ContinueFrom:         &continueFrom,
	}
}

// turnRequest is what the agent prepared for one turn.
type turnRequest struct {
	mode   turnMode
	prompt model.UserMessage

	settings             TurnSettings
	tools                []ai.Tool
	activeTools          []ai.Tool
	builtinToolFiltering *BuiltinToolFiltering

	responseFormat *model.ResponseFormat
	output         ai.Output
	telemetry      ai.TelemetryOptions
	callbacks      lifecycleCallbacks
	stopConditions []ai.StopCondition

	approvalContinuations []model.ToolApprovalResponsePart
	resultContinuations   []model.ToolResultPart
}

// startPromptTurn starts a fresh turn.
func (s *AgentSession) startPromptTurn(ctx context.Context, request turnRequest) (*StreamResult, int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := s.requireReusableLocked(); err != nil {
		return nil, 0, err
	}

	if err := s.requirePromptableLocked(); err != nil {
		return nil, 0, err
	}

	settings := request.settings
	s.persistedSettings = &settings
	s.activeSettings = &activeTurnSettings{
		persisted:            settings,
		tools:                request.tools,
		activeTools:          request.activeTools,
		builtinToolFiltering: request.builtinToolFiltering,
	}

	return s.startTrackedTurnLocked(ctx, request)
}

// startContinueTurn continues the unfinished turn with the settings frozen
// when it began.
func (s *AgentSession) startContinueTurn(ctx context.Context, request turnRequest) (*StreamResult, int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := s.requireReusableLocked(); err != nil {
		return nil, 0, err
	}

	if err := s.requireContinuableLocked(); err != nil {
		return nil, 0, err
	}

	active, err := s.resolveActiveSettingsLocked(request)
	if err != nil {
		return nil, 0, err
	}

	request.settings = active.persisted
	request.tools = active.tools
	request.activeTools = active.activeTools
	request.builtinToolFiltering = active.builtinToolFiltering

	return s.startTrackedTurnLocked(ctx, request)
}

// startTrackedTurnLocked starts tracking a turn and starts its driver. The
// caller holds mu.
func (s *AgentSession) startTrackedTurnLocked(ctx context.Context, request turnRequest) (*StreamResult, int, error) {
	s.turnSequence++
	turnID := s.turnSequence

	s.activeTurnSequence = turnID
	s.suspension = nil
	s.turn = turnRunning
	s.clearActivePromptControlLocked(0)
	s.activeControl = &promptControlWaiter{
		turnID: turnID,
		ready:  make(chan struct{}),
	}

	stream := startTurn(ctx, turnSetup{
		harness:               s.harness,
		session:               s.underlying,
		mode:                  request.mode,
		prompt:                request.prompt,
		settings:              request.settings,
		tools:                 request.tools,
		activeTools:           request.activeTools,
		builtinToolFiltering:  request.builtinToolFiltering,
		sandbox:               RestrictedSession(s.sandbox),
		sessionWorkDir:        s.sessionWorkDir,
		responseFormat:        request.responseFormat,
		output:                request.output,
		telemetry:             request.telemetry,
		callbacks:             request.callbacks,
		stopConditions:        request.stopConditions,
		toolApproval:          s.toolApproval,
		pendingToolApprovals:  s.pendingApprovals.values(),
		pendingToolResults:    s.pendingResults.values(),
		approvalContinuations: request.approvalContinuations,
		resultContinuations:   request.resultContinuations,
		hooks:                 s.turnHooks(turnID),
	})

	s.activeTurn = stream

	return stream, turnID, nil
}

// turnHooks connects a turn's driver to the session.
func (s *AgentSession) turnHooks(turnID int) turnHooks {
	return turnHooks{
		onPendingToolApproval: func(approval PendingToolApproval) {
			s.mu.Lock()
			defer s.mu.Unlock()

			s.pendingApprovals.set(approval.ApprovalID, approval)
			s.markAwaitingLocked(turnAwaitingApproval)
		},
		onToolApprovalSettled: func(approvalID string) {
			s.mu.Lock()
			defer s.mu.Unlock()

			s.pendingApprovals.delete(approvalID)
		},
		onPendingToolResult: func(pending PendingToolResult) {
			s.mu.Lock()
			defer s.mu.Unlock()

			s.pendingResults.set(pending.ToolCallID, pending)
			s.markAwaitingLocked(turnAwaitingToolResult)
		},
		onToolResultSettled: func(toolCallID string) {
			s.mu.Lock()
			defer s.mu.Unlock()

			s.pendingResults.delete(toolCallID)
		},
		onTurnFinished: func() {
			s.finishTrackedTurn(turnID)
		},
		onTurnFailed: func() {
			s.finishTrackedTurn(turnID)
		},
		onPromptControlAvailable: func(control PromptControl) {
			s.setPromptControl(turnID, control)
		},
		isTurnSuspending: func() bool {
			s.mu.Lock()
			defer s.mu.Unlock()

			return s.activeTurnSequence == turnID && s.suspension != nil
		},
		onStopConditionMet: func(ctx context.Context) error {
			return s.captureStopConditionBoundary(ctx, turnID)
		},
	}
}

// waitForPromptControl waits until the turn's adapter started or the turn
// ended.
func (s *AgentSession) waitForPromptControl(ctx context.Context, turnID int) error {
	s.mu.Lock()
	waiter := s.activeControl
	s.mu.Unlock()

	if waiter == nil || waiter.turnID != turnID {
		return nil
	}

	select {
	case <-waiter.ready:
		return nil
	case <-ctx.Done():
		return context.Cause(ctx)
	}
}

// setPromptControl delivers the control of the running turn.
func (s *AgentSession) setPromptControl(turnID int, control PromptControl) {
	s.mu.Lock()
	defer s.mu.Unlock()

	waiter := s.activeControl
	isCurrent := s.state == sessionActive &&
		s.turn == turnRunning &&
		s.activeTurnSequence == turnID &&
		waiter != nil &&
		waiter.turnID == turnID

	if !isCurrent {
		return
	}

	settleWaiter(waiter, control)
}

// finishTrackedTurn returns the session to idle after its turn ended.
func (s *AgentSession) finishTrackedTurn(turnID int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.state != sessionActive || s.activeTurnSequence != turnID {
		return
	}

	s.clearActivePromptControlLocked(turnID)
	s.pendingApprovals.clear()
	s.pendingResults.clear()
	s.suspension = nil
	s.activeSettings = nil
	s.persistedSettings = nil
	s.turn = turnIdle
}

// markAwaitingLocked records that the turn waits for host input. The caller
// holds mu.
func (s *AgentSession) markAwaitingLocked(state turnState) {
	if s.state != sessionActive {
		return
	}

	s.clearActivePromptControlLocked(0)
	s.turn = state
}

// clearActivePromptControlLocked releases the waiter of a turn, or of any
// turn when turnID is zero. The caller holds mu.
func (s *AgentSession) clearActivePromptControlLocked(turnID int) {
	waiter := s.activeControl
	if waiter == nil {
		return
	}

	if turnID != 0 && waiter.turnID != turnID {
		return
	}

	settleWaiter(waiter, nil)
	s.activeControl = nil
}

// settleWaiter delivers control to a waiter once.
func settleWaiter(waiter *promptControlWaiter, control PromptControl) {
	if waiter.settled {
		return
	}

	waiter.settled = true
	waiter.control = control
	close(waiter.ready)
}

// resolveActiveSettingsLocked returns the frozen settings of the unfinished
// turn, rebuilding them from persisted state after a resume. Every tool the
// turn offered must still be available. The caller holds mu.
func (s *AgentSession) resolveActiveSettingsLocked(request turnRequest) (*activeTurnSettings, error) {
	if s.activeSettings != nil {
		return s.activeSettings, nil
	}

	persisted := request.settings
	if s.persistedSettings != nil {
		persisted = *s.persistedSettings
	}

	s.persistedSettings = &persisted

	activeTools := make([]ai.Tool, 0, len(persisted.Tools))

	for _, spec := range persisted.Tools {
		tool, found := findTool(request.activeTools, spec.Name)
		if !found {
			return nil, fmt.Errorf("harness agent cannot continue turn because tool %q is no longer available", spec.Name)
		}

		activeTools = append(activeTools, *tool)
	}

	s.activeSettings = &activeTurnSettings{
		persisted:            persisted,
		tools:                request.tools,
		activeTools:          activeTools,
		builtinToolFiltering: request.builtinToolFiltering,
	}

	return s.activeSettings, nil
}

// requirePromptableLocked fails when the session cannot start a new turn.
// The caller holds mu.
func (s *AgentSession) requirePromptableLocked() error {
	switch s.turn {
	case turnIdle:
		return nil
	case turnRunning:
		return fmt.Errorf("harness session %s already has a turn in progress", s.sessionID)
	default:
		return fmt.Errorf("harness session %s has an unfinished turn and must be continued before accepting a new prompt", s.sessionID)
	}
}

// requireContinuableLocked fails when the session has no turn to continue.
// The caller holds mu.
func (s *AgentSession) requireContinuableLocked() error {
	switch s.turn {
	case turnAwaitingApproval, turnAwaitingToolResult, turnSuspended:
		return nil
	case turnRunning:
		return fmt.Errorf("harness session %s already has a turn in progress", s.sessionID)
	default:
		return fmt.Errorf("harness session %s has no unfinished turn to continue", s.sessionID)
	}
}

// requireReusableLocked fails when the session ended. The caller holds mu.
func (s *AgentSession) requireReusableLocked() error {
	if s.state != sessionActive || s.underlying == nil {
		return s.endedError()
	}

	return nil
}

// reusableSession returns the adapter session of an active session.
func (s *AgentSession) reusableSession() (Session, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := s.requireReusableLocked(); err != nil {
		return nil, err
	}

	return s.underlying, nil
}

// endedError reports use of a session that ended.
func (s *AgentSession) endedError() error {
	return fmt.Errorf("harness session %s has ended and cannot be reused", s.sessionID)
}

// endLocalHandle ends the local handle.
func (s *AgentSession) endLocalHandle(state sessionState) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.endLocalHandleLocked(state)
}

// endLocalHandleLocked ends the local handle. The caller holds mu.
func (s *AgentSession) endLocalHandleLocked(state sessionState) {
	s.clearActivePromptControlLocked(0)
	s.state = state
	s.underlying = nil
	s.sandbox = nil
}

// errSessionRequired reports a call without a session.
var errSessionRequired = errors.New("harness agent: a session is required; create one with CreateSession")

// orderedPending is a map that keeps insertion order, as JavaScript maps do.
type orderedPending[Value any] struct {
	keys    []string
	entries map[string]Value
}

// set inserts or replaces a value, keeping the position of an existing key.
func (o *orderedPending[Value]) set(key string, value Value) {
	if o.entries == nil {
		o.entries = make(map[string]Value)
	}

	if _, exists := o.entries[key]; !exists {
		o.keys = append(o.keys, key)
	}

	o.entries[key] = value
}

// delete removes a key.
func (o *orderedPending[Value]) delete(key string) {
	if _, exists := o.entries[key]; !exists {
		return
	}

	delete(o.entries, key)

	for index, existing := range o.keys {
		if existing == key {
			o.keys = append(o.keys[:index], o.keys[index+1:]...)

			return
		}
	}
}

// clear removes every key.
func (o *orderedPending[Value]) clear() {
	o.keys = nil
	o.entries = nil
}

// len returns the number of keys.
func (o *orderedPending[Value]) len() int {
	return len(o.keys)
}

// values returns the values in insertion order, or nil when empty.
func (o *orderedPending[Value]) values() []Value {
	if len(o.keys) == 0 {
		return nil
	}

	values := make([]Value, 0, len(o.keys))
	for _, key := range o.keys {
		values = append(values, o.entries[key])
	}

	return values
}
