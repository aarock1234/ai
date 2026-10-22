package pi

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"path"
	"sync"
	"time"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/sandbox"
)

// Grace periods of the session lifecycle.
const (
	shutdownGracePeriod = 5 * time.Second
	finishCallTimeout   = 30 * time.Second
)

// errSessionStopped reports a session whose local instance ended.
var errSessionStopped = errors.New("pi session has been stopped")

// session is a Pi session over one relay launch.
type session struct {
	harness   *Harness
	sessionID string
	isResume  bool

	sandbox  sandbox.Session
	layout   layout
	launchID string
	conn     *connection

	// relay is nil when the session attached to a relay another process
	// spawned.
	relay sandbox.Process

	fingerprint           string
	setup                 providerSetup
	credentials           *storedCredentials
	credentialEnvironment map[string]string
	builtinToolNames      []string
	diagnostics           *diagnosticsFollower
	observability         *harness.Observability

	mu sync.Mutex

	turn       *turn
	translator *translator

	// resultDialogs and approvalDialogs map tool call IDs to the dialogs Pi
	// waits on; earlyResults and earlyApprovals hold answers submitted before
	// their dialog arrived.
	resultDialogs   map[string]string
	approvalDialogs map[string]string
	earlyResults    map[string]hostToolResult
	earlyApprovals  map[string]approvalDecision

	// pendingCompactions are compactions reported while no turn ran,
	// delivered on the next turn.
	pendingCompactions []harness.StreamPart

	sessionFileName string
	toolsSignature  string
	models          []modelInfo
	model           *modelInfo
	modelSelected   bool
	turnBaseline    *sessionTokens

	// liveTurn makes the next ContinueTurn attach to the turn the relay kept
	// running; rerunContinue makes it re-drive Pi's thread instead.
	liveTurn      bool
	rerunContinue bool

	// abortDone is closed when the last abort command was answered.
	abortDone chan struct{}

	stopped bool

	// workers are the goroutines that finish turns and abort runs.
	workers sync.WaitGroup
}

// SessionID returns the harness session ID.
func (s *session) SessionID() string {
	return s.sessionID
}

// IsResume reports whether the session resumed from lifecycle state.
func (s *session) IsResume() bool {
	return s.isResume
}

// handleRecord handles a record that is not a command response. It runs on
// the connection's reading goroutine.
func (s *session) handleRecord(recordType string, record []byte) {
	switch recordType {
	case recordExtensionUIRequest:
		var request extensionUIRequest
		if err := json.Unmarshal(record, &request); err == nil {
			s.handleUIRequest(request)
		}
	case recordExtensionError:
		var event extensionErrorEvent
		if err := json.Unmarshal(record, &event); err == nil {
			s.reportExtensionError(event)
		}
	default:
		var event sessionEvent
		if err := json.Unmarshal(record, &event); err == nil {
			s.handleEvent(event)
		}
	}
}

// handleEvent translates a session event for the active turn, and finishes
// the turn when Pi settled.
func (s *session) handleEvent(event sessionEvent) {
	s.mu.Lock()
	parts := s.translator.translate(event)
	parts = append(parts, s.earlyFinishStepLocked()...)
	current := s.turn
	s.mu.Unlock()

	s.deliver(current, parts)

	if event.Type == "agent_settled" && current != nil && current.beginFinish() {
		s.finishInBackground(current)
	}
}

// handleUIRequest answers the adapter's host-input dialogs, reports file
// changes, and dismisses every other dialog, which no user can answer.
func (s *session) handleUIRequest(request extensionUIRequest) {
	if request.Method == "setStatus" && request.StatusKey == fileChangeStatusKey {
		s.handleFileChange(request.StatusText)

		return
	}

	if request.Method == "input" && request.Title == dialogTitle {
		var dialog dialogRequest
		if err := json.Unmarshal([]byte(request.Placeholder), &dialog); err == nil {
			s.handleDialog(request.ID, dialog)

			return
		}
	}

	if request.isDialog() {
		s.respond(cancelledResponse{
			Type:      recordExtensionUIResponse,
			ID:        request.ID,
			Cancelled: true,
		})
	}
}

// handleDialog records a host-input dialog, or answers it with an answer the
// agent already submitted.
func (s *session) handleDialog(requestID string, dialog dialogRequest) {
	switch dialog.Kind {
	case dialogToolResult:
		s.handleResultDialog(requestID, dialog)
	case dialogToolApproval:
		s.handleApprovalDialog(requestID, dialog)
	default:
		s.respond(cancelledResponse{
			Type:      recordExtensionUIResponse,
			ID:        requestID,
			Cancelled: true,
		})
	}
}

// handleResultDialog waits for an application tool result.
func (s *session) handleResultDialog(requestID string, dialog dialogRequest) {
	s.mu.Lock()
	early, answered := s.earlyResults[dialog.ToolCallID]
	if answered {
		delete(s.earlyResults, dialog.ToolCallID)
	} else {
		s.resultDialogs[dialog.ToolCallID] = requestID
	}

	parts := s.earlyFinishStepLocked()
	current := s.turn
	s.mu.Unlock()

	if answered {
		s.respondValue(requestID, early)

		return
	}

	s.deliver(current, parts)
}

// handleApprovalDialog asks the agent to approve a built-in tool call.
func (s *session) handleApprovalDialog(requestID string, dialog dialogRequest) {
	s.mu.Lock()
	early, answered := s.earlyApprovals[dialog.ToolCallID]
	if answered {
		delete(s.earlyApprovals, dialog.ToolCallID)
	} else {
		s.approvalDialogs[dialog.ToolCallID] = requestID
	}

	parts := s.earlyFinishStepLocked()
	current := s.turn
	s.mu.Unlock()

	if answered {
		s.respondValue(requestID, early)

		return
	}

	request := harness.ToolApprovalRequestPart{
		ApprovalID: dialog.ToolCallID,
		ToolCallID: dialog.ToolCallID,
	}

	s.deliver(current, append([]harness.StreamPart{request}, parts...))
}

// handleFileChange reports a workspace file change of a built-in tool.
func (s *session) handleFileChange(statusText string) {
	var change fileChangeStatus
	if err := json.Unmarshal([]byte(statusText), &change); err != nil {
		return
	}

	event := harness.FileModified
	if change.Event == "create" {
		event = harness.FileCreated
	}

	s.mu.Lock()
	current := s.turn
	s.mu.Unlock()

	s.deliver(current, []harness.StreamPart{
		harness.FileChangePart{
			Event: event,
			Path:  change.Path,
		},
	})
}

// earlyFinishStepLocked returns the early finish step of a tool batch that
// waits on host input. The caller holds s.mu.
func (s *session) earlyFinishStepLocked() []harness.StreamPart {
	if s.turn == nil {
		return nil
	}

	return s.translator.earlyFinishStep(func(toolCallID string) bool {
		_, waitsForResult := s.resultDialogs[toolCallID]
		_, waitsForApproval := s.approvalDialogs[toolCallID]

		return waitsForResult || waitsForApproval
	})
}

// deliver forwards parts to the running turn. Without one, compactions wait
// for the next turn and other parts have no consumer.
func (s *session) deliver(current *turn, parts []harness.StreamPart) {
	for _, part := range parts {
		if current != nil && current.forward(part) {
			continue
		}

		if _, isCompaction := part.(harness.CompactionPart); isCompaction {
			s.mu.Lock()
			s.pendingCompactions = append(s.pendingCompactions, part)
			s.mu.Unlock()
		}
	}
}

// respondValue answers a dialog with a JSON value.
func (s *session) respondValue(requestID string, value any) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return
	}

	s.respond(valueResponse{
		Type:  recordExtensionUIResponse,
		ID:    requestID,
		Value: string(encoded),
	})
}

// respond writes an extension UI response.
func (s *session) respond(record any) {
	if err := s.conn.send(context.Background(), record); err != nil {
		slog.Default().Error("pi extension ui response failed", slog.String("session", s.sessionID), slog.Any("error", err))
	}
}

// submitToolResult answers the dialog of an application tool call.
func (s *session) submitToolResult(ctx context.Context, submission harness.ToolResultSubmission) error {
	result, err := convertToolResult(submission)
	if err != nil {
		return err
	}

	s.mu.Lock()
	s.translator.recordHostToolResult(submission.ToolCallID, encodeOutput(submission.Output))

	requestID, waiting := s.resultDialogs[submission.ToolCallID]
	if waiting {
		delete(s.resultDialogs, submission.ToolCallID)
	} else {
		s.earlyResults[submission.ToolCallID] = result
	}
	s.mu.Unlock()

	if !waiting {
		return nil
	}

	return s.sendValue(ctx, requestID, result)
}

// submitToolApproval answers the approval dialog of a built-in tool call.
func (s *session) submitToolApproval(ctx context.Context, submission harness.ToolApprovalSubmission) error {
	decision := approvalDecision{
		Approved: submission.Approved,
		Reason:   submission.Reason,
	}

	s.mu.Lock()
	requestID, waiting := s.approvalDialogs[submission.ApprovalID]
	if waiting {
		delete(s.approvalDialogs, submission.ApprovalID)
	} else {
		s.earlyApprovals[submission.ApprovalID] = decision
	}
	s.mu.Unlock()

	if !waiting {
		return nil
	}

	return s.sendValue(ctx, requestID, decision)
}

// sendValue answers a dialog with a JSON value.
func (s *session) sendValue(ctx context.Context, requestID string, value any) error {
	encoded, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("encode pi dialog response: %w", err)
	}

	return s.conn.send(ctx, valueResponse{
		Type:  recordExtensionUIResponse,
		ID:    requestID,
		Value: string(encoded),
	})
}

// steer queues a steering message for the running turn.
func (s *session) steer(ctx context.Context, text string) error {
	_, err := s.conn.call(ctx, steerCommand{
		Type:    "steer",
		Message: text,
	})

	return err
}

// finishInBackground finishes a settled turn on a worker goroutine, because
// reading the turn's usage needs a command response from the reading
// goroutine.
func (s *session) finishInBackground(current *turn) {
	s.workers.Go(func() {
		s.finishTurn(current)
	})
}

// finishTurn ends a turn Pi settled: with the error of its last assistant
// message, or with its usage, the difference of Pi's session token counts.
func (s *session) finishTurn(current *turn) {
	s.mu.Lock()
	terminal := s.translator.LastAssistantError
	reasoning := s.translator.TurnReasoningTokens
	baseline := s.turnBaseline
	s.mu.Unlock()

	if terminal != "" {
		current.fail(errors.New(terminal))

		return
	}

	usage := messageUsage{
		Reasoning: reasoning,
	}

	ctx, cancel := context.WithTimeout(context.Background(), finishCallTimeout)
	defer cancel()

	if after, err := s.sessionTokens(ctx); err == nil && baseline != nil {
		usage.Input = after.Input - baseline.Input
		usage.Output = after.Output - baseline.Output
		usage.CacheRead = after.CacheRead - baseline.CacheRead
		usage.CacheWrite = after.CacheWrite - baseline.CacheWrite
	}

	current.finish(harness.FinishPart{
		FinishReason: model.FinishReason{
			Unified: model.FinishStop,
		},
		TotalUsage: harnessUsage(usage),
	}, nil)
}

// sessionTokens returns Pi's cumulative token counts.
func (s *session) sessionTokens(ctx context.Context) (sessionTokens, error) {
	reply, err := s.conn.call(ctx, simpleCommand{
		Type: "get_session_stats",
	})
	if err != nil {
		return sessionTokens{}, err
	}

	var stats sessionStatsData
	if err := json.Unmarshal(reply.Data, &stats); err != nil {
		return sessionTokens{}, fmt.Errorf("decode pi session stats: %w", err)
	}

	return stats.Tokens, nil
}

// abortRun aborts Pi's current run. Later turns wait for Pi to become idle.
func (s *session) abortRun() {
	done := make(chan struct{})

	s.mu.Lock()
	s.abortDone = done
	s.mu.Unlock()

	s.workers.Go(func() {
		defer close(done)

		ctx, cancel := context.WithTimeout(context.Background(), finishCallTimeout)
		defer cancel()

		_, _ = s.conn.call(ctx, simpleCommand{
			Type: "abort",
		})
	})
}

// waitForAbort waits until the last abort was answered.
func (s *session) waitForAbort(ctx context.Context) error {
	s.mu.Lock()
	done := s.abortDone
	s.mu.Unlock()

	if done == nil {
		return nil
	}

	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return context.Cause(ctx)
	}
}

// release clears a settled turn and the dialogs it waited on.
func (s *session) release(current *turn) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.turn == current {
		s.turn = nil
	}
}

// Compact asks Pi to compact its context. Pi reports the compaction, which
// the active or next turn emits.
func (s *session) Compact(ctx context.Context, customInstructions string) error {
	if err := s.ensureUsable(); err != nil {
		return err
	}

	if err := s.conn.start(ctx); err != nil {
		return err
	}

	_, err := s.conn.call(ctx, compactCommand{
		Type:               "compact",
		CustomInstructions: customInstructions,
	})

	return err
}

// ensureUsable fails when the local session ended.
func (s *session) ensureUsable() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.stopped {
		return errSessionStopped
	}

	return nil
}

// markStopped records the end of the local session, failing when it already
// ended.
func (s *session) markStopped() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.stopped {
		return errSessionStopped
	}

	s.stopped = true

	return nil
}

// SuspendTurn freezes the active turn. With live reattachment Pi keeps
// running the turn in the sandbox and the next process attaches at the
// cursor; otherwise Pi stops and the next process re-drives its thread from
// the session file, recomputing the work in flight, as upstream.
func (s *session) SuspendTurn(ctx context.Context) (harness.ContinueTurnState, error) {
	if err := s.markStopped(); err != nil {
		return harness.ContinueTurnState{}, err
	}

	data, err := s.leave(ctx, s.harness.reattachLiveRuntime())
	if err != nil {
		return harness.ContinueTurnState{}, err
	}

	return harness.NewContinueTurnState(HarnessID, data), nil
}

// Detach disconnects from Pi before a later turn. With live reattachment Pi
// keeps running; otherwise the session stops.
func (s *session) Detach(ctx context.Context) (harness.ResumeSessionState, error) {
	if err := s.markStopped(); err != nil {
		return harness.ResumeSessionState{}, err
	}

	data, err := s.leave(ctx, s.harness.reattachLiveRuntime())
	if err != nil {
		return harness.ResumeSessionState{}, err
	}

	return harness.NewResumeSessionState(HarnessID, data), nil
}

// Stop stops Pi, whose session file the next start opens.
func (s *session) Stop(ctx context.Context) (harness.ResumeSessionState, error) {
	if err := s.markStopped(); err != nil {
		return harness.ResumeSessionState{}, err
	}

	data, err := s.leave(ctx, false)
	if err != nil {
		return harness.ResumeSessionState{}, err
	}

	return harness.NewResumeSessionState(HarnessID, data), nil
}

// Destroy stops Pi without returning lifecycle state.
func (s *session) Destroy(ctx context.Context) error {
	if err := s.markStopped(); err != nil {
		return nil
	}

	s.failActiveTurn(errSessionStopped)
	signalRelay(context.WithoutCancel(ctx), s.sandbox, s.conn.stateDir())
	s.shutdown(ctx)

	return nil
}

// leave ends the local session: it keeps Pi running and returns its cursor,
// or shuts Pi down and returns its session file.
func (s *session) leave(ctx context.Context, keepRunning bool) (json.RawMessage, error) {
	if keepRunning {
		return s.freeze(ctx)
	}

	s.mu.Lock()
	suspendedTurn := s.turn
	s.mu.Unlock()

	if suspendedTurn != nil {
		s.abortRun()
		suspendedTurn.settle(nil)
	}

	_ = s.waitForAbort(ctx)

	if err := s.conn.closeInput(ctx); err != nil {
		signalRelay(context.WithoutCancel(ctx), s.sandbox, s.conn.stateDir())
	}

	s.shutdown(ctx)

	s.mu.Lock()
	data := stateData{
		SessionFileName:              s.sessionFileName,
		SandboxCredentialEnvironment: s.credentialEnvironment,
	}
	s.mu.Unlock()

	return encodeState(data)
}

// freeze stops reading at a record boundary while Pi keeps running, and
// returns the cursor with the host state of the active turn.
func (s *session) freeze(ctx context.Context) (json.RawMessage, error) {
	s.workers.Wait()

	offset, sequence := s.conn.freeze()
	s.diagnostics.stop()

	s.mu.Lock()
	active := s.turn
	runtime := &liveRuntime{
		LaunchID:                 s.launchID,
		OutboxOffset:             offset,
		InputSequence:            sequence,
		ConfigurationFingerprint: s.fingerprint,
		TurnActive:               active != nil && active.isRunning(),
		PendingDialogs:           s.pendingDialogsLocked(),
		Translator:               s.translator,
		TurnBaseline:             s.turnBaseline,
		ToolsSignature:           s.toolsSignature,
		Model:                    s.model,
		ModelSelected:            s.modelSelected,
	}

	data := stateData{
		SessionFileName:              s.sessionFileName,
		Runtime:                      runtime,
		SandboxCredentialEnvironment: s.credentialEnvironment,
	}
	s.mu.Unlock()

	if active != nil {
		active.settle(nil)
	}

	s.saveCredentials(ctx)

	return encodeState(data)
}

// pendingDialogsLocked lists the dialogs Pi waits on. The caller holds s.mu.
func (s *session) pendingDialogsLocked() []pendingDialog {
	var dialogs []pendingDialog

	for toolCallID, requestID := range s.resultDialogs {
		dialogs = append(dialogs, pendingDialog{
			RequestID:  requestID,
			Kind:       dialogToolResult,
			ToolCallID: toolCallID,
		})
	}

	for toolCallID, requestID := range s.approvalDialogs {
		dialogs = append(dialogs, pendingDialog{
			RequestID:  requestID,
			Kind:       dialogToolApproval,
			ToolCallID: toolCallID,
		})
	}

	return dialogs
}

// failActiveTurn ends the active turn with err and aborts Pi's run.
func (s *session) failActiveTurn(err error) {
	s.mu.Lock()
	current := s.turn
	s.mu.Unlock()

	if current == nil {
		return
	}

	s.abortRun()
	current.fail(err)
}

// shutdown waits for Pi to exit, kills what remains, saves the credentials
// and removes the launch's state.
func (s *session) shutdown(ctx context.Context) {
	// An attached session reads only once it needs Pi; following the output
	// observes Pi's exit.
	_ = s.conn.start(ctx)

	waitCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), shutdownGracePeriod)
	exited := s.conn.waitEnded(waitCtx)
	cancel()

	if !exited {
		signalRelay(context.WithoutCancel(ctx), s.sandbox, s.conn.stateDir())
	}

	s.conn.stop()
	s.workers.Wait()
	s.diagnostics.stop()

	if s.relay != nil {
		waitCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), shutdownGracePeriod)
		_, _ = s.relay.Wait(waitCtx)
		cancel()

		_ = s.relay.Kill(context.WithoutCancel(ctx))
	}

	s.saveCredentials(ctx)
	removeRelayState(context.WithoutCancel(ctx), s.sandbox, s.conn.stateDir())
}

// saveCredentials saves the credential document when Pi changed it, for
// example after refreshing an OAuth token.
func (s *session) saveCredentials(ctx context.Context) {
	if s.credentials == nil {
		return
	}

	document, err := s.sandbox.ReadBinaryFile(context.WithoutCancel(ctx), sandbox.ReadFileOptions{
		Path: path.Join(s.layout.agentDir(), "auth.json"),
	})
	if err != nil {
		return
	}

	if err := s.credentials.saveIfChanged(ctx, document); err != nil {
		slog.Default().ErrorContext(ctx, "pi credential save failed", slog.String("session", s.sessionID), slog.Any("error", err))
	}
}

// The session implements the harness session.
var _ harness.Session = (*session)(nil)
