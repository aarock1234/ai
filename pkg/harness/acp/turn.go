package acp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/bridge"
)

// capabilityUnsupportedErrorName is the serialized name of the bridge's
// capability error.
const capabilityUnsupportedErrorName = "AI_HarnessBridgeCapabilityUnsupportedError"

// turn is the control of one prompt turn. Its goroutine reads the channel,
// forwards the turn's events and answers the bridge's classification
// requests; it is the only caller of emit.
type turn struct {
	session *session
	emit    func(part harness.StreamPart)

	done chan struct{}
	err  error

	// readCtx bounds the channel reads; stopReading ends them early.
	readCtx     context.Context
	stopReading context.CancelCauseFunc

	// stopAbortWatch stops watching the turn context.
	stopAbortWatch func() bool

	// openBlock and dynamicToolCalls belong to the goroutine.
	openBlock        *openBlock
	dynamicToolCalls map[string]bool

	mu                sync.Mutex
	finished          bool
	abortRequested    bool
	abortErr          error
	activeQuestions   map[string]activeQuestion
	questionToolCalls map[string]string
}

// openBlock is the text or reasoning block the turn forwarded last.
type openBlock struct {
	reasoning bool
	id        string
}

// activeQuestion is a question request the implementation awaits an
// answer to.
type activeQuestion struct {
	requestID     string
	nativeRequest json.RawMessage
}

// startTurn starts the goroutine of a turn. Cancelling ctx aborts the turn.
func (s *session) startTurn(ctx context.Context, emit func(part harness.StreamPart)) *turn {
	readCtx, stopReading := context.WithCancelCause(context.Background())

	t := &turn{
		session:           s,
		emit:              emit,
		done:              make(chan struct{}),
		readCtx:           readCtx,
		stopReading:       stopReading,
		dynamicToolCalls:  make(map[string]bool),
		activeQuestions:   make(map[string]activeQuestion),
		questionToolCalls: make(map[string]string),
	}

	t.stopAbortWatch = context.AfterFunc(ctx, func() {
		t.abort(context.Cause(ctx))
	})

	s.turns.Go(t.run)

	return t
}

// Done is closed once every event of the turn was emitted.
func (t *turn) Done() <-chan struct{} {
	return t.done
}

// Err returns the failure of the turn after Done is closed.
func (t *turn) Err() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	return t.err
}

// SubmitToolApproval answers a built-in tool approval request.
func (t *turn) SubmitToolApproval(_ context.Context, submission harness.ToolApprovalSubmission) error {
	return t.session.channel.Send(bridge.NewToolApprovalResponseCommand(submission))
}

// SubmitToolResult submits the result of an application tool call, or the
// answer to a native question request.
func (t *turn) SubmitToolResult(_ context.Context, submission harness.ToolResultSubmission) error {
	if t.isQuestionAnswer(submission) {
		return t.submitQuestionAnswer(submission)
	}

	command, err := bridge.NewToolResultCommand(submission)
	if err != nil {
		return err
	}

	return t.session.channel.Send(command)
}

// abort asks the bridge to cancel the turn. When the command cannot be
// sent, the turn ends with the cause.
func (t *turn) abort(cause error) {
	t.mu.Lock()
	if t.finished || t.abortRequested {
		t.mu.Unlock()

		return
	}

	t.abortRequested = true
	t.abortErr = cause
	t.mu.Unlock()

	err := t.session.channel.Send(bridge.SimpleCommand{
		Type: "abort",
	})
	if err != nil {
		t.stopReading(cause)
	}
}

// run reads the channel until the turn ends.
func (t *turn) run() {
	for {
		frame, err := t.session.channel.Next(t.readCtx)
		if err != nil {
			t.settle(t.endedError(err))

			return
		}

		if t.handle(frame) {
			return
		}
	}
}

// endedError returns the outcome of a turn whose channel ended: none for a
// suspension, the abort cause for an aborted turn, and a closed bridge
// otherwise.
func (t *turn) endedError(err error) error {
	t.session.finishTurn()

	if closed, ok := errors.AsType[*bridge.ClosedError](err); ok && closed.Reason == bridge.CloseReasonSuspended {
		return nil
	}

	t.closeForwardedBlock()

	if abortErr := t.requestedAbort(); abortErr != nil {
		return abortErr
	}

	return fmt.Errorf("%s acp bridge closed before turn end", t.session.harnessID())
}

// handle processes one frame and reports whether the turn ended.
func (t *turn) handle(frame bridge.Frame) bool {
	switch message := frame.Message.(type) {
	case bridge.Thread:
		t.session.setACPSessionID(message.ThreadID)
	case bridge.CustomMessage:
		return t.handleCustom(message)
	case bridge.StreamMessage:
		return t.handlePart(message.Part)
	}

	return false
}

// handlePart forwards one event, marking MCP tool calls dynamic. Finish and
// error events end the turn.
func (t *turn) handlePart(part harness.StreamPart) bool {
	switch typed := part.(type) {
	case harness.ToolCallPart:
		if t.dynamicToolCalls[typed.ToolCallID] {
			typed.Dynamic = true
		}

		t.forward(typed)
	case harness.ToolResultPart:
		if t.dynamicToolCalls[typed.ToolCallID] {
			typed.Dynamic = true
		}

		delete(t.dynamicToolCalls, typed.ToolCallID)
		t.forward(typed)
	case harness.FinishPart:
		t.session.finishTurn()
		t.closeForwardedBlock()
		t.forward(typed)
		t.settle(t.requestedAbort())

		return true
	case harness.ErrorPart:
		t.session.finishTurn()
		t.closeForwardedBlock()

		err := t.session.config.harness.bridgeError(typed.Err)
		t.forward(harness.ErrorPart{
			Err: err,
		})
		t.settle(err)

		return true
	case harness.StreamStartPart,
		harness.TextStartPart,
		harness.TextDeltaPart,
		harness.TextEndPart,
		harness.ReasoningStartPart,
		harness.ReasoningDeltaPart,
		harness.ReasoningEndPart,
		harness.ToolApprovalRequestPart,
		harness.FileChangePart,
		harness.FinishStepPart,
		harness.RawPart:
		t.forward(typed)
	}

	return false
}

// handleCustom answers the ACP classification frames.
func (t *turn) handleCustom(message bridge.CustomMessage) bool {
	switch message.Type {
	case frameToolCallCandidate:
		return t.classifyToolCall(message.Data)
	case frameQuestionRequest:
		return t.handleQuestionRequest(message.Data)
	case frameQuestionResolved:
		return t.resolveQuestion(message.Data)
	default:
		return false
	}
}

// classifyToolCall tells the bridge whether to suppress a tool call that
// carries a native question request, and remembers MCP tool calls.
func (t *turn) classifyToolCall(data json.RawMessage) bool {
	candidate, err := decodeToolCallCandidate(data)
	if err != nil {
		return t.fail(err)
	}

	settings := t.session.config.harness.settings

	suppress := false
	if questions := settings.AskUserQuestions; questions != nil && questions.IsNativeToolCall != nil {
		suppress = questions.IsNativeToolCall(candidate.ToolCall)
	}

	t.dynamicToolCalls[candidate.ToolCall.ToolCallID] = settings.IsMCPToolCall != nil && settings.IsMCPToolCall(candidate.ToolCall)

	command, err := classificationCommand(candidate.RequestID, suppressionOutput{
		Suppress: suppress,
	})
	if err != nil {
		return t.fail(err)
	}

	_ = t.session.channel.Send(command)

	return false
}

// resolveQuestion forgets a question request the implementation stopped
// waiting for.
func (t *turn) resolveQuestion(data json.RawMessage) bool {
	requestID, err := decodeQuestionResolved(data)
	if err != nil {
		return t.fail(err)
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	toolCallID, ok := t.questionToolCalls[requestID]
	if ok {
		delete(t.activeQuestions, toolCallID)
		delete(t.questionToolCalls, requestID)
	}

	return false
}

// forward emits an event, tracking the open text or reasoning block.
func (t *turn) forward(part harness.StreamPart) {
	switch typed := part.(type) {
	case harness.TextStartPart:
		t.openBlock = &openBlock{
			id: typed.ID,
		}
	case harness.ReasoningStartPart:
		t.openBlock = &openBlock{
			reasoning: true,
			id:        typed.ID,
		}
	case harness.TextEndPart:
		t.closeBlockWithID(typed.ID)
	case harness.ReasoningEndPart:
		t.closeBlockWithID(typed.ID)
	}

	t.emit(part)
}

// closeBlockWithID forgets the open block when it has the ID.
func (t *turn) closeBlockWithID(id string) {
	if t.openBlock != nil && t.openBlock.id == id {
		t.openBlock = nil
	}
}

// closeForwardedBlock closes the open block before the turn ends.
func (t *turn) closeForwardedBlock() {
	block := t.openBlock
	if block == nil {
		return
	}

	t.openBlock = nil

	if block.reasoning {
		t.forward(harness.ReasoningEndPart{
			ID: block.id,
		})

		return
	}

	t.forward(harness.TextEndPart{
		ID: block.id,
	})
}

// fail ends the turn with err and asks the bridge to cancel it.
func (t *turn) fail(err error) bool {
	t.closeForwardedBlock()
	t.forward(harness.ErrorPart{
		Err: err,
	})

	_ = t.session.channel.Send(bridge.SimpleCommand{
		Type: "abort",
	})

	t.settle(err)

	return true
}

// requestedAbort returns the cause of a requested abort, or nil.
func (t *turn) requestedAbort() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if !t.abortRequested {
		return nil
	}

	return t.abortErr
}

// settle ends the turn with its outcome.
func (t *turn) settle(err error) {
	t.mu.Lock()
	t.finished = true
	t.err = err
	t.mu.Unlock()

	t.stopAbortWatch()
	t.stopReading(nil)
	close(t.done)
}

// bridgeError translates the bridge's capability error into a
// harness.CapabilityUnsupportedError.
func (h *Harness) bridgeError(err error) error {
	remote, ok := errors.AsType[*bridge.RemoteError](err)
	if !ok {
		return err
	}

	var serialized struct {
		Name    string  `json:"name"`
		Message *string `json:"message"`
	}

	if json.Unmarshal(remote.Value, &serialized) != nil || serialized.Name != capabilityUnsupportedErrorName || serialized.Message == nil {
		return err
	}

	return h.unsupported(*serialized.Message, nil)
}
