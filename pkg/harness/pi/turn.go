package pi

import (
	"context"
	"sync"

	"github.com/aarock1234/ai/pkg/harness"
)

// turn is one prompt turn. Parts reach it from the connection's reading
// goroutine until Pi settles; the session then finishes it with the turn's
// usage. It implements harness.PromptControl, harness.ToolApprovalSubmitter
// and harness.UserMessageSubmitter.
type turn struct {
	session *session
	ctx     context.Context
	emit    func(part harness.StreamPart)

	mu sync.Mutex

	// promptSent is set once Pi may run the turn, so that its settlement
	// finishes the turn.
	promptSent bool

	// finishing is set when Pi settled: the session emits the last part.
	finishing bool

	settled   bool
	err       error
	stopAbort func() bool
	done      chan struct{}
}

// newTurn returns a turn governed by ctx: cancelling ctx aborts Pi's run and
// fails the turn with ctx's cause.
func newTurn(ctx context.Context, owner *session, emit func(part harness.StreamPart)) *turn {
	current := &turn{
		session: owner,
		ctx:     ctx,
		emit:    emit,
		done:    make(chan struct{}),
	}

	current.mu.Lock()
	current.stopAbort = context.AfterFunc(ctx, current.abort)
	current.mu.Unlock()

	return current
}

// forward emits a part of Pi's run unless the turn is finishing or ended.
func (t *turn) forward(part harness.StreamPart) bool {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.settled || t.finishing {
		return false
	}

	t.emit(part)

	return true
}

// markPromptSent records that Pi may run the turn.
func (t *turn) markPromptSent() {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.promptSent = true
}

// beginFinish marks a sent turn as finishing and reports whether it was
// running.
func (t *turn) beginFinish() bool {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.settled || t.finishing || !t.promptSent {
		return false
	}

	t.finishing = true

	return true
}

// isRunning reports a sent turn that has not settled.
func (t *turn) isRunning() bool {
	t.mu.Lock()
	defer t.mu.Unlock()

	return t.promptSent && !t.settled
}

// finish emits the turn's last part and ends it with err.
func (t *turn) finish(part harness.StreamPart, err error) {
	t.mu.Lock()
	if !t.settled {
		t.emit(part)
	}
	t.mu.Unlock()

	t.settle(err)
}

// fail reports err as the turn's last part and ends the turn.
func (t *turn) fail(err error) {
	t.finish(harness.ErrorPart{
		Err: err,
	}, err)
}

// abort aborts Pi's run and fails the turn with ctx's cause.
func (t *turn) abort() {
	if t.isSettled() {
		return
	}

	t.session.abortRun()
	t.settle(context.Cause(t.ctx))
}

// settle ends the turn once.
func (t *turn) settle(err error) {
	t.mu.Lock()
	if t.settled {
		t.mu.Unlock()

		return
	}

	t.settled = true
	t.err = err
	stopAbort := t.stopAbort
	close(t.done)
	t.mu.Unlock()

	if stopAbort != nil {
		stopAbort()
	}

	t.session.release(t)
}

// isSettled reports whether the turn ended.
func (t *turn) isSettled() bool {
	t.mu.Lock()
	defer t.mu.Unlock()

	return t.settled
}

// SubmitToolResult answers the tool call's pending dialog.
func (t *turn) SubmitToolResult(ctx context.Context, submission harness.ToolResultSubmission) error {
	return t.session.submitToolResult(ctx, submission)
}

// SubmitToolApproval answers the built-in tool's approval dialog.
func (t *turn) SubmitToolApproval(ctx context.Context, submission harness.ToolApprovalSubmission) error {
	return t.session.submitToolApproval(ctx, submission)
}

// SubmitUserMessage steers the running turn: Pi delivers the message after
// the current assistant turn finished its tool calls.
func (t *turn) SubmitUserMessage(ctx context.Context, text string) error {
	return t.session.steer(ctx, text)
}

// Done is closed when the turn ended.
func (t *turn) Done() <-chan struct{} {
	return t.done
}

// Err returns the turn's failure after Done is closed.
func (t *turn) Err() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	return t.err
}

// The turn implements the prompt control and its optional capabilities.
var (
	_ harness.PromptControl         = (*turn)(nil)
	_ harness.ToolApprovalSubmitter = (*turn)(nil)
	_ harness.UserMessageSubmitter  = (*turn)(nil)
)
