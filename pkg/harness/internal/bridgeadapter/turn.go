package bridgeadapter

import (
	"context"
	"sync"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/bridge"
)

// TurnOptions configure Runtime.BeginTurn.
type TurnOptions struct {
	// Emit receives the turn's events.
	Emit func(part harness.StreamPart)

	// AcceptUserMessages offers interactive user messages on every turn.
	// Otherwise they are offered when the bridge's hello announced user
	// message responses.
	AcceptUserMessages bool
}

// Turn is one prompt turn wired into the runtime's frames. It ends with the
// turn's finish or error part, when ctx is cancelled, or when the channel
// ends.
type Turn struct {
	runtime *Runtime
	ctx     context.Context
	emit    func(part harness.StreamPart)

	// submitter is nil when the turn does not accept user messages.
	submitter       *bridge.UserMessageSubmitter
	removeReconnect func()

	mu        sync.Mutex
	settled   bool
	err       error
	stopAbort func() bool
	done      chan struct{}
}

// BeginTurn wires a turn into the runtime's frames and returns it. Ctx
// governs the turn: cancelling it aborts the bridge's turn and fails the
// turn with ctx's cause. A turn whose ctx is already cancelled is settled on
// return; the adapter must not start it in the bridge.
func (r *Runtime) BeginTurn(ctx context.Context, options TurnOptions) *Turn {
	turn := &Turn{
		runtime: r,
		ctx:     ctx,
		emit:    options.Emit,
		done:    make(chan struct{}),
	}

	if options.AcceptUserMessages || r.userMessageResponses.Load() {
		turn.submitter = bridge.NewUserMessageSubmitter(func(command bridge.UserMessageCommand) error {
			return r.channel.Send(command)
		})
		turn.removeReconnect = r.channel.OnReconnect(turn.submitter.Resend)
	}

	if ctx.Err() != nil {
		turn.abort()

		return turn
	}

	turn.mu.Lock()
	turn.stopAbort = context.AfterFunc(ctx, turn.abort)
	turn.mu.Unlock()

	r.activate(turn)

	return turn
}

// Control returns the turn's prompt control. It accepts user messages when
// the turn does.
func (t *Turn) Control() harness.PromptControl {
	if t.submitter != nil {
		return userMessageTurn{
			Turn: t,
		}
	}

	return t
}

// Settled reports whether the turn already ended.
func (t *Turn) Settled() bool {
	return t.isSettled()
}

// Fail ends the turn with err, for example when its start command could not
// be sent.
func (t *Turn) Fail(err error) {
	t.settle(err)
}

// SubmitToolResult sends the result of an application tool call.
func (t *Turn) SubmitToolResult(_ context.Context, submission harness.ToolResultSubmission) error {
	command, err := bridge.NewToolResultCommand(submission)
	if err != nil {
		return err
	}

	return t.runtime.channel.Send(command)
}

// SubmitToolApproval answers a built-in tool approval request.
func (t *Turn) SubmitToolApproval(_ context.Context, submission harness.ToolApprovalSubmission) error {
	return t.runtime.channel.Send(bridge.NewToolApprovalResponseCommand(submission))
}

// Done is closed when the turn ended.
func (t *Turn) Done() <-chan struct{} {
	return t.done
}

// Err returns the turn's failure after Done is closed.
func (t *Turn) Err() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	return t.err
}

// forward emits a part unless the turn ended.
func (t *Turn) forward(part harness.StreamPart) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.settled {
		return
	}

	t.emit(part)
}

// abort asks the bridge to abort the turn and fails it with ctx's cause.
func (t *Turn) abort() {
	if t.isSettled() {
		return
	}

	_ = t.runtime.channel.Send(bridge.SimpleCommand{
		Type: "abort",
	})

	t.settle(context.Cause(t.ctx))
}

// settle ends the turn once.
func (t *Turn) settle(err error) {
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

	if t.submitter != nil {
		t.submitter.Close(err)
		t.removeReconnect()
	}

	t.runtime.release(t)
}

// isSettled reports whether the turn ended.
func (t *Turn) isSettled() bool {
	t.mu.Lock()
	defer t.mu.Unlock()

	return t.settled
}

// userMessageTurn is the control of a turn that accepts user messages.
type userMessageTurn struct {
	*Turn
}

// SubmitUserMessage sends a user message and waits until the bridge
// accepted it.
func (t userMessageTurn) SubmitUserMessage(ctx context.Context, text string) error {
	return t.submitter.Submit(ctx, text)
}
