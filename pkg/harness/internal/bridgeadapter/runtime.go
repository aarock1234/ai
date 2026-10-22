package bridgeadapter

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/bridge"
	"github.com/aarock1234/ai/pkg/sandbox"
)

// shutdownGracePeriod bounds the waits for the bridge's stop reply and for
// its process to exit, as upstream.
const shutdownGracePeriod = 5 * time.Second

// Hooks observe frames of a session. They run on the runtime's routing
// goroutine and must not block.
type Hooks struct {
	// OnFinish observes the finish part of each turn before it is emitted.
	OnFinish func(part harness.FinishPart)

	// OnThread observes the bridge's thread announcements.
	OnThread func(threadID string)
}

// Runtime is one session's connection to its bridge: it routes the bridge's
// frames into the active turn and implements the session lifecycle shared by
// the bridge adapters.
//
// A Runtime owns the goroutine that reads the channel and, for a bridge it
// spawned, the goroutines that consume the process output. Stop and Destroy
// wait for all of them. Detach and SuspendTurn leave the bridge running, so
// they wait only for the routing goroutine; the output goroutines end when
// the bridge process exits.
type Runtime struct {
	harnessID string
	sessionID string
	channel   *bridge.Channel
	hooks     Hooks

	strategy  Strategy
	port      int
	token     string
	sandboxID string
	sandbox   sandbox.Session
	homeDir   string

	// bridgeProcess is nil when the runtime attached to a bridge another
	// process spawned.
	bridgeProcess *BridgeProcess

	userMessageResponses *atomic.Bool

	// deliverMu orders frame delivery, so that frames held for the first
	// turn are delivered before the frames that follow them.
	deliverMu sync.Mutex

	mu          sync.Mutex
	turn        *Turn
	holding     bool
	held        []harness.StreamPart
	closed      bool
	closeReason string
	stopped     bool
	destroyed   chan struct{}

	stopReplies chan bridge.Stop
	routed      chan struct{}
}

// Strategy returns how the runtime reached its bridge.
func (r *Runtime) Strategy() Strategy {
	return r.strategy
}

// Sandbox returns the restricted view of the session's sandbox.
func (r *Runtime) Sandbox() sandbox.Session {
	return r.sandbox
}

// HomeDir returns the sandbox's HOME directory.
func (r *Runtime) HomeDir() string {
	return r.homeDir
}

// Send sends a command to the bridge.
func (r *Runtime) Send(command any) error {
	return r.channel.Send(command)
}

// start starts the routing goroutine.
func (r *Runtime) start() {
	go func() {
		defer close(r.routed)

		r.route()
	}()
}

// route reads frames until the channel ends.
func (r *Runtime) route() {
	for {
		frame, err := r.channel.Next(context.Background())
		if err != nil {
			r.handleClosed(err)

			return
		}

		r.routeFrame(frame)
	}
}

// routeFrame delivers one frame to its consumer.
func (r *Runtime) routeFrame(frame bridge.Frame) {
	switch message := frame.Message.(type) {
	case bridge.StreamMessage:
		r.routeStreamPart(message.Part)
	case bridge.UserMessageResponse:
		if turn := r.activeTurn(); turn != nil && turn.submitter != nil {
			turn.submitter.HandleResponse(message)
		}
	case bridge.Stop:
		select {
		case r.stopReplies <- message:
		default:
		}
	case bridge.Thread:
		if r.hooks.OnThread != nil {
			r.hooks.OnThread(message.ThreadID)
		}
	}
}

// routeStreamPart delivers a stream part to the active turn, holds it for
// the first turn of a continued session, or drops it between turns.
func (r *Runtime) routeStreamPart(part harness.StreamPart) {
	r.deliverMu.Lock()
	defer r.deliverMu.Unlock()

	r.mu.Lock()
	turn := r.turn
	if turn == nil && r.holding {
		r.held = append(r.held, part)
	}
	r.mu.Unlock()

	if turn != nil {
		r.deliver(turn, part)
	}
}

// deliver emits a part to a turn and settles the turn on its terminal
// parts.
func (r *Runtime) deliver(turn *Turn, part harness.StreamPart) {
	switch typed := part.(type) {
	case harness.FinishPart:
		if r.hooks.OnFinish != nil {
			r.hooks.OnFinish(typed)
		}

		turn.forward(part)
		turn.settle(nil)
	case harness.ErrorPart:
		turn.forward(part)
		turn.settle(typed.Err)
	default:
		turn.forward(part)
	}
}

// handleClosed ends the active turn when the channel ends: cleanly after a
// suspension, which leaves the turn running in the bridge, and with an error
// after any other close.
func (r *Runtime) handleClosed(err error) {
	reason := bridge.CloseReasonClosed
	if closed, ok := errors.AsType[*bridge.ClosedError](err); ok {
		reason = closed.Reason
	}

	r.mu.Lock()
	r.closed = true
	r.closeReason = reason
	r.holding = false
	r.held = nil
	turn := r.turn
	r.mu.Unlock()

	if turn != nil {
		turn.settle(r.closeError(reason))
	}
}

// closeError is the outcome of a turn whose channel ended for reason.
func (r *Runtime) closeError(reason string) error {
	if reason == bridge.CloseReasonSuspended {
		return nil
	}

	return errClosedBeforeFinish(r.harnessID)
}

// activeTurn returns the active turn, or nil.
func (r *Runtime) activeTurn() *Turn {
	r.mu.Lock()
	defer r.mu.Unlock()

	return r.turn
}

// activate makes turn the active turn and delivers the frames held for it.
// A turn that already settled, or one that starts after the channel ended,
// is not activated.
func (r *Runtime) activate(turn *Turn) {
	r.deliverMu.Lock()
	defer r.deliverMu.Unlock()

	r.mu.Lock()
	if r.closed {
		reason := r.closeReason
		r.mu.Unlock()

		turn.settle(r.closeError(reason))

		return
	}

	if turn.isSettled() {
		r.mu.Unlock()

		return
	}

	r.turn = turn
	held := r.held
	r.held = nil
	r.holding = false
	r.mu.Unlock()

	for _, part := range held {
		r.deliver(turn, part)
	}
}

// release clears turn when it is the active turn.
func (r *Runtime) release(turn *Turn) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.turn == turn {
		r.turn = nil
	}
}

// coordinates returns the coordinates of the live bridge with the cursor.
func (r *Runtime) coordinates(cursor int64) Coordinates {
	return Coordinates{
		Port:            r.port,
		Token:           r.token,
		LastSeenEventID: cursor,
		SandboxID:       r.sandboxID,
	}
}

// markStopped records the end of the local session, failing when it already
// ended.
func (r *Runtime) markStopped(failure string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.stopped {
		return fmt.Errorf("%s session %s %s", r.harnessID, r.sessionID, failure)
	}

	r.stopped = true

	return nil
}

// Detach freezes the channel without stopping the bridge and returns its
// coordinates, so that a later process can reattach before its next turn.
func (r *Runtime) Detach() (Coordinates, error) {
	if err := r.markStopped("is already stopped; cannot detach"); err != nil {
		return Coordinates{}, err
	}

	return r.suspend(), nil
}

// SuspendTurn freezes the host at the channel's cursor while the bridge
// keeps the active turn running and buffers its later events for the next
// process to replay. The active turn ends cleanly after the frames already
// received were delivered.
func (r *Runtime) SuspendTurn() (Coordinates, error) {
	if err := r.markStopped("is stopped; cannot suspend"); err != nil {
		return Coordinates{}, err
	}

	return r.suspend(), nil
}

// suspend suspends the channel and waits for the routing goroutine to
// deliver the frames already received.
func (r *Runtime) suspend() Coordinates {
	cursor := r.channel.Suspend()
	<-r.routed

	return r.coordinates(cursor)
}

// Stop asks the bridge for its resume data, then stops the bridge. A channel
// that already closed has no bridge to answer, so the data is empty.
func (r *Runtime) Stop(ctx context.Context) (json.RawMessage, error) {
	if err := r.markStopped("is already stopped; cannot stop"); err != nil {
		return nil, err
	}

	r.channel.BeginClose()

	data, err := r.requestStop(ctx)
	r.shutdown(ctx)

	if err != nil {
		return nil, err
	}

	return data, nil
}

// requestStop sends the stop command and waits for the bridge's reply.
func (r *Runtime) requestStop(ctx context.Context) (json.RawMessage, error) {
	if r.channel.Closed() {
		return json.RawMessage("{}"), nil
	}

	if err := r.channel.Send(bridge.SimpleCommand{
		Type: "stop",
	}); err != nil {
		return nil, fmt.Errorf("stop %s session %s: %w", r.harnessID, r.sessionID, err)
	}

	timer := time.NewTimer(shutdownGracePeriod)
	defer timer.Stop()

	select {
	case reply := <-r.stopReplies:
		return reply.Data, nil
	case <-timer.C:
		return nil, fmt.Errorf("%s session %s did not reply to stop within %s", r.harnessID, r.sessionID, shutdownGracePeriod)
	case <-ctx.Done():
		return nil, context.Cause(ctx)
	}
}

// Destroy stops the bridge without returning lifecycle state. It does
// nothing after Detach, SuspendTurn or Stop, and a second call waits for the
// first.
func (r *Runtime) Destroy(ctx context.Context) error {
	r.mu.Lock()
	if r.stopped {
		destroyed := r.destroyed
		r.mu.Unlock()

		if destroyed != nil {
			<-destroyed
		}

		return nil
	}

	r.stopped = true
	r.destroyed = make(chan struct{})
	r.mu.Unlock()

	defer close(r.destroyed)

	r.channel.BeginClose()
	sendDestroy(r.channel)

	r.shutdown(ctx)

	return nil
}

// shutdown gives the bridge process a moment to exit on its own, kills it,
// closes the channel and waits for every goroutine of the runtime.
func (r *Runtime) shutdown(ctx context.Context) {
	if r.bridgeProcess != nil {
		waitCtx, cancel := context.WithTimeout(ctx, shutdownGracePeriod)
		_, _ = r.bridgeProcess.Process().Wait(waitCtx)
		cancel()

		r.bridgeProcess.Kill(ctx)
	}

	_ = r.channel.Close()
	<-r.routed

	if r.bridgeProcess != nil {
		r.bridgeProcess.WaitOutput()
	}
}

// errClosedBeforeFinish reports a bridge connection that ended mid-turn.
func errClosedBeforeFinish(harnessID string) error {
	return errors.New(harnessID + " bridge closed before the turn finished")
}
