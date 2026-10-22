package tui

import (
	"context"
	"sync"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/ai/ui"
	"github.com/aarock1234/ai/pkg/model"
)

// streamPartMsg delivers one part of a local agent's stream to the UI.
type streamPartMsg struct {
	part ai.StreamPart
}

// chunkMsg delivers one chunk of a remote agent's UI message stream to the
// UI.
type chunkMsg struct {
	chunk ui.Chunk
}

// turnFinishedMsg reports the end of a turn. A local turn carries the
// messages the agent generated, and a remote turn the assistant message it
// streamed. Err is the error of the turn, if any.
type turnFinishedMsg struct {
	responseMessages []model.Message
	responseMessage  *ui.Message
	err              error
}

// turnFunc streams one turn, sending its events and then a
// turnFinishedMsg. It returns early once ctx is canceled.
type turnFunc func(ctx context.Context, events chan<- tea.Msg)

// turnRunner streams turns on a goroutine of their own. RunAgent owns it:
// it stops the running turn and waits for it before returning.
type turnRunner struct {
	ctx context.Context

	wg sync.WaitGroup

	mu         sync.Mutex
	cancelTurn context.CancelFunc
}

func newTurnRunner(ctx context.Context) *turnRunner {
	return &turnRunner{
		ctx: ctx,
	}
}

// start streams one turn. The returned channel receives the turn's events
// and is closed when the turn ends.
func (r *turnRunner) start(turn turnFunc) <-chan tea.Msg {
	turnCtx, cancel := context.WithCancel(r.ctx)
	events := make(chan tea.Msg)

	r.mu.Lock()
	r.cancelTurn = cancel
	r.mu.Unlock()

	r.wg.Go(func() {
		defer cancel()
		defer close(events)

		turn(turnCtx, events)
	})

	return events
}

// stop cancels the running turn.
func (r *turnRunner) stop() {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.cancelTurn != nil {
		r.cancelTurn()
	}
}

// wait returns once every turn has ended.
func (r *turnRunner) wait() {
	r.wg.Wait()
}

// send delivers a message to the UI and reports whether it was delivered
// before the turn was cancelled.
func send(ctx context.Context, events chan<- tea.Msg, msg tea.Msg) bool {
	select {
	case events <- msg:
		return true
	case <-ctx.Done():
		return false
	}
}

// waitForTurnEvent returns a command that delivers the next event of a
// turn. It delivers nothing once the turn's channel is closed.
func waitForTurnEvent(events <-chan tea.Msg) tea.Cmd {
	return func() tea.Msg {
		return <-events
	}
}
