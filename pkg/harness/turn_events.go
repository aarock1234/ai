package harness

import (
	"context"
	"sync"
)

// turnEvents bridges an adapter's emit callback into a pull-based queue. It
// is the Go form of upstream toHarnessStream. The queue has no size limit,
// so emit never blocks the adapter: an adapter may emit from a goroutine
// that waits on a tool result the agent is about to submit.
//
// The queue closes when the control's Done channel closes; a turn failure is
// delivered as a final ErrorPart rather than as a separate error, so the
// reader needs a single loop.
type turnEvents struct {
	mu     sync.Mutex
	parts  []StreamPart
	closed bool

	// ready receives a value when parts were added or the queue closed.
	ready chan struct{}

	// stop ends the watcher early when the reader is done.
	stop     chan struct{}
	stopOnce sync.Once
	watcher  sync.WaitGroup
}

// newTurnEvents returns an open, empty queue.
func newTurnEvents() *turnEvents {
	return &turnEvents{
		ready: make(chan struct{}, 1),
		stop:  make(chan struct{}),
	}
}

// emit appends a part. Parts emitted after the queue closed are dropped.
func (e *turnEvents) emit(part StreamPart) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed {
		return
	}

	e.parts = append(e.parts, part)
	e.signal()
}

// watch closes the queue when control's turn ends, appending its failure as
// an ErrorPart. The watcher stops early when release is called.
func (e *turnEvents) watch(control PromptControl) {
	e.watcher.Go(func() {
		select {
		case <-control.Done():
			e.closeWithError(control.Err())
		case <-e.stop:
		}
	})
}

// closeWithError closes the queue, appending err as an ErrorPart when it is
// not nil.
func (e *turnEvents) closeWithError(err error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed {
		return
	}

	if err != nil {
		e.parts = append(e.parts, ErrorPart{
			Err: err,
		})
	}

	e.closed = true
	e.signal()
}

// next returns the next part. It reports false when the queue is closed and
// drained, and returns ctx's error when ctx ends first.
func (e *turnEvents) next(ctx context.Context) (StreamPart, bool, error) {
	for {
		e.mu.Lock()
		if len(e.parts) > 0 {
			part := e.parts[0]
			e.parts[0] = nil
			e.parts = e.parts[1:]
			e.mu.Unlock()

			return part, true, nil
		}

		closed := e.closed
		e.mu.Unlock()

		if closed {
			return nil, false, nil
		}

		select {
		case <-e.ready:
		case <-ctx.Done():
			return nil, false, context.Cause(ctx)
		}
	}
}

// release stops the watcher and waits for it. The reader calls it once it
// stops reading.
func (e *turnEvents) release() {
	e.stopOnce.Do(func() {
		close(e.stop)
	})

	e.watcher.Wait()
}

// signal wakes the reader. The caller holds mu.
func (e *turnEvents) signal() {
	select {
	case e.ready <- struct{}{}:
	default:
	}
}
