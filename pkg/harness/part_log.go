package harness

import (
	"sync"

	"github.com/aarock1234/ai/pkg/ai"
)

// loggedPart is one stream part with the partial output it completes.
type loggedPart struct {
	part ai.StreamPart

	// partialOutput is set when the part changed the partial output.
	partialOutput    any
	hasPartialOutput bool
}

// partLog is an append-only, replayable log of the parts of one turn. Every
// reader starts at the first part, so any number of streams can be read, in
// any order, like upstream's teed streams. It is safe for concurrent use.
type partLog struct {
	mu     sync.Mutex
	cond   *sync.Cond
	parts  []loggedPart
	closed bool
}

// newPartLog returns an empty, open log.
func newPartLog() *partLog {
	log := &partLog{}
	log.cond = sync.NewCond(&log.mu)

	return log
}

// append adds parts. Parts appended after close are dropped.
func (l *partLog) append(parts ...loggedPart) {
	l.mu.Lock()
	defer l.mu.Unlock()

	if l.closed {
		return
	}

	l.parts = append(l.parts, parts...)
	l.cond.Broadcast()
}

// close ends the log.
func (l *partLog) close() {
	l.mu.Lock()
	defer l.mu.Unlock()

	l.closed = true
	l.cond.Broadcast()
}

// at returns the part at index, blocking until it exists. It reports false
// once the log is closed and index is past its end.
func (l *partLog) at(index int) (loggedPart, bool) {
	l.mu.Lock()
	defer l.mu.Unlock()

	for index >= len(l.parts) && !l.closed {
		l.cond.Wait()
	}

	if index >= len(l.parts) {
		return loggedPart{}, false
	}

	return l.parts[index], true
}

// all yields every logged part in order until the log closes or yield
// returns false.
func (l *partLog) all(yield func(part loggedPart) bool) {
	for index := 0; ; index++ {
		part, ok := l.at(index)
		if !ok || !yield(part) {
			return
		}
	}
}
