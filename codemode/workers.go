package codemode

import (
	"errors"
	"fmt"
	"sync"
)

// DefaultMaxWorkers is the process-wide number of invocations that may
// execute at the same time unless SetMaxWorkers changed it.
const DefaultMaxWorkers = 32

// workerSlots counts the invocations executing in this process. Like
// upstream's worker pool, it is shared by every invocation of the package.
var workerSlots = struct {
	mu         sync.Mutex
	active     int
	maxWorkers int
}{
	maxWorkers: DefaultMaxWorkers,
}

// SetMaxWorkers sets how many invocations may execute at the same time in
// this process, across every tool and Run call. An invocation that would
// exceed the cap fails immediately with an Error whose code is
// ErrorCodeConcurrency; invocations are not queued. Zero restores
// DefaultMaxWorkers. Lowering the cap does not stop invocations that are
// already executing.
//
// Upstream derives its default from the available memory and caps it at
// 32; the Go interpreter cannot bound the memory of a program, so the
// default is that cap.
func SetMaxWorkers(maxWorkers int) error {
	if maxWorkers < 0 {
		return errors.New("max workers must not be negative")
	}

	workerSlots.mu.Lock()
	defer workerSlots.mu.Unlock()

	workerSlots.maxWorkers = positiveOr(maxWorkers, DefaultMaxWorkers)

	return nil
}

// acquireWorker reserves an execution slot. The returned function releases
// it and must be called exactly once.
func acquireWorker() (func(), error) {
	workerSlots.mu.Lock()
	defer workerSlots.mu.Unlock()

	if workerSlots.active >= workerSlots.maxWorkers {
		return nil, newConcurrencyError(workerSlots.maxWorkers)
	}

	workerSlots.active++

	return sync.OnceFunc(releaseWorker), nil
}

func releaseWorker() {
	workerSlots.mu.Lock()
	defer workerSlots.mu.Unlock()

	workerSlots.active--
}

func newConcurrencyError(maxWorkers int) *Error {
	return &Error{
		Code:    ErrorCodeConcurrency,
		Message: fmt.Sprintf("max workers limit reached (%d)", maxWorkers),
		Limit:   int64(maxWorkers),
	}
}
