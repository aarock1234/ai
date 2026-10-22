// Package poll paces the status requests of asynchronous provider jobs, such
// as image, video and transcription generations. It ports the upstream
// delay helper with an abort signal.
package poll

import (
	"context"
	"time"
)

// Wait pauses for delay or until ctx ends. It returns the context's cause
// when ctx ends first. A delay of zero or less does not pause and only
// reports whether ctx has already ended.
func Wait(ctx context.Context, delay time.Duration) error {
	if delay <= 0 {
		return context.Cause(ctx)
	}

	timer := time.NewTimer(delay)
	defer timer.Stop()

	select {
	case <-ctx.Done():
		return context.Cause(ctx)
	case <-timer.C:
		return nil
	}
}
