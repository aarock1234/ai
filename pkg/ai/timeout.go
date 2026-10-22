package ai

import (
	"context"
	"fmt"
	"maps"
	"slices"
	"time"
)

// TimeoutConfiguration limits steps, model output chunks and tool executions.
// Zero durations apply no limit. The total timeout of a call comes from its
// context, which replaces upstream's totalMs.
type TimeoutConfiguration struct {
	// Step limits each complete step, including the model call, its retries
	// and local tool execution.
	Step time.Duration

	// FirstChunk limits the wait for the first output chunk of each model
	// call. Only streaming calls enforce it.
	FirstChunk time.Duration

	// Chunk limits the gap between output chunks, from the first output
	// until the model response ends. Only streaming calls enforce it.
	Chunk time.Duration

	// Tool is the default limit of each tool execution.
	Tool time.Duration

	// Tools overrides Tool for the named tools.
	Tools map[string]time.Duration
}

// TimeoutKind names the limit that a TimeoutError reports.
type TimeoutKind string

// Timeout kinds.
const (
	TimeoutStep       TimeoutKind = "step"
	TimeoutFirstChunk TimeoutKind = "first chunk"
	TimeoutChunk      TimeoutKind = "chunk"
	TimeoutTool       TimeoutKind = "tool"
)

// TimeoutError reports that a step, chunk or tool timeout elapsed. It is the
// cancellation cause of the affected context and matches
// context.DeadlineExceeded with errors.Is.
type TimeoutError struct {
	Kind    TimeoutKind
	Timeout time.Duration
}

// Error describes the elapsed limit.
func (e *TimeoutError) Error() string {
	return fmt.Sprintf("%s timeout of %s exceeded", e.Kind, e.Timeout)
}

// Is reports whether target is context.DeadlineExceeded.
func (e *TimeoutError) Is(target error) bool {
	return target == context.DeadlineExceeded
}

// namedTimeout is one limit of a configuration, named as its option.
type namedTimeout struct {
	parameter string
	timeout   time.Duration
}

// validate rejects negative durations.
func (c TimeoutConfiguration) validate() error {
	limits := []namedTimeout{
		{
			parameter: "timeout.step",
			timeout:   c.Step,
		},
		{
			parameter: "timeout.firstChunk",
			timeout:   c.FirstChunk,
		},
		{
			parameter: "timeout.chunk",
			timeout:   c.Chunk,
		},
		{
			parameter: "timeout.tool",
			timeout:   c.Tool,
		},
	}

	for _, toolName := range slices.Sorted(maps.Keys(c.Tools)) {
		limits = append(limits, namedTimeout{
			parameter: "timeout.tools." + toolName,
			timeout:   c.Tools[toolName],
		})
	}

	for _, limit := range limits {
		if limit.timeout < 0 {
			return &InvalidArgumentError{
				Parameter: limit.parameter,
				Value:     limit.timeout,
				Message:   "timeouts must be >= 0",
			}
		}
	}

	return nil
}

// toolTimeout returns the limit of one tool's executions.
func (c TimeoutConfiguration) toolTimeout(toolName string) time.Duration {
	if timeout, ok := c.Tools[toolName]; ok {
		return timeout
	}

	return c.Tool
}

// withTimeout derives a context that ends with a TimeoutError once timeout
// elapses. A zero timeout returns ctx unchanged.
func withTimeout(ctx context.Context, kind TimeoutKind, timeout time.Duration) (context.Context, context.CancelFunc) {
	if timeout <= 0 {
		return ctx, func() {}
	}

	return context.WithTimeoutCause(ctx, timeout, &TimeoutError{
		Kind:    kind,
		Timeout: timeout,
	})
}

// streamTimers abort a stream when its step, first-chunk or chunk timeout
// elapses, as upstream's merged abort signal does. The timers are armed and
// stopped on the consumer's goroutine; a firing timer only cancels the
// stream.
type streamTimers struct {
	timeouts TimeoutConfiguration
	abort    context.CancelCauseFunc

	step       *time.Timer
	firstChunk *time.Timer
	chunk      *time.Timer
}

func newStreamTimers(timeouts TimeoutConfiguration, abort context.CancelCauseFunc) *streamTimers {
	return &streamTimers{
		timeouts: timeouts,
		abort:    abort,
	}
}

// startStep arms the step timeout.
func (t *streamTimers) startStep() {
	t.step = t.arm(TimeoutStep, t.timeouts.Step)
}

// startFirstChunk arms the first-chunk timeout when a model stream opens.
func (t *streamTimers) startFirstChunk() {
	stopTimer(t.firstChunk)
	t.firstChunk = t.arm(TimeoutFirstChunk, t.timeouts.FirstChunk)
}

// observeOutput clears the first-chunk timeout and restarts the chunk
// timeout after an output chunk.
func (t *streamTimers) observeOutput() {
	stopTimer(t.firstChunk)
	t.firstChunk = nil

	stopTimer(t.chunk)
	t.chunk = t.arm(TimeoutChunk, t.timeouts.Chunk)
}

// stopOutput clears the timeouts of the model output.
func (t *streamTimers) stopOutput() {
	stopTimer(t.firstChunk)
	stopTimer(t.chunk)

	t.firstChunk = nil
	t.chunk = nil
}

// stop clears every timeout of the step.
func (t *streamTimers) stop() {
	t.stopOutput()

	stopTimer(t.step)
	t.step = nil
}

func (t *streamTimers) arm(kind TimeoutKind, timeout time.Duration) *time.Timer {
	if timeout <= 0 || t.abort == nil {
		return nil
	}

	return time.AfterFunc(timeout, func() {
		t.abort(&TimeoutError{
			Kind:    kind,
			Timeout: timeout,
		})
	})
}

func stopTimer(timer *time.Timer) {
	if timer != nil {
		timer.Stop()
	}
}
