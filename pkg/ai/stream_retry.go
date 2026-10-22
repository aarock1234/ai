package ai

import (
	"context"
	"errors"
	"iter"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// streamAttempts reads the model call of a streamed step. When stream
// retries are enabled, a provider error after the stream opened reruns the
// model call: the parts of the failed attempt that are still held back are
// dropped, open text and reasoning blocks are closed, and a retry boundary
// separates the attempts. It also drives the first-chunk and chunk
// timeouts.
type streamAttempts struct {
	run    *streamTextRun
	timers *streamTimers

	// open opens one model call, retrying failures to open it.
	open func(ctx context.Context) (*LanguageModelCallStream, error)

	// current is the stream of the running attempt.
	current *LanguageModelCallStream

	automaticRetries int
	callbackRetries  int
}

// attemptOutcome tells whether an attempt goes on and how it ended.
type attemptOutcome int

const (
	// attemptContinues means the attempt reads its next part.
	attemptContinues attemptOutcome = iota

	// attemptCompleted means the model stream ended.
	attemptCompleted

	// attemptStopped means the consumer stopped.
	attemptStopped

	// attemptRetry means the failed attempt must be rerun.
	attemptRetry
)

// continuesIf returns attemptContinues when the consumer accepted the part
// and attemptStopped otherwise.
func continuesIf(accepted bool) attemptOutcome {
	if accepted {
		return attemptContinues
	}

	return attemptStopped
}

// newStreamAttempts prepares the model call attempts of a step.
func (r *streamTextRun) newStreamAttempts(prepared preparedStep, timers *streamTimers) *streamAttempts {
	call := r.languageModelCall(prepared, r.retention.collectContent)

	return &streamAttempts{
		run:    r,
		timers: timers,
		open: func(ctx context.Context) (*LanguageModelCallStream, error) {
			return retryCall(ctx, r.retry, func(ctx context.Context) (*LanguageModelCallStream, error) {
				return openLanguageModelCall(ctx, call)
			})
		},
	}
}

// start releases the previous attempt's stream, opens the next attempt's
// model call and arms its first-chunk timeout. The previous stream stays
// current until the next one opens so that its response metadata remains
// readable when opening fails.
func (a *streamAttempts) start(ctx context.Context) error {
	a.close()

	stream, err := a.open(ctx)
	if err != nil {
		return err
	}

	a.current = stream
	a.timers.startFirstChunk()

	return nil
}

// close releases the current attempt's stream. Closing a stream more than
// once is safe.
func (a *streamAttempts) close() {
	if a.current == nil {
		return
	}

	// The attempt is over; a release failure cannot change its outcome.
	_ = a.current.Close()
}

// parts yields the parts of every attempt, separated by retry boundaries.
func (a *streamAttempts) parts(ctx context.Context) iter.Seq[LanguageModelStreamPart] {
	return func(yield func(LanguageModelStreamPart) bool) {
		boundaryPending := false

		for {
			if a.forward(ctx, boundaryPending, yield) != attemptRetry {
				return
			}

			if err := a.start(ctx); err != nil {
				yield(ErrorPart{
					Err: err,
				})

				return
			}

			boundaryPending = true
		}
	}
}

// forward passes on the parts of the current attempt. A retry boundary
// opens a retried attempt; it carries the warnings of the attempt's start
// part.
func (a *streamAttempts) forward(ctx context.Context, boundaryPending bool, yield func(LanguageModelStreamPart) bool) attemptOutcome {
	attempt := &attemptParts{
		yield:       yield,
		bufferTools: a.buffersToolParts(),
	}

	for part := range a.current.Parts(ctx) {
		if boundaryPending {
			boundaryPending = false

			if !yield(streamRetryBoundary{
				warnings: startWarnings(part),
			}) {
				return attemptStopped
			}
		}

		if outcome := a.route(ctx, attempt, part); outcome != attemptContinues {
			return outcome
		}
	}

	if boundaryPending && !yield(streamRetryBoundary{}) {
		return attemptStopped
	}

	a.timers.stopOutput()

	if !attempt.flush() {
		return attemptStopped
	}

	return attemptCompleted
}

// route handles one part of an attempt and tells whether the attempt goes
// on.
func (a *streamAttempts) route(ctx context.Context, attempt *attemptParts, part LanguageModelStreamPart) attemptOutcome {
	if end, isEnd := part.(ModelCallEndPart); isEnd {
		a.timers.stopOutput()

		return continuesIf(attempt.flush() && attempt.emit(end))
	}

	if isLanguageModelOutputPart(part) {
		a.timers.observeOutput()
	}

	errorPart, isError := part.(ErrorPart)
	if !isError {
		if attempt.holdsBack(part) {
			attempt.hold(part)

			return attemptContinues
		}

		return continuesIf(attempt.emit(part))
	}

	a.timers.stopOutput()

	if a.run.streamRetries == nil {
		return continuesIf(attempt.flush() && attempt.emit(errorPart))
	}

	if a.shouldRetry(ctx, errorPart) {
		if !attempt.discard() {
			return attemptStopped
		}

		return attemptRetry
	}

	errorPart.callbacksHandled = true

	return continuesIf(attempt.flush() && attempt.emit(errorPart))
}

// shouldRetry passes a stream error to OnChunk and OnError and decides
// whether to retry: automatically while automatic retries remain, and once
// more when OnError requests it. Tool choice violations are never retried.
func (a *streamAttempts) shouldRetry(ctx context.Context, part ErrorPart) bool {
	options := a.run.options

	if options.OnChunk != nil {
		options.OnChunk(ctx, StreamTextChunkEvent{
			Chunk: part,
		})
	}

	retryRequested := false

	if options.OnError != nil {
		options.OnError(ctx, StreamTextErrorEvent{
			Error: part.Err,
			requestRetry: func() {
				retryRequested = true
			},
		})
	}

	if _, ok := errors.AsType[*ToolChoiceViolationError](part.Err); ok {
		return false
	}

	if a.automaticRetries < *a.run.streamRetries {
		a.automaticRetries++

		return true
	}

	if retryRequested && a.callbackRetries < 1 {
		a.callbackRetries++

		return true
	}

	return false
}

// buffersToolParts reports whether a retry is possible, in which case tool
// parts are held back until the model call ends so that a failed attempt
// never runs tools.
func (a *streamAttempts) buffersToolParts() bool {
	retries := a.run.streamRetries
	if retries == nil {
		return false
	}

	return *retries > 0 || a.run.options.OnError != nil
}

// startWarnings returns the warnings of a model call start part.
func startWarnings(part LanguageModelStreamPart) []model.Warning {
	if start, ok := part.(ModelCallStartPart); ok {
		return start.Warnings
	}

	return nil
}

// attemptParts passes on the parts of one attempt. Once a tool part arrives
// while retries are possible, it and every later part are held back until
// the model call ends.
type attemptParts struct {
	yield       func(LanguageModelStreamPart) bool
	bufferTools bool

	held []LanguageModelStreamPart

	// openText and openReasoning are the IDs of the blocks that started but
	// did not end, in start order.
	openText      []string
	openReasoning []string
}

func (p *attemptParts) holdsBack(part LanguageModelStreamPart) bool {
	return p.bufferTools && (isToolStreamPart(part) || len(p.held) > 0)
}

func (p *attemptParts) hold(part LanguageModelStreamPart) {
	p.held = append(p.held, part)
}

// emit passes a part on and tracks the open text and reasoning blocks.
func (p *attemptParts) emit(part LanguageModelStreamPart) bool {
	switch typed := part.(type) {
	case TextStartPart:
		p.openText = append(p.openText, typed.ID)
	case TextEndPart:
		p.openText = slices.DeleteFunc(p.openText, isID(typed.ID))
	case ReasoningStartPart:
		p.openReasoning = append(p.openReasoning, typed.ID)
	case ReasoningEndPart:
		p.openReasoning = slices.DeleteFunc(p.openReasoning, isID(typed.ID))
	}

	return p.yield(part)
}

// flush passes on the held-back parts.
func (p *attemptParts) flush() bool {
	held := p.held
	p.held = nil

	for _, part := range held {
		if !p.emit(part) {
			return false
		}
	}

	return true
}

// discard drops the held-back parts of a failed attempt and closes its
// open text and reasoning blocks. It reports false when the consumer
// stopped.
func (p *attemptParts) discard() bool {
	p.held = nil

	for _, id := range p.openText {
		if !p.yield(TextEndPart{
			ID: id,
		}) {
			return false
		}
	}

	for _, id := range p.openReasoning {
		if !p.yield(ReasoningEndPart{
			ID: id,
		}) {
			return false
		}
	}

	p.openText = nil
	p.openReasoning = nil

	return true
}

func isID(id string) func(candidate string) bool {
	return func(candidate string) bool {
		return candidate == id
	}
}

// isToolStreamPart reports whether a part belongs to a tool call.
func isToolStreamPart(part LanguageModelStreamPart) bool {
	switch part.(type) {
	case ToolInputStartPart, ToolInputDeltaPart, ToolInputEndPart, ToolCall,
		ToolApprovalRequest, ToolApprovalResponse, ToolResult, ToolError:
		return true
	default:
		return false
	}
}
