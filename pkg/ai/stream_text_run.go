package ai

import (
	"context"
	"errors"
	"iter"
	"slices"
	"sync/atomic"

	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/pkg/model"
)

// errFinishStepDropped reports a stream transform that swallowed a step's
// FinishStepPart, which the multi-step loop needs to continue.
var errFinishStepDropped = errors.New("stream transform did not forward the finish-step part")

// streamTextRun is the generation behind a StreamTextResult. All of its
// state is used on the consumer's goroutine.
type streamTextRun struct {
	*textGeneration

	options           StreamTextOptions
	hasExplicitOutput bool
	retention         streamRetention

	// streamRetries is the number of automatic stream retries; nil disables
	// stream retries.
	streamRetries *int

	// abort cancels the stream, for example when a timeout elapses.
	abort context.CancelCauseFunc

	recorder *streamRecorder

	// textIDs and reasoningIDs keep part IDs unique across steps, because
	// providers reuse them in every call.
	textIDs      *partIDReserver
	reasoningIDs *partIDReserver

	deferred deferredToolCalls
}

// streamTextRunSetup holds what a stream run starts from.
type streamTextRunSetup struct {
	generation    *textGeneration
	options       StreamTextOptions
	retention     streamRetention
	streamRetries *int
}

func newStreamTextRun(setup streamTextRunSetup) *streamTextRun {
	return &streamTextRun{
		textGeneration:    setup.generation,
		options:           setup.options,
		hasExplicitOutput: setup.options.Output != nil,
		retention:         setup.retention,
		streamRetries:     setup.streamRetries,
		textIDs:           newPartIDReserver(),
		reasoningIDs:      newPartIDReserver(),
		deferred:          make(deferredToolCalls),
	}
}

// streamExecution describes how a consumer receives the stream.
type streamExecution struct {
	// trackPartialOutput parses partial outputs from the streamed text.
	trackPartialOutput bool

	// deliver receives each part until it returns false.
	deliver func(part enrichedPart) bool

	// abort cancels the stream's context.
	abort context.CancelCauseFunc
}

// execute runs the stream pipeline: the producer, abort detection, the
// transforms, partial output parsing and the recorder, and returns the
// outcome.
func (r *streamTextRun) execute(ctx context.Context, execution streamExecution) streamOutcome {
	r.recorder = newStreamRecorder(r)
	r.abort = execution.abort

	source := r.abortable(ctx, func(yield func(StreamPart) bool) {
		r.produce(ctx, yield)
	})

	transformed := r.applyTransforms(ctx, source)
	enriched := enrichWithPartialOutput(partialOutputEnrichment{
		parts:   transformed,
		output:  r.output,
		enabled: execution.trackPartialOutput,
	})

	for part := range enriched {
		if boundary, isBoundary := part.part.(streamRetryBoundaryPart); isBoundary {
			r.recorder.resetAttempt(boundary)

			continue
		}

		invalidParts := r.recorder.record(ctx, part.part)

		if !deliverParts(execution.deliver, part, invalidParts) {
			return streamOutcome{
				err: ErrStreamClosed,
			}
		}
	}

	return r.recorder.finish(ctx)
}

// deliverParts delivers a part followed by the error parts its recording
// produced. It reports false when the consumer stopped.
func deliverParts(deliver func(part enrichedPart) bool, part enrichedPart, errorParts []StreamPart) bool {
	if !deliver(part) {
		return false
	}

	for _, errorPart := range errorParts {
		enriched := enrichedPart{
			part: errorPart,
		}

		if !deliver(enriched) {
			return false
		}
	}

	return true
}

// abortable ends the stream with an AbortPart as soon as ctx is cancelled.
func (r *streamTextRun) abortable(ctx context.Context, source iter.Seq[StreamPart]) iter.Seq[StreamPart] {
	return func(yield func(StreamPart) bool) {
		for part := range source {
			if ctx.Err() != nil {
				break
			}

			if !yield(part) {
				return
			}
		}

		if ctx.Err() == nil {
			return
		}

		reason := context.Cause(ctx)
		r.recorder.aborted = true

		abortCtx := context.WithoutCancel(ctx)
		event := StreamTextAbortEvent{
			CallID: r.callID,
			Steps:  r.recorder.steps,
			Reason: reason,
		}

		if r.options.OnAbort != nil {
			r.options.OnAbort(abortCtx, event)
		}

		r.telemetryDispatcher.onGenerateTextAbort(abortCtx, event)

		yield(AbortPart{
			Reason: reason.Error(),
		})
	}
}

// applyTransforms applies the configured transforms behind a gate that
// stops the source once a transform calls StopStream. Transforms may call
// StopStream from their own goroutines. When stream retries are enabled,
// each model call attempt is transformed independently: the transforms
// restart after every retry boundary, as upstream does.
func (r *streamTextRun) applyTransforms(ctx context.Context, source iter.Seq[StreamPart]) iter.Seq[StreamPart] {
	if len(r.options.Transforms) == 0 {
		return source
	}

	var stopped atomic.Bool

	if r.streamRetries == nil {
		return r.transformSegment(ctx, gatedParts(source, &stopped), &stopped)
	}

	return func(yield func(StreamPart) bool) {
		next, stop := iter.Pull(source)
		defer stop()

		for {
			segment := &attemptSegment{
				next:    next,
				stopped: &stopped,
			}

			for part := range r.transformSegment(ctx, segment.parts, &stopped) {
				if !yield(part) {
					return
				}
			}

			if segment.boundary == nil || !yield(*segment.boundary) {
				return
			}
		}
	}
}

// transformSegment passes parts through every transform.
func (r *streamTextRun) transformSegment(ctx context.Context, parts iter.Seq[StreamPart], stopped *atomic.Bool) iter.Seq[StreamPart] {
	for _, transform := range r.options.Transforms {
		parts = transform(ctx, StreamTransformInput{
			Parts: parts,
			Tools: r.tools,
			StopStream: func() {
				stopped.Store(true)
			},
		})
	}

	return parts
}

// gatedParts stops passing parts on once a transform stopped the stream.
func gatedParts(source iter.Seq[StreamPart], stopped *atomic.Bool) iter.Seq[StreamPart] {
	return func(yield func(StreamPart) bool) {
		for part := range source {
			if stopped.Load() || !yield(part) {
				return
			}
		}
	}
}

// attemptSegment yields the parts of one model call attempt and records the
// retry boundary that ends it.
type attemptSegment struct {
	next    func() (StreamPart, bool)
	stopped *atomic.Bool

	// boundary is set when the segment ended at a retry boundary.
	boundary *streamRetryBoundaryPart
}

func (s *attemptSegment) parts(yield func(StreamPart) bool) {
	for !s.stopped.Load() {
		part, ok := s.next()
		if !ok {
			return
		}

		if boundary, isBoundary := part.(streamRetryBoundaryPart); isBoundary {
			s.boundary = &boundary

			return
		}

		if !yield(part) {
			return
		}
	}
}

// produce generates the parts of every step.
func (r *streamTextRun) produce(ctx context.Context, yield func(StreamPart) bool) {
	if !yield(StartPart{}) {
		return
	}

	r.notifyStart(ctx, r.startEvent("ai.streamText"), r.options.OnStart)

	initialResponseMessages, ok := r.runInitialApprovals(ctx, yield)
	if !ok {
		return
	}

	r.recorder.initialResponseMessages = initialResponseMessages

	input := stepInput{
		system:           r.prompt.System,
		messages:         slices.Concat(r.initialPrompt.messages, initialResponseMessages),
		responseMessages: initialResponseMessages,
	}

	var usage LanguageModelUsage

	for {
		input.number = len(r.recorder.steps)
		input.steps = r.recorder.steps

		outcome, ok := r.streamStep(ctx, input, yield)
		if !ok {
			return
		}

		if len(r.recorder.steps) != input.number+1 {
			if r.recorder.stepErr == nil {
				yield(ErrorPart{
					Err: errFinishStepDropped,
				})
			}

			return
		}

		step := r.recorder.steps[input.number]
		usage = addLanguageModelUsage(usage, step.Usage)

		if !r.shouldContinue(ctx, outcome.progress, r.deferred, r.recorder.steps) {
			yield(FinishPart{
				FinishReason:    outcome.finishReason,
				RawFinishReason: outcome.rawFinishReason,
				TotalUsage:      usage,
			})

			return
		}

		input = stepInput{
			system:           outcome.prepared.system,
			messages:         slices.Concat(outcome.prepared.messages, step.Response.Messages),
			responseMessages: slices.Concat(input.responseMessages, step.Response.Messages),
		}
	}
}

// runInitialApprovals executes the tool calls approved in the prompt,
// streaming their denials, errors and results, and returns the tool message
// that reports them. It reports false when the stream must end.
func (r *streamTextRun) runInitialApprovals(ctx context.Context, yield func(StreamPart) bool) ([]model.Message, bool) {
	approvals, err := r.resolveInitialApprovals(ctx)
	if err != nil {
		r.fail(ctx, err, yield)

		return nil, false
	}

	if approvals.isEmpty() {
		return nil, true
	}

	if !yieldInitialApprovalOutcomes(approvals, yield) {
		return nil, false
	}

	outputs, ok := r.streamToolExecutions(ctx, streamedToolExecution{
		execution: r.toolExecution(r.tools, r.initialPrompt.messages),
		calls:     approvals.approvedCalls(),
		yield:     yield,
	})
	if !ok {
		return nil, false
	}

	message, hasMessage, err := r.initialToolMessage(ctx, initialToolMessageInput{
		outputs: outputs.outputs,
		invalid: approvals.invalid,
		denied:  approvals.deniedLocal,
	})
	if err != nil {
		r.fail(ctx, err, yield)

		return nil, false
	}

	if !hasMessage {
		return nil, true
	}

	return []model.Message{
		message,
	}, true
}

// yieldInitialApprovalOutcomes streams the denials of the prompt's approvals
// and the errors of its invalid approvals. It reports false when the
// consumer stopped.
func yieldInitialApprovalOutcomes(approvals initialApprovals, yield func(StreamPart) bool) bool {
	denied := slices.Concat(approvals.deniedLocal, approvals.deniedProviderExecuted)

	for _, approval := range denied {
		denial := ToolOutputDeniedPart{
			ToolCallID:       approval.call.ToolCallID,
			ToolName:         approval.call.ToolName,
			ProviderExecuted: approval.call.ProviderExecuted,
		}

		if !yield(denial) {
			return false
		}
	}

	for _, approval := range approvals.invalid {
		toolError := ToolError{
			ToolCallID:   approval.call.ToolCallID,
			ToolName:     approval.call.ToolName,
			Input:        approval.call.Input,
			Error:        approval.err,
			Dynamic:      approval.call.Dynamic,
			Title:        approval.call.Title,
			ToolMetadata: approval.call.ToolMetadata,
		}

		if !yield(toolError) {
			return false
		}
	}

	return true
}

// fail ends the stream with an error part and reports the error to the
// telemetry integrations.
func (r *streamTextRun) fail(ctx context.Context, err error, yield func(StreamPart) bool) {
	r.telemetryDispatcher.onError(ctx, TelemetryErrorEvent{
		CallID: r.callID,
		Error:  err,
	})

	yield(ErrorPart{
		Err: err,
	})
}

// partIDReserver hands out unique part IDs, replacing reused ones with
// generated IDs.
type partIDReserver struct {
	used map[string]bool
}

func newPartIDReserver() *partIDReserver {
	return &partIDReserver{
		used: make(map[string]bool),
	}
}

func (r *partIDReserver) reserve(providerID string) string {
	if !r.used[providerID] {
		r.used[providerID] = true

		return providerID
	}

	for {
		generated := id.New("aitxt")
		if !r.used[generated] {
			r.used[generated] = true

			return generated
		}
	}
}
