package ai

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"sync"

	"github.com/aarock1234/ai/pkg/model"
)

// ErrStreamClosed reports a stream that was closed or abandoned before it
// completed.
var ErrStreamClosed = errors.New("stream closed before completion")

// StreamTextOptions configures StreamText.
type StreamTextOptions struct {
	Model model.LanguageModel

	CallSettings
	Prompt

	// Tools are the tools the model may call, in declaration order.
	Tools []Tool

	// ToolChoice controls tool selection. Nil selects automatic choice.
	ToolChoice *model.ToolChoice

	// ActiveTools restricts the tools available to the model without
	// changing the declared tools. Nil applies no restriction.
	ActiveTools []string

	// ToolOrder lists tool names to send first; the other tools follow in
	// name order. Nil keeps declaration order.
	ToolOrder []string

	// ToolCallConcurrency limits the tool calls that run at once within a
	// step. Zero means no limit.
	ToolCallConcurrency int

	// StopWhen ends the tool-calling loop when any condition is met. Nil
	// stops after one step.
	StopWhen []StopCondition

	// Output specifies the shape of the output. Nil generates text.
	Output Output

	// ToolApproval decides whether tool calls need approval: a
	// ToolApprovalFunc for every call, or ToolApprovals per tool. Its
	// decisions replace the NeedsApproval functions of the tools.
	ToolApproval ToolApprovalConfiguration

	// ToolApprovalSecret signs tool approval requests and verifies the
	// approvals in the prompt.
	ToolApprovalSecret []byte

	// ToolCallers maps tool names to the callers that may call them, as in
	// GenerateTextOptions.
	ToolCallers map[string][]string

	ProviderOptions model.ProviderOptions

	// PrepareStep adjusts the settings of each step.
	PrepareStep PrepareStepFunc

	// RepairToolCall attempts to repair tool calls that fail to parse.
	RepairToolCall ToolCallRepairFunc

	// RefineToolInput maps tool names to refinements of their validated
	// inputs.
	RefineToolInput map[string]ToolInputRefinement

	// Transforms are applied to the stream in order, for example
	// SmoothStream. They must keep the structure of the stream intact.
	Transforms []StreamTransform

	// Download resolves file URLs in the prompt. Nil downloads the URLs the
	// model does not support.
	Download DownloadFunction

	// Include selects the payloads retained in step results and the stream.
	Include StreamTextInclude

	// StreamRetries is the number of automatic retries of a step whose model
	// stream reports a provider error after it opened. A retry reruns only
	// the current model call; completed steps and their tool results are
	// kept. Output of the failed attempt that was already streamed cannot be
	// retracted, but it is excluded from the step result, the parsed output
	// and later steps. While retries are possible, tool parts are held back
	// until the model call ends.
	//
	// Nil disables stream retries. Zero disables automatic retries but lets
	// OnError request one retry through StreamTextErrorEvent.Retry, which it
	// may also do once after the automatic retries are exhausted.
	StreamRetries *int

	// Timeout limits each step, the first output chunk and the gaps between
	// output chunks of each model call, and tool executions. An elapsed
	// limit aborts the stream with a TimeoutError. The total timeout comes
	// from ctx.
	Timeout TimeoutConfiguration

	// Telemetry sends the lifecycle events of the generation to telemetry
	// integrations.
	Telemetry TelemetryOptions

	// LogWarnings receives the warnings of each step. Nil discards them; they
	// are always available on the step results.
	LogWarnings LogWarningsFunc

	// OnChunk is called for each part before the consumer receives it.
	OnChunk func(ctx context.Context, event StreamTextChunkEvent)

	// OnError is called for each error part.
	OnError func(ctx context.Context, event StreamTextErrorEvent)

	// OnStart is called before the first model call.
	OnStart func(ctx context.Context, event GenerateTextStartEvent)

	// OnStepStart is called before each step.
	OnStepStart func(ctx context.Context, event GenerateTextStepStartEvent)

	// OnLanguageModelCallStart is called immediately before each model call
	// attempt and OnLanguageModelCallEnd when its response has finished,
	// before any client-side tool runs.
	OnLanguageModelCallStart func(ctx context.Context, event LanguageModelCallStartEvent)
	OnLanguageModelCallEnd   func(ctx context.Context, event LanguageModelCallEndEvent)

	// OnToolExecutionStart and OnToolExecutionEnd are called around each
	// tool execution. They may be called concurrently.
	OnToolExecutionStart func(ctx context.Context, event ToolExecutionStartEvent)
	OnToolExecutionEnd   func(ctx context.Context, event ToolExecutionEndEvent)

	// OnStepEnd is called after each step.
	OnStepEnd func(ctx context.Context, step StepResult)

	// OnEnd is called after the final step, unless the stream fails or is
	// aborted.
	OnEnd func(ctx context.Context, event GenerateTextEndEvent)

	// OnAbort is called when the context is cancelled or the result is
	// closed while streaming.
	OnAbort func(ctx context.Context, event StreamTextAbortEvent)
}

// StreamTextInclude selects payloads to retain. Request payloads are dropped
// by default to reduce memory use.
type StreamTextInclude struct {
	RequestBody     bool
	RequestMessages bool

	// RawChunks emits RawPart parts with the provider's native events.
	RawChunks bool
}

// StreamTextChunkEvent carries one stream part.
type StreamTextChunkEvent struct {
	Chunk StreamPart
}

// StreamTextErrorEvent carries the error of an error part.
type StreamTextErrorEvent struct {
	Error error

	// requestRetry records a retry request. Nil when the error cannot be
	// retried.
	requestRetry func()
}

// Retry asks StreamText to retry the current model call after a provider
// stream error. It only has an effect when StreamRetries is set; see
// StreamTextOptions.StreamRetries.
func (e StreamTextErrorEvent) Retry() {
	if e.requestRetry != nil {
		e.requestRetry()
	}
}

// StreamTextAbortEvent describes an aborted stream.
type StreamTextAbortEvent struct {
	CallID string

	// Steps are the steps completed before the abort.
	Steps []StepResult

	Reason error
}

// StreamTransform transforms the stream of parts. StopStream ends the
// source stream; parts the transform yields afterwards still pass through.
type StreamTransform func(ctx context.Context, input StreamTransformInput) iter.Seq[StreamPart]

// StreamTransformInput is the input of a StreamTransform.
type StreamTransformInput struct {
	Parts iter.Seq[StreamPart]
	Tools []Tool

	// StopStream stops the source stream.
	StopStream func()
}

// StreamTextResult is a streaming text generation with a single consumer.
// Generation starts when one of its streams is ranged over, runs on the
// consumer's goroutine (tools run on their own goroutines) and stops when the
// consumer stops. Breaking out of the range releases the provider stream and
// waits for running tools.
//
// Range over exactly one of FullStream, TextStream, PartialOutputStream and
// ElementStream, once. Result, Output and ConsumeStream consume the stream if
// no stream was ranged over; otherwise they wait for it to end. Do not call
// them while ranging over a stream on the same goroutine.
type StreamTextResult struct {
	run    *streamTextRun
	parent context.Context

	mu      sync.Mutex
	claimed bool
	cancel  context.CancelCauseFunc
	done    chan struct{}

	outcome streamOutcome
}

// streamOutcome is the final state of a stream.
type streamOutcome struct {
	// result is nil when the stream does not replay its steps.
	result    *GenerateTextResult
	summary   StreamTextSummary
	err       error
	output    any
	outputErr error
}

// streamProjection identifies the stream a consumer chose.
type streamProjection int

const (
	projectFullStream streamProjection = iota
	projectTextStream
	projectPartialOutputStream
	projectElementStream
)

// StreamText generates text and calls tools with a language model,
// streaming the output. Option errors are returned immediately; errors
// during generation are reported as ErrorPart parts and by Result.
//
// Cancellation and deadlines come from ctx. Cancelling it while streaming
// ends the stream with an AbortPart.
func StreamText(ctx context.Context, options StreamTextOptions) (*StreamTextResult, error) {
	return newStreamTextResult(ctx, options, streamRetention{
		replay:         true,
		collectContent: true,
	})
}

// streamRetention selects what a stream keeps while it runs.
type streamRetention struct {
	// replay keeps the steps for the aggregated result.
	replay bool

	// collectContent records the content of each step.
	collectContent bool
}

func newStreamTextResult(ctx context.Context, options StreamTextOptions, retention streamRetention) (*StreamTextResult, error) {
	generation, err := newTextGeneration(newTextGenerationConfig(options.generateTextOptions()))
	if err != nil {
		return nil, err
	}

	streamRetries, err := prepareStreamRetries(options.StreamRetries)
	if err != nil {
		return nil, err
	}

	return &StreamTextResult{
		run: newStreamTextRun(streamTextRunSetup{
			generation:    generation,
			options:       options,
			retention:     retention,
			streamRetries: streamRetries,
		}),
		parent: ctx,
		done:   make(chan struct{}),
	}, nil
}

// prepareStreamRetries validates the StreamRetries option. Nil disables
// stream retries.
func prepareStreamRetries(streamRetries *int) (*int, error) {
	if streamRetries != nil && *streamRetries < 0 {
		return nil, &InvalidArgumentError{
			Parameter: "streamRetries",
			Value:     *streamRetries,
			Message:   "streamRetries must be >= 0",
		}
	}

	return streamRetries, nil
}

// generateTextOptions returns the generation settings that StreamText shares
// with GenerateText. Stream-only settings and callbacks are left out.
func (o StreamTextOptions) generateTextOptions() GenerateTextOptions {
	return GenerateTextOptions{
		Model:               o.Model,
		CallSettings:        o.CallSettings,
		Prompt:              o.Prompt,
		Tools:               o.Tools,
		ToolChoice:          o.ToolChoice,
		ActiveTools:         o.ActiveTools,
		ToolOrder:           o.ToolOrder,
		ToolCallConcurrency: o.ToolCallConcurrency,
		StopWhen:            o.StopWhen,
		Output:              o.Output,
		ToolApproval:        o.ToolApproval,
		ToolApprovalSecret:  o.ToolApprovalSecret,
		ToolCallers:         o.ToolCallers,
		ProviderOptions:     o.ProviderOptions,
		PrepareStep:         o.PrepareStep,
		RepairToolCall:      o.RepairToolCall,
		RefineToolInput:     o.RefineToolInput,
		Download:            o.Download,
		Include: GenerateTextInclude{
			RequestBody:     o.Include.RequestBody,
			RequestMessages: o.Include.RequestMessages,
		},
		Timeout:                  o.Timeout,
		Telemetry:                o.Telemetry,
		LogWarnings:              o.LogWarnings,
		OnLanguageModelCallStart: o.OnLanguageModelCallStart,
		OnLanguageModelCallEnd:   o.OnLanguageModelCallEnd,
		OnToolExecutionStart:     o.OnToolExecutionStart,
		OnToolExecutionEnd:       o.OnToolExecutionEnd,
	}
}

// FullStream returns every part of the stream. A second stream iteration
// yields a single ErrorPart wrapping model.ErrStreamConsumed.
func (r *StreamTextResult) FullStream() iter.Seq[StreamPart] {
	return func(yield func(StreamPart) bool) {
		claimed := r.consume(projectFullStream, func(part enrichedPart) bool {
			return yield(part.part)
		})

		if !claimed {
			yield(ErrorPart{
				Err: model.ErrStreamConsumed,
			})
		}
	}
}

// TextStream returns the text deltas of the stream. Errors are reported by
// Result.
func (r *StreamTextResult) TextStream() iter.Seq[string] {
	return func(yield func(string) bool) {
		r.consume(projectTextStream, func(part enrichedPart) bool {
			delta, ok := part.part.(TextDeltaPart)
			if !ok {
				return true
			}

			return yield(delta.Text)
		})
	}
}

// PartialOutputStream returns the partial outputs parsed from the streamed
// text whenever they change: growing strings for text output, partial
// values for object, array, choice and JSON outputs.
func (r *StreamTextResult) PartialOutputStream() iter.Seq[any] {
	return func(yield func(any) bool) {
		r.consume(projectPartialOutputStream, func(part enrichedPart) bool {
			if !part.hasPartial {
				return true
			}

			return yield(part.partial)
		})
	}
}

// ElementStream returns each element of an ArrayOutput as soon as it is
// complete. It yields a single error for outputs without elements, and an
// error when the elements exceed the maximum count.
func (r *StreamTextResult) ElementStream() iter.Seq2[any, error] {
	return func(yield func(any, error) bool) {
		elements, ok := r.run.output.(ElementOutput)
		if !ok {
			yield(nil, fmt.Errorf("%w: element streams in %s mode", model.ErrUnsupported, r.run.output.Name()))

			return
		}

		publish := elements.ElementPublisher()

		r.consume(projectElementStream, func(part enrichedPart) bool {
			if !part.hasPartial {
				return true
			}

			completed, err := publish(part.partial)
			for _, element := range completed {
				if !yield(element, nil) {
					return false
				}
			}

			if err != nil {
				yield(nil, err)

				return false
			}

			return true
		})
	}
}

// ConsumeStream runs the stream to its end without reading it and returns
// the error of Result.
func (r *StreamTextResult) ConsumeStream() error {
	_, err := r.Result()

	return err
}

// Result returns the aggregated result once the stream has ended. It fails
// when no step completed, when the stream was aborted or closed, and when
// the consumer stopped early.
func (r *StreamTextResult) Result() (*GenerateTextResult, error) {
	r.mu.Lock()
	claimed := r.claimed
	r.mu.Unlock()

	if !claimed {
		r.consume(projectFullStream, func(enrichedPart) bool {
			return true
		})
	}

	<-r.done

	return r.outcome.result, r.outcome.err
}

// Output returns the output parsed from the final step once the stream has
// ended.
func (r *StreamTextResult) Output() (any, error) {
	if _, err := r.Result(); err != nil {
		return nil, err
	}

	return r.outcome.output, r.outcome.outputErr
}

// Close abandons the stream. A running stream is cancelled and ends with an
// AbortPart; a stream that has not started will not start. Close is safe to
// call more than once and from any goroutine.
func (r *StreamTextResult) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.cancel != nil {
		r.cancel(ErrStreamClosed)

		return nil
	}

	if !r.claimed {
		r.claimed = true
		r.outcome.err = ErrStreamClosed
		close(r.done)
	}

	return nil
}

// consume claims the stream and runs it, passing each part to deliver until
// it returns false. It reports false when the stream was already claimed.
func (r *StreamTextResult) consume(projection streamProjection, deliver func(part enrichedPart) bool) bool {
	ctx, ok := r.claim()
	if !ok {
		return false
	}

	defer r.finish()

	r.outcome = r.run.execute(ctx, streamExecution{
		trackPartialOutput: r.run.hasExplicitOutput || projection.readsPartialOutput(),
		deliver:            deliver,
		abort:              r.cancel,
	})

	return true
}

// readsPartialOutput reports whether the projection is built from the
// partial outputs of the stream.
func (p streamProjection) readsPartialOutput() bool {
	return p == projectPartialOutputStream || p == projectElementStream
}

func (r *StreamTextResult) claim() (context.Context, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.claimed {
		return nil, false
	}

	ctx, cancel := context.WithCancelCause(r.parent)
	r.claimed = true
	r.cancel = cancel

	return ctx, true
}

func (r *StreamTextResult) finish() {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.cancel(nil)
	close(r.done)
}
