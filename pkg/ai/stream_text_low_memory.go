package ai

import (
	"context"
	"iter"

	"github.com/aarock1234/ai/pkg/model"
)

// StreamTextLowMemoryResult is a streaming text generation that keeps no
// steps for an aggregated result. It offers the streams of StreamTextResult
// and a summary of the finished stream, but no final content.
type StreamTextLowMemoryResult struct {
	result *StreamTextResult
}

// StreamTextSummary describes a finished stream.
type StreamTextSummary struct {
	// TotalUsage is the summed usage of every step.
	TotalUsage LanguageModelUsage

	// FinishReason is the finish reason of the final step.
	FinishReason    model.UnifiedFinishReason
	RawFinishReason string
}

// StreamTextLowMemory streams text like StreamText, with the same options
// and execution, but does not retain the steps once the stream has ended.
// Step content is only recorded when the options need it: with tools, a
// tool choice, an output, PrepareStep, StopWhen, the finish, step, abort and
// language-model-call-end callbacks, or enabled telemetry. It is upstream's
// experimental_streamTextLowMemory.
//
// Cancellation and deadlines come from ctx.
func StreamTextLowMemory(ctx context.Context, options StreamTextOptions) (*StreamTextLowMemoryResult, error) {
	result, err := newStreamTextResult(ctx, options, streamRetention{
		replay:         false,
		collectContent: needsStepContent(options),
	})
	if err != nil {
		return nil, err
	}

	return &StreamTextLowMemoryResult{
		result: result,
	}, nil
}

// needsStepContent reports whether the options use the content of steps.
// It is decided before the stream starts.
func needsStepContent(options StreamTextOptions) bool {
	usesTools := len(options.Tools) > 0 || options.ToolChoice != nil
	shapesSteps := options.Output != nil || options.PrepareStep != nil || options.StopWhen != nil
	observesSteps := options.OnEnd != nil || options.OnStepEnd != nil || options.OnAbort != nil || options.OnLanguageModelCallEnd != nil

	return usesTools || shapesSteps || observesSteps || newTelemetryDispatcher(options.Telemetry).enabled()
}

// FullStream returns every part of the stream, as StreamTextResult.FullStream
// does.
func (r *StreamTextLowMemoryResult) FullStream() iter.Seq[StreamPart] {
	return r.result.FullStream()
}

// TextStream returns the text deltas of the stream.
func (r *StreamTextLowMemoryResult) TextStream() iter.Seq[string] {
	return r.result.TextStream()
}

// PartialOutputStream returns the partial outputs parsed from the streamed
// text.
func (r *StreamTextLowMemoryResult) PartialOutputStream() iter.Seq[any] {
	return r.result.PartialOutputStream()
}

// ElementStream returns each element of an ArrayOutput as soon as it is
// complete.
func (r *StreamTextLowMemoryResult) ElementStream() iter.Seq2[any, error] {
	return r.result.ElementStream()
}

// ConsumeStream runs the stream to its end without reading it and returns
// the error of Summary.
func (r *StreamTextLowMemoryResult) ConsumeStream() error {
	_, err := r.Summary()

	return err
}

// Summary returns the usage and finish reason once the stream has ended. It
// consumes the stream if no stream was ranged over and fails as
// StreamTextResult.Result does.
func (r *StreamTextLowMemoryResult) Summary() (StreamTextSummary, error) {
	if _, err := r.result.Result(); err != nil {
		return StreamTextSummary{}, err
	}

	return r.result.outcome.summary, nil
}

// Close abandons the stream, as StreamTextResult.Close does.
func (r *StreamTextLowMemoryResult) Close() error {
	return r.result.Close()
}
