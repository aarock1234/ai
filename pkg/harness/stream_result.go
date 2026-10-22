package harness

import (
	"context"
	"fmt"
	"iter"
	"sync"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/ai/ui"
	"github.com/aarock1234/ai/pkg/model"
)

// StreamResult is the result of one harness turn, the Go form of the
// upstream HarnessStreamTextResult. Its parts use the ai stream part types,
// so ai and ai/ui consumers handle them like StreamText parts.
//
// The turn runs in a goroutine owned by the result and does not depend on
// consumption: parts are recorded as they arrive, and every stream replays
// them from the start, so any number of streams can be ranged over in any
// order. Breaking out of a range does not stop the turn; cancel the context
// passed to Agent.Stream or call Close to stop it. Result, Output and
// ConsumeStream wait for the turn to end.
type StreamResult struct {
	result *turnResult
	tools  []ai.Tool

	cancel    context.CancelCauseFunc
	driver    sync.WaitGroup
	closeOnce sync.Once
}

// FullStream yields every part of the turn: an ai.StartPart, the parts of
// each step framed by ai.StartStepPart and ai.FinishStepPart, and a final
// ai.FinishPart, ai.AbortPart or ai.ErrorPart.
func (r *StreamResult) FullStream() iter.Seq[ai.StreamPart] {
	return func(yield func(ai.StreamPart) bool) {
		r.result.log.all(func(part loggedPart) bool {
			return yield(part.part)
		})
	}
}

// TextStream yields the text deltas of the turn.
func (r *StreamResult) TextStream() iter.Seq[string] {
	return func(yield func(string) bool) {
		r.result.log.all(func(part loggedPart) bool {
			delta, ok := part.part.(ai.TextDeltaPart)
			if !ok {
				return true
			}

			return yield(delta.Text)
		})
	}
}

// PartialOutputStream yields the partial output of the agent's configured
// ai.Output each time it changes. It yields nothing when the agent has no
// output.
func (r *StreamResult) PartialOutputStream() iter.Seq[any] {
	return func(yield func(any) bool) {
		r.result.log.all(func(part loggedPart) bool {
			if !part.hasPartialOutput {
				return true
			}

			return yield(part.partialOutput)
		})
	}
}

// ElementStream yields each element of the agent's ai.ArrayOutput as soon
// as it is complete. It yields a single error when the agent's output has
// no elements, and an error when the elements exceed the maximum count.
func (r *StreamResult) ElementStream() iter.Seq2[any, error] {
	return func(yield func(any, error) bool) {
		elements, ok := r.result.output.(ai.ElementOutput)
		if !ok {
			yield(nil, fmt.Errorf("%w: element streams in %s mode", model.ErrUnsupported, outputName(r.result.output)))

			return
		}

		publish := elements.ElementPublisher()

		r.result.log.all(func(part loggedPart) bool {
			if !part.hasPartialOutput {
				return true
			}

			completed, err := publish(part.partialOutput)
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

// outputName names an output mode for errors; no output is text.
func outputName(output ai.Output) string {
	if output == nil {
		return "text"
	}

	return output.Name()
}

// ToUIMessageStream converts the turn into a UI message stream. Tools
// defaults to the turn's tools.
func (r *StreamResult) ToUIMessageStream(ctx context.Context, options ui.MessageStreamOptions) iter.Seq2[ui.Chunk, error] {
	if options.Tools == nil {
		options.Tools = r.tools
	}

	return ui.ToMessageStream(ctx, r.FullStream(), options)
}

// Done is closed when the turn has ended.
func (r *StreamResult) Done() <-chan struct{} {
	return r.result.done
}

// ConsumeStream waits for the turn to end and returns its failure.
func (r *StreamResult) ConsumeStream() error {
	<-r.result.done
	r.driver.Wait()

	return r.result.outcome.err
}

// Result waits for the turn to end and returns the aggregated result, whose
// steps, usage and response messages mirror GenerateText's. A cancelled or
// failed turn returns its error.
func (r *StreamResult) Result() (*ai.GenerateTextResult, error) {
	if err := r.ConsumeStream(); err != nil {
		return nil, err
	}

	return r.result.outcome.result, nil
}

// Output waits for the turn to end and returns the output parsed by the
// agent's ai.Output, or the final text when the agent has none.
func (r *StreamResult) Output() (any, error) {
	result, err := r.Result()
	if err != nil {
		return nil, err
	}

	if r.result.outcome.outputErr != nil {
		return nil, r.result.outcome.outputErr
	}

	return result.Output, nil
}

// Close cancels the turn if it is still running and waits for it to end. A
// cancelled turn ends with an ai.AbortPart.
func (r *StreamResult) Close() error {
	r.closeOnce.Do(func() {
		r.cancel(ai.ErrStreamClosed)
	})

	r.driver.Wait()

	return nil
}
