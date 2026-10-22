package ai

import (
	"context"
	"errors"
	"iter"
	"sync"
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

// singleConsumerStream runs a stream on the goroutine of its single
// consumer. The run starts when the parts are ranged over or the result is
// awaited, and stops when the consumer stops or the stream is closed.
// StreamTranscribe and StreamTranslate share it.
type singleConsumerStream[Part, Result any] struct {
	parent context.Context

	// run produces the parts, passing each to deliver until it returns
	// false, and returns the final result.
	run func(ctx context.Context, deliver func(Part) bool) (*Result, error)

	mu      sync.Mutex
	claimed bool
	cancel  context.CancelCauseFunc
	done    chan struct{}

	result *Result
	err    error
}

func newSingleConsumerStream[Part, Result any](
	parent context.Context,
	run func(ctx context.Context, deliver func(Part) bool) (*Result, error),
) *singleConsumerStream[Part, Result] {
	return &singleConsumerStream[Part, Result]{
		parent: parent,
		run:    run,
		done:   make(chan struct{}),
	}
}

// parts returns the stream's parts. A second iteration yields only
// consumed, the part that reports the stream as already consumed.
func (s *singleConsumerStream[Part, Result]) parts(consumed Part) iter.Seq[Part] {
	return func(yield func(Part) bool) {
		if !s.consume(yield) {
			yield(consumed)
		}
	}
}

// wait consumes the stream unless it was claimed and returns its result
// once it has ended.
func (s *singleConsumerStream[Part, Result]) wait() (*Result, error) {
	s.consume(func(Part) bool {
		return true
	})

	<-s.done

	return s.result, s.err
}

// close cancels a running stream with ErrStreamClosed, or ends a stream
// that has not started so that it never starts. It is safe to call more
// than once and from any goroutine.
func (s *singleConsumerStream[Part, Result]) close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.cancel != nil {
		s.cancel(ErrStreamClosed)

		return nil
	}

	if !s.claimed {
		s.claimed = true
		s.err = ErrStreamClosed
		close(s.done)
	}

	return nil
}

// consume claims the stream and runs it, passing each part to deliver
// until it returns false. It reports false when the stream was already
// claimed.
func (s *singleConsumerStream[Part, Result]) consume(deliver func(Part) bool) bool {
	ctx, ok := s.claim()
	if !ok {
		return false
	}

	defer s.finish()

	s.result, s.err = s.run(ctx, deliver)

	return true
}

func (s *singleConsumerStream[Part, Result]) claim() (context.Context, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.claimed {
		return nil, false
	}

	ctx, cancel := context.WithCancelCause(s.parent)
	s.claimed = true
	s.cancel = cancel

	return ctx, true
}

func (s *singleConsumerStream[Part, Result]) finish() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.cancel(nil)
	close(s.done)
}

// isStreamClosed reports whether a run ended because the consumer stopped
// or closed the stream, which is not a failure of the stream.
func isStreamClosed(ctx context.Context, err error) bool {
	return errors.Is(err, ErrStreamClosed) || errors.Is(context.Cause(ctx), ErrStreamClosed)
}

// forwardStreamParts passes the parts that collect keeps for the consumer
// to deliver. It returns ErrStreamClosed when the consumer stops.
func forwardStreamParts[Part any](parts iter.Seq[Part], collect func(Part) bool, deliver func(Part) bool) error {
	for part := range parts {
		if !collect(part) {
			continue
		}

		if !deliver(part) {
			return ErrStreamClosed
		}
	}

	return nil
}

// streamResponseCollector records the response, warnings and last error of
// a model stream from its stream start, response metadata and error parts.
type streamResponseCollector struct {
	response model.ResponseInfo
	warnings []model.Warning
	err      error
}

// newStreamResponseCollector starts from the response of the connection,
// filling in the start time and model ID when the provider left them out.
func newStreamResponseCollector(response model.ResponseInfo, startedAt time.Time, modelID string) streamResponseCollector {
	if response.Timestamp.IsZero() {
		response.Timestamp = startedAt
	}

	if response.ModelID == "" {
		response.ModelID = modelID
	}

	return streamResponseCollector{
		response: response,
	}
}

func (c *streamResponseCollector) start(start model.StreamStart) {
	c.warnings = start.Warnings
}

func (c *streamResponseCollector) responseMetadata(metadata model.ResponseMetadata) {
	if metadata.ID != "" {
		c.response.ID = metadata.ID
	}

	if !metadata.Timestamp.IsZero() {
		c.response.Timestamp = metadata.Timestamp
	}

	if metadata.ModelID != "" {
		c.response.ModelID = metadata.ModelID
	}
}

func (c *streamResponseCollector) streamError(part model.StreamError) {
	c.err = part.Err
}

// failure explains a stream that ended without its finish part: its last
// error, the context's cause, or noContent.
func (c *streamResponseCollector) failure(ctx context.Context, noContent error) error {
	if c.err != nil {
		return c.err
	}

	if ctx.Err() != nil {
		return context.Cause(ctx)
	}

	return noContent
}
