package ui

import (
	"context"
	"iter"
	"sync"

	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/pkg/ai"
)

// CreateMessageStreamOptions configure CreateMessageStream.
type CreateMessageStreamOptions struct {
	// Execute writes chunks to the stream and merges other streams into it.
	// It runs on its own goroutine and must return once ctx is done. A
	// returned error is reported to the client as an error chunk.
	Execute func(ctx context.Context, writer *Writer) error

	// OnError formats errors for the client. Nil sends a generic message so
	// that server errors do not leak.
	OnError func(err error) string

	// OriginalMessages are the messages of the conversation. When set,
	// persistence mode is assumed and the response message receives an ID.
	OriginalMessages []Message

	// OnStepFinish is called with the response message after each step.
	OnStepFinish func(ctx context.Context, event StepFinishEvent)

	// OnFinish is called once the stream has ended.
	OnFinish func(ctx context.Context, event FinishEvent)

	// GenerateID generates the ID of the response message. Nil generates
	// random IDs.
	GenerateID func() string
}

// CreateMessageStream returns a UI message stream whose chunks Execute
// writes. The stream ends once Execute has returned and every merged stream
// has ended. Ranging over it starts Execute; breaking out of the range
// cancels the context passed to Execute and waits for Execute and the merged
// streams to stop. Merged streams must therefore be created with that
// context, so that cancelling it ends them promptly.
func CreateMessageStream(ctx context.Context, options CreateMessageStreamOptions) iter.Seq2[Chunk, error] {
	return func(yield func(Chunk, error) bool) {
		if options.Execute == nil {
			yield(nil, &ai.InvalidArgumentError{
				Parameter: "execute",
				Message:   "execute must not be nil",
			})

			return
		}

		generateID := options.GenerateID
		if generateID == nil {
			generateID = func() string {
				return id.New("")
			}
		}

		streamCtx, cancel := context.WithCancel(ctx)
		defer cancel()

		writer := newWriter(streamCtx, errorFormatter(options.OnError))
		writer.start(func() {
			if err := options.Execute(streamCtx, writer); err != nil {
				writer.handleError(err)
			}
		})

		defer func() {
			cancel()
			<-writer.done
		}()

		finished := handleMessageStreamFinish(ctx, writer.chunks(), streamFinishOptions{
			messageID:        generateID(),
			originalMessages: options.OriginalMessages,
			onStepFinish:     options.OnStepFinish,
			onFinish:         options.OnFinish,
			onError:          writer.onError,
			outcome:          writer.outcome,
		})

		for chunk, err := range finished {
			if !yield(chunk, err) {
				return
			}
		}
	}
}

// Writer writes chunks to a stream created by CreateMessageStream. It is
// safe for concurrent use.
type Writer struct {
	ctx     context.Context
	onError func(err error) string

	// pending receives the chunks for the consumer.
	pending chan Chunk

	// done is closed once every writing goroutine has finished.
	done chan struct{}

	mu       sync.Mutex
	active   int
	finished bool
	declared Outcome
	failure  *Outcome
}

func newWriter(ctx context.Context, onError func(err error) string) *Writer {
	return &Writer{
		ctx:     ctx,
		onError: onError,
		pending: make(chan Chunk),
		done:    make(chan struct{}),
		declared: Outcome{
			Status: OutcomeUnknown,
		},
	}
}

// Write sends a chunk to the stream. It blocks until the consumer receives
// the chunk, and drops it once the stream has been cancelled or has ended.
func (w *Writer) Write(chunk Chunk) {
	select {
	case w.pending <- chunk:
	case <-w.ctx.Done():
	case <-w.done:
	}
}

// Merge forwards the chunks of another stream on its own goroutine. The
// stream is not read when the writer has been cancelled or has ended. An
// error of the merged stream is reported as an error chunk. Create the
// stream with the context passed to Execute: the writer waits for it to end
// after cancellation.
func (w *Writer) Merge(stream iter.Seq2[Chunk, error]) {
	w.start(func() {
		for chunk, err := range stream {
			if err != nil {
				w.handleError(err)

				return
			}

			if !w.send(chunk) {
				return
			}
		}
	})
}

// OnError formats an error with the stream's error formatter. Pass it to
// streams that are merged into this one, so that their errors are formatted
// once.
func (w *Writer) OnError(err error) string {
	return w.onError(err)
}

// SetOutcome declares the outcome of the stream. The first declared outcome
// is kept; fatal failures of Execute and of merged streams override it.
// Declaring an outcome neither writes a chunk nor ends the stream.
func (w *Writer) SetOutcome(outcome Outcome) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.declared.Status == OutcomeUnknown && outcome.Status != OutcomeUnknown {
		w.declared = outcome
	}
}

// start runs work on a goroutine that the stream waits for. Work is dropped
// when the stream has ended or has been cancelled.
func (w *Writer) start(work func()) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.finished {
		return
	}

	if w.ctx.Err() != nil {
		w.finishIfIdle()

		return
	}

	w.active++

	go func() {
		defer w.stop()

		work()
	}()
}

// stop records a finished goroutine and ends the stream after the last one.
func (w *Writer) stop() {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.active--
	w.finishIfIdle()
}

// finishIfIdle ends the stream when no goroutine is writing. The caller
// holds mu.
func (w *Writer) finishIfIdle() {
	if w.active > 0 || w.finished {
		return
	}

	w.finished = true
	close(w.done)
}

// send delivers a chunk and reports whether the stream is still open.
func (w *Writer) send(chunk Chunk) bool {
	select {
	case w.pending <- chunk:
		return true
	case <-w.ctx.Done():
		return false
	}
}

// handleError records a fatal failure and reports it to the client.
func (w *Writer) handleError(err error) {
	w.mu.Lock()
	w.failure = &Outcome{
		Status: OutcomeFailed,
		Error:  err,
	}
	w.mu.Unlock()

	w.send(ErrorChunk{
		ErrorText: w.onError(err),
	})
}

func (w *Writer) outcome() Outcome {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.failure != nil {
		return *w.failure
	}

	return w.declared
}

// chunks yields the written chunks until every writing goroutine has
// finished.
func (w *Writer) chunks() iter.Seq2[Chunk, error] {
	return func(yield func(Chunk, error) bool) {
		for {
			select {
			case chunk := <-w.pending:
				if !yield(chunk, nil) {
					return
				}
			case <-w.done:
				return
			}
		}
	}
}
