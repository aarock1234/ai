// Package chatsse converts OpenAI-style server-sent event streams into
// language model stream parts. Providers supply the per-event state machine;
// the package owns reading, the "[DONE]" terminator and error reporting.
package chatsse

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"

	"github.com/aarock1234/ai/internal/sse"
	"github.com/aarock1234/ai/pkg/model"
)

// doneMarker is the data of the event that ends an OpenAI-style stream.
const doneMarker = "[DONE]"

// Processor is the state machine of one streaming response.
type Processor interface {
	// Process converts one event. An error ends the stream after the
	// returned parts.
	Process(event sse.Event) ([]model.StreamPart, error)

	// Fail records that the stream failed, so that the closing parts report
	// an error finish reason, and returns the StreamError part for err.
	Fail(err error) model.StreamPart

	// Flush returns the closing parts after the last event, including the
	// Finish part. An error is reported after the returned parts.
	Flush() ([]model.StreamPart, error)
}

// Parts reads server-sent events from body and yields start, the processor's
// parts and its closing parts. Events whose data is "[DONE]" are skipped. A
// read or processing failure stops reading and yields the processor's error
// part; the closing parts, with the Finish part, follow in every case. A read
// failure caused by the cancellation of ctx reports the context's cause. Parts
// does not close body.
func Parts(ctx context.Context, body io.Reader, start model.StreamStart, processor Processor) iter.Seq[model.StreamPart] {
	return func(yield func(model.StreamPart) bool) {
		if !yield(start) {
			return
		}

		if !readEvents(ctx, body, processor, yield) {
			return
		}

		parts, err := processor.Flush()
		if !yieldAll(yield, parts) {
			return
		}

		if err != nil {
			yield(model.StreamError{
				Err: err,
			})
		}
	}
}

// readEvents yields the parts of every event until the body ends or reading
// or processing fails. It reports false when the consumer stopped.
func readEvents(ctx context.Context, body io.Reader, processor Processor, yield func(model.StreamPart) bool) bool {
	for event, err := range sse.Read(body) {
		if err != nil {
			return yield(processor.Fail(readError(ctx, err)))
		}

		if event.Data == doneMarker {
			continue
		}

		parts, err := processor.Process(event)
		if !yieldAll(yield, parts) {
			return false
		}

		if err != nil {
			return yield(processor.Fail(err))
		}
	}

	return true
}

// readError reports the context's cause when ctx ended the read.
func readError(ctx context.Context, err error) error {
	if ctx.Err() != nil {
		return context.Cause(ctx)
	}

	return fmt.Errorf("read stream: %w", err)
}

// Raw returns the event data as a raw chunk part when it is valid JSON.
func Raw(event sse.Event) (model.Raw, bool) {
	if !json.Valid([]byte(event.Data)) {
		return model.Raw{}, false
	}

	return model.Raw{
		Value: json.RawMessage(event.Data),
	}, true
}

// yieldAll yields each part and reports whether the consumer wants more.
func yieldAll(yield func(model.StreamPart) bool, parts []model.StreamPart) bool {
	for _, part := range parts {
		if !yield(part) {
			return false
		}
	}

	return true
}
