package ui

import (
	"errors"
	"fmt"
	"io"
	"iter"
	"maps"
	"net/http"
	"sync"
	"time"

	"github.com/aarock1234/ai/internal/sse"
	"github.com/aarock1234/ai/pkg/ai"
)

// SSE frames of UI message streams.
var (
	streamOpenComment = []byte(": stream-open\n\n")
	keepAliveComment  = []byte(": keep-alive\n\n")
	doneFrame         = []byte("data: " + doneData + "\n\n")
)

// doneData is the data of the frame that ends a UI message stream.
const doneData = "[DONE]"

// MessageStreamHeaders returns the response headers of UI message streams.
// The x-vercel-ai-ui-message-stream header identifies the protocol version
// to the client.
func MessageStreamHeaders() http.Header {
	return http.Header{
		"Content-Type":                  {"text/event-stream"},
		"Cache-Control":                 {"no-cache"},
		"Connection":                    {"keep-alive"},
		"X-Vercel-Ai-Ui-Message-Stream": {"v1"},
		"X-Accel-Buffering":             {"no"},
	}
}

// ResponseInit configures the HTTP response of a stream.
type ResponseInit struct {
	// Status is the status code. Zero means 200.
	Status int

	// Headers are added to the response. They take precedence over the
	// default headers of the same names.
	Headers http.Header
}

// MessageStreamResponseInit configures the HTTP response of a UI message
// stream.
type MessageStreamResponseInit struct {
	ResponseInit

	// KeepAlive sends an opening SSE comment immediately and a keep-alive
	// comment whenever the stream has been idle for this long. Zero disables
	// keep-alive comments. With keep-alive, the stream is read on its own
	// goroutine, which the response waits for after a write failure; create
	// the stream with the request's context so that a disconnect ends it.
	KeepAlive time.Duration

	// ConsumeSSE receives a copy of every SSE frame, for example for logging.
	// It is skipped after its first write error, which does not affect the
	// response.
	ConsumeSSE io.Writer
}

// MessageStreamResponseOptions configure PipeMessageStreamToResponse and
// CreateMessageStreamResponse.
type MessageStreamResponseOptions struct {
	Stream iter.Seq2[Chunk, error]

	MessageStreamResponseInit
}

// PipeMessageStreamToResponse writes a UI message stream to w as Server-Sent
// Events, flushing after every frame, and ends it with a [DONE] frame. It
// returns when the stream has ended, with the error of the stream or of a
// failed write; it stops reading the stream after a failed write, such as
// when the client disconnects.
func PipeMessageStreamToResponse(w http.ResponseWriter, options MessageStreamResponseOptions) error {
	if err := validateKeepAlive(options.KeepAlive); err != nil {
		return err
	}

	writeResponseHead(w, options.ResponseInit, MessageStreamHeaders())

	destination := &frameDestination{
		writer: w,
		flush:  responseFlusher(w),
		tee:    options.ConsumeSSE,
	}

	return pipeFrames(destination, sseFrames(options.Stream), options.KeepAlive)
}

// CreateMessageStreamResponse returns an HTTP response whose body streams a
// UI message stream as Server-Sent Events. The stream starts on the first
// read of the body; closing the body stops it and waits for it to end.
func CreateMessageStreamResponse(options MessageStreamResponseOptions) (*http.Response, error) {
	if err := validateKeepAlive(options.KeepAlive); err != nil {
		return nil, err
	}

	body := newStreamBody(func(w io.Writer) error {
		destination := &frameDestination{
			writer: w,
			tee:    options.ConsumeSSE,
		}

		return pipeFrames(destination, sseFrames(options.Stream), options.KeepAlive)
	})

	return newStreamResponse(options.ResponseInit, MessageStreamHeaders(), body), nil
}

// ParseMessageStream decodes the UI message stream in an SSE response body,
// such as one written by PipeMessageStreamToResponse, skipping the [DONE]
// terminator. It does not close body.
func ParseMessageStream(body io.Reader) iter.Seq2[Chunk, error] {
	return func(yield func(Chunk, error) bool) {
		for event, err := range sse.Read(body) {
			if err != nil {
				yield(nil, fmt.Errorf("read ui message stream: %w", err))

				return
			}

			if event.Data == doneData {
				continue
			}

			chunk, err := UnmarshalChunk([]byte(event.Data))
			if err != nil {
				yield(nil, err)

				return
			}

			if !yield(chunk, nil) {
				return
			}
		}
	}
}

// sseFrames encodes each chunk as an SSE data frame and ends with a [DONE]
// frame. A stream error ends the frames without [DONE].
func sseFrames(stream iter.Seq2[Chunk, error]) iter.Seq2[[]byte, error] {
	return func(yield func([]byte, error) bool) {
		for chunk, err := range stream {
			if err != nil {
				yield(nil, err)

				return
			}

			encoded, err := encodeJSON(chunk)
			if err != nil {
				yield(nil, fmt.Errorf("encode ui message chunk: %w", err))

				return
			}

			frame := make([]byte, 0, len(encoded)+8)
			frame = append(frame, "data: "...)
			frame = append(frame, encoded...)
			frame = append(frame, "\n\n"...)

			if !yield(frame, nil) {
				return
			}
		}

		yield(doneFrame, nil)
	}
}

// frameDestination writes stream frames to a response.
type frameDestination struct {
	writer io.Writer

	// flush sends buffered data to the client. Nil means no flushing.
	flush func() error

	// tee receives a copy of every frame. Nil means none.
	tee io.Writer
}

func (d *frameDestination) write(frame []byte) error {
	if _, err := d.writer.Write(frame); err != nil {
		return fmt.Errorf("write stream frame: %w", err)
	}

	if d.tee != nil {
		if _, err := d.tee.Write(frame); err != nil {
			d.tee = nil
		}
	}

	if d.flush == nil {
		return nil
	}

	if err := d.flush(); err != nil {
		return fmt.Errorf("flush stream frame: %w", err)
	}

	return nil
}

// pipeFrames writes every frame to destination, with keep-alive comments
// when keepAlive is positive.
func pipeFrames(destination *frameDestination, frames iter.Seq2[[]byte, error], keepAlive time.Duration) error {
	if keepAlive > 0 {
		return pipeFramesWithKeepAlive(destination, frames, keepAlive)
	}

	for frame, err := range frames {
		if err != nil {
			return err
		}

		if err := destination.write(frame); err != nil {
			return err
		}
	}

	return nil
}

// frameResult is one frame read by the producer of
// pipeFramesWithKeepAlive.
type frameResult struct {
	frame []byte
	err   error
}

// pipeFramesWithKeepAlive reads the frames on a producer goroutine so that
// keep-alive comments can be written while the stream is idle. On return the
// producer has stopped reading and has finished.
func pipeFramesWithKeepAlive(destination *frameDestination, frames iter.Seq2[[]byte, error], keepAlive time.Duration) error {
	if err := destination.write(streamOpenComment); err != nil {
		return err
	}

	results := make(chan frameResult)
	stop := make(chan struct{})

	var producer sync.WaitGroup
	producer.Go(func() {
		defer close(results)

		for frame, err := range frames {
			select {
			case results <- frameResult{
				frame: frame,
				err:   err,
			}:
			case <-stop:
				return
			}
		}
	})

	defer func() {
		close(stop)
		producer.Wait()
	}()

	idle := time.NewTimer(keepAlive)
	defer idle.Stop()

	for {
		select {
		case result, open := <-results:
			if !open {
				return nil
			}

			if result.err != nil {
				return result.err
			}

			if err := destination.write(result.frame); err != nil {
				return err
			}

		case <-idle.C:
			if err := destination.write(keepAliveComment); err != nil {
				return err
			}
		}

		idle.Reset(keepAlive)
	}
}

func validateKeepAlive(keepAlive time.Duration) error {
	if keepAlive < 0 {
		return &ai.InvalidArgumentError{
			Parameter: "keepAlive",
			Value:     keepAlive,
			Message:   "keepAlive must not be negative",
		}
	}

	return nil
}

// writeResponseHead sets the default headers that init does not override,
// adds the headers of init and writes the status.
func writeResponseHead(w http.ResponseWriter, init ResponseInit, defaults http.Header) {
	header := w.Header()

	maps.Copy(header, responseHeaders(init.Headers, defaults))

	w.WriteHeader(responseStatus(init.Status))
}

// responseHeaders returns headers with the defaults whose names it lacks.
func responseHeaders(headers, defaults http.Header) http.Header {
	merged := headers.Clone()
	if merged == nil {
		merged = make(http.Header)
	}

	for name, values := range defaults {
		if len(merged.Values(name)) == 0 {
			merged[name] = values
		}
	}

	return merged
}

func responseStatus(status int) int {
	if status == 0 {
		return http.StatusOK
	}

	return status
}

// responseFlusher returns a function that flushes w, treating writers
// without flushing support as flushed.
func responseFlusher(w http.ResponseWriter) func() error {
	controller := http.NewResponseController(w)

	return func() error {
		err := controller.Flush()
		if errors.Is(err, http.ErrNotSupported) {
			return nil
		}

		return err
	}
}

// newStreamResponse returns a response with a streamed body of unknown
// length.
func newStreamResponse(init ResponseInit, defaults http.Header, body io.ReadCloser) *http.Response {
	status := responseStatus(init.Status)

	return &http.Response{
		Status:        fmt.Sprintf("%d %s", status, http.StatusText(status)),
		StatusCode:    status,
		Proto:         "HTTP/1.1",
		ProtoMajor:    1,
		ProtoMinor:    1,
		Header:        responseHeaders(init.Headers, defaults),
		Body:          body,
		ContentLength: -1,
	}
}

// streamBody is a response body produced by a write function on its own
// goroutine. The function starts on the first read; Close stops it by
// failing its writes and waits for it to return.
type streamBody struct {
	write func(w io.Writer) error

	reader *io.PipeReader
	writer *io.PipeWriter

	start   sync.Once
	writing sync.WaitGroup
}

func newStreamBody(write func(w io.Writer) error) *streamBody {
	reader, writer := io.Pipe()

	return &streamBody{
		write:  write,
		reader: reader,
		writer: writer,
	}
}

// Read reads the streamed body, starting the stream on the first call.
func (b *streamBody) Read(p []byte) (int, error) {
	b.start.Do(func() {
		b.writing.Go(func() {
			b.writer.CloseWithError(b.write(b.writer))
		})
	})

	return b.reader.Read(p)
}

// Close stops the stream and waits for it to end. A body that was never
// read does not start the stream.
func (b *streamBody) Close() error {
	b.start.Do(func() {})

	err := b.reader.Close()
	b.writing.Wait()

	return err
}
