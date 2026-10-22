package ai

import (
	"context"
	"fmt"
	"io"
	"iter"
	"net/http"
	"time"

	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/pkg/model"
)

// StreamTranscribeOptions configures StreamTranscribe.
type StreamTranscribeOptions struct {
	// Model must implement model.StreamingTranscriptionModel.
	Model model.TranscriptionModel

	// Audio supplies raw audio bytes in InputAudioFormat, such as a
	// microphone capture or an io.Pipe fed by one. It is read until io.EOF
	// while transcripts stream back. The caller keeps ownership: the stream
	// never closes Audio and stops reading it when the stream ends, but a
	// Read already in progress is not interrupted.
	Audio io.Reader

	// InputAudioFormat describes Audio, such as {Type: "audio/pcm", Rate:
	// 24000}.
	InputAudioFormat model.AudioFormat

	Headers         http.Header
	ProviderOptions model.ProviderOptions

	// IncludeRawChunks asks the provider to emit model.Raw parts.
	IncludeRawChunks bool

	// Telemetry sends the lifecycle events of the stream to the integrations
	// that implement TranscriptionTelemetry.
	Telemetry TelemetryOptions

	// LogWarnings receives the warnings of the stream when it starts. Nil
	// discards them; they are always returned on the result.
	LogWarnings LogWarningsFunc
}

// StreamTranscribeResult is a streaming transcription with a single
// consumer. The transcription starts when FullStream is ranged over or
// Result is called, runs on the consumer's goroutine, and stops when the
// consumer stops. The streaming transcription API is experimental and may
// change.
//
// Range over FullStream once, before calling Result, when both the parts
// and the final transcript are needed. Result alone consumes the stream.
// Do not call Result while ranging over FullStream on the same goroutine.
type StreamTranscribeResult struct {
	model   model.StreamingTranscriptionModel
	options StreamTranscribeOptions

	callID    string
	telemetry telemetryDispatcher

	stream *singleConsumerStream[model.TranscriptionStreamPart, TranscribeResult]
}

// StreamTranscribe transcribes live audio as it arrives. Option errors and
// models without streaming support are reported immediately; errors during
// the transcription are reported as model.StreamError parts and by Result.
//
// Cancellation and deadlines come from ctx.
func StreamTranscribe(ctx context.Context, options StreamTranscribeOptions) (*StreamTranscribeResult, error) {
	if err := options.validate(); err != nil {
		return nil, err
	}

	streaming, ok := options.Model.(model.StreamingTranscriptionModel)
	if !ok {
		return nil, fmt.Errorf(
			"%w: the %s model %q does not support streaming transcription",
			model.ErrUnsupported,
			options.Model.Provider(),
			options.Model.ModelID(),
		)
	}

	result := &StreamTranscribeResult{
		model:     streaming,
		options:   options,
		callID:    id.New("call"),
		telemetry: newTelemetryDispatcher(options.Telemetry),
	}

	result.stream = newSingleConsumerStream(ctx, result.run)

	return result, nil
}

func (o StreamTranscribeOptions) validate() error {
	if o.Model == nil {
		return &InvalidArgumentError{
			Parameter: "model",
			Message:   "model is required",
		}
	}

	if o.Audio == nil {
		return &InvalidArgumentError{
			Parameter: "audio",
			Message:   "audio is required",
		}
	}

	if o.InputAudioFormat.Type == "" {
		return &InvalidArgumentError{
			Parameter: "inputAudioFormat",
			Value:     o.InputAudioFormat,
			Message:   "input audio format type is required",
		}
	}

	return nil
}

// FullStream returns the transcript parts as they arrive:
// model.TranscriptDelta, model.TranscriptPartial, model.TranscriptFinal,
// model.Raw and model.StreamError. Stream start, response metadata and
// finish parts are folded into Result. A second iteration yields a single
// model.StreamError wrapping model.ErrStreamConsumed.
func (r *StreamTranscribeResult) FullStream() iter.Seq[model.TranscriptionStreamPart] {
	return r.stream.parts(model.StreamError{
		Err: model.ErrStreamConsumed,
	})
}

// Result returns the final transcript once the stream has ended. It fails
// with NoTranscriptGeneratedError when the stream ended without a
// transcript, with the stream's error when it failed, and with
// ErrStreamClosed when the stream was closed or the consumer stopped early.
func (r *StreamTranscribeResult) Result() (*TranscribeResult, error) {
	return r.stream.wait()
}

// Close abandons the stream. A running stream is cancelled; a stream that
// has not started will not start. Close is safe to call more than once and
// from any goroutine.
func (r *StreamTranscribeResult) Close() error {
	return r.stream.close()
}

// run opens the model stream, reports it to telemetry, delivers its
// transcript parts and returns the final transcript. A consumer that stops
// or closes the stream is not reported as an error.
func (r *StreamTranscribeResult) run(ctx context.Context, deliver func(model.TranscriptionStreamPart) bool) (*TranscribeResult, error) {
	audio := &countingReader{
		reader: r.options.Audio,
	}

	r.telemetry.onStreamTranscriptionStart(ctx, r.startEvent())

	result, finish, err := r.transcribe(ctx, audio, deliver)
	if err != nil {
		if !isStreamClosed(ctx, err) {
			r.telemetry.onError(ctx, TelemetryErrorEvent{
				CallID: r.callID,
				Error:  err,
			})
		}

		return nil, err
	}

	r.telemetry.onStreamTranscriptionEnd(ctx, r.endEvent(result, finish, audio.count()))

	return result, nil
}

// transcribe opens the model stream, delivers its transcript parts and
// folds the others into the final transcript.
func (r *StreamTranscribeResult) transcribe(
	ctx context.Context,
	audio io.Reader,
	deliver func(model.TranscriptionStreamPart) bool,
) (*TranscribeResult, *model.TranscriptFinish, error) {
	startedAt := time.Now()

	stream, err := r.model.StreamTranscribe(ctx, model.TranscriptionStreamOptions{
		Audio:            audio,
		InputAudioFormat: r.options.InputAudioFormat,
		Headers:          r.options.Headers,
		ProviderOptions:  r.options.ProviderOptions,
		IncludeRawChunks: r.options.IncludeRawChunks,
	})
	if err != nil {
		deliver(model.StreamError{
			Err: err,
		})

		return nil, nil, err
	}

	defer closeTranscriptionStream(stream)

	collector := &transcriptCollector{
		streamResponseCollector: newStreamResponseCollector(stream.Response, startedAt, r.model.ModelID()),
	}

	collect := func(part model.TranscriptionStreamPart) bool {
		r.logStreamWarnings(ctx, part)

		return collector.collect(part)
	}

	if err := forwardStreamParts(stream.Parts(), collect, deliver); err != nil {
		return nil, nil, err
	}

	result, err := collector.result(ctx)
	if err != nil {
		return nil, nil, err
	}

	return result, collector.finish, nil
}

// logStreamWarnings passes the warnings of a stream start part to
// LogWarnings when it arrives.
func (r *StreamTranscribeResult) logStreamWarnings(ctx context.Context, part model.TranscriptionStreamPart) {
	start, ok := part.(model.StreamStart)
	if !ok {
		return
	}

	r.options.LogWarnings.log(ctx, WarningsEvent{
		Warnings: start.Warnings,
		Provider: r.model.Provider(),
		ModelID:  r.model.ModelID(),
	})
}

// closeTranscriptionStream releases a stream whose outcome is decided. Its
// error is ignored: the transcript has already been read or abandoned.
func closeTranscriptionStream(stream *model.TranscriptionStreamResult) {
	_ = stream.Close()
}

// transcriptCollector folds stream start, response metadata, finish and
// error parts into the final transcript.
type transcriptCollector struct {
	streamResponseCollector

	finish *model.TranscriptFinish
}

// collect records one part and reports whether it goes to the consumer.
func (c *transcriptCollector) collect(part model.TranscriptionStreamPart) bool {
	switch typed := part.(type) {
	case model.StreamStart:
		c.start(typed)

		return false

	case model.ResponseMetadata:
		c.responseMetadata(typed)

		return false

	case model.TranscriptFinish:
		c.finish = &typed

		return false

	case model.StreamError:
		c.streamError(typed)

		return true

	default:
		return true
	}
}

// result returns the final transcript. A stream without a finish part
// fails with its last error, the context's cause, or
// NoTranscriptGeneratedError.
func (c *transcriptCollector) result(ctx context.Context) (*TranscribeResult, error) {
	if c.finish == nil {
		return nil, c.failure(ctx, c.noTranscript())
	}

	if c.finish.Text == "" {
		return nil, c.noTranscript()
	}

	return &TranscribeResult{
		Text:     c.finish.Text,
		Segments: c.finish.Segments,
		Language: c.finish.Language,
		Duration: c.finish.Duration,
		Warnings: c.warnings,
		Responses: []model.ResponseInfo{
			c.response,
		},
		ProviderMetadata: c.finish.ProviderMetadata,
	}, nil
}

func (c *transcriptCollector) noTranscript() error {
	return &NoTranscriptGeneratedError{
		Responses: []model.ResponseInfo{
			c.response,
		},
	}
}
