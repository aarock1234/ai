package ai

import (
	"context"
	"io"
	"iter"
	"net/http"
	"time"

	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/pkg/model"
)

// StreamTranslateOptions configures StreamTranslate.
type StreamTranslateOptions struct {
	Model model.SpeechTranslationModel

	// Audio supplies raw audio bytes in InputAudioFormat, such as a
	// microphone capture or an io.Pipe fed by one. It is read until io.EOF
	// while translations stream back. The caller keeps ownership: the stream
	// never closes Audio and stops reading it when the stream ends, but a
	// Read already in progress is not interrupted.
	Audio io.Reader

	// InputAudioFormat describes Audio, such as {Type: "audio/pcm", Rate:
	// 24000}.
	InputAudioFormat model.AudioFormat

	// TargetLanguage is the BCP-47 style language to translate into, such as
	// "en", "es" or "fr-CA". Supported values are provider-specific.
	TargetLanguage string

	// SourceLanguage is the language of the audio. Empty asks the provider
	// to detect it.
	SourceLanguage string

	// OutputAudioFormat is nil for the provider default.
	OutputAudioFormat *model.AudioFormat

	Headers         http.Header
	ProviderOptions model.ProviderOptions

	// IncludeRawChunks asks the provider to emit model.Raw parts.
	IncludeRawChunks bool

	// Telemetry sends the lifecycle events of the stream to the integrations
	// that implement TranslationTelemetry. Upstream's streamTranslate has no
	// telemetry; the Go port reports it like StreamTranscribe.
	Telemetry TelemetryOptions

	// LogWarnings receives the warnings of the stream when it starts. Nil
	// discards them; they are always returned on the result.
	LogWarnings LogWarningsFunc
}

// TranslateResult is the final outcome of a speech translation stream.
type TranslateResult struct {
	// SourceText is the final transcript of the source audio.
	SourceText string

	// TranslationText is the final translated text. It may be empty for
	// providers that produce only audio.
	TranslationText string

	// Duration is the length of the source audio, or zero when unknown.
	Duration time.Duration

	// Usage is nil when the provider did not report it.
	Usage *model.SpeechTranslationUsage

	Warnings []model.Warning

	// Responses holds the response of the stream, which always has a
	// timestamp and a model ID. It is a list like TranscribeResult's.
	Responses []model.ResponseInfo

	ProviderMetadata model.ProviderMetadata
}

// NoTranslationGeneratedError reports a speech translation stream that
// ended without translated audio or text. It wraps
// model.ErrNoContentGenerated.
type NoTranslationGeneratedError struct {
	Responses []model.ResponseInfo
}

// Error returns a fixed message.
func (e *NoTranslationGeneratedError) Error() string {
	return "no translation generated"
}

// Unwrap returns model.ErrNoContentGenerated.
func (e *NoTranslationGeneratedError) Unwrap() error {
	return model.ErrNoContentGenerated
}

// StreamTranslateResult is a speech translation stream with a single
// consumer. The translation starts when FullStream is ranged over or Result
// is called, runs on the consumer's goroutine, and stops when the consumer
// stops. The speech translation API is experimental and may change.
//
// Range over FullStream once, before calling Result, when both the parts
// and the final texts are needed. Result alone consumes the stream. Do not
// call Result while ranging over FullStream on the same goroutine.
type StreamTranslateResult struct {
	options StreamTranslateOptions

	callID    string
	telemetry telemetryDispatcher

	stream *singleConsumerStream[model.SpeechTranslationStreamPart, TranslateResult]
}

// StreamTranslate translates live speech into audio and text in another
// language. Option errors are reported immediately; errors during the
// translation are reported as model.StreamError parts and by Result.
//
// Cancellation and deadlines come from ctx.
func StreamTranslate(ctx context.Context, options StreamTranslateOptions) (*StreamTranslateResult, error) {
	if err := options.validate(); err != nil {
		return nil, err
	}

	result := &StreamTranslateResult{
		options:   options,
		callID:    id.New("call"),
		telemetry: newTelemetryDispatcher(options.Telemetry),
	}

	result.stream = newSingleConsumerStream(ctx, result.run)

	return result, nil
}

func (o StreamTranslateOptions) validate() error {
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

	if o.TargetLanguage == "" {
		return &InvalidArgumentError{
			Parameter: "targetLanguage",
			Message:   "target language is required",
		}
	}

	return nil
}

// FullStream returns the translation parts as they arrive:
// model.TranslationAudio, model.OutputTextDelta, model.OutputTextFinal,
// model.SourceTranscriptDelta, model.SourceTranscriptPartial,
// model.SourceTranscriptFinal, model.Raw and model.StreamError. Stream
// start, response metadata and finish parts are folded into Result. A
// second iteration yields a single model.StreamError wrapping
// model.ErrStreamConsumed.
func (r *StreamTranslateResult) FullStream() iter.Seq[model.SpeechTranslationStreamPart] {
	return r.stream.parts(model.StreamError{
		Err: model.ErrStreamConsumed,
	})
}

// Result returns the final texts once the stream has ended. It fails with
// NoTranslationGeneratedError when the stream ended without a translation,
// with the stream's error when it failed, and with ErrStreamClosed when the
// stream was closed or the consumer stopped early.
func (r *StreamTranslateResult) Result() (*TranslateResult, error) {
	return r.stream.wait()
}

// Close abandons the stream. A running stream is cancelled; a stream that
// has not started will not start. Close is safe to call more than once and
// from any goroutine.
func (r *StreamTranslateResult) Close() error {
	return r.stream.close()
}

// run reports the stream to telemetry, delivers its translation parts and
// returns the final texts. A consumer that stops or closes the stream is
// not reported as an error.
func (r *StreamTranslateResult) run(ctx context.Context, deliver func(model.SpeechTranslationStreamPart) bool) (*TranslateResult, error) {
	audio := &countingReader{
		reader: r.options.Audio,
	}

	r.telemetry.onStreamTranslationStart(ctx, r.startEvent())

	result, err := r.translate(ctx, audio, deliver)
	if err != nil {
		if !isStreamClosed(ctx, err) {
			r.telemetry.onError(ctx, TelemetryErrorEvent{
				CallID: r.callID,
				Error:  err,
			})
		}

		return nil, err
	}

	r.telemetry.onStreamTranslationEnd(ctx, r.endEvent(result, audio.count()))

	return result, nil
}

// translate opens the model stream, delivers its translation parts and
// folds the others into the final texts.
func (r *StreamTranslateResult) translate(
	ctx context.Context,
	audio io.Reader,
	deliver func(model.SpeechTranslationStreamPart) bool,
) (*TranslateResult, error) {
	startedAt := time.Now()

	stream, err := r.options.Model.StreamTranslation(ctx, model.SpeechTranslationStreamOptions{
		Audio:             audio,
		InputAudioFormat:  r.options.InputAudioFormat,
		TargetLanguage:    r.options.TargetLanguage,
		SourceLanguage:    r.options.SourceLanguage,
		OutputAudioFormat: r.options.OutputAudioFormat,
		Headers:           r.options.Headers,
		ProviderOptions:   r.options.ProviderOptions,
		IncludeRawChunks:  r.options.IncludeRawChunks,
	})
	if err != nil {
		deliver(model.StreamError{
			Err: err,
		})

		return nil, err
	}

	defer closeTranslationStream(stream)

	collector := &translationCollector{
		streamResponseCollector: newStreamResponseCollector(stream.Response, startedAt, r.options.Model.ModelID()),
	}

	collect := func(part model.SpeechTranslationStreamPart) bool {
		r.logStreamWarnings(ctx, part)

		return collector.collect(part)
	}

	if err := forwardStreamParts(stream.Parts(), collect, deliver); err != nil {
		return nil, err
	}

	return collector.result(ctx)
}

// logStreamWarnings passes the warnings of a stream start part to
// LogWarnings when it arrives.
func (r *StreamTranslateResult) logStreamWarnings(ctx context.Context, part model.SpeechTranslationStreamPart) {
	start, ok := part.(model.StreamStart)
	if !ok {
		return
	}

	r.options.LogWarnings.log(ctx, WarningsEvent{
		Warnings: start.Warnings,
		Provider: r.options.Model.Provider(),
		ModelID:  r.options.Model.ModelID(),
	})
}

// closeTranslationStream releases a stream whose outcome is decided. Its
// error is ignored: the translation has already been read or abandoned.
func closeTranslationStream(stream *model.SpeechTranslationStreamResult) {
	_ = stream.Close()
}

// translationCollector folds stream start, response metadata, finish and
// error parts into the final result.
type translationCollector struct {
	streamResponseCollector

	finish *model.TranslationFinish

	// hasAudio records that translated audio arrived, which makes an empty
	// output text a success.
	hasAudio bool
}

// collect records one part and reports whether it goes to the consumer.
func (c *translationCollector) collect(part model.SpeechTranslationStreamPart) bool {
	switch typed := part.(type) {
	case model.StreamStart:
		c.start(typed)

		return false

	case model.ResponseMetadata:
		c.responseMetadata(typed)

		return false

	case model.TranslationFinish:
		c.finish = &typed

		return false

	case model.TranslationAudio:
		c.hasAudio = true

		return true

	case model.StreamError:
		c.streamError(typed)

		return true

	default:
		return true
	}
}

// result returns the final texts. A stream without a finish part fails with
// its last error, the context's cause, or NoTranslationGeneratedError.
func (c *translationCollector) result(ctx context.Context) (*TranslateResult, error) {
	if c.finish == nil {
		return nil, c.failure(ctx, c.noTranslation())
	}

	if !c.hasAudio && c.finish.OutputText == "" {
		return nil, c.noTranslation()
	}

	return &TranslateResult{
		SourceText:      c.finish.SourceText,
		TranslationText: c.finish.OutputText,
		Duration:        c.finish.Duration,
		Usage:           c.finish.Usage,
		Warnings:        c.warnings,
		Responses: []model.ResponseInfo{
			c.response,
		},
		ProviderMetadata: c.finish.ProviderMetadata,
	}, nil
}

func (c *translationCollector) noTranslation() error {
	return &NoTranslationGeneratedError{
		Responses: []model.ResponseInfo{
			c.response,
		},
	}
}
