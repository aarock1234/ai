package ai

import (
	"context"
	"net/http"
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

// streamTranslateOperationID identifies StreamTranslate in telemetry.
const streamTranslateOperationID = "ai.streamTranslate"

// TranslationTelemetry receives the lifecycle events of StreamTranslate.
// The speech translation API is experimental.
type TranslationTelemetry interface {
	// OnStreamTranslationStart is called when a StreamTranslate stream
	// starts.
	OnStreamTranslationStart(ctx context.Context, event TelemetryEvent[TranslationStartEvent])

	// OnStreamTranslationEnd is called when a StreamTranslate stream
	// finishes with a translation.
	OnStreamTranslationEnd(ctx context.Context, event TelemetryEvent[TranslationEndEvent])
}

// TranslationStartEvent describes a StreamTranslate stream before the model
// is called. The audio itself is left out.
type TranslationStartEvent struct {
	CallID string

	// OperationID is "ai.streamTranslate".
	OperationID string

	Provider string
	ModelID  string

	InputAudioFormat model.AudioFormat
	TargetLanguage   string
	SourceLanguage   string

	// OutputAudioFormat is nil for the provider default.
	OutputAudioFormat *model.AudioFormat

	Headers         http.Header
	ProviderOptions model.ProviderOptions
}

// TranslationEndEvent describes a StreamTranslate stream that finished with
// a translation. The audio itself is left out.
type TranslationEndEvent struct {
	CallID      string
	OperationID string
	Provider    string
	ModelID     string

	// AudioByteLength is the size of the source audio that was read.
	AudioByteLength int

	SourceText      string
	TranslationText string

	// Duration is the length of the source audio, or zero when unknown.
	Duration time.Duration

	// Usage is nil when the provider did not report it.
	Usage *model.SpeechTranslationUsage

	Warnings         []model.Warning
	ProviderMetadata model.ProviderMetadata
	Response         model.ResponseInfo
}

// OnStreamTranslationStart does nothing.
func (NopTelemetry) OnStreamTranslationStart(context.Context, TelemetryEvent[TranslationStartEvent]) {
}

// OnStreamTranslationEnd does nothing.
func (NopTelemetry) OnStreamTranslationEnd(context.Context, TelemetryEvent[TranslationEndEvent]) {}

func (d telemetryDispatcher) onStreamTranslationStart(ctx context.Context, event TranslationStartEvent) {
	notifyTelemetry(ctx, d, event, TranslationTelemetry.OnStreamTranslationStart)
}

func (d telemetryDispatcher) onStreamTranslationEnd(ctx context.Context, event TranslationEndEvent) {
	notifyTelemetry(ctx, d, event, TranslationTelemetry.OnStreamTranslationEnd)
}

// startEvent describes the stream before the model is called.
func (r *StreamTranslateResult) startEvent() TranslationStartEvent {
	return TranslationStartEvent{
		CallID:            r.callID,
		OperationID:       streamTranslateOperationID,
		Provider:          r.options.Model.Provider(),
		ModelID:           r.options.Model.ModelID(),
		InputAudioFormat:  r.options.InputAudioFormat,
		TargetLanguage:    r.options.TargetLanguage,
		SourceLanguage:    r.options.SourceLanguage,
		OutputAudioFormat: r.options.OutputAudioFormat,
		Headers:           r.options.Headers,
		ProviderOptions:   r.options.ProviderOptions,
	}
}

// endEvent describes a stream that finished with a translation, after
// audioByteLength bytes of audio were read.
func (r *StreamTranslateResult) endEvent(result *TranslateResult, audioByteLength int) TranslationEndEvent {
	return TranslationEndEvent{
		CallID:           r.callID,
		OperationID:      streamTranslateOperationID,
		Provider:         r.options.Model.Provider(),
		ModelID:          r.options.Model.ModelID(),
		AudioByteLength:  audioByteLength,
		SourceText:       result.SourceText,
		TranslationText:  result.TranslationText,
		Duration:         result.Duration,
		Usage:            result.Usage,
		Warnings:         result.Warnings,
		ProviderMetadata: result.ProviderMetadata,
		Response:         result.Responses[0],
	}
}
