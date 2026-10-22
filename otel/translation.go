package otel

import (
	"context"
	"encoding/json"

	"go.opentelemetry.io/otel/attribute"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// Upstream's streamTranslate reports no telemetry. The Go port records it
// like StreamTranscribe: one operation span with the input audio, the
// languages, the source transcript and the translated text.

// translationUsage is the JSON shape of upstream's speech translation
// usage, with the input audio duration in seconds.
type translationUsage struct {
	InputAudioSeconds *float64 `json:"inputAudioSeconds,omitempty"`
	InputAudioTokens  *int     `json:"inputAudioTokens,omitempty"`
	OutputAudioTokens *int     `json:"outputAudioTokens,omitempty"`
	InputTextTokens   *int     `json:"inputTextTokens,omitempty"`
	OutputTextTokens  *int     `json:"outputTextTokens,omitempty"`
}

// OnStreamTranslationStart starts the operation span of StreamTranslate.
func (t *OpenTelemetry) OnStreamTranslationStart(ctx context.Context, event ai.TelemetryEvent[ai.TranslationStartEvent]) {
	t.startAudioOperation(ctx, event.TelemetryMetadata, translationAudioStart(event.Event))
}

// OnStreamTranslationEnd records the translation and ends the operation
// span.
func (t *OpenTelemetry) OnStreamTranslationEnd(_ context.Context, event ai.TelemetryEvent[ai.TranslationEndEvent]) {
	t.endAudioOperation(translationAudioEnd(event.Event))
}

// OnStreamTranslationStart starts the root span of StreamTranslate.
func (l *Legacy) OnStreamTranslationStart(ctx context.Context, event ai.TelemetryEvent[ai.TranslationStartEvent]) {
	l.startAudioOperation(ctx, event.TelemetryMetadata, translationAudioStart(event.Event))
}

// OnStreamTranslationEnd records the translation and ends the root span.
func (l *Legacy) OnStreamTranslationEnd(_ context.Context, event ai.TelemetryEvent[ai.TranslationEndEvent]) {
	l.endAudioOperation(translationAudioEnd(event.Event))
}

func translationAudioStart(start ai.TranslationStartEvent) audioStart {
	return audioStart{
		callID:         start.CallID,
		operationID:    start.OperationID,
		provider:       start.Provider,
		modelID:        start.ModelID,
		headers:        start.Headers,
		audioMediaType: start.InputAudioFormat.Type,
		targetLanguage: start.TargetLanguage,
		sourceLanguage: start.SourceLanguage,
	}
}

func translationAudioEnd(end ai.TranslationEndEvent) audioEnd {
	sourceText := end.SourceText
	translationText := end.TranslationText

	return audioEnd{
		callID: end.CallID,
		inputAudio: &audioDescription{
			byteLength: end.AudioByteLength,
		},
		outputText:       &translationText,
		sourceText:       &sourceText,
		usage:            encodeTranslationUsage(end.Usage),
		providerMetadata: end.ProviderMetadata,
	}
}

// encodeTranslationUsage returns the usage in upstream's JSON shape, or nil
// when the provider reported none.
func encodeTranslationUsage(usage *model.SpeechTranslationUsage) json.RawMessage {
	if usage == nil {
		return nil
	}

	wire := translationUsage{
		InputAudioTokens:  usage.InputAudioTokens,
		OutputAudioTokens: usage.OutputAudioTokens,
		InputTextTokens:   usage.InputTextTokens,
		OutputTextTokens:  usage.OutputTextTokens,
	}

	if usage.InputAudioDuration != nil {
		wire.InputAudioSeconds = new(usage.InputAudioDuration.Seconds())
	}

	encoded, err := json.Marshal(wire)
	if err != nil {
		return nil
	}

	return encoded
}

// setSourceText records the transcript of the source audio of a
// translation. The key differs between the integrations.
func setSourceText(attributes *spanAttributes, sourceText *string, key string) {
	if sourceText == nil {
		return
	}

	text := *sourceText

	attributes.setOutput(key, func() attribute.Value {
		return attribute.StringValue(text)
	})
}
