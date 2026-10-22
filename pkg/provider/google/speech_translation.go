package google

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/aarock1234/ai/internal/livetranscription"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// translationFinishGrace is the trailing output silence, after the
	// input audio ended, that finishes a translation. Live translation is
	// continuous and does not complete turns.
	translationFinishGrace = time.Second

	// translationInputRate is the only input audio rate.
	translationInputRate = 16000

	// translationOutputRate is the rate of the Live API output audio.
	translationOutputRate = 24000
)

var (
	_ model.SpeechTranslationModel    = (*SpeechTranslationModel)(nil)
	_ model.SpeechTranslationProvider = (*Provider)(nil)
)

// SpeechTranslationOptions are the Google settings of a speech translation,
// read from ProviderOptions["google"].
type SpeechTranslationOptions struct {
	// EchoTargetLanguage echoes input audio already in the target language
	// instead of producing silence.
	EchoTargetLanguage *bool `json:"echoTargetLanguage,omitempty"`
}

// SpeechTranslationModel translates live speech with a Gemini Live
// translation model, such as gemini-3.5-live-translate-preview, over the
// Live API WebSocket. Input audio is 16 kHz 16-bit PCM, output audio is
// 24 kHz 16-bit PCM, and the source language is detected. The speech
// translation contract is experimental and may change.
type SpeechTranslationModel struct {
	modelID  string
	provider *Provider
}

// SpeechTranslation returns a Live API speech translation model. Speech
// translation is experimental.
func (p *Provider) SpeechTranslation(modelID string) *SpeechTranslationModel {
	return &SpeechTranslationModel{
		modelID:  modelID,
		provider: p,
	}
}

// SpeechTranslationModel returns SpeechTranslation(modelID).
func (p *Provider) SpeechTranslationModel(modelID string) (model.SpeechTranslationModel, error) {
	return p.SpeechTranslation(modelID), nil
}

// Provider returns the provider identifier,
// "google.generative-ai.speech-translation" by default.
func (m *SpeechTranslationModel) Provider() string {
	return m.provider.name + ".speech-translation"
}

// ModelID returns the model ID.
func (m *SpeechTranslationModel) ModelID() string {
	return m.modelID
}

// translationSetup is the Live API setup of a translation.
type translationSetup struct {
	Model                    string                      `json:"model"`
	GenerationConfig         translationGenerationConfig `json:"generationConfig"`
	InputAudioTranscription  emptyObject                 `json:"inputAudioTranscription"`
	OutputAudioTranscription emptyObject                 `json:"outputAudioTranscription"`
}

type translationGenerationConfig struct {
	ResponseModalities []string                  `json:"responseModalities"`
	TranslationConfig  RealtimeTranslationConfig `json:"translationConfig"`
}

// StreamTranslation opens the Live API WebSocket. The audio is sent once
// the server acknowledged the setup, and the stream finishes after the
// input audio ended and one second of trailing output silence arrived, or
// when a pending finish meets a server close. The API key is required: it
// authenticates the WebSocket as a query parameter.
func (m *SpeechTranslationModel) StreamTranslation(ctx context.Context, options model.SpeechTranslationStreamOptions) (*model.SpeechTranslationStreamResult, error) {
	if err := validateSpeechTranslation(options, m.modelID); err != nil {
		return nil, err
	}

	timestamp := time.Now()

	googleOptions, _, err := model.ProviderValue[SpeechTranslationOptions](options.ProviderOptions, Namespace)
	if err != nil {
		return nil, err
	}

	setup, err := json.Marshal(translationSetup{
		Model: modelPath(m.modelID),
		GenerationConfig: translationGenerationConfig{
			ResponseModalities: []string{
				"AUDIO",
			},
			TranslationConfig: RealtimeTranslationConfig{
				TargetLanguageCode: options.TargetLanguage,
				EchoTargetLanguage: googleOptions.EchoTargetLanguage,
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("encode translation setup: %w", err)
	}

	setupMessage, err := json.Marshal(realtimeSetupMessage{
		Setup: setup,
	})
	if err != nil {
		return nil, fmt.Errorf("encode translation setup: %w", err)
	}

	connection, err := m.provider.liveDialOptions(ctx, options.Headers, "streaming translation")
	if err != nil {
		return nil, err
	}

	conn, err := livetranscription.Dial(ctx, connection)
	if err != nil {
		return nil, err
	}

	protocol := &translationProtocol{
		warnings:     speechTranslationWarnings(options),
		setupMessage: setupMessage,
		finishGrace:  translationFinishGrace,
	}

	result := livetranscription.NewTranslationResult(ctx, livetranscription.TranslationSetup{
		Conn:             conn,
		Protocol:         protocol,
		Audio:            options.Audio,
		IncludeRawChunks: options.IncludeRawChunks,
		KeepPendingTimer: true,
		Name:             "google live translation",
	})
	result.Request = model.RequestInfo{
		Body: setup,
	}
	result.Response = model.ResponseInfo{
		ResponseMetadata: model.ResponseMetadata{
			Timestamp: timestamp,
			ModelID:   m.modelID,
		},
		Headers: conn.ResponseHeader(),
	}

	return result, nil
}

// validateSpeechTranslation requires a target language, audio and 16 kHz
// PCM input.
func validateSpeechTranslation(options model.SpeechTranslationStreamOptions, modelID string) error {
	if options.TargetLanguage == "" {
		return fmt.Errorf("%w: targetLanguage is required for translation model %q", model.ErrInvalidArgument, modelID)
	}

	if options.Audio == nil {
		return fmt.Errorf("%w: speech translation requires audio", model.ErrInvalidArgument)
	}

	format := options.InputAudioFormat
	if format.Type != "audio/pcm" || (format.Rate != 0 && format.Rate != translationInputRate) {
		return fmt.Errorf(
			"%w: the Gemini Live translation API only supports 16kHz 16-bit PCM input audio",
			model.ErrInvalidArgument,
		)
	}

	return nil
}

// speechTranslationWarnings reports settings the Live API does not take.
func speechTranslationWarnings(options model.SpeechTranslationStreamOptions) []model.Warning {
	var warnings []model.Warning

	if options.SourceLanguage != "" {
		warnings = append(warnings, model.UnsupportedWarning(
			"sourceLanguage",
			"The Gemini Live translation API auto-detects the source language and does not accept a source language.",
		))
	}

	if options.OutputAudioFormat != nil {
		warnings = append(warnings, model.UnsupportedWarning(
			"outputAudioFormat",
			"The Gemini Live API always outputs 24kHz 16-bit PCM audio and does not accept an output audio format.",
		))
	}

	return warnings
}
