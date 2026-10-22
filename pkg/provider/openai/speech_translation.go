package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"time"

	"github.com/aarock1234/ai/internal/livetranscription"
	"github.com/aarock1234/ai/internal/websocket"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// translationPath is the realtime translation WebSocket endpoint.
	translationPath = "/realtime/translations"

	// translationSampleRate is the only input and output sample rate.
	translationSampleRate = 24000

	// translationTranscriptionModel transcribes the source audio.
	translationTranscriptionModel = "gpt-realtime-whisper"
)

// SpeechTranslationModel translates live speech with the OpenAI Realtime
// translation API, such as gpt-realtime-translate. Input and output audio
// are 24 kHz 16-bit PCM, and the source language is detected. The speech
// translation contract is experimental and may change.
type SpeechTranslationModel struct {
	modelID string
	config  config
}

func newSpeechTranslationModel(modelID string, c config) *SpeechTranslationModel {
	return &SpeechTranslationModel{
		modelID: modelID,
		config:  c,
	}
}

// Provider returns the provider identifier, such as
// "openai.speech-translation".
func (m *SpeechTranslationModel) Provider() string {
	return m.config.modelProvider("speech-translation")
}

// ModelID returns the model identifier.
func (m *SpeechTranslationModel) ModelID() string {
	return m.modelID
}

// translationSessionUpdate configures a translation session.
type translationSessionUpdate struct {
	Type    string             `json:"type"`
	Session translationSession `json:"session"`
}

type translationSession struct {
	Audio translationAudio `json:"audio"`
}

type translationAudio struct {
	Input  translationAudioInput  `json:"input"`
	Output translationAudioOutput `json:"output"`
}

type translationAudioInput struct {
	Transcription translationTranscription `json:"transcription"`

	// NoiseReduction is always null, which turns noise reduction off.
	NoiseReduction *struct{} `json:"noise_reduction"`
}

type translationTranscription struct {
	Model string `json:"model"`
}

type translationAudioOutput struct {
	Language string `json:"language"`
}

// StreamTranslation opens the translation WebSocket. The audio is sent as
// it is read, and the session closes once Audio reaches io.EOF and the
// translation of the remaining audio has arrived.
func (m *SpeechTranslationModel) StreamTranslation(ctx context.Context, options model.SpeechTranslationStreamOptions) (*model.SpeechTranslationStreamResult, error) {
	if err := validateTranslationOptions(options); err != nil {
		return nil, err
	}

	timestamp := time.Now()

	sessionUpdate, err := json.Marshal(translationSessionUpdate{
		Type: "session.update",
		Session: translationSession{
			Audio: translationAudio{
				Input: translationAudioInput{
					Transcription: translationTranscription{
						Model: translationTranscriptionModel,
					},
				},
				Output: translationAudioOutput{
					Language: options.TargetLanguage,
				},
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("encode translation session: %w", err)
	}

	conn, err := m.dialTranslation(ctx, options.Headers)
	if err != nil {
		return nil, err
	}

	protocol := &realtimeTranslationProtocol{
		warnings:      translationWarnings(options),
		sessionUpdate: sessionUpdate,
	}

	result := livetranscription.NewTranslationResult(ctx, livetranscription.TranslationSetup{
		Conn:             conn,
		Protocol:         protocol,
		Audio:            options.Audio,
		IncludeRawChunks: options.IncludeRawChunks,
		Name:             "openai realtime translation",
	})
	result.Request = model.RequestInfo{
		Body: sessionUpdate,
	}
	result.Response = model.ResponseInfo{
		Timestamp: timestamp,
		ModelID:   m.modelID,
		Headers:   conn.ResponseHeader(),
	}

	return result, nil
}

// validateTranslationOptions requires a target language, audio and 24 kHz
// PCM input.
func validateTranslationOptions(options model.SpeechTranslationStreamOptions) error {
	if options.TargetLanguage == "" {
		return fmt.Errorf("%w: targetLanguage is required for speech translation", model.ErrInvalidArgument)
	}

	if options.Audio == nil {
		return fmt.Errorf("%w: speech translation requires audio", model.ErrInvalidArgument)
	}

	format := options.InputAudioFormat
	if format.Type != "audio/pcm" || (format.Rate != 0 && format.Rate != translationSampleRate) {
		return fmt.Errorf(
			"%w: the OpenAI Realtime translation API only supports 24kHz 16-bit PCM input audio",
			model.ErrInvalidArgument,
		)
	}

	if _, _, err := model.ProviderValue[map[string]json.RawMessage](options.ProviderOptions, Namespace); err != nil {
		return err
	}

	return nil
}

// translationWarnings reports settings the translation API does not take.
func translationWarnings(options model.SpeechTranslationStreamOptions) []model.Warning {
	var warnings []model.Warning

	if options.SourceLanguage != "" {
		warnings = append(warnings, model.UnsupportedWarning(
			"sourceLanguage",
			"The OpenAI Realtime translation API auto-detects the source language and does not accept a source language.",
		))
	}

	if options.OutputAudioFormat != nil {
		warnings = append(warnings, model.UnsupportedWarning(
			"outputAudioFormat",
			"The OpenAI Realtime translation API always outputs 24kHz 16-bit PCM audio and does not accept an output audio format.",
		))
	}

	return warnings
}

// dialTranslation opens the translation WebSocket with the provider's
// authentication and headers.
func (m *SpeechTranslationModel) dialTranslation(ctx context.Context, callHeaders http.Header) (*websocket.Conn, error) {
	request, err := m.config.request(ctx, translationPath+"?model="+url.QueryEscape(m.modelID), callHeaders)
	if err != nil {
		return nil, err
	}

	target, err := livetranscription.WebSocketURL(request.URL)
	if err != nil {
		return nil, err
	}

	return dialRealtimeSocket(ctx, request, target)
}
