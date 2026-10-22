package fal

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/download"
	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// Speech output formats. Speech is always downloaded from the returned URL.
const (
	speechOutputURL = "url"
	speechOutputHex = "hex"
)

// LanguageBoost improves recognition of a language.
type LanguageBoost string

// languageBoosts are the accepted language boosts.
var languageBoosts = []LanguageBoost{
	"Chinese",
	"Chinese,Yue",
	"English",
	"Arabic",
	"Russian",
	"Spanish",
	"French",
	"Portuguese",
	"German",
	"Turkish",
	"Dutch",
	"Ukrainian",
	"Vietnamese",
	"Indonesian",
	"Japanese",
	"Italian",
	"Korean",
	"Thai",
	"Polish",
	"Romanian",
	"Greek",
	"Czech",
	"Finnish",
	"Hindi",
	"auto",
}

// Emotion is the emotion of generated speech.
type Emotion string

// Emotions.
const (
	EmotionHappy     Emotion = "happy"
	EmotionSad       Emotion = "sad"
	EmotionAngry     Emotion = "angry"
	EmotionFearful   Emotion = "fearful"
	EmotionDisgusted Emotion = "disgusted"
	EmotionSurprised Emotion = "surprised"
	EmotionNeutral   Emotion = "neutral"
)

// emotions are the accepted emotions.
var emotions = []Emotion{
	EmotionHappy,
	EmotionSad,
	EmotionAngry,
	EmotionFearful,
	EmotionDisgusted,
	EmotionSurprised,
	EmotionNeutral,
}

// VoiceSetting configures the voice of MiniMax speech models.
type VoiceSetting struct {
	Speed                *float64 `json:"speed,omitempty"`
	Vol                  *float64 `json:"vol,omitempty"`
	VoiceID              *string  `json:"voice_id,omitempty"`
	Pitch                *float64 `json:"pitch,omitempty"`
	EnglishNormalization *bool    `json:"english_normalization,omitempty"`
	Emotion              *Emotion `json:"emotion,omitempty"`
}

// SpeechOptions are the provider options of speech models, read from the
// "fal" namespace. Every member, including model-specific members that are
// not fields here, is sent in the request body unchanged.
type SpeechOptions struct {
	VoiceSetting *VoiceSetting `json:"voice_setting,omitempty"`

	// AudioSetting holds model-specific audio settings.
	AudioSetting map[string]json.RawMessage `json:"audio_setting,omitempty"`

	LanguageBoost *LanguageBoost `json:"language_boost,omitempty"`

	// PronunciationDict maps words to pronunciations.
	PronunciationDict map[string]string `json:"pronunciation_dict,omitempty"`
}

// SpeechModel is a fal.ai text-to-speech model.
type SpeechModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "fal.speech" by default.
func (m *SpeechModel) Provider() string {
	return m.provider.name + ".speech"
}

// ModelID returns the model identifier.
func (m *SpeechModel) ModelID() string {
	return m.modelID
}

type speechRequest struct {
	Text         string   `json:"text"`
	OutputFormat string   `json:"output_format"`
	Voice        string   `json:"voice,omitempty"`
	Speed        *float64 `json:"speed,omitempty"`
}

type speechResponse struct {
	Audio      speechAudio `json:"audio"`
	DurationMs *float64    `json:"duration_ms"`
	RequestID  *string     `json:"request_id"`
}

type speechAudio struct {
	URL string `json:"url"`
}

// Generate generates speech and downloads the audio file.
func (m *SpeechModel) Generate(ctx context.Context, options model.SpeechOptions) (*model.SpeechResult, error) {
	timestamp := time.Now()

	body, err := speechBody(options)
	if err != nil {
		return nil, err
	}

	requestURL := speechBaseURL + "/" + m.modelID

	request, err := m.provider.apiRequest(requestURL, options.Headers, parseError)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, err
	}

	var decoded speechResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	audio, err := download.Fetch(ctx, download.Request{
		URL:           decoded.Audio.URL,
		TrustedOrigin: requestURL,
		Client:        m.provider.httpClient,
	})
	if err != nil {
		return nil, err
	}

	return &model.SpeechResult{
		Audio:    audio.Data,
		Warnings: unsupportedSpeechSettings(options),
		Request: model.RequestInfo{
			Body: response.RequestBody,
		},
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: timestamp,
				ModelID:   m.modelID,
			},
			Headers: response.Headers,
			Body:    response.Body,
		},
	}, nil
}

// speechBody builds the request. The provider options are spread over the
// standard fields.
func speechBody(options model.SpeechOptions) (map[string]json.RawMessage, error) {
	speechOptions, _, err := model.ProviderValue[SpeechOptions](options.ProviderOptions, Namespace)
	if err != nil {
		return nil, err
	}

	if err := speechOptions.validate(); err != nil {
		return nil, err
	}

	passthrough, _, err := model.ProviderValue[map[string]json.RawMessage](options.ProviderOptions, Namespace)
	if err != nil {
		return nil, err
	}

	outputFormat := speechOutputURL
	if options.OutputFormat == speechOutputHex {
		outputFormat = speechOutputHex
	}

	body, err := jsonobject.Members(speechRequest{
		Text:         options.Text,
		OutputFormat: outputFormat,
		Voice:        options.Voice,
		Speed:        options.Speed,
	})
	if err != nil {
		return nil, fmt.Errorf("encode request body: %w", err)
	}

	maps.Copy(body, passthrough)

	return body, nil
}

func (o SpeechOptions) validate() error {
	if o.LanguageBoost != nil && !slices.Contains(languageBoosts, *o.LanguageBoost) {
		return fmt.Errorf("%w: fal language_boost %q", model.ErrInvalidArgument, *o.LanguageBoost)
	}

	if o.VoiceSetting != nil && o.VoiceSetting.Emotion != nil && !slices.Contains(emotions, *o.VoiceSetting.Emotion) {
		return fmt.Errorf("%w: fal voice_setting.emotion %q", model.ErrInvalidArgument, *o.VoiceSetting.Emotion)
	}

	return nil
}

func unsupportedSpeechSettings(options model.SpeechOptions) []model.Warning {
	var warnings []model.Warning

	if options.Language != "" {
		warnings = append(warnings, model.UnsupportedWarning(
			"language",
			"fal speech models don't support 'language' directly; consider providerOptions.fal.language_boost",
		))
	}

	if options.OutputFormat != "" && options.OutputFormat != speechOutputURL && options.OutputFormat != speechOutputHex {
		warnings = append(warnings, model.UnsupportedWarning(
			"outputFormat",
			fmt.Sprintf("Unsupported outputFormat: %s. Using 'url' instead.", options.OutputFormat),
		))
	}

	return warnings
}

var _ model.SpeechModel = (*SpeechModel)(nil)
