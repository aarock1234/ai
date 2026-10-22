package elevenlabs

import (
	"context"
	"net/url"
	"strconv"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// speechPath is the text-to-speech endpoint, followed by the voice ID.
	speechPath = "/v1/text-to-speech/"

	// defaultVoiceID is the voice used when the call sets none.
	defaultVoiceID = "21m00Tcm4TlvDq8ikWAM"

	// defaultOutputFormat is the format used when the call sets none.
	defaultOutputFormat = "mp3_44100_128"
)

// speechFormatAliases map short output formats to ElevenLabs formats. Other
// formats are sent unchanged.
var speechFormatAliases = map[string]string{
	"mp3":       "mp3_44100_128",
	"mp3_32":    "mp3_44100_32",
	"mp3_64":    "mp3_44100_64",
	"mp3_96":    "mp3_44100_96",
	"mp3_128":   "mp3_44100_128",
	"mp3_192":   "mp3_44100_192",
	"pcm":       "pcm_44100",
	"pcm_16000": "pcm_16000",
	"pcm_22050": "pcm_22050",
	"pcm_24000": "pcm_24000",
	"pcm_44100": "pcm_44100",
	"ulaw":      "ulaw_8000",
}

// SpeechModel is an ElevenLabs text-to-speech model.
type SpeechModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "elevenlabs.speech" by default.
func (m *SpeechModel) Provider() string {
	return m.provider.name + ".speech"
}

// ModelID returns the model identifier.
func (m *SpeechModel) ModelID() string {
	return m.modelID
}

type speechRequest struct {
	Text                            string                     `json:"text"`
	ModelID                         string                     `json:"model_id,omitempty"`
	LanguageCode                    string                     `json:"language_code,omitempty"`
	VoiceSettings                   *voiceSettingsRequest      `json:"voice_settings,omitempty"`
	PronunciationDictionaryLocators []dictionaryLocatorRequest `json:"pronunciation_dictionary_locators,omitempty"`
	Seed                            *int64                     `json:"seed,omitempty"`
	PreviousText                    string                     `json:"previous_text,omitempty"`
	NextText                        string                     `json:"next_text,omitempty"`
	PreviousRequestIDs              []string                   `json:"previous_request_ids,omitempty"`
	NextRequestIDs                  []string                   `json:"next_request_ids,omitempty"`
	ApplyTextNormalization          TextNormalization          `json:"apply_text_normalization,omitempty"`
	ApplyLanguageTextNormalization  *bool                      `json:"apply_language_text_normalization,omitempty"`
}

type voiceSettingsRequest struct {
	Stability       *float64 `json:"stability,omitempty"`
	SimilarityBoost *float64 `json:"similarity_boost,omitempty"`
	Style           *float64 `json:"style,omitempty"`
	UseSpeakerBoost *bool    `json:"use_speaker_boost,omitempty"`
	Speed           *float64 `json:"speed,omitempty"`
}

type dictionaryLocatorRequest struct {
	PronunciationDictionaryID string `json:"pronunciation_dictionary_id"`
	VersionID                 string `json:"version_id,omitempty"`
}

// Generate converts text into audio. The output format is sent as a query
// parameter; short formats such as "mp3" map to ElevenLabs formats.
func (m *SpeechModel) Generate(ctx context.Context, options model.SpeechOptions) (*model.SpeechResult, error) {
	timestamp := time.Now()

	elevenLabsOptions, err := readSpeechOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	body := m.speechRequest(options, elevenLabsOptions)

	var warnings []model.Warning
	if options.Instructions != "" {
		warnings = append(warnings, model.UnsupportedWarning(
			"instructions",
			"ElevenLabs speech models do not support instructions. Instructions parameter was ignored.",
		))
	}

	request, err := m.provider.apiRequest(speechURLPath(options, elevenLabsOptions), options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, err
	}

	return &model.SpeechResult{
		Audio:    response.Body,
		Warnings: warnings,
		Request: model.RequestInfo{
			Body: response.RequestBody,
		},
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: timestamp,
				ModelID:   m.modelID,
			},
			Headers: response.Headers,
		},
	}, nil
}

func (m *SpeechModel) speechRequest(options model.SpeechOptions, elevenLabsOptions SpeechOptions) speechRequest {
	body := speechRequest{
		Text:                            options.Text,
		ModelID:                         m.modelID,
		LanguageCode:                    options.Language,
		VoiceSettings:                   voiceSettings(options.Speed, elevenLabsOptions.VoiceSettings),
		PronunciationDictionaryLocators: dictionaryLocators(elevenLabsOptions.PronunciationDictionaryLocators),
		Seed:                            elevenLabsOptions.Seed,
		PreviousRequestIDs:              elevenLabsOptions.PreviousRequestIDs,
		NextRequestIDs:                  elevenLabsOptions.NextRequestIDs,
		ApplyTextNormalization:          elevenLabsOptions.ApplyTextNormalization,
		ApplyLanguageTextNormalization:  elevenLabsOptions.ApplyLanguageTextNormalization,
	}

	if body.LanguageCode == "" && elevenLabsOptions.LanguageCode != nil {
		body.LanguageCode = *elevenLabsOptions.LanguageCode
	}

	if elevenLabsOptions.PreviousText != nil {
		body.PreviousText = *elevenLabsOptions.PreviousText
	}

	if elevenLabsOptions.NextText != nil {
		body.NextText = *elevenLabsOptions.NextText
	}

	return body
}

// voiceSettings combines the call speed with the voice settings option. It
// returns nil when neither sets anything.
func voiceSettings(speed *float64, settings *VoiceSettings) *voiceSettingsRequest {
	converted := voiceSettingsRequest{
		Speed: speed,
	}

	if settings != nil {
		converted.Stability = settings.Stability
		converted.SimilarityBoost = settings.SimilarityBoost
		converted.Style = settings.Style
		converted.UseSpeakerBoost = settings.UseSpeakerBoost
	}

	if converted == (voiceSettingsRequest{}) {
		return nil
	}

	return &converted
}

func dictionaryLocators(locators []PronunciationDictionaryLocator) []dictionaryLocatorRequest {
	if locators == nil {
		return nil
	}

	converted := make([]dictionaryLocatorRequest, 0, len(locators))
	for _, locator := range locators {
		request := dictionaryLocatorRequest{
			PronunciationDictionaryID: locator.PronunciationDictionaryID,
		}

		if locator.VersionID != nil {
			request.VersionID = *locator.VersionID
		}

		converted = append(converted, request)
	}

	return converted
}

// speechURLPath returns the endpoint path for the voice with the output
// format and logging query parameters.
func speechURLPath(options model.SpeechOptions, elevenLabsOptions SpeechOptions) string {
	voiceID := options.Voice
	if voiceID == "" {
		voiceID = defaultVoiceID
	}

	query := url.Values{}
	query.Set("output_format", speechOutputFormat(options.OutputFormat))

	if elevenLabsOptions.EnableLogging != nil {
		query.Set("enable_logging", strconv.FormatBool(*elevenLabsOptions.EnableLogging))
	}

	return speechPath + url.PathEscape(voiceID) + "?" + query.Encode()
}

func speechOutputFormat(outputFormat string) string {
	if outputFormat == "" {
		return defaultOutputFormat
	}

	if format, ok := speechFormatAliases[outputFormat]; ok {
		return format
	}

	return outputFormat
}

var _ model.SpeechModel = (*SpeechModel)(nil)
