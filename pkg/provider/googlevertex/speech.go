package googlevertex

import (
	"context"
	"encoding/base64"
	"fmt"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/google"
)

var _ model.SpeechModel = (*SpeechModel)(nil)

const (
	// cloudSynthesizeURL is the Cloud Text-to-Speech endpoint, which uses a
	// single non-regional host unlike Speech-to-Text and Vertex AI.
	cloudSynthesizeURL = "https://texttospeech.googleapis.com/v1/text:synthesize"

	cloudDefaultVoice    = "Kore"
	cloudDefaultLanguage = "en-US"

	// chirpVoiceInfix appears in Chirp 3: HD voice names, which have the
	// form "<locale>-Chirp3-HD-<voice>", such as "en-US-Chirp3-HD-Kore".
	chirpVoiceInfix = "Chirp3-HD"
)

// SpeechOptions are the options of a Gemini speech call, read from
// ProviderOptions["googleVertex"], ["vertex"] or ["google"]. Chirp models
// take no provider options.
type SpeechOptions = google.SpeechOptions

// SpeechMetadata is the metadata of a Chirp speech result, stored under
// ProviderMetadata["google"].
type SpeechMetadata struct {
	MimeType string `json:"mimeType"`
}

// SpeechModel converts text into speech with Gemini text-to-speech, or with
// Cloud Text-to-Speech for Chirp 3: HD voices. It is safe for concurrent
// use.
type SpeechModel struct {
	modelID  string
	provider string
	gemini   *google.SpeechModel
	cloud    *cloudSpeechModel
}

// Provider returns "google.vertex.speech".
func (m *SpeechModel) Provider() string {
	return m.provider
}

// ModelID returns the model ID.
func (m *SpeechModel) ModelID() string {
	return m.modelID
}

// Generate converts text into audio.
func (m *SpeechModel) Generate(ctx context.Context, options model.SpeechOptions) (*model.SpeechResult, error) {
	if m.cloud != nil {
		return m.cloud.generate(ctx, options)
	}

	return m.gemini.Generate(ctx, options)
}

// cloudSpeechModel synthesizes Chirp 3: HD voices with the Cloud
// Text-to-Speech text:synthesize endpoint and the provider's credentials.
type cloudSpeechModel struct {
	modelID  string
	settings *settings
}

// synthesizeRequest is the body of POST text:synthesize.
type synthesizeRequest struct {
	Input       synthesizeInput       `json:"input"`
	Voice       synthesizeVoice       `json:"voice"`
	AudioConfig synthesizeAudioConfig `json:"audioConfig"`
}

type synthesizeInput struct {
	Text string `json:"text"`
}

type synthesizeVoice struct {
	LanguageCode string `json:"languageCode"`
	Name         string `json:"name"`
}

type synthesizeAudioConfig struct {
	AudioEncoding string   `json:"audioEncoding"`
	SpeakingRate  *float64 `json:"speakingRate,omitempty"`
}

type synthesizeResponse struct {
	AudioContent *string `json:"audioContent"`
}

// generate returns LINEAR16 WAV audio. Empty audio is returned as it is, so
// the caller reports that no speech was generated.
func (m *cloudSpeechModel) generate(ctx context.Context, options model.SpeechOptions) (*model.SpeechResult, error) {
	timestamp := time.Now()

	if err := m.settings.requireTokenSource("Chirp speech models"); err != nil {
		return nil, err
	}

	voiceName, languageCode := chirpVoice(options.Voice, options.Language)
	warnings := unsupportedCloudSpeechSettings(options)

	body := synthesizeRequest{
		Input: synthesizeInput{
			Text: options.Text,
		},
		Voice: synthesizeVoice{
			LanguageCode: languageCode,
			Name:         voiceName,
		},
		AudioConfig: synthesizeAudioConfig{
			AudioEncoding: "LINEAR16",
			SpeakingRate:  options.Speed,
		},
	}

	request, err := m.settings.request(ctx, cloudSynthesizeURL, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, err
	}

	var decoded synthesizeResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	audio := []byte{}
	if decoded.AudioContent != nil {
		audio, err = base64.StdEncoding.DecodeString(*decoded.AudioContent)
		if err != nil {
			return nil, fmt.Errorf("%w: audio content: %w", model.ErrInvalidResponse, err)
		}
	}

	return &model.SpeechResult{
		Audio: audio,
		ProviderMetadata: model.ProviderMetadata{
			google.Namespace: SpeechMetadata{
				MimeType: "audio/wav",
			},
		},
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
			Body:    response.Body,
		},
	}, nil
}

// chirpVoice returns the voice name and language code. A fully qualified
// Chirp 3: HD voice name is used as it is, with its locale prefix as the
// default language; otherwise the name is composed from the language and the
// plain voice name.
func chirpVoice(voice string, language string) (string, string) {
	if voice == "" {
		voice = cloudDefaultVoice
	}

	if strings.Contains(voice, chirpVoiceInfix) {
		prefix, _, _ := strings.Cut(voice, chirpVoiceInfix)
		locale := strings.TrimSuffix(prefix, "-")

		languageCode := language
		if languageCode == "" {
			languageCode = locale
		}

		if languageCode == "" {
			languageCode = cloudDefaultLanguage
		}

		return voice, languageCode
	}

	languageCode := language
	if languageCode == "" {
		languageCode = cloudDefaultLanguage
	}

	return languageCode + "-" + chirpVoiceInfix + "-" + voice, languageCode
}

// unsupportedCloudSpeechSettings reports settings Chirp voices ignore.
// LINEAR16 responses are WAV files, so only the wav format is supported.
func unsupportedCloudSpeechSettings(options model.SpeechOptions) []model.Warning {
	var warnings []model.Warning

	if options.Instructions != "" {
		warnings = append(warnings, model.UnsupportedWarning(
			"instructions",
			"Google Cloud Text-to-Speech Chirp 3: HD voices do not support the `instructions` option. It was ignored.",
		))
	}

	if options.OutputFormat != "" && options.OutputFormat != "wav" {
		warnings = append(warnings, model.UnsupportedWarning(
			"outputFormat",
			"Unsupported output format: "+options.OutputFormat+". Using wav instead.",
		))
	}

	return warnings
}
