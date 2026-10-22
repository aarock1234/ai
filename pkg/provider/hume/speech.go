package hume

import (
	"context"
	"fmt"
	"slices"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// speechPath is the file synthesis endpoint.
	speechPath = "/v0/tts/file"

	// defaultVoiceID is the Hume voice used when the call sets none.
	defaultVoiceID = "d8ab67c6-953d-4bd8-9370-8fa53a0f1453"

	// defaultFormat is the audio format used when the call sets none or an
	// unsupported one.
	defaultFormat = "mp3"
)

// speechFormats are the audio formats Hume returns.
var speechFormats = []string{
	"mp3",
	"pcm",
	"wav",
}

// SpeechModel is the Hume text-to-speech model.
type SpeechModel struct {
	provider *Provider
}

// Provider returns the provider identifier, "hume.speech" by default.
func (m *SpeechModel) Provider() string {
	return m.provider.name + ".speech"
}

// ModelID returns the empty model identifier: Hume has a single speech model.
func (m *SpeechModel) ModelID() string {
	return ""
}

type speechRequest struct {
	Utterances []utteranceRequest `json:"utterances"`
	Context    *contextRequest    `json:"context,omitempty"`
	Format     formatRequest      `json:"format"`
}

type utteranceRequest struct {
	Text            string   `json:"text"`
	Description     *string  `json:"description,omitempty"`
	Speed           *float64 `json:"speed,omitempty"`
	TrailingSilence *float64 `json:"trailing_silence,omitempty"`
	Voice           *Voice   `json:"voice,omitempty"`
}

type contextRequest struct {
	GenerationID *string            `json:"generation_id,omitempty"`
	Utterances   []utteranceRequest `json:"utterances,omitzero"`
}

type formatRequest struct {
	Type string `json:"type"`
}

// Generate converts text into audio. The voice defaults to a built-in Hume
// voice and the format to mp3, which also replaces unsupported formats.
func (m *SpeechModel) Generate(ctx context.Context, options model.SpeechOptions) (*model.SpeechResult, error) {
	timestamp := time.Now()

	body, warnings, err := prepareSpeechRequest(options)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(speechPath, options.Headers)
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
			},
			Headers: response.Headers,
		},
	}, nil
}

func prepareSpeechRequest(options model.SpeechOptions) (speechRequest, []model.Warning, error) {
	humeOptions, err := readSpeechOptions(options.ProviderOptions)
	if err != nil {
		return speechRequest{}, nil, err
	}

	var warnings []model.Warning

	voiceID := options.Voice
	if voiceID == "" {
		voiceID = defaultVoiceID
	}

	utterance := utteranceRequest{
		Text:  options.Text,
		Speed: options.Speed,
		Voice: &Voice{
			ID:       voiceID,
			Provider: VoiceHumeAI,
		},
	}

	if options.Instructions != "" {
		utterance.Description = &options.Instructions
	}

	body := speechRequest{
		Utterances: []utteranceRequest{
			utterance,
		},
		Context: speechContext(humeOptions.Context),
		Format: formatRequest{
			Type: defaultFormat,
		},
	}

	if options.OutputFormat != "" {
		if slices.Contains(speechFormats, options.OutputFormat) {
			body.Format.Type = options.OutputFormat
		} else {
			warnings = append(warnings, model.UnsupportedWarning(
				"outputFormat",
				fmt.Sprintf("Unsupported output format: %s. Using mp3 instead.", options.OutputFormat),
			))
		}
	}

	if options.Language != "" {
		warnings = append(warnings, model.UnsupportedWarning(
			"language",
			fmt.Sprintf("Hume speech models do not support language selection. Language parameter %q was ignored.", options.Language),
		))
	}

	return body, warnings, nil
}

// speechContext converts the context option to its wire form, or returns nil
// when it is absent.
func speechContext(options *SpeechContext) *contextRequest {
	if options == nil {
		return nil
	}

	if options.GenerationID != nil {
		return &contextRequest{
			GenerationID: options.GenerationID,
		}
	}

	// The wire form has the same fields with snake_case JSON names.
	utterances := make([]utteranceRequest, 0, len(options.Utterances))
	for _, utterance := range options.Utterances {
		utterances = append(utterances, utteranceRequest(utterance))
	}

	return &contextRequest{
		Utterances: utterances,
	}
}

var _ model.SpeechModel = (*SpeechModel)(nil)
