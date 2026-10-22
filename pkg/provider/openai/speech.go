package openai

import (
	"context"
	"fmt"
	"slices"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	speechDefaultVoice  = "alloy"
	speechDefaultFormat = "mp3"
)

// speechOutputFormats are the audio formats the speech endpoint returns.
var speechOutputFormats = []string{
	"mp3",
	"opus",
	"aac",
	"flac",
	"wav",
	"pcm",
}

// SpeechOptions are the speech settings read from the "openai" provider
// options namespace. They override the matching call options.
type SpeechOptions struct {
	// Instructions guide the speaking style, for example "Speak in a slow and
	// steady tone". They do not work with tts-1 or tts-1-hd.
	Instructions string `json:"instructions,omitempty"`

	// Speed of the generated audio, from 0.25 to 4.0. The API default is 1.0.
	Speed *float64 `json:"speed,omitempty"`
}

// SpeechModel is an OpenAI text-to-speech model.
type SpeechModel struct {
	modelID string
	config  config
}

func newSpeechModel(modelID string, c config) *SpeechModel {
	return &SpeechModel{
		modelID: modelID,
		config:  c,
	}
}

// Provider returns the provider identifier, such as "openai.speech".
func (m *SpeechModel) Provider() string {
	return m.config.modelProvider("speech")
}

// ModelID returns the model identifier.
func (m *SpeechModel) ModelID() string {
	return m.modelID
}

// speechRequest is the body of a speech request.
type speechRequest struct {
	Model          string   `json:"model"`
	Input          string   `json:"input"`
	Voice          string   `json:"voice"`
	ResponseFormat string   `json:"response_format"`
	Speed          *float64 `json:"speed,omitempty"`
	Instructions   string   `json:"instructions,omitempty"`
}

// Generate converts the text into audio. The voice defaults to "alloy" and
// the format to mp3, which also replaces unsupported formats.
func (m *SpeechModel) Generate(ctx context.Context, options model.SpeechOptions) (*model.SpeechResult, error) {
	timestamp := time.Now()

	body, warnings, err := m.prepareRequest(options)
	if err != nil {
		return nil, err
	}

	request, err := m.config.request(ctx, "/audio/speech", options.Headers)
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

func (m *SpeechModel) prepareRequest(options model.SpeechOptions) (speechRequest, []model.Warning, error) {
	openaiOptions, _, err := model.ProviderValue[SpeechOptions](options.ProviderOptions, Namespace)
	if err != nil {
		return speechRequest{}, nil, err
	}

	if err := openaiOptions.validate(); err != nil {
		return speechRequest{}, nil, err
	}

	var warnings []model.Warning

	body := speechRequest{
		Model:          m.modelID,
		Input:          options.Text,
		Voice:          options.Voice,
		ResponseFormat: speechDefaultFormat,
		Speed:          options.Speed,
		Instructions:   options.Instructions,
	}

	if body.Voice == "" {
		body.Voice = speechDefaultVoice
	}

	if options.OutputFormat != "" {
		if slices.Contains(speechOutputFormats, options.OutputFormat) {
			body.ResponseFormat = options.OutputFormat
		} else {
			warnings = append(warnings, model.UnsupportedWarning(
				"outputFormat",
				fmt.Sprintf("Unsupported output format: %s. Using mp3 instead.", options.OutputFormat),
			))
		}
	}

	if openaiOptions.Speed != nil {
		body.Speed = openaiOptions.Speed
	}

	if openaiOptions.Instructions != "" {
		body.Instructions = openaiOptions.Instructions
	}

	if options.Language != "" {
		warnings = append(warnings, model.UnsupportedWarning(
			"language",
			fmt.Sprintf("OpenAI speech models do not support language selection. Language parameter %q was ignored.", options.Language),
		))
	}

	return body, warnings, nil
}

func (o SpeechOptions) validate() error {
	if o.Speed == nil {
		return nil
	}

	speed := *o.Speed
	if speed < 0.25 || speed > 4.0 {
		return fmt.Errorf("%w: openai speech options: speed must be between 0.25 and 4.0, got %g", model.ErrInvalidArgument, speed)
	}

	return nil
}
