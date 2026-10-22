package mistral

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// speechPath is the text-to-speech endpoint.
	speechPath = "/audio/speech"

	// redactedAudio replaces reference audio in the recorded request body.
	redactedAudio = "[redacted]"
)

// SpeechFormat is an audio format of generated speech.
type SpeechFormat string

// Speech formats.
const (
	SpeechPCM  SpeechFormat = "pcm"
	SpeechWAV  SpeechFormat = "wav"
	SpeechMP3  SpeechFormat = "mp3"
	SpeechFLAC SpeechFormat = "flac"
	SpeechOpus SpeechFormat = "opus"
)

// SpeechModel is a Mistral text-to-speech model.
type SpeechModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "mistral.speech" by default.
func (m *SpeechModel) Provider() string {
	return m.provider.name + ".speech"
}

// ModelID returns the model identifier.
func (m *SpeechModel) ModelID() string {
	return m.modelID
}

type speechRequest struct {
	Model          string       `json:"model"`
	Input          string       `json:"input"`
	VoiceID        string       `json:"voice_id,omitempty"`
	RefAudio       *string      `json:"ref_audio,omitempty"`
	ResponseFormat SpeechFormat `json:"response_format"`
	Stream         bool         `json:"stream"`
}

type speechResponse struct {
	AudioData string `json:"audio_data"`
}

// Generate converts text into audio. Reference audio replaces the voice.
func (m *SpeechModel) Generate(ctx context.Context, options model.SpeechOptions) (*model.SpeechResult, error) {
	timestamp := time.Now()

	speechOptions, err := readSpeechOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	format, warnings := speechFormat(options.OutputFormat)
	warnings = append(warnings, unsupportedSpeechSettings(options)...)

	body := speechRequest{
		Model:          m.modelID,
		Input:          options.Text,
		RefAudio:       speechOptions.RefAudio,
		ResponseFormat: format,
	}

	if speechOptions.RefAudio == nil {
		body.VoiceID = options.Voice
	}

	recordedBody, err := redactedSpeechBody(body)
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

	var decoded speechResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	audio, err := base64.StdEncoding.DecodeString(decoded.AudioData)
	if err != nil {
		return nil, fmt.Errorf("%w: decode audio: %w", model.ErrInvalidResponse, err)
	}

	return &model.SpeechResult{
		Audio:    audio,
		Warnings: warnings,
		Request: model.RequestInfo{
			Body: recordedBody,
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

// speechFormat returns the requested format, falling back to MP3 with a
// warning when Mistral does not support it.
func speechFormat(outputFormat string) (SpeechFormat, []model.Warning) {
	switch format := SpeechFormat(outputFormat); format {
	case "":
		return SpeechMP3, nil

	case SpeechPCM, SpeechWAV, SpeechMP3, SpeechFLAC, SpeechOpus:
		return format, nil

	default:
		return SpeechMP3, []model.Warning{
			model.UnsupportedWarning(
				"outputFormat",
				fmt.Sprintf("Unsupported output format: %s. Using mp3 instead.", outputFormat),
			),
		}
	}
}

func unsupportedSpeechSettings(options model.SpeechOptions) []model.Warning {
	var warnings []model.Warning

	if options.Instructions != "" {
		warnings = append(warnings, model.UnsupportedWarning(
			"instructions",
			"Mistral speech models do not support the `instructions` option. Use a reference audio clip to guide delivery.",
		))
	}

	if options.Speed != nil {
		warnings = append(warnings, model.UnsupportedWarning(
			"speed",
			"Mistral speech models do not support the `speed` option. It was ignored.",
		))
	}

	if options.Language != "" {
		warnings = append(warnings, model.UnsupportedWarning(
			"language",
			"Mistral speech models do not support the `language` option. Language is inferred from the input text and voice.",
		))
	}

	return warnings
}

// redactedSpeechBody encodes the request for the result without the
// reference audio, which can be large and personal.
func redactedSpeechBody(body speechRequest) (json.RawMessage, error) {
	if body.RefAudio != nil {
		body.RefAudio = new(redactedAudio)
	}

	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("encode request body: %w", err)
	}

	return encoded, nil
}

var _ model.SpeechModel = (*SpeechModel)(nil)
