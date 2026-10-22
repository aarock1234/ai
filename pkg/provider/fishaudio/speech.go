package fishaudio

import (
	"context"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// speechPath is the text-to-speech endpoint.
	speechPath = "/v1/tts"

	// defaultFormat is the format used when the call sets none or an
	// unsupported one.
	defaultFormat = "mp3"

	// minSpeed and maxSpeed bound the prosody speed.
	minSpeed = 0.5
	maxSpeed = 2.0

	// s1ModelID is the model that ignores loudness normalization.
	s1ModelID = "s1"
)

// speechFormats are the audio formats Fish Audio returns.
var speechFormats = []string{
	"wav",
	"pcm",
	"mp3",
	"opus",
}

// SpeechModel is a Fish Audio text-to-speech model.
type SpeechModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "fish-audio.speech" by default.
func (m *SpeechModel) Provider() string {
	return m.provider.name + ".speech"
}

// ModelID returns the model identifier.
func (m *SpeechModel) ModelID() string {
	return m.modelID
}

type speechRequest struct {
	Text                      string          `json:"text"`
	ReferenceID               *ReferenceID    `json:"reference_id,omitempty"`
	Prosody                   *prosodyRequest `json:"prosody,omitempty"`
	Format                    string          `json:"format"`
	SampleRate                *int            `json:"sample_rate,omitempty"`
	MP3Bitrate                *int            `json:"mp3_bitrate,omitempty"`
	OpusBitrate               *int            `json:"opus_bitrate,omitempty"`
	Latency                   Latency         `json:"latency,omitempty"`
	Temperature               *float64        `json:"temperature,omitempty"`
	TopP                      *float64        `json:"top_p,omitempty"`
	ChunkLength               *int            `json:"chunk_length,omitempty"`
	MinChunkLength            *int            `json:"min_chunk_length,omitempty"`
	Normalize                 *bool           `json:"normalize,omitempty"`
	MaxNewTokens              *int            `json:"max_new_tokens,omitempty"`
	RepetitionPenalty         *float64        `json:"repetition_penalty,omitempty"`
	ConditionOnPreviousChunks *bool           `json:"condition_on_previous_chunks,omitempty"`
	EarlyStopThreshold        *float64        `json:"early_stop_threshold,omitempty"`
	Features                  []string        `json:"features,omitempty"`
}

type prosodyRequest struct {
	Speed             *float64 `json:"speed,omitempty"`
	Volume            *float64 `json:"volume,omitempty"`
	NormalizeLoudness *bool    `json:"normalize_loudness,omitempty"`
}

// Generate converts text into audio. The model is selected with the model
// header rather than a body field.
func (m *SpeechModel) Generate(ctx context.Context, options model.SpeechOptions) (*model.SpeechResult, error) {
	timestamp := time.Now()

	fishOptions, err := readSpeechOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	body, warnings := m.speechRequest(options, fishOptions)

	modelHeader := http.Header{
		"Model": {
			m.modelID,
		},
	}

	request, err := m.provider.apiRequest(speechPath, modelHeader, options.Headers)
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

func (m *SpeechModel) speechRequest(options model.SpeechOptions, fishOptions SpeechOptions) (speechRequest, []model.Warning) {
	format, warnings := speechFormat(options.OutputFormat)

	prosody, prosodyWarnings := m.prosody(options.Speed, fishOptions)
	warnings = append(warnings, prosodyWarnings...)
	warnings = append(warnings, unsupportedSettings(options)...)

	body := speechRequest{
		Text:                      options.Text,
		ReferenceID:               fishOptions.ReferenceID,
		Prosody:                   prosody,
		Format:                    format,
		SampleRate:                fishOptions.SampleRate,
		Latency:                   fishOptions.Latency,
		Temperature:               fishOptions.Temperature,
		TopP:                      fishOptions.TopP,
		ChunkLength:               fishOptions.ChunkLength,
		MinChunkLength:            fishOptions.MinChunkLength,
		Normalize:                 fishOptions.Normalize,
		MaxNewTokens:              fishOptions.MaxNewTokens,
		RepetitionPenalty:         fishOptions.RepetitionPenalty,
		ConditionOnPreviousChunks: fishOptions.ConditionOnPreviousChunks,
		EarlyStopThreshold:        fishOptions.EarlyStopThreshold,
		Features:                  fishOptions.Features,
	}

	if body.ReferenceID == nil && options.Voice != "" {
		body.ReferenceID = &ReferenceID{
			ID: options.Voice,
		}
	}

	if fishOptions.MP3Bitrate != nil {
		if format == "mp3" {
			body.MP3Bitrate = fishOptions.MP3Bitrate
		} else {
			warnings = append(warnings, bitrateWarning("mp3Bitrate", "mp3", format))
		}
	}

	if fishOptions.OpusBitrate != nil {
		if format == "opus" {
			body.OpusBitrate = fishOptions.OpusBitrate
		} else {
			warnings = append(warnings, bitrateWarning("opusBitrate", "opus", format))
		}
	}

	return body, warnings
}

// speechFormat returns the requested format, falling back to mp3 with a
// warning when Fish Audio does not support it.
func speechFormat(outputFormat string) (string, []model.Warning) {
	if outputFormat == "" {
		return defaultFormat, nil
	}

	normalized := strings.ToLower(outputFormat)
	if slices.Contains(speechFormats, normalized) {
		return normalized, nil
	}

	return defaultFormat, []model.Warning{
		model.UnsupportedWarning(
			"outputFormat",
			fmt.Sprintf(
				"Fish Audio does not support the output format %q. Falling back to %s. Supported formats are %s.",
				outputFormat,
				defaultFormat,
				strings.Join(speechFormats, ", "),
			),
		),
	}
}

// prosody returns the speed, volume and loudness settings, or nil when none
// apply.
func (m *SpeechModel) prosody(speed *float64, fishOptions SpeechOptions) (*prosodyRequest, []model.Warning) {
	var warnings []model.Warning

	prosody := prosodyRequest{
		Volume: fishOptions.Volume,
	}

	if speed != nil {
		if *speed >= minSpeed && *speed <= maxSpeed {
			prosody.Speed = speed
		} else {
			warnings = append(warnings, model.UnsupportedWarning(
				"speed",
				fmt.Sprintf("Fish Audio speed must be between %g and %g. The speed option was ignored.", minSpeed, maxSpeed),
			))
		}
	}

	if fishOptions.NormalizeLoudness != nil {
		if m.modelID == s1ModelID {
			warnings = append(warnings, model.UnsupportedWarning(
				"providerOptions.fishAudio.normalizeLoudness",
				"Fish Audio ignores normalizeLoudness on s1. It is supported by the S2 family (s2-pro, s2.1-pro).",
			))
		} else {
			prosody.NormalizeLoudness = fishOptions.NormalizeLoudness
		}
	}

	if prosody == (prosodyRequest{}) {
		return nil, warnings
	}

	return &prosody, warnings
}

func unsupportedSettings(options model.SpeechOptions) []model.Warning {
	var warnings []model.Warning

	if options.Language != "" {
		warnings = append(warnings, model.UnsupportedWarning(
			"language",
			"Fish Audio infers the language from the input text and the selected voice, and has no language parameter. The language option was ignored.",
		))
	}

	if options.Instructions != "" {
		warnings = append(warnings, model.UnsupportedWarning(
			"instructions",
			"Fish Audio does not support instructions. The instructions option was ignored.",
		))
	}

	return warnings
}

func bitrateWarning(option string, appliesTo string, format string) model.Warning {
	return model.UnsupportedWarning(
		"providerOptions.fishAudio."+option,
		fmt.Sprintf("%s only applies to %s output. The option was ignored for %s output.", option, appliesTo, format),
	)
}

var _ model.SpeechModel = (*SpeechModel)(nil)
