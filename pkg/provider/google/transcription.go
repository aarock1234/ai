package google

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/optional"
	"github.com/aarock1234/ai/internal/wirejson"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

var _ model.TranscriptionModel = (*TranscriptionModel)(nil)

// TranscriptionOptions are the Google provider options of a
// transcription call, read from ProviderOptions["google"].
type TranscriptionOptions struct {
	// LanguageCodes are BCP-47 hints for the languages in the audio. Without
	// them, the language is detected automatically.
	LanguageCodes []string `json:"languageCodes,omitempty"`

	// CustomVocabulary biases recognition toward specific terms.
	CustomVocabulary []string `json:"customVocabulary,omitempty"`

	// WordTimestamp enables word-level timestamps, reported as segments.
	WordTimestamp *bool `json:"wordTimestamp,omitempty"`

	// Diarization enables speaker diarization.
	Diarization *bool `json:"diarization,omitempty"`

	Mode TranscriptionMode `json:"mode,omitempty"`
}

// TranscriptionMode formats the transcript.
type TranscriptionMode string

// Transcription modes.
const (
	// TranscriptionVerbatim keeps filler words, repetitions and false
	// starts. It is the default.
	TranscriptionVerbatim TranscriptionMode = "VERBATIM"

	// TranscriptionSmart removes disfluencies and formats lists, numbers,
	// dates and paragraphs.
	TranscriptionSmart TranscriptionMode = "SMART"
)

var transcriptionModes = []TranscriptionMode{
	TranscriptionVerbatim,
	TranscriptionSmart,
}

// TranscriptionMetadata is the Google metadata of a transcription, stored
// under ProviderMetadata["google"].
type TranscriptionMetadata struct {
	Usage json.RawMessage `json:"usage"`
}

// TranscriptionModel transcribes audio files with unary Gemini transcription
// models, such as gemini-3.5-transcribe, through the Interactions API, and
// live audio with live models, such as gemini-3.5-transcribe-live, through
// the Live API. It is safe for concurrent use.
type TranscriptionModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider name, "google.generative-ai.transcription"
// by default.
func (m *TranscriptionModel) Provider() string {
	return m.provider.name + ".transcription"
}

// ModelID returns the model ID.
func (m *TranscriptionModel) ModelID() string {
	return m.modelID
}

type transcriptionRequest struct {
	Model            string                   `json:"model"`
	Input            []transcriptionInput     `json:"input"`
	GenerationConfig *transcriptionGeneration `json:"generation_config,omitempty"`
}

type transcriptionInput struct {
	Type     string `json:"type"`
	Data     string `json:"data"`
	MimeType string `json:"mime_type"`
}

type transcriptionGeneration struct {
	TranscriptionConfig transcriptionConfig `json:"transcription_config"`
}

type transcriptionConfig struct {
	LanguageCodes    []string                 `json:"language_codes,omitempty"`
	CustomVocabulary []string                 `json:"custom_vocabulary,omitempty"`
	Mode             *transcriptionModeConfig `json:"mode,omitempty"`
}

type transcriptionModeConfig struct {
	Type                   string   `json:"type"`
	DiarizationMode        string   `json:"diarization_mode,omitempty"`
	TimestampGranularities []string `json:"timestamp_granularities,omitempty"`
}

type transcriptionResponse struct {
	Steps []transcriptionStep `json:"steps"`
	Usage json.RawMessage     `json:"usage"`
}

type transcriptionStep struct {
	Content []transcriptionContent `json:"content"`
}

type transcriptionContent struct {
	Type        string                    `json:"type"`
	Text        *string                   `json:"text"`
	Annotations []transcriptionAnnotation `json:"annotations"`
}

type transcriptionAnnotation struct {
	Type        string  `json:"type"`
	Text        *string `json:"text"`
	StartOffset *string `json:"start_offset"`
	EndOffset   *string `json:"end_offset"`
}

// Transcribe transcribes the audio. Word timestamps become segments. Live
// models, whose IDs contain "-live", only support StreamTranscribe.
func (m *TranscriptionModel) Transcribe(ctx context.Context, options model.TranscriptionOptions) (*model.TranscriptionResult, error) {
	if strings.Contains(m.modelID, "-live") {
		return nil, fmt.Errorf("%w: model %q only supports streaming transcription; use a unary model such as gemini-3.5-transcribe", model.ErrInvalidArgument, m.modelID)
	}

	googleOptions, _, err := model.ProviderValue[TranscriptionOptions](options.ProviderOptions, Namespace)
	if err != nil {
		return nil, err
	}

	if err := validateEnum("mode", googleOptions.Mode, transcriptionModes); err != nil {
		return nil, err
	}

	body := transcriptionRequest{
		Model: m.modelID,
		Input: []transcriptionInput{
			{
				Type:     "audio",
				Data:     base64.StdEncoding.EncodeToString(options.Audio),
				MimeType: options.MediaType,
			},
		},
		GenerationConfig: transcriptionGenerationConfig(googleOptions),
	}

	request, err := m.provider.request(ctx, "interactions", options.Headers)
	if err != nil {
		return nil, err
	}

	raw, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, err
	}

	var response transcriptionResponse
	if err := apicall.DecodeJSON(raw.Body, &response); err != nil {
		return nil, err
	}

	text, segments := transcriptText(response)

	result := &model.TranscriptionResult{
		Text:     text,
		Segments: segments,
		Request: model.RequestInfo{
			Body: raw.RequestBody,
		},
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: time.Now(),
				ModelID:   m.modelID,
			},
			Headers: raw.Headers,
			Body:    raw.Body,
		},
	}

	if wirejson.HasValue(response.Usage) {
		result.Usage = response.Usage
		result.ProviderMetadata = model.ProviderMetadata{
			Namespace: TranscriptionMetadata{
				Usage: response.Usage,
			},
		}
	}

	return result, nil
}

// transcriptionGenerationConfig expresses diarization and word timestamps
// inside the mode, as the Interactions API expects.
func transcriptionGenerationConfig(options TranscriptionOptions) *transcriptionGeneration {
	config := transcriptionConfig{
		LanguageCodes:    options.LanguageCodes,
		CustomVocabulary: options.CustomVocabulary,
	}

	diarization := optional.Or(options.Diarization, false)
	wordTimestamps := optional.Or(options.WordTimestamp, false)

	if options.Mode != "" || diarization || wordTimestamps {
		mode := options.Mode
		if mode == "" {
			mode = TranscriptionVerbatim
		}

		config.Mode = &transcriptionModeConfig{
			Type: strings.ToLower(string(mode)),
		}

		if diarization {
			config.Mode.DiarizationMode = "speaker"
		}

		if wordTimestamps {
			config.Mode.TimestampGranularities = []string{
				"word",
			}
		}
	}

	isEmpty := config.LanguageCodes == nil && config.CustomVocabulary == nil && config.Mode == nil
	if isEmpty {
		return nil
	}

	return &transcriptionGeneration{
		TranscriptionConfig: config,
	}
}

// transcriptText joins the text contents and collects word annotations with
// valid offsets as segments.
func transcriptText(response transcriptionResponse) (string, []model.TranscriptionSegment) {
	var text strings.Builder
	var segments []model.TranscriptionSegment

	for _, step := range response.Steps {
		for _, item := range step.Content {
			if item.Type != "text" || item.Text == nil {
				continue
			}

			text.WriteString(*item.Text)

			for _, annotation := range item.Annotations {
				segment, ok := wordSegment(annotation)
				if ok {
					segments = append(segments, segment)
				}
			}
		}
	}

	return text.String(), segments
}

func wordSegment(annotation transcriptionAnnotation) (model.TranscriptionSegment, bool) {
	if annotation.Type != "word_info" || annotation.Text == nil {
		return model.TranscriptionSegment{}, false
	}

	start, startOK := parseOffset(annotation.StartOffset)
	end, endOK := parseOffset(annotation.EndOffset)
	if !startOK || !endOK {
		return model.TranscriptionSegment{}, false
	}

	return model.TranscriptionSegment{
		Text:  *annotation.Text,
		Start: start,
		End:   end,
	}, true
}

// parseOffset parses a duration offset such as "1s" or "9.400s".
func parseOffset(offset *string) (time.Duration, bool) {
	if offset == nil {
		return 0, false
	}

	seconds, err := strconv.ParseFloat(strings.TrimSuffix(*offset, "s"), 64)
	if err != nil {
		return 0, false
	}

	return time.Duration(seconds * float64(time.Second)), true
}
