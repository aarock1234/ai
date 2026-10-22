package googlevertex

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/google"
)

// GeminiTranscriptionMode selects the transcript formatting of Gemini
// transcription models.
type GeminiTranscriptionMode string

// Gemini transcription modes.
const (
	// GeminiTranscriptionVerbatim keeps filler words, repetitions and false
	// starts. It is the default.
	GeminiTranscriptionVerbatim GeminiTranscriptionMode = "VERBATIM"

	// GeminiTranscriptionSmart removes disfluencies, applies
	// self-corrections, and formats lists, numbers, dates and paragraphs.
	GeminiTranscriptionSmart GeminiTranscriptionMode = "SMART"
)

// GeminiTranscriptionOptions are the options of a Gemini transcription call,
// read from ProviderOptions["googleVertex"], ["vertex"] or ["google"]. They
// map onto Google's AudioTranscriptionConfig.
type GeminiTranscriptionOptions struct {
	// LanguageCodes are BCP-47 hints for the languages in the audio. Empty
	// detects the language automatically.
	LanguageCodes []string `json:"languageCodes,omitempty"`

	// CustomVocabulary biases recognition toward these phrases.
	CustomVocabulary []string `json:"customVocabulary,omitempty"`

	// WordTimestamp enables word timestamps, which become the result
	// segments.
	WordTimestamp *bool `json:"wordTimestamp,omitempty"`

	// Diarization enables speaker diarization.
	Diarization *bool `json:"diarization,omitempty"`

	// Mode selects the transcript formatting.
	Mode GeminiTranscriptionMode `json:"mode,omitempty"`
}

func (o GeminiTranscriptionOptions) validate() error {
	switch o.Mode {
	case "", GeminiTranscriptionVerbatim, GeminiTranscriptionSmart:
		return nil
	default:
		return fmt.Errorf("%w: unsupported Gemini transcription mode %q", model.ErrInvalidArgument, o.Mode)
	}
}

// isEmpty reports whether no option is set.
func (o GeminiTranscriptionOptions) isEmpty() bool {
	return o.LanguageCodes == nil &&
		o.CustomVocabulary == nil &&
		o.WordTimestamp == nil &&
		o.Diarization == nil &&
		o.Mode == ""
}

// GeminiTranscriptionMetadata is the metadata of a Gemini transcription,
// stored under ProviderMetadata["google"].
type GeminiTranscriptionMetadata struct {
	UsageMetadata json.RawMessage `json:"usageMetadata"`
}

// geminiTranscriptionRequest is the body of POST models/{model}:generateContent.
type geminiTranscriptionRequest struct {
	Contents         []geminiContent         `json:"contents"`
	GenerationConfig *geminiGenerationConfig `json:"generationConfig,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	InlineData geminiInlineData `json:"inlineData"`
}

type geminiInlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

type geminiGenerationConfig struct {
	AudioTranscriptionConfig GeminiTranscriptionOptions `json:"audioTranscriptionConfig"`
}

type geminiTranscriptionResponse struct {
	Candidates []struct {
		Content *struct {
			Parts []geminiResponsePart `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
	UsageMetadata json.RawMessage `json:"usageMetadata"`
}

type geminiResponsePart struct {
	Text               *string                   `json:"text"`
	AudioTranscription *geminiAudioTranscription `json:"audioTranscription"`
}

type geminiAudioTranscription struct {
	Text         *string `json:"text"`
	LanguageCode *string `json:"languageCode"`
	SpeakerLabel *string `json:"speakerLabel"`
	Words        []struct {
		Word        *string `json:"word"`
		StartOffset *string `json:"startOffset"`
		EndOffset   *string `json:"endOffset"`
	} `json:"words"`
}

// transcribeWithGemini transcribes with a unary Gemini transcription model.
// Live models, such as gemini-3.5-transcribe-live, only stream.
func (m *TranscriptionModel) transcribeWithGemini(ctx context.Context, options model.TranscriptionOptions) (*model.TranscriptionResult, error) {
	timestamp := time.Now()

	if strings.Contains(m.modelID, "-live") {
		return nil, fmt.Errorf(
			"%w: model %q only supports streaming transcription; use a unary model such as gemini-3.5-transcribe",
			model.ErrInvalidArgument,
			m.modelID,
		)
	}

	geminiOptions, _, err := readOptions[GeminiTranscriptionOptions](options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	if err := geminiOptions.validate(); err != nil {
		return nil, err
	}

	baseURL, err := m.settings.resolveBaseURL(false)
	if err != nil {
		return nil, err
	}

	request, err := m.settings.request(ctx, baseURL+"/models/"+m.modelID+":generateContent", options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, geminiTranscriptionBody(options, geminiOptions))
	if err != nil {
		return nil, err
	}

	var decoded geminiTranscriptionResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	result := geminiTranscriptionResult(decoded)
	result.Response = model.ResponseInfo{
		Timestamp: timestamp,
		ModelID:   m.modelID,
		Headers:   response.Headers,
		Body:      response.Body,
	}

	return result, nil
}

func geminiTranscriptionBody(options model.TranscriptionOptions, geminiOptions GeminiTranscriptionOptions) geminiTranscriptionRequest {
	body := geminiTranscriptionRequest{
		Contents: []geminiContent{
			{
				Role: "user",
				Parts: []geminiPart{
					{
						InlineData: geminiInlineData{
							MimeType: options.MediaType,
							Data:     base64.StdEncoding.EncodeToString(options.Audio),
						},
					},
				},
			},
		},
	}

	if !geminiOptions.isEmpty() {
		body.GenerationConfig = &geminiGenerationConfig{
			AudioTranscriptionConfig: geminiOptions,
		}
	}

	return body
}

// geminiTranscriptionResult prefers plain text parts and falls back to the
// audio transcription text. Timed words become segments.
func geminiTranscriptionResult(response geminiTranscriptionResponse) *model.TranscriptionResult {
	var parts []geminiResponsePart
	if len(response.Candidates) > 0 && response.Candidates[0].Content != nil {
		parts = response.Candidates[0].Content.Parts
	}

	var plainText strings.Builder
	var transcriptionText strings.Builder
	result := &model.TranscriptionResult{}

	for _, responsePart := range parts {
		plainText.WriteString(stringValue(responsePart.Text))

		transcription := responsePart.AudioTranscription
		if transcription == nil {
			continue
		}

		transcriptionText.WriteString(stringValue(transcription.Text))

		if result.Language == "" {
			result.Language = stringValue(transcription.LanguageCode)
		}

		result.Segments = append(result.Segments, transcriptionSegments(*transcription)...)
	}

	result.Text = plainText.String()
	if result.Text == "" {
		result.Text = transcriptionText.String()
	}

	if isJSONObject(response.UsageMetadata) {
		result.Usage = slices.Clone(response.UsageMetadata)
		result.ProviderMetadata = model.ProviderMetadata{
			google.Namespace: GeminiTranscriptionMetadata{
				UsageMetadata: result.Usage,
			},
		}
	}

	return result
}

func transcriptionSegments(transcription geminiAudioTranscription) []model.TranscriptionSegment {
	var segments []model.TranscriptionSegment

	for _, word := range transcription.Words {
		start, hasStart := parseDurationSeconds(word.StartOffset)
		end, hasEnd := parseDurationSeconds(word.EndOffset)

		if word.Word == nil || !hasStart || !hasEnd {
			continue
		}

		segments = append(segments, model.TranscriptionSegment{
			Text:  *word.Word,
			Start: start,
			End:   end,
		})
	}

	return segments
}

func isJSONObject(value json.RawMessage) bool {
	trimmed := bytes.TrimSpace(value)

	return len(trimmed) > 0 && trimmed[0] == '{'
}
