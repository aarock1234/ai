package groq

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// transcriptionPath is the transcription endpoint.
const transcriptionPath = "/audio/transcriptions"

// textResponseFormat returns the transcript as plain text.
const textResponseFormat = "text"

// TranscriptionModel is a Groq Whisper transcription model.
type TranscriptionModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "groq.transcription" by default.
func (m *TranscriptionModel) Provider() string {
	return m.provider.name + ".transcription"
}

// ModelID returns the model identifier.
func (m *TranscriptionModel) ModelID() string {
	return m.modelID
}

// transcriptionResponse is a JSON transcription. Language, duration,
// segments and words are reported with the "verbose_json" format.
type transcriptionResponse struct {
	Text     string                 `json:"text"`
	XGroq    *transcriptionXGroq    `json:"x_groq"`
	Task     *string                `json:"task"`
	Language *string                `json:"language"`
	Duration *float64               `json:"duration"`
	Segments []transcriptionSegment `json:"segments"`
	Words    []transcriptionWord    `json:"words"`
}

type transcriptionXGroq struct {
	ID string `json:"id"`
}

type transcriptionSegment struct {
	ID               int     `json:"id"`
	Seek             int     `json:"seek"`
	Start            float64 `json:"start"`
	End              float64 `json:"end"`
	Text             string  `json:"text"`
	Tokens           []int   `json:"tokens"`
	Temperature      float64 `json:"temperature"`
	AvgLogprob       float64 `json:"avg_logprob"`
	CompressionRatio float64 `json:"compression_ratio"`
	NoSpeechProb     float64 `json:"no_speech_prob"`
}

type transcriptionWord struct {
	Word  string  `json:"word"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}

// Transcribe converts audio into text.
func (m *TranscriptionModel) Transcribe(ctx context.Context, options model.TranscriptionOptions) (*model.TranscriptionResult, error) {
	timestamp := time.Now()

	transcriptionOptions, err := readTranscriptionOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(transcriptionPath, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostForm(ctx, request, m.formFields(options, transcriptionOptions))
	if err != nil {
		return nil, err
	}

	result := &model.TranscriptionResult{
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: timestamp,
				ModelID:   m.modelID,
			},
			Headers: response.Headers,
			Body:    response.Body,
		},
	}

	if valueOrEmpty(transcriptionOptions.ResponseFormat) == textResponseFormat {
		return textTranscription(result, response.Body)
	}

	var decoded transcriptionResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	if decoded.XGroq == nil {
		return nil, fmt.Errorf("%w: transcription response is missing x_groq", model.ErrInvalidResponse)
	}

	result.Text = decoded.Text
	result.Segments = transcriptSegments(decoded)
	result.Language = valueOrEmpty(decoded.Language)

	if decoded.Duration != nil {
		result.Duration = seconds(*decoded.Duration)
	}

	return result, nil
}

// textTranscription completes a result from a plain-text response. The
// response body is recorded as a JSON string.
func textTranscription(result *model.TranscriptionResult, body []byte) (*model.TranscriptionResult, error) {
	text := string(body)

	encoded, err := json.Marshal(text)
	if err != nil {
		return nil, fmt.Errorf("encode transcription text: %w", err)
	}

	result.Text = text
	result.Segments = []model.TranscriptionSegment{}
	result.Response.Body = encoded

	return result, nil
}

// formFields returns the multipart fields of a transcription request. The
// timestamp granularities repeat their field with brackets.
func (m *TranscriptionModel) formFields(options model.TranscriptionOptions, transcriptionOptions TranscriptionOptions) []apicall.FormField {
	fields := []apicall.FormField{
		{
			Name:  "model",
			Value: m.modelID,
		},
		{
			Name:      "file",
			Filename:  "audio." + audioExtension(options.MediaType),
			MediaType: options.MediaType,
			Data:      options.Audio,
		},
	}

	if transcriptionOptions.Language != nil {
		fields = append(fields, formValue("language", *transcriptionOptions.Language))
	}

	if transcriptionOptions.Prompt != nil {
		fields = append(fields, formValue("prompt", *transcriptionOptions.Prompt))
	}

	if transcriptionOptions.ResponseFormat != nil {
		fields = append(fields, formValue("response_format", *transcriptionOptions.ResponseFormat))
	}

	if transcriptionOptions.Temperature != nil {
		fields = append(fields, formValue("temperature", strconv.FormatFloat(*transcriptionOptions.Temperature, 'f', -1, 64)))
	}

	for _, granularity := range transcriptionOptions.TimestampGranularities {
		fields = append(fields, formValue("timestamp_granularities[]", granularity))
	}

	return fields
}

func formValue(name string, value string) apicall.FormField {
	return apicall.FormField{
		Name:  name,
		Value: value,
	}
}

// audioExtension returns the file extension of an audio media type.
func audioExtension(mediaType string) string {
	_, subtype, _ := strings.Cut(strings.ToLower(mediaType), "/")

	switch subtype {
	case "mpeg":
		return "mp3"
	case "x-wav":
		return "wav"
	case "opus":
		return "ogg"
	case "mp4", "x-m4a":
		return "m4a"
	default:
		return subtype
	}
}

// transcriptSegments prefers segments and falls back to word timestamps.
func transcriptSegments(response transcriptionResponse) []model.TranscriptionSegment {
	if response.Segments != nil {
		segments := make([]model.TranscriptionSegment, 0, len(response.Segments))
		for _, segment := range response.Segments {
			segments = append(segments, model.TranscriptionSegment{
				Text:  segment.Text,
				Start: seconds(segment.Start),
				End:   seconds(segment.End),
			})
		}

		return segments
	}

	segments := make([]model.TranscriptionSegment, 0, len(response.Words))
	for _, word := range response.Words {
		segments = append(segments, model.TranscriptionSegment{
			Text:  word.Word,
			Start: seconds(word.Start),
			End:   seconds(word.End),
		})
	}

	return segments
}

func seconds(value float64) time.Duration {
	return time.Duration(value * float64(time.Second))
}

var _ model.TranscriptionModel = (*TranscriptionModel)(nil)
