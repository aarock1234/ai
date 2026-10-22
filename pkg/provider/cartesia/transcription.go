package cartesia

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// transcriptionPath is the batch speech-to-text endpoint.
	transcriptionPath = "/stt"

	// batchModelFamily is the model family that supports batch
	// transcription, including dated snapshots.
	batchModelFamily = "ink-whisper"
)

// TranscriptionModel is a Cartesia transcription model.
type TranscriptionModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "cartesia.transcription" by
// default.
func (m *TranscriptionModel) Provider() string {
	return m.provider.name + ".transcription"
}

// ModelID returns the model identifier.
func (m *TranscriptionModel) ModelID() string {
	return m.modelID
}

type transcriptionResponse struct {
	Text     string              `json:"text"`
	Language *string             `json:"language"`
	Duration *float64            `json:"duration"`
	Words    []transcriptionWord `json:"words"`
}

type transcriptionWord struct {
	Word  string  `json:"word"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}

// Transcribe converts audio into text. Models outside the Ink Whisper family
// only support streaming and return an error wrapping model.ErrUnsupported.
func (m *TranscriptionModel) Transcribe(ctx context.Context, options model.TranscriptionOptions) (*model.TranscriptionResult, error) {
	if !isBatchModel(m.modelID) {
		return nil, fmt.Errorf("%w: non-streaming transcription with %s", model.ErrUnsupported, m.modelID)
	}

	timestamp := time.Now()

	transcriptionOptions, err := readTranscriptionOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	var warnings []model.Warning
	if transcriptionOptions.Streaming != nil {
		warnings = append(warnings, model.UnsupportedWarning(
			"providerOptions.cartesia.streaming",
			"Cartesia batch transcription does not support streaming options.",
		))
	}

	request, err := m.provider.apiRequest(transcriptionPath, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostForm(ctx, request, m.formFields(options, transcriptionOptions))
	if err != nil {
		return nil, err
	}

	var decoded transcriptionResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	result := &model.TranscriptionResult{
		Text:     decoded.Text,
		Segments: transcriptSegments(decoded.Words),
		Warnings: warnings,
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: timestamp,
				ModelID:   m.modelID,
			},
			Headers: response.Headers,
			Body:    response.Body,
		},
	}

	if decoded.Language != nil {
		result.Language = *decoded.Language
	}

	if decoded.Duration != nil {
		result.Duration = seconds(*decoded.Duration)
	}

	return result, nil
}

func (m *TranscriptionModel) formFields(options model.TranscriptionOptions, transcriptionOptions TranscriptionOptions) []apicall.FormField {
	fields := []apicall.FormField{
		formValue("model", m.modelID),
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

	for _, granularity := range transcriptionOptions.TimestampGranularities {
		fields = append(fields, formValue("timestamp_granularities[]", string(granularity)))
	}

	return fields
}

// isBatchModel reports whether modelID belongs to the Ink Whisper family.
// Other models use the streaming protocol regardless of their name.
func isBatchModel(modelID string) bool {
	return modelID == batchModelFamily || strings.HasPrefix(modelID, batchModelFamily+"-")
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

func transcriptSegments(words []transcriptionWord) []model.TranscriptionSegment {
	segments := make([]model.TranscriptionSegment, 0, len(words))
	for _, word := range words {
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
