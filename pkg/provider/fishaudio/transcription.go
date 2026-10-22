package fishaudio

import (
	"context"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// transcriptionPath is the speech-to-text endpoint.
	transcriptionPath = "/v1/asr"

	// defaultTranscriptionModelID labels the single ASR model. It is not
	// sent to the API.
	defaultTranscriptionModelID = "transcribe-1"
)

// TranscriptionModel is the Fish Audio speech-to-text model.
type TranscriptionModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "fish-audio.transcription" by
// default.
func (m *TranscriptionModel) Provider() string {
	return m.provider.name + ".transcription"
}

// ModelID returns the model identifier.
func (m *TranscriptionModel) ModelID() string {
	return m.modelID
}

// TranscriptionMetadata is the provider metadata of a transcription.
type TranscriptionMetadata struct {
	// Language is the human-readable name of the detected language, such as
	// "English". Its form is not guaranteed; branch on the result language
	// code instead.
	Language string `json:"language"`
}

type transcriptionResponse struct {
	Text         string                 `json:"text"`
	Language     *string                `json:"language"`
	LanguageCode *string                `json:"language_code"`
	Duration     *float64               `json:"duration"`
	Segments     []transcriptionSegment `json:"segments"`
}

type transcriptionSegment struct {
	Text  string  `json:"text"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}

// Transcribe converts audio into text. Timestamps are requested unless the
// ignoreTimestamps option is set.
func (m *TranscriptionModel) Transcribe(ctx context.Context, options model.TranscriptionOptions) (*model.TranscriptionResult, error) {
	timestamp := time.Now()

	fishOptions, _, err := model.ProviderValue[TranscriptionOptions](options.ProviderOptions, Namespace)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(transcriptionPath, nil, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostForm(ctx, request, formFields(options, fishOptions))
	if err != nil {
		return nil, err
	}

	var decoded transcriptionResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	result := &model.TranscriptionResult{
		Text:     decoded.Text,
		Segments: transcriptSegments(decoded.Segments),
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: timestamp,
				ModelID:   m.modelID,
			},
			Headers: response.Headers,
			Body:    response.Body,
		},
	}

	if decoded.LanguageCode != nil {
		result.Language = *decoded.LanguageCode
	}

	if decoded.Duration != nil {
		result.Duration = seconds(*decoded.Duration)
	}

	if decoded.Language != nil {
		result.ProviderMetadata = model.ProviderMetadata{
			Namespace: TranscriptionMetadata{
				Language: *decoded.Language,
			},
		}
	}

	return result, nil
}

func formFields(options model.TranscriptionOptions, fishOptions TranscriptionOptions) []apicall.FormField {
	fields := []apicall.FormField{
		{
			Name:      "audio",
			Filename:  "audio." + audioExtension(options.MediaType),
			MediaType: options.MediaType,
			Data:      options.Audio,
		},
	}

	if fishOptions.Language != nil {
		fields = append(fields, apicall.FormField{
			Name:  "language",
			Value: *fishOptions.Language,
		})
	}

	ignoreTimestamps := false
	if fishOptions.IgnoreTimestamps != nil {
		ignoreTimestamps = *fishOptions.IgnoreTimestamps
	}

	return append(fields, apicall.FormField{
		Name:  "ignore_timestamps",
		Value: strconv.FormatBool(ignoreTimestamps),
	})
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

func transcriptSegments(segments []transcriptionSegment) []model.TranscriptionSegment {
	converted := make([]model.TranscriptionSegment, 0, len(segments))
	for _, segment := range segments {
		converted = append(converted, model.TranscriptionSegment{
			Text:  segment.Text,
			Start: seconds(segment.Start),
			End:   seconds(segment.End),
		})
	}

	return converted
}

func seconds(value float64) time.Duration {
	return time.Duration(value * float64(time.Second))
}

var _ model.TranscriptionModel = (*TranscriptionModel)(nil)
