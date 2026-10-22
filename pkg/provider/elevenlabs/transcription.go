package elevenlabs

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// transcriptionPath is the speech-to-text endpoint.
	transcriptionPath = "/v1/speech-to-text"

	// realtimeModelID is the model that only supports streaming
	// transcription.
	realtimeModelID = "scribe_v2_realtime"
)

// TranscriptionModel is an ElevenLabs speech-to-text model.
type TranscriptionModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "elevenlabs.transcription" by
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
	Words []Word `json:"words"`
}

// WordType classifies a transcribed word.
type WordType string

// Word types.
const (
	WordTypeWord       WordType = "word"
	WordTypeSpacing    WordType = "spacing"
	WordTypeAudioEvent WordType = "audio_event"
)

// Word is one word, space or audio event of a transcript. Times are in
// seconds.
type Word struct {
	Text       string      `json:"text"`
	Type       WordType    `json:"type"`
	Start      *float64    `json:"start,omitempty"`
	End        *float64    `json:"end,omitempty"`
	SpeakerID  *string     `json:"speaker_id,omitempty"`
	Characters []Character `json:"characters,omitempty"`
}

// Character is one character of a word, with times in seconds.
type Character struct {
	Text  string   `json:"text"`
	Start *float64 `json:"start,omitempty"`
	End   *float64 `json:"end,omitempty"`
}

type transcriptionResponse struct {
	LanguageCode        string  `json:"language_code"`
	LanguageProbability float64 `json:"language_probability"`
	Text                string  `json:"text"`
	Words               []Word  `json:"words"`
}

// Transcribe converts audio into text. The realtime model only supports
// streaming and returns an error wrapping model.ErrUnsupported.
func (m *TranscriptionModel) Transcribe(ctx context.Context, options model.TranscriptionOptions) (*model.TranscriptionResult, error) {
	if m.modelID == realtimeModelID {
		return nil, fmt.Errorf("%w: non-streaming transcription with %s", model.ErrUnsupported, m.modelID)
	}

	timestamp := time.Now()

	transcriptionOptions, present, err := readTranscriptionOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	var warnings []model.Warning
	if transcriptionOptions.Streaming != nil {
		warnings = append(warnings, model.UnsupportedWarning(
			"providerOptions.elevenlabs.streaming",
			"ElevenLabs batch transcription does not support streaming options.",
		))
	}

	request, err := m.provider.apiRequest(transcriptionPath, options.Headers)
	if err != nil {
		return nil, err
	}

	fields := m.formFields(options, transcriptionOptions, present)

	response, err := apicall.PostForm(ctx, request, fields)
	if err != nil {
		return nil, err
	}

	var decoded transcriptionResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	if err := validateWords(decoded.Words); err != nil {
		return nil, err
	}

	result := &model.TranscriptionResult{
		Text:     decoded.Text,
		Segments: transcriptSegments(decoded.Words),
		Language: decoded.LanguageCode,
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

	if len(decoded.Words) > 0 {
		if end := decoded.Words[len(decoded.Words)-1].End; end != nil {
			result.Duration = seconds(*end)
		}
	}

	if decoded.Words != nil {
		result.ProviderMetadata = model.ProviderMetadata{
			Namespace: TranscriptionMetadata{
				Words: decoded.Words,
			},
		}
	}

	return result, nil
}

// formFields returns the multipart fields of a transcription request. When
// the provider options are present, their defaults are sent as well.
func (m *TranscriptionModel) formFields(options model.TranscriptionOptions, transcriptionOptions TranscriptionOptions, present bool) []apicall.FormField {
	diarize := true
	if transcriptionOptions.Diarize != nil {
		diarize = *transcriptionOptions.Diarize
	}

	fields := []apicall.FormField{
		formValue("model_id", m.modelID),
		{
			Name:      "file",
			Filename:  "audio." + audioExtension(options.MediaType),
			MediaType: options.MediaType,
			Data:      options.Audio,
		},
		formValue("diarize", strconv.FormatBool(diarize)),
	}

	if !present {
		return fields
	}

	if transcriptionOptions.LanguageCode != nil {
		fields = append(fields, formValue("language_code", *transcriptionOptions.LanguageCode))
	}

	tagAudioEvents := true
	if transcriptionOptions.TagAudioEvents != nil {
		tagAudioEvents = *transcriptionOptions.TagAudioEvents
	}

	fields = append(fields, formValue("tag_audio_events", strconv.FormatBool(tagAudioEvents)))

	if transcriptionOptions.NumSpeakers != nil {
		fields = append(fields, formValue("num_speakers", strconv.Itoa(*transcriptionOptions.NumSpeakers)))
	}

	granularity := transcriptionOptions.TimestampsGranularity
	if granularity == "" {
		granularity = TimestampsWord
	}

	fileFormat := transcriptionOptions.FileFormat
	if fileFormat == "" {
		fileFormat = FileOther
	}

	return append(
		fields,
		formValue("timestamps_granularity", string(granularity)),
		formValue("file_format", string(fileFormat)),
	)
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

func validateWords(words []Word) error {
	for _, word := range words {
		switch word.Type {
		case WordTypeWord, WordTypeSpacing, WordTypeAudioEvent:
		default:
			return fmt.Errorf("%w: elevenlabs word type %q", model.ErrInvalidResponse, word.Type)
		}
	}

	return nil
}

// transcriptSegments returns one segment per word, with missing times as
// zero.
func transcriptSegments(words []Word) []model.TranscriptionSegment {
	segments := make([]model.TranscriptionSegment, 0, len(words))
	for _, word := range words {
		segment := model.TranscriptionSegment{
			Text: word.Text,
		}

		if word.Start != nil {
			segment.Start = seconds(*word.Start)
		}

		if word.End != nil {
			segment.End = seconds(*word.End)
		}

		segments = append(segments, segment)
	}

	return segments
}

func seconds(value float64) time.Duration {
	return time.Duration(value * float64(time.Second))
}

var _ model.TranscriptionModel = (*TranscriptionModel)(nil)
