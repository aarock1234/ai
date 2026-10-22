package mistral

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

// TranscriptionModel is a Mistral audio transcription model.
type TranscriptionModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "mistral.transcription" by default.
func (m *TranscriptionModel) Provider() string {
	return m.provider.name + ".transcription"
}

// ModelID returns the model identifier.
func (m *TranscriptionModel) ModelID() string {
	return m.modelID
}

// TranscriptionMetadata is the provider metadata of a transcription.
type TranscriptionMetadata struct {
	Usage *TranscriptionUsage `json:"usage,omitempty"`

	// Segments are the segments that carry a type, score or speaker.
	Segments []SegmentMetadata `json:"segments,omitempty"`
}

// TranscriptionUsage is the usage of a transcription.
type TranscriptionUsage struct {
	PromptTokens       *int     `json:"promptTokens,omitempty"`
	CompletionTokens   *int     `json:"completionTokens,omitempty"`
	TotalTokens        *int     `json:"totalTokens,omitempty"`
	PromptAudioSeconds *float64 `json:"promptAudioSeconds,omitempty"`
	RequestCount       *int     `json:"requestCount,omitempty"`
}

// SegmentMetadata describes one transcript segment.
type SegmentMetadata struct {
	Text        string   `json:"text"`
	StartSecond float64  `json:"startSecond"`
	EndSecond   float64  `json:"endSecond"`
	Type        string   `json:"type,omitempty"`
	Score       *float64 `json:"score,omitempty"`
	SpeakerID   string   `json:"speakerId,omitempty"`
}

type transcriptionResponse struct {
	Model    string                 `json:"model"`
	Text     string                 `json:"text"`
	Language *string                `json:"language"`
	Segments []transcriptionSegment `json:"segments"`
	Usage    json.RawMessage        `json:"usage"`
}

type transcriptionSegment struct {
	Type      *string  `json:"type"`
	Text      string   `json:"text"`
	Start     float64  `json:"start"`
	End       float64  `json:"end"`
	Score     *float64 `json:"score"`
	SpeakerID *string  `json:"speaker_id"`
}

type transcriptionUsage struct {
	PromptTokens       *int     `json:"prompt_tokens"`
	CompletionTokens   *int     `json:"completion_tokens"`
	TotalTokens        *int     `json:"total_tokens"`
	PromptAudioSeconds *float64 `json:"prompt_audio_seconds"`
	RequestCount       *int     `json:"request_count"`
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

	var decoded transcriptionResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	var usage *transcriptionUsage
	if len(decoded.Usage) > 0 && !isJSONNull(decoded.Usage) {
		usage = &transcriptionUsage{}
		if err := json.Unmarshal(decoded.Usage, usage); err != nil {
			return nil, fmt.Errorf("%w: transcription usage: %w", model.ErrInvalidResponse, err)
		}
	}

	result := &model.TranscriptionResult{
		Text:             decoded.Text,
		Segments:         transcriptSegments(decoded.Segments),
		Duration:         transcriptDuration(usage, decoded.Segments),
		ProviderMetadata: transcriptionMetadata(usage, decoded.Segments),
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: timestamp,
				ModelID:   decoded.Model,
			},
			Headers: response.Headers,
			Body:    response.Body,
		},
	}

	if decoded.Language != nil {
		result.Language = *decoded.Language
	}

	if usage != nil {
		result.Usage = decoded.Usage
	}

	return result, nil
}

// formFields returns the multipart fields of a transcription request. List
// options repeat their field.
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

	if transcriptionOptions.Temperature != nil {
		fields = append(fields, formValue("temperature", strconv.FormatFloat(*transcriptionOptions.Temperature, 'f', -1, 64)))
	}

	for _, granularity := range transcriptionOptions.TimestampGranularities {
		fields = append(fields, formValue("timestamp_granularities", string(granularity)))
	}

	if transcriptionOptions.Diarize != nil {
		fields = append(fields, formValue("diarize", strconv.FormatBool(*transcriptionOptions.Diarize)))
	}

	for _, term := range transcriptionOptions.ContextBias {
		fields = append(fields, formValue("context_bias", term))
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

// transcriptDuration prefers the billed audio duration and falls back to the
// end of the last segment.
func transcriptDuration(usage *transcriptionUsage, segments []transcriptionSegment) time.Duration {
	if usage != nil && usage.PromptAudioSeconds != nil {
		return seconds(*usage.PromptAudioSeconds)
	}

	if len(segments) == 0 {
		return 0
	}

	return seconds(segments[len(segments)-1].End)
}

func seconds(value float64) time.Duration {
	return time.Duration(value * float64(time.Second))
}

// transcriptionMetadata reports usage and the segments with a type, score or
// speaker. It returns nil when there is neither.
func transcriptionMetadata(usage *transcriptionUsage, segments []transcriptionSegment) model.ProviderMetadata {
	var metadata TranscriptionMetadata

	if usage != nil {
		metadata.Usage = &TranscriptionUsage{
			PromptTokens:       usage.PromptTokens,
			CompletionTokens:   usage.CompletionTokens,
			TotalTokens:        usage.TotalTokens,
			PromptAudioSeconds: usage.PromptAudioSeconds,
			RequestCount:       usage.RequestCount,
		}
	}

	for _, segment := range segments {
		if segment.Type == nil && segment.Score == nil && segment.SpeakerID == nil {
			continue
		}

		metadata.Segments = append(metadata.Segments, segmentMetadata(segment))
	}

	if metadata.Usage == nil && metadata.Segments == nil {
		return nil
	}

	return model.ProviderMetadata{
		Namespace: metadata,
	}
}

func segmentMetadata(segment transcriptionSegment) SegmentMetadata {
	converted := SegmentMetadata{
		Text:        segment.Text,
		StartSecond: segment.Start,
		EndSecond:   segment.End,
		Score:       segment.Score,
	}

	if segment.Type != nil {
		converted.Type = *segment.Type
	}

	if segment.SpeakerID != nil {
		converted.SpeakerID = *segment.SpeakerID
	}

	return converted
}

var _ model.TranscriptionModel = (*TranscriptionModel)(nil)
