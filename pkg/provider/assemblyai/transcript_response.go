package assemblyai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"time"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// transcriptStatus is the processing state of a transcript.
type transcriptStatus string

// Transcript statuses.
const (
	statusQueued     transcriptStatus = "queued"
	statusProcessing transcriptStatus = "processing"
	statusCompleted  transcriptStatus = "completed"
	statusFailed     transcriptStatus = "error"
)

func (s transcriptStatus) validate() error {
	switch s {
	case statusQueued, statusProcessing, statusCompleted, statusFailed:
		return nil
	default:
		return fmt.Errorf("%w: assemblyai transcript status %q", model.ErrInvalidResponse, s)
	}
}

// transcriptResponse is a transcript read while polling.
type transcriptResponse struct {
	ID                       string            `json:"id"`
	Status                   transcriptStatus  `json:"status"`
	Text                     *string           `json:"text"`
	LanguageCode             *string           `json:"language_code"`
	SpeechModelUsed          *string           `json:"speech_model_used"`
	Words                    []transcriptWord  `json:"words"`
	Utterances               []utterance       `json:"utterances"`
	SentimentAnalysisResults []sentimentResult `json:"sentiment_analysis_results"`
	Entities                 []entity          `json:"entities"`
	ContentSafetyLabels      json.RawMessage   `json:"content_safety_labels"`
	IABCategoriesResult      json.RawMessage   `json:"iab_categories_result"`
	AutoHighlightsResult     json.RawMessage   `json:"auto_highlights_result"`
	AudioDuration            *float64          `json:"audio_duration"`
	Error                    *string           `json:"error"`
}

// rawTranscriptParts are the audio intelligence results exactly as
// AssemblyAI returned them, so that metadata keeps every field.
type rawTranscriptParts struct {
	Utterances               json.RawMessage `json:"utterances"`
	SentimentAnalysisResults json.RawMessage `json:"sentiment_analysis_results"`
	Entities                 json.RawMessage `json:"entities"`
}

// transcriptWord is a word with times in milliseconds.
type transcriptWord struct {
	Start      float64  `json:"start"`
	End        float64  `json:"end"`
	Text       string   `json:"text"`
	Confidence *float64 `json:"confidence"`
	Speaker    *string  `json:"speaker"`
	Channel    *string  `json:"channel"`
}

type utterance struct {
	Start      float64          `json:"start"`
	End        float64          `json:"end"`
	Text       string           `json:"text"`
	Confidence *float64         `json:"confidence"`
	Speaker    *string          `json:"speaker"`
	Channel    *string          `json:"channel"`
	Words      []transcriptWord `json:"words"`
}

type sentimentResult struct {
	Text       string   `json:"text"`
	Start      *float64 `json:"start"`
	End        *float64 `json:"end"`
	Sentiment  string   `json:"sentiment"`
	Confidence *float64 `json:"confidence"`
	Speaker    *string  `json:"speaker"`
}

type entity struct {
	EntityType string   `json:"entity_type"`
	Text       string   `json:"text"`
	Start      *float64 `json:"start"`
	End        *float64 `json:"end"`
}

// TranscriptionMetadata carries the diarization and audio intelligence
// results that segments cannot represent, exactly as AssemblyAI returned
// them. Times inside them are in milliseconds, unlike segment times.
type TranscriptionMetadata struct {
	Utterances               json.RawMessage `json:"utterances,omitempty"`
	SentimentAnalysisResults json.RawMessage `json:"sentimentAnalysisResults,omitempty"`
	Entities                 json.RawMessage `json:"entities,omitempty"`
	ContentSafetyLabels      json.RawMessage `json:"contentSafetyLabels,omitempty"`
	IABCategoriesResult      json.RawMessage `json:"iabCategoriesResult,omitempty"`
	AutoHighlightsResult     json.RawMessage `json:"autoHighlightsResult,omitempty"`
}

// transcriptionResult converts a completed transcript. The response headers
// and body are those of the final poll.
func transcriptionResult(completed completedTranscript) (*model.TranscriptionResult, error) {
	transcript := completed.transcript

	metadata, err := transcriptionMetadata(transcript, completed.body)
	if err != nil {
		return nil, err
	}

	result := &model.TranscriptionResult{
		Segments:         transcriptSegments(transcript.Words),
		Duration:         transcriptDuration(transcript),
		ProviderMetadata: metadata,
		Response: model.ResponseInfo{
			Headers: completed.headers,
			Body:    completed.body,
		},
	}

	if transcript.Text != nil {
		result.Text = *transcript.Text
	}

	if transcript.LanguageCode != nil {
		result.Language = *transcript.LanguageCode
	}

	return result, nil
}

// transcriptSegments converts word times from milliseconds.
func transcriptSegments(words []transcriptWord) []model.TranscriptionSegment {
	segments := make([]model.TranscriptionSegment, 0, len(words))
	for _, word := range words {
		segments = append(segments, model.TranscriptionSegment{
			Text:  word.Text,
			Start: milliseconds(word.Start),
			End:   milliseconds(word.End),
		})
	}

	return segments
}

// transcriptDuration prefers the audio duration, in seconds, and falls back
// to the end of the last word.
func transcriptDuration(transcript transcriptResponse) time.Duration {
	if transcript.AudioDuration != nil {
		return time.Duration(*transcript.AudioDuration * float64(time.Second))
	}

	if len(transcript.Words) == 0 {
		return 0
	}

	return milliseconds(transcript.Words[len(transcript.Words)-1].End)
}

func milliseconds(value float64) time.Duration {
	return time.Duration(value * float64(time.Millisecond))
}

// transcriptionMetadata gathers the present audio intelligence results. It
// returns nil when there are none.
func transcriptionMetadata(transcript transcriptResponse, body json.RawMessage) (model.ProviderMetadata, error) {
	var raw rawTranscriptParts
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("%w: %w", model.ErrInvalidResponse, err)
	}

	var metadata TranscriptionMetadata

	if transcript.Utterances != nil {
		metadata.Utterances = raw.Utterances
	}

	if transcript.SentimentAnalysisResults != nil {
		metadata.SentimentAnalysisResults = raw.SentimentAnalysisResults
	}

	if transcript.Entities != nil {
		metadata.Entities = raw.Entities
	}

	var err error

	metadata.ContentSafetyLabels, err = objectResult("content_safety_labels", transcript.ContentSafetyLabels)
	if err != nil {
		return nil, err
	}

	metadata.IABCategoriesResult, err = objectResult("iab_categories_result", transcript.IABCategoriesResult)
	if err != nil {
		return nil, err
	}

	metadata.AutoHighlightsResult, err = objectResult("auto_highlights_result", transcript.AutoHighlightsResult)
	if err != nil {
		return nil, err
	}

	if metadata.isEmpty() {
		return nil, nil
	}

	return model.ProviderMetadata{
		Namespace: metadata,
	}, nil
}

// objectResult returns an audio intelligence object, or nil when it is
// missing or null.
func objectResult(name string, value json.RawMessage) (json.RawMessage, error) {
	if isAbsent(value) {
		return nil, nil
	}

	if !jsonobject.IsObject(value) {
		return nil, fmt.Errorf("%w: assemblyai %s must be an object", model.ErrInvalidResponse, name)
	}

	return value, nil
}

func (m TranscriptionMetadata) isEmpty() bool {
	return m.Utterances == nil &&
		m.SentimentAnalysisResults == nil &&
		m.Entities == nil &&
		m.ContentSafetyLabels == nil &&
		m.IABCategoriesResult == nil &&
		m.AutoHighlightsResult == nil
}

// isAbsent reports whether a raw member is missing or null.
func isAbsent(value json.RawMessage) bool {
	trimmed := bytes.TrimSpace(value)

	return len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null"))
}
