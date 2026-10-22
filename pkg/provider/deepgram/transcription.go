package deepgram

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"strconv"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// transcriptionPath is the pre-recorded transcription endpoint.
const transcriptionPath = "/v1/listen"

// TranscriptionModel is a Deepgram pre-recorded transcription model.
type TranscriptionModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "deepgram.transcription" by
// default.
func (m *TranscriptionModel) Provider() string {
	return m.provider.name + ".transcription"
}

// ModelID returns the model identifier.
func (m *TranscriptionModel) ModelID() string {
	return m.modelID
}

type transcriptionResponse struct {
	Metadata *transcriptionMetadata `json:"metadata"`
	Results  *transcriptionResults  `json:"results"`
}

type transcriptionMetadata struct {
	Duration float64 `json:"duration"`
}

type transcriptionResults struct {
	Channels []transcriptionChannel `json:"channels"`
}

type transcriptionChannel struct {
	DetectedLanguage *string                    `json:"detected_language"`
	Alternatives     []transcriptionAlternative `json:"alternatives"`
}

type transcriptionAlternative struct {
	Transcript string              `json:"transcript"`
	Words      []transcriptionWord `json:"words"`
}

type transcriptionWord struct {
	Word  string  `json:"word"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}

type transcriptionUsage struct {
	Seconds float64 `json:"seconds"`
}

// Transcribe converts audio into text. The audio is sent as the request body
// and the options as query parameters.
func (m *TranscriptionModel) Transcribe(ctx context.Context, options model.TranscriptionOptions) (*model.TranscriptionResult, error) {
	timestamp := time.Now()

	transcriptionOptions, err := readTranscriptionOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	query := m.transcriptionQuery(transcriptionOptions)

	request, err := m.provider.apiRequest(transcriptionPath+"?"+query.Encode(), options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostBody(ctx, request, options.Audio, options.MediaType)
	if err != nil {
		return nil, err
	}

	var decoded transcriptionResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
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

	if err := applyTranscript(result, decoded.Results); err != nil {
		return nil, err
	}

	if decoded.Metadata != nil {
		result.Duration = seconds(decoded.Metadata.Duration)

		usage, err := json.Marshal(transcriptionUsage{
			Seconds: decoded.Metadata.Duration,
		})
		if err != nil {
			return nil, fmt.Errorf("encode transcription usage: %w", err)
		}

		result.Usage = usage
	}

	return result, nil
}

// transcriptionQuery returns the model and the set provider options as query
// parameters. Redaction lists are joined with commas.
func (m *TranscriptionModel) transcriptionQuery(options TranscriptionOptions) url.Values {
	query := url.Values{}
	query.Set("model", m.modelID)

	setBool(query, "detect_entities", options.DetectEntities)
	setBool(query, "detect_language", options.DetectLanguage)
	setBool(query, "diarize", options.Diarize)
	setBool(query, "filler_words", options.FillerWords)
	setBool(query, "intents", options.Intents)
	setString(query, "keyterm", options.Keyterm)
	setString(query, "language", options.Language)
	setBool(query, "paragraphs", options.Paragraphs)
	setBool(query, "punctuate", options.Punctuate)

	if options.Redact != nil {
		query.Set("redact", options.Redact.joined())
	}

	setString(query, "replace", options.Replace)
	setString(query, "search", options.Search)
	setBool(query, "sentiment", options.Sentiment)
	setBool(query, "smart_format", options.SmartFormat)

	if options.Summarize != "" {
		query.Set("summarize", string(options.Summarize))
	}

	setBool(query, "topics", options.Topics)
	setBool(query, "utterances", options.Utterances)

	if options.UttSplit != nil {
		query.Set("utt_split", formatNumber(*options.UttSplit))
	}

	return query
}

func setBool(query url.Values, key string, value *bool) {
	if value != nil {
		query.Set(key, strconv.FormatBool(*value))
	}
}

func setString(query url.Values, key string, value *string) {
	if value != nil {
		query.Set(key, *value)
	}
}

// applyTranscript sets the text, words and detected language of the first
// channel's first alternative.
func applyTranscript(result *model.TranscriptionResult, results *transcriptionResults) error {
	if results == nil || len(results.Channels) == 0 {
		return nil
	}

	channel := results.Channels[0]
	if channel.DetectedLanguage != nil {
		result.Language = *channel.DetectedLanguage
	}

	if len(channel.Alternatives) == 0 {
		return fmt.Errorf("%w: deepgram channel has no alternatives", model.ErrInvalidResponse)
	}

	alternative := channel.Alternatives[0]
	result.Text = alternative.Transcript

	result.Segments = make([]model.TranscriptionSegment, 0, len(alternative.Words))
	for _, word := range alternative.Words {
		result.Segments = append(result.Segments, model.TranscriptionSegment{
			Text:  word.Word,
			Start: seconds(word.Start),
			End:   seconds(word.End),
		})
	}

	return nil
}

func seconds(value float64) time.Duration {
	return time.Duration(value * float64(time.Second))
}

var _ model.TranscriptionModel = (*TranscriptionModel)(nil)
