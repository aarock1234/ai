package azure

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"slices"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

const speechTranscriptionPath = "/speechtotext/transcriptions:transcribe?api-version=2025-10-15"

// isoLanguagePattern matches a two-letter ISO 639-1 language code.
var isoLanguagePattern = regexp.MustCompile(`^[a-z]{2}$`)

// TranscriptionMetadata is the Azure metadata of an Azure Speech
// transcription, stored under ProviderMetadata["azure"].
type TranscriptionMetadata struct {
	Phrases []TranscriptionPhrase `json:"phrases"`
}

// TranscriptionPhrase is one recognized phrase. Nil fields were not
// reported.
type TranscriptionPhrase struct {
	Text                 string                    `json:"text"`
	OffsetMilliseconds   *float64                  `json:"offsetMilliseconds,omitempty"`
	DurationMilliseconds *float64                  `json:"durationMilliseconds,omitempty"`
	Locale               *string                   `json:"locale,omitempty"`
	Speaker              *int                      `json:"speaker,omitempty"`
	Confidence           *float64                  `json:"confidence,omitempty"`
	Words                []TranscriptionPhraseWord `json:"words,omitempty"`
}

// TranscriptionPhraseWord is one timed word of a phrase.
type TranscriptionPhraseWord struct {
	Text                 string   `json:"text"`
	OffsetMilliseconds   *float64 `json:"offsetMilliseconds,omitempty"`
	DurationMilliseconds *float64 `json:"durationMilliseconds,omitempty"`
}

// speechTranscriptionModel transcribes audio with Azure Speech fast
// transcription.
type speechTranscriptionModel struct {
	modelID  string
	settings *settings
}

// transcriptionDefinition is the "definition" form field.
type transcriptionDefinition struct {
	EnhancedMode enhancedMode              `json:"enhancedMode"`
	Locales      []string                  `json:"locales,omitempty"`
	Diarization  *TranscriptionDiarization `json:"diarization,omitempty"`
	PhraseList   *TranscriptionPhraseList  `json:"phraseList,omitempty"`
}

type enhancedMode struct {
	Enabled      bool                     `json:"enabled"`
	Model        string                   `json:"model"`
	ModelOptions enhancedModeModelOptions `json:"modelOptions"`
}

type enhancedModeModelOptions struct {
	Timestamps      TranscriptionTimestamps `json:"timestamps,omitempty"`
	TranscribeStyle TranscribeStyle         `json:"transcribeStyle,omitempty"`
}

// speechTranscriptionResponse is a fast transcription response.
type speechTranscriptionResponse struct {
	CombinedPhrases []struct {
		Text string `json:"text"`
	} `json:"combinedPhrases"`
	DurationMilliseconds *float64              `json:"durationMilliseconds"`
	Phrases              []TranscriptionPhrase `json:"phrases"`
}

func (m *speechTranscriptionModel) transcribe(ctx context.Context, options model.TranscriptionOptions, azureOptions TranscriptionOptions) (*model.TranscriptionResult, error) {
	timestamp := time.Now()

	definition, err := json.Marshal(m.definition(azureOptions))
	if err != nil {
		return nil, fmt.Errorf("encode transcription definition: %w", err)
	}

	requestURL, err := m.settings.speechURL(speechTranscriptionPath)
	if err != nil {
		return nil, err
	}

	request, err := m.settings.serviceRequest(apiSpeech, requestURL, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostForm(ctx, request, []apicall.FormField{
		{
			Name:      "audio",
			Filename:  "audio." + audioExtension(options.MediaType),
			MediaType: options.MediaType,
			Data:      options.Audio,
		},
		formValue("definition", string(definition)),
	})
	if err != nil {
		return nil, err
	}

	var decoded speechTranscriptionResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	phrases := decoded.Phrases
	if phrases == nil {
		phrases = []TranscriptionPhrase{}
	}

	result := &model.TranscriptionResult{
		Text:     combinedText(decoded),
		Segments: phraseSegments(phrases),
		Language: reportedLanguage(phrases),
		ProviderMetadata: model.ProviderMetadata{
			Namespace: TranscriptionMetadata{
				Phrases: phrases,
			},
		},
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: timestamp,
				ModelID:   m.modelID,
			},
			Headers: response.Headers,
			Body:    response.Body,
		},
	}

	if decoded.DurationMilliseconds != nil {
		result.Duration = milliseconds(*decoded.DurationMilliseconds)
	}

	return result, nil
}

// definition describes the request. Timestamps default to segments, except
// for models that only accept Azure's default.
func (m *speechTranscriptionModel) definition(azureOptions TranscriptionOptions) transcriptionDefinition {
	maiModel, isMAIModel := maiTranscribeModels[strings.ToLower(m.modelID)]

	modelName := m.modelID
	if isMAIModel {
		modelName = maiModel.name
	}

	timestamps := azureOptions.Timestamps
	if timestamps == "" && (!isMAIModel || maiModel.supportsTimestamps) {
		timestamps = TranscriptionTimestampsSegment
	}

	return transcriptionDefinition{
		EnhancedMode: enhancedMode{
			Enabled: true,
			Model:   modelName,
			ModelOptions: enhancedModeModelOptions{
				Timestamps:      timestamps,
				TranscribeStyle: azureOptions.TranscribeStyle,
			},
		},
		Locales:     azureOptions.Locales,
		Diarization: azureOptions.Diarization,
		PhraseList:  azureOptions.PhraseList,
	}
}

func combinedText(response speechTranscriptionResponse) string {
	texts := make([]string, 0, len(response.CombinedPhrases))
	for _, phrase := range response.CombinedPhrases {
		texts = append(texts, phrase.Text)
	}

	return strings.Join(texts, " ")
}

// phraseSegments returns the timed phrases.
func phraseSegments(phrases []TranscriptionPhrase) []model.TranscriptionSegment {
	segments := make([]model.TranscriptionSegment, 0, len(phrases))

	for _, phrase := range phrases {
		if phrase.OffsetMilliseconds == nil || phrase.DurationMilliseconds == nil {
			continue
		}

		start := *phrase.OffsetMilliseconds
		end := start + *phrase.DurationMilliseconds

		segments = append(segments, model.TranscriptionSegment{
			Text:  phrase.Text,
			Start: milliseconds(start),
			End:   milliseconds(end),
		})
	}

	return segments
}

// reportedLanguage returns the ISO 639-1 language of the phrases. Every
// reported locale is considered, so mixed results such as English and
// Filipino are not collapsed to one language.
func reportedLanguage(phrases []TranscriptionPhrase) string {
	var languages []string

	for _, phrase := range phrases {
		if phrase.Locale == nil || *phrase.Locale == "" {
			continue
		}

		prefix, _, _ := strings.Cut(*phrase.Locale, "-")
		language := strings.ToLower(prefix)

		if !slices.Contains(languages, language) {
			languages = append(languages, language)
		}
	}

	if len(languages) != 1 || !isoLanguagePattern.MatchString(languages[0]) {
		return ""
	}

	return languages[0]
}

func milliseconds(value float64) time.Duration {
	return time.Duration(value * float64(time.Millisecond))
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
