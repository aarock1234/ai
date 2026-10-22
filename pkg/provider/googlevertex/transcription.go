package googlevertex

import (
	"context"
	"encoding/base64"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/optional"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/googlevertex/internal/vertex"
)

var _ model.TranscriptionModel = (*TranscriptionModel)(nil)

// TranscriptionOptions are the options of a Cloud Speech-to-Text call, read
// from ProviderOptions["googleVertex"], ["vertex"] or ["google"].
type TranscriptionOptions struct {
	// LanguageCodes are the BCP-47 languages to recognize, such as
	// ["en-US"], or ["auto"] to detect the language. The default is
	// ["auto"]; telephony models need an explicit supported language.
	LanguageCodes []string `json:"languageCodes,omitempty"`

	// EnableAutomaticPunctuation adds punctuation. The default is true.
	EnableAutomaticPunctuation *bool `json:"enableAutomaticPunctuation,omitempty"`

	// EnableWordTimeOffsets includes word timestamps, which become the
	// result segments. The default is true; it can reduce the quality and
	// speed of Chirp models.
	EnableWordTimeOffsets *bool `json:"enableWordTimeOffsets,omitempty"`

	// Region is the Speech-to-Text region, such as "us", "eu" or
	// "us-central1". The default is the provider location. Speech-to-Text
	// regions differ from Vertex AI regions, and Chirp is not available in
	// the global location.
	Region string `json:"region,omitempty"`
}

// TranscriptionModel transcribes audio with Gemini transcription models
// through the generateContent API, or with Cloud Speech-to-Text for other
// models such as chirp_2, chirp_3 and telephony. Live Gemini models stream
// through the Live API. All require a token source. It is safe for
// concurrent use.
type TranscriptionModel struct {
	modelID  string
	settings *settings
}

// Provider returns "google.vertex.transcription".
func (m *TranscriptionModel) Provider() string {
	return m.settings.name + ".transcription"
}

// ModelID returns the model ID.
func (m *TranscriptionModel) ModelID() string {
	return m.modelID
}

// Transcribe converts audio into text.
func (m *TranscriptionModel) Transcribe(ctx context.Context, options model.TranscriptionOptions) (*model.TranscriptionResult, error) {
	if err := m.settings.requireTokenSource("transcription models"); err != nil {
		return nil, err
	}

	if strings.HasPrefix(m.modelID, "gemini") {
		return m.transcribeWithGemini(ctx, options)
	}

	return m.recognize(ctx, options)
}

// recognizeRequest is the body of POST recognizers/_:recognize.
type recognizeRequest struct {
	Config  recognitionConfig `json:"config"`
	Content string            `json:"content"`
}

type recognitionConfig struct {
	Model         string   `json:"model"`
	LanguageCodes []string `json:"languageCodes"`

	// AutoDecodingConfig detects the audio encoding.
	AutoDecodingConfig struct{} `json:"autoDecodingConfig"`

	Features recognitionFeatures `json:"features"`
}

type recognitionFeatures struct {
	EnableWordTimeOffsets      bool `json:"enableWordTimeOffsets"`
	EnableAutomaticPunctuation bool `json:"enableAutomaticPunctuation"`
}

type recognizeResponse struct {
	Results []struct {
		Alternatives []struct {
			Transcript *string `json:"transcript"`
			Words      []struct {
				Word        *string `json:"word"`
				StartOffset *string `json:"startOffset"`
				EndOffset   *string `json:"endOffset"`
			} `json:"words"`
		} `json:"alternatives"`
		LanguageCode *string `json:"languageCode"`
	} `json:"results"`
	Metadata *struct {
		TotalBilledDuration *string `json:"totalBilledDuration"`
	} `json:"metadata"`
}

// recognize transcribes with Cloud Speech-to-Text v2.
func (m *TranscriptionModel) recognize(ctx context.Context, options model.TranscriptionOptions) (*model.TranscriptionResult, error) {
	timestamp := time.Now()

	transcriptionOptions, _, err := readOptions[TranscriptionOptions](options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	requestURL, err := m.recognizeURL(transcriptionOptions.Region)
	if err != nil {
		return nil, err
	}

	languageCodes := transcriptionOptions.LanguageCodes
	if languageCodes == nil {
		languageCodes = []string{
			"auto",
		}
	}

	body := recognizeRequest{
		Config: recognitionConfig{
			Model:         m.modelID,
			LanguageCodes: languageCodes,
			Features: recognitionFeatures{
				EnableWordTimeOffsets:      optional.Or(transcriptionOptions.EnableWordTimeOffsets, true),
				EnableAutomaticPunctuation: optional.Or(transcriptionOptions.EnableAutomaticPunctuation, true),
			},
		},
		Content: base64.StdEncoding.EncodeToString(options.Audio),
	}

	request, err := m.settings.request(ctx, requestURL, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, err
	}

	var decoded recognizeResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	result := recognitionResult(decoded)
	result.Response = model.ResponseInfo{
		ResponseMetadata: model.ResponseMetadata{
			Timestamp: timestamp,
			ModelID:   m.modelID,
		},
		Headers: response.Headers,
		Body:    response.Body,
	}

	return result, nil
}

// recognizeURL returns the regional recognize endpoint of the project.
func (m *TranscriptionModel) recognizeURL(requestedRegion string) (string, error) {
	region := requestedRegion
	if region == "" {
		location, err := vertex.LoadLocation(m.settings.location, "")
		if err != nil {
			return "", err
		}

		region = location
	}

	if !vertex.IsHostnameLabel(region) {
		return "", fmt.Errorf(
			"%w: invalid Google Cloud Speech-to-Text region; expected a single DNS label (letters, digits, and hyphens)",
			model.ErrInvalidArgument,
		)
	}

	project, err := vertex.LoadProject(m.settings.project)
	if err != nil {
		return "", err
	}

	host := region + "-speech.googleapis.com"
	if region == "global" {
		host = "speech.googleapis.com"
	}

	return "https://" + host + "/v2/projects/" + project + "/locations/" + region + "/recognizers/_:recognize", nil
}

// recognitionResult joins the primary alternatives of the sequential results
// and collects their timed words.
func recognitionResult(response recognizeResponse) *model.TranscriptionResult {
	transcripts := make([]string, 0, len(response.Results))
	var segments []model.TranscriptionSegment

	for _, recognized := range response.Results {
		if len(recognized.Alternatives) == 0 {
			transcripts = append(transcripts, "")

			continue
		}

		alternative := recognized.Alternatives[0]
		transcripts = append(transcripts, stringValue(alternative.Transcript))

		for _, word := range alternative.Words {
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
	}

	result := &model.TranscriptionResult{
		Text:     strings.TrimSpace(strings.Join(transcripts, " ")),
		Segments: segments,
	}

	if len(response.Results) > 0 {
		result.Language = isoLanguage(response.Results[0].LanguageCode)
	}

	if response.Metadata != nil {
		duration, ok := parseDurationSeconds(response.Metadata.TotalBilledDuration)
		if ok {
			result.Duration = duration
		}
	}

	return result
}

// isoLanguage returns the two-letter language of a BCP-47 tag, or an empty
// string when the language has no two-letter code.
func isoLanguage(tag *string) string {
	if tag == nil {
		return ""
	}

	primary, _, _ := strings.Cut(strings.ReplaceAll(*tag, "_", "-"), "-")
	primary = strings.ToLower(primary)

	if len(primary) != 2 || strings.Trim(primary, "abcdefghijklmnopqrstuvwxyz") != "" {
		return ""
	}

	return primary
}

// parseDurationSeconds parses a Google duration such as "1.200s".
func parseDurationSeconds(value *string) (time.Duration, bool) {
	if value == nil {
		return 0, false
	}

	seconds, err := strconv.ParseFloat(strings.TrimSuffix(*value, "s"), 64)
	if err != nil {
		return 0, false
	}

	return time.Duration(seconds * float64(time.Second)), true
}

func stringValue(value *string) string {
	if value == nil {
		return ""
	}

	return *value
}
