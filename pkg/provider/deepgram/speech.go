package deepgram

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// speechPath is the text-to-speech endpoint.
	speechPath = "/v1/speak"

	// defaultOutputFormat is the format used when the call sets none.
	defaultOutputFormat = "mp3"

	// defaultLanguage completes voice family model IDs without a language.
	defaultLanguage = "en"

	// autoLanguage requests language detection, which speech does not
	// support.
	autoLanguage = "auto"
)

// voiceFamilyIDs are the model IDs that compose the voice and language into
// the Deepgram model, as "<family>-<voice>-<language>".
var voiceFamilyIDs = []string{
	"aura",
	"aura-2",
}

// SpeechModel is a Deepgram text-to-speech model.
type SpeechModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "deepgram.speech" by default.
func (m *SpeechModel) Provider() string {
	return m.provider.name + ".speech"
}

// ModelID returns the model identifier.
func (m *SpeechModel) ModelID() string {
	return m.modelID
}

// SpeechMetadata is the provider metadata of a speech generation, read from
// the Deepgram response headers.
type SpeechMetadata struct {
	ModelName             string   `json:"modelName,omitempty"`
	ModelUUID             string   `json:"modelUuid,omitempty"`
	AdditionalModelUUIDs  []string `json:"additionalModelUuids,omitempty"`
	CharCount             *float64 `json:"charCount,omitempty"`
	BreaksApplied         *float64 `json:"breaksApplied,omitempty"`
	PronunciationsApplied *float64 `json:"pronunciationsApplied,omitempty"`
	PronunciationWarnings string   `json:"pronunciationWarnings,omitempty"`
	RequestID             string   `json:"requestId,omitempty"`
}

type speechRequest struct {
	Text string `json:"text"`
}

type speechUsage struct {
	Characters float64 `json:"characters"`
}

// Generate converts text into audio. The output format maps to Deepgram's
// encoding, container and sample rate query parameters.
func (m *SpeechModel) Generate(ctx context.Context, options model.SpeechOptions) (*model.SpeechResult, error) {
	timestamp := time.Now()

	query, warnings, err := m.speechQuery(options)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(speechPath+"?"+query.Encode(), options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, speechRequest{
		Text: options.Text,
	})
	if err != nil {
		return nil, err
	}

	metadata := speechMetadata(response.Headers)

	result := &model.SpeechResult{
		Audio:    response.Body,
		Warnings: warnings,
		ProviderMetadata: model.ProviderMetadata{
			Namespace: metadata,
		},
		Request: model.RequestInfo{
			Body: response.RequestBody,
		},
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: timestamp,
				ModelID:   m.modelID,
			},
			Headers: response.Headers,
		},
	}

	if metadata.CharCount != nil {
		usage, err := json.Marshal(speechUsage{
			Characters: *metadata.CharCount,
		})
		if err != nil {
			return nil, fmt.Errorf("encode speech usage: %w", err)
		}

		result.Usage = usage
	}

	return result, nil
}

// speechQuery returns the query parameters of a speech request and the
// warnings for ignored settings.
func (m *SpeechModel) speechQuery(options model.SpeechOptions) (url.Values, []model.Warning, error) {
	deepgramOptions, err := readSpeechOptions(options.ProviderOptions)
	if err != nil {
		return nil, nil, err
	}

	upstreamModelID, warnings, err := m.upstreamModelID(options)
	if err != nil {
		return nil, nil, err
	}

	outputFormat := options.OutputFormat
	if outputFormat == "" {
		outputFormat = defaultOutputFormat
	}

	settings := outputFormatSettings(outputFormat)
	warnings = append(warnings, settings.applyOptions(deepgramOptions)...)

	query := url.Values{}
	query.Set("model", upstreamModelID)
	setQueryValue(query, "encoding", settings.encoding)
	setQueryValue(query, "container", settings.container)
	setQueryValue(query, "sample_rate", settings.sampleRate)
	setQueryValue(query, "bit_rate", settings.bitRate)

	if deepgramOptions.Callback != nil {
		setQueryValue(query, "callback", *deepgramOptions.Callback)
	}

	setQueryValue(query, "callback_method", string(deepgramOptions.CallbackMethod))

	if deepgramOptions.MIPOptOut != nil {
		query.Set("mip_opt_out", strconv.FormatBool(*deepgramOptions.MIPOptOut))
	}

	setQueryValue(query, "tag", deepgramOptions.Tag.joined())

	if options.Speed != nil {
		query.Set("speed", formatNumber(*options.Speed))
	}

	warnings = append(warnings, m.ignoredSettings(options, upstreamModelID)...)

	return query, warnings, nil
}

// upstreamModelID composes "<family>-<voice>-<language>" for voice family
// model IDs and returns other model IDs unchanged.
func (m *SpeechModel) upstreamModelID(options model.SpeechOptions) (string, []model.Warning, error) {
	if !slices.Contains(voiceFamilyIDs, m.modelID) {
		return m.modelID, nil, nil
	}

	voice := strings.TrimSpace(options.Voice)
	if voice == "" {
		return "", nil, fmt.Errorf(
			"%w: deepgram speech model %q requires a voice to be set, such as \"thalia\"",
			model.ErrInvalidArgument,
			m.modelID,
		)
	}

	var warnings []model.Warning

	language := options.Language
	if language == autoLanguage {
		warnings = append(warnings, model.CompatibilityWarning(
			"language",
			"Deepgram TTS models do not support automatic language detection. Language \"en\" was used instead.",
		))
	}

	if language == "" || language == autoLanguage {
		language = defaultLanguage
	}

	return m.modelID + "-" + voice + "-" + language, warnings, nil
}

// ignoredSettings warns about the voice and language of full voice model
// IDs, which already embed them, and about instructions.
func (m *SpeechModel) ignoredSettings(options model.SpeechOptions, upstreamModelID string) []model.Warning {
	var warnings []model.Warning

	composed := upstreamModelID != m.modelID

	if !composed && options.Voice != "" && options.Voice != m.modelID {
		warnings = append(warnings, model.UnsupportedWarning(
			"voice",
			fmt.Sprintf(
				"Deepgram TTS models embed the voice in the model ID. The voice parameter %q was ignored. Use the model ID to select a voice (e.g., \"aura-2-helena-en\").",
				options.Voice,
			),
		))
	}

	if !composed && options.Language != "" {
		warnings = append(warnings, model.UnsupportedWarning(
			"language",
			fmt.Sprintf(
				"Deepgram TTS models are language-specific via the model ID. Language parameter %q was ignored. Select a model with the appropriate language suffix (e.g., \"-en\" for English).",
				options.Language,
			),
		))
	}

	if options.Instructions != "" {
		warnings = append(warnings, model.UnsupportedWarning(
			"instructions",
			"Deepgram TTS REST API does not support instructions. Instructions parameter was ignored.",
		))
	}

	return warnings
}

func setQueryValue(query url.Values, key string, value string) {
	if value != "" {
		query.Set(key, value)
	}
}

// speechMetadata reads the usage and model details Deepgram reports in
// response headers. The project ID is deliberately excluded.
func speechMetadata(headers http.Header) SpeechMetadata {
	metadata := SpeechMetadata{
		ModelName:             headers.Get("Dg-Model-Name"),
		ModelUUID:             headers.Get("Dg-Model-Uuid"),
		CharCount:             headerNumber(headers, "Dg-Char-Count"),
		BreaksApplied:         headerNumber(headers, "Dg-Breaks-Applied"),
		PronunciationsApplied: headerNumber(headers, "Dg-Pronunciations-Applied"),
		PronunciationWarnings: headers.Get("Dg-Pronunciation-Warnings"),
		RequestID:             headers.Get("Dg-Request-Id"),
	}

	if values := headers.Values("Dg-Additional-Model-Uuids"); len(values) > 0 {
		metadata.AdditionalModelUUIDs = strings.Split(values[0], ",")
	}

	return metadata
}

// headerNumber parses a numeric header, returning nil when it is absent or
// malformed.
func headerNumber(headers http.Header, name string) *float64 {
	values := headers.Values(name)
	if len(values) == 0 {
		return nil
	}

	text := strings.TrimSpace(values[0])
	if text == "" {
		zero := 0.0

		return &zero
	}

	value, err := strconv.ParseFloat(text, 64)
	if err != nil {
		return nil
	}

	return &value
}

var _ model.SpeechModel = (*SpeechModel)(nil)
