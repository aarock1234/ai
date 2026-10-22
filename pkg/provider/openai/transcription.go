package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	transcriptionWhisperModel     = "whisper-1"
	transcriptionDiarizationModel = "gpt-4o-transcribe-diarize"
	transcriptionRealtimeModel    = "gpt-realtime-whisper"
)

// transcriptionLanguageCodePattern matches ISO-639-1 language codes.
var transcriptionLanguageCodePattern = regexp.MustCompile(`^[a-z]{2}$`)

// transcriptionLanguageCodes maps the language names that whisper-1 reports
// to ISO-639-1 codes.
var transcriptionLanguageCodes = map[string]string{
	"afrikaans":   "af",
	"arabic":      "ar",
	"armenian":    "hy",
	"azerbaijani": "az",
	"belarusian":  "be",
	"bosnian":     "bs",
	"bulgarian":   "bg",
	"catalan":     "ca",
	"chinese":     "zh",
	"croatian":    "hr",
	"czech":       "cs",
	"danish":      "da",
	"dutch":       "nl",
	"english":     "en",
	"estonian":    "et",
	"finnish":     "fi",
	"french":      "fr",
	"galician":    "gl",
	"german":      "de",
	"greek":       "el",
	"hebrew":      "he",
	"hindi":       "hi",
	"hungarian":   "hu",
	"icelandic":   "is",
	"indonesian":  "id",
	"italian":     "it",
	"japanese":    "ja",
	"kannada":     "kn",
	"kazakh":      "kk",
	"korean":      "ko",
	"latvian":     "lv",
	"lithuanian":  "lt",
	"macedonian":  "mk",
	"malay":       "ms",
	"marathi":     "mr",
	"maori":       "mi",
	"nepali":      "ne",
	"norwegian":   "no",
	"persian":     "fa",
	"polish":      "pl",
	"portuguese":  "pt",
	"romanian":    "ro",
	"russian":     "ru",
	"serbian":     "sr",
	"slovak":      "sk",
	"slovenian":   "sl",
	"spanish":     "es",
	"swahili":     "sw",
	"swedish":     "sv",
	"tagalog":     "tl",
	"tamil":       "ta",
	"thai":        "th",
	"turkish":     "tr",
	"ukrainian":   "uk",
	"urdu":        "ur",
	"vietnamese":  "vi",
	"welsh":       "cy",
}

// transcriptionExtensions maps audio subtypes whose file extension differs.
var transcriptionExtensions = map[string]string{
	"mpeg":  "mp3",
	"x-wav": "wav",
	"opus":  "ogg",
	"mp4":   "m4a",
	"x-m4a": "m4a",
}

// TranscriptionOptions are the transcription settings read from the "openai"
// provider options namespace. When the namespace is present, Temperature
// defaults to 0 and TimestampGranularities to segment timestamps.
type TranscriptionOptions struct {
	// Include lists additional information to include in the response.
	Include []string `json:"include,omitempty"`

	// Language is the ISO-639-1 language of the input audio.
	Language string `json:"language,omitempty"`

	// Prompt guides the style or continues a previous audio segment.
	Prompt string `json:"prompt,omitempty"`

	// Temperature is the sampling temperature, from 0 to 1.
	Temperature *float64 `json:"temperature,omitempty"`

	// TimestampGranularities selects word or segment timestamps.
	TimestampGranularities []TimestampGranularity `json:"timestampGranularities,omitempty"`

	// ResponseFormat overrides the response format chosen for the model.
	ResponseFormat TranscriptionResponseFormat `json:"responseFormat,omitempty"`

	// ChunkingStrategy controls how the audio is split before transcription.
	// gpt-4o-transcribe-diarize defaults to automatic chunking.
	ChunkingStrategy *ChunkingStrategy `json:"chunkingStrategy,omitempty"`

	// Streaming configures realtime transcription models, such as
	// gpt-realtime-whisper, which stream over a WebSocket.
	Streaming *TranscriptionStreamingOptions `json:"streaming,omitempty"`
}

// TimestampGranularity is the granularity of transcript timestamps.
type TimestampGranularity string

// Timestamp granularities.
const (
	TimestampWord    TimestampGranularity = "word"
	TimestampSegment TimestampGranularity = "segment"
)

// TranscriptionResponseFormat is the format of a transcription response.
type TranscriptionResponseFormat string

// Transcription response formats.
const (
	TranscriptionJSON        TranscriptionResponseFormat = "json"
	TranscriptionVerboseJSON TranscriptionResponseFormat = "verbose_json"
	TranscriptionDiarized    TranscriptionResponseFormat = "diarized_json"
)

// ChunkingStrategy is "auto" chunking, or server-side voice activity
// detection with optional tuning.
type ChunkingStrategy struct {
	// Auto selects automatic chunking; the other fields are then ignored.
	Auto bool

	// Threshold is the voice detection sensitivity, from 0 to 1.
	Threshold *float64

	// PrefixPaddingMs is the audio kept before detected speech.
	PrefixPaddingMs *int

	// SilenceDurationMs is the silence that ends a chunk.
	SilenceDurationMs *int
}

// chunkingStrategyObject is the JSON object form of a ChunkingStrategy.
type chunkingStrategyObject struct {
	Type              string   `json:"type"`
	Threshold         *float64 `json:"threshold,omitempty"`
	PrefixPaddingMs   *int     `json:"prefixPaddingMs,omitempty"`
	SilenceDurationMs *int     `json:"silenceDurationMs,omitempty"`
}

// chunkingStrategyWire is the JSON form sent to the API.
type chunkingStrategyWire struct {
	Type              string   `json:"type"`
	Threshold         *float64 `json:"threshold,omitempty"`
	PrefixPaddingMs   *int     `json:"prefix_padding_ms,omitempty"`
	SilenceDurationMs *int     `json:"silence_duration_ms,omitempty"`
}

// MarshalJSON encodes "auto" or a server_vad object.
func (s ChunkingStrategy) MarshalJSON() ([]byte, error) {
	if s.Auto {
		return json.Marshal("auto")
	}

	return json.Marshal(chunkingStrategyObject{
		Type:              "server_vad",
		Threshold:         s.Threshold,
		PrefixPaddingMs:   s.PrefixPaddingMs,
		SilenceDurationMs: s.SilenceDurationMs,
	})
}

// UnmarshalJSON accepts "auto" or a server_vad object.
func (s *ChunkingStrategy) UnmarshalJSON(data []byte) error {
	var mode string
	if err := json.Unmarshal(data, &mode); err == nil {
		if mode != "auto" {
			return fmt.Errorf("unknown chunking strategy %q", mode)
		}

		*s = ChunkingStrategy{
			Auto: true,
		}

		return nil
	}

	var object chunkingStrategyObject
	if err := json.Unmarshal(data, &object); err != nil {
		return err
	}

	if object.Type != "server_vad" {
		return fmt.Errorf("unknown chunking strategy type %q", object.Type)
	}

	*s = ChunkingStrategy{
		Threshold:         object.Threshold,
		PrefixPaddingMs:   object.PrefixPaddingMs,
		SilenceDurationMs: object.SilenceDurationMs,
	}

	return nil
}

// formValue returns the chunking_strategy form value.
func (s ChunkingStrategy) formValue() (string, error) {
	if s.Auto {
		return "auto", nil
	}

	encoded, err := json.Marshal(chunkingStrategyWire{
		Type:              "server_vad",
		Threshold:         s.Threshold,
		PrefixPaddingMs:   s.PrefixPaddingMs,
		SilenceDurationMs: s.SilenceDurationMs,
	})
	if err != nil {
		return "", fmt.Errorf("encode chunking strategy: %w", err)
	}

	return string(encoded), nil
}

// TranscriptionStreamingOptions configure realtime transcription.
type TranscriptionStreamingOptions struct {
	// Delay trades latency for accuracy: "minimal", "low", "medium", "high"
	// or "xhigh".
	Delay string `json:"delay,omitempty"`

	// Include lists additional fields of realtime events.
	Include []string `json:"include,omitempty"`
}

// TranscriptionMetadata is the "openai" provider metadata of a
// transcription result.
type TranscriptionMetadata struct {
	// Segments are the speaker-labeled segments of a diarized transcript.
	Segments []TranscriptionSpeakerSegment `json:"segments,omitempty"`

	// Languages are the languages detected in the audio.
	Languages []TranscriptionLanguage `json:"languages,omitempty"`
}

// TranscriptionSpeakerSegment is a diarized segment.
type TranscriptionSpeakerSegment struct {
	Text        string  `json:"text"`
	StartSecond float64 `json:"startSecond"`
	EndSecond   float64 `json:"endSecond"`
	Speaker     string  `json:"speaker"`
}

// TranscriptionLanguage is a detected language.
type TranscriptionLanguage struct {
	Code string `json:"code"`
}

// TranscriptionModel is an OpenAI speech-to-text model.
type TranscriptionModel struct {
	modelID string
	config  config
}

func newTranscriptionModel(modelID string, c config) *TranscriptionModel {
	return &TranscriptionModel{
		modelID: modelID,
		config:  c,
	}
}

// Provider returns the provider identifier, such as "openai.transcription".
func (m *TranscriptionModel) Provider() string {
	return m.config.modelProvider("transcription")
}

// ModelID returns the model identifier.
func (m *TranscriptionModel) ModelID() string {
	return m.modelID
}

// transcriptionResponse is a transcription response in any of the JSON
// formats.
type transcriptionResponse struct {
	Text      *string                 `json:"text"`
	Language  string                  `json:"language"`
	Languages []TranscriptionLanguage `json:"languages"`
	Duration  *float64                `json:"duration"`
	Words     []transcriptionWord     `json:"words"`
	Segments  []transcriptionSegment  `json:"segments"`
	Usage     json.RawMessage         `json:"usage"`
}

// transcriptionWord is a timed word.
type transcriptionWord struct {
	Word  string  `json:"word"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}

// transcriptionSegment is a timed segment; diarized segments name a speaker.
type transcriptionSegment struct {
	Text    string  `json:"text"`
	Start   float64 `json:"start"`
	End     float64 `json:"end"`
	Speaker *string `json:"speaker"`
}

// Transcribe converts audio into text. Realtime models only stream; use
// StreamTranscribe for them.
func (m *TranscriptionModel) Transcribe(ctx context.Context, options model.TranscriptionOptions) (*model.TranscriptionResult, error) {
	if transcriptionIsRealtimeModel(m.modelID) {
		return nil, fmt.Errorf("%w: non-streaming transcription with %s", model.ErrUnsupported, m.modelID)
	}

	timestamp := time.Now()

	fields, warnings, err := m.prepareFields(options)
	if err != nil {
		return nil, err
	}

	request, err := m.config.request(ctx, "/audio/transcriptions", options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostForm(ctx, request, fields)
	if err != nil {
		return nil, err
	}

	var decoded transcriptionResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	if decoded.Text == nil {
		return nil, fmt.Errorf("%w: response has no text", model.ErrInvalidResponse)
	}

	return &model.TranscriptionResult{
		Text:             *decoded.Text,
		Segments:         decoded.segments(),
		Language:         decoded.languageCode(),
		Duration:         transcriptionSeconds(decoded.Duration),
		Usage:            transcriptionUsage(decoded.Usage),
		ProviderMetadata: decoded.metadata(),
		Warnings:         warnings,
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: timestamp,
				ModelID:   m.modelID,
			},
			Headers: response.Headers,
			Body:    response.Body,
		},
	}, nil
}

func transcriptionIsRealtimeModel(modelID string) bool {
	return modelID == transcriptionRealtimeModel || strings.HasPrefix(modelID, transcriptionRealtimeModel+"-")
}

// prepareFields builds the multipart form. whisper-1 always returns
// verbose_json; other models choose a format that includes segments when
// they can.
func (m *TranscriptionModel) prepareFields(options model.TranscriptionOptions) ([]apicall.FormField, []model.Warning, error) {
	openaiOptions, hasOptions, err := model.ProviderValue[TranscriptionOptions](options.ProviderOptions, Namespace)
	if err != nil {
		return nil, nil, err
	}

	if err := openaiOptions.validate(); err != nil {
		return nil, nil, err
	}

	var warnings []model.Warning
	if openaiOptions.Streaming != nil {
		warnings = append(warnings, model.UnsupportedWarning(
			"providerOptions.openai.streaming",
			"streaming options only apply to realtime transcription models",
		))
	}

	mediaType, _ := mediatype.Resolve(options.MediaType, options.Audio)
	if mediaType == "" {
		mediaType = options.MediaType
	}

	form := transcriptionForm{
		fields: []apicall.FormField{
			{
				Name:  "model",
				Value: m.modelID,
			},
			{
				Name:      "file",
				Filename:  "audio." + transcriptionExtension(mediaType),
				MediaType: mediaType,
				Data:      options.Audio,
			},
		},
	}

	isWhisper := m.modelID == transcriptionWhisperModel
	isDiarization := m.modelID == transcriptionDiarizationModel

	if isWhisper {
		form.add("response_format", string(TranscriptionVerboseJSON))
	}

	switch {
	case hasOptions:
		m.addOptionFields(&form, openaiOptions.withDefaults(), isWhisper, isDiarization)
	case isDiarization:
		form.add("response_format", string(TranscriptionDiarized))
	}

	chunkingStrategy := openaiOptions.ChunkingStrategy
	if chunkingStrategy == nil && isDiarization {
		chunkingStrategy = &ChunkingStrategy{
			Auto: true,
		}
	}

	if chunkingStrategy != nil {
		value, err := chunkingStrategy.formValue()
		if err != nil {
			return nil, nil, err
		}

		form.add("chunking_strategy", value)
	}

	return form.fields, warnings, nil
}

func (m *TranscriptionModel) addOptionFields(
	form *transcriptionForm,
	options TranscriptionOptions,
	isWhisper bool,
	isDiarization bool,
) {
	for _, include := range options.Include {
		form.add("include[]", include)
	}

	form.add("language", options.Language)
	form.add("prompt", options.Prompt)

	if !isWhisper {
		form.add("response_format", string(m.responseFormat(options, isDiarization)))
	}

	if options.Temperature != nil {
		form.add("temperature", strconv.FormatFloat(*options.Temperature, 'f', -1, 64))
	}

	for _, granularity := range options.TimestampGranularities {
		form.add("timestamp_granularities[]", string(granularity))
	}
}

// responseFormat prefers verbose_json, which includes segments, for models
// that support it.
func (m *TranscriptionModel) responseFormat(options TranscriptionOptions, isDiarization bool) TranscriptionResponseFormat {
	switch {
	case options.ResponseFormat != "":
		return options.ResponseFormat
	case isDiarization:
		return TranscriptionDiarized
	case m.modelID == "gpt-4o-transcribe", m.modelID == "gpt-4o-mini-transcribe":
		return TranscriptionJSON
	default:
		return TranscriptionVerboseJSON
	}
}

// withDefaults applies the defaults of a present options namespace.
func (o TranscriptionOptions) withDefaults() TranscriptionOptions {
	if o.Temperature == nil {
		o.Temperature = new(0.0)
	}

	if o.TimestampGranularities == nil {
		o.TimestampGranularities = []TimestampGranularity{
			TimestampSegment,
		}
	}

	return o
}

func (o TranscriptionOptions) validate() error {
	if o.Temperature != nil {
		temperature := *o.Temperature
		if temperature < 0 || temperature > 1 {
			return transcriptionOptionError("temperature must be between 0 and 1, got %g", temperature)
		}
	}

	for _, granularity := range o.TimestampGranularities {
		if granularity != TimestampWord && granularity != TimestampSegment {
			return transcriptionOptionError("unknown timestamp granularity %q", granularity)
		}
	}

	switch o.ResponseFormat {
	case "", TranscriptionJSON, TranscriptionVerboseJSON, TranscriptionDiarized:
	default:
		return transcriptionOptionError("unknown responseFormat %q", o.ResponseFormat)
	}

	if err := o.ChunkingStrategy.validate(); err != nil {
		return err
	}

	return o.Streaming.validate()
}

func (s *ChunkingStrategy) validate() error {
	if s == nil || s.Auto {
		return nil
	}

	if s.Threshold != nil {
		threshold := *s.Threshold
		if threshold < 0 || threshold > 1 {
			return transcriptionOptionError("chunkingStrategy.threshold must be between 0 and 1, got %g", threshold)
		}
	}

	if s.PrefixPaddingMs != nil && *s.PrefixPaddingMs < 0 {
		return transcriptionOptionError("chunkingStrategy.prefixPaddingMs must not be negative")
	}

	if s.SilenceDurationMs != nil && *s.SilenceDurationMs < 0 {
		return transcriptionOptionError("chunkingStrategy.silenceDurationMs must not be negative")
	}

	return nil
}

func transcriptionOptionError(format string, args ...any) error {
	return fmt.Errorf("%w: openai transcription options: %s", model.ErrInvalidArgument, fmt.Sprintf(format, args...))
}

// transcriptionForm collects multipart fields in order.
type transcriptionForm struct {
	fields []apicall.FormField
}

// add appends a value field unless value is empty.
func (f *transcriptionForm) add(name, value string) {
	if value == "" {
		return
	}

	f.fields = append(f.fields, apicall.FormField{
		Name:  name,
		Value: value,
	})
}

// transcriptionExtension returns the file extension of an audio media type.
func transcriptionExtension(mediaType string) string {
	_, subtype, _ := strings.Cut(strings.ToLower(mediaType), "/")

	if extension, ok := transcriptionExtensions[subtype]; ok {
		return extension
	}

	return subtype
}

// segments returns the timed segments, or the timed words when the response
// has no segments.
func (r transcriptionResponse) segments() []model.TranscriptionSegment {
	if r.Segments != nil {
		segments := make([]model.TranscriptionSegment, 0, len(r.Segments))
		for _, segment := range r.Segments {
			segments = append(segments, model.TranscriptionSegment{
				Text:  segment.Text,
				Start: transcriptionDuration(segment.Start),
				End:   transcriptionDuration(segment.End),
			})
		}

		return segments
	}

	segments := make([]model.TranscriptionSegment, 0, len(r.Words))
	for _, word := range r.Words {
		segments = append(segments, model.TranscriptionSegment{
			Text:  word.Word,
			Start: transcriptionDuration(word.Start),
			End:   transcriptionDuration(word.End),
		})
	}

	return segments
}

// languageCode returns the detected language as an ISO-639-1 code. Language
// names are mapped, and a single detected language is used as a fallback.
func (r transcriptionResponse) languageCode() string {
	if transcriptionLanguageCodePattern.MatchString(r.Language) {
		return r.Language
	}

	if code, ok := transcriptionLanguageCodes[r.Language]; ok {
		return code
	}

	if len(r.Languages) == 1 && transcriptionLanguageCodePattern.MatchString(r.Languages[0].Code) {
		return r.Languages[0].Code
	}

	return ""
}

// metadata returns diarized segments and detected languages, or nil when
// the response has neither.
func (r transcriptionResponse) metadata() model.ProviderMetadata {
	var speakerSegments []TranscriptionSpeakerSegment

	for _, segment := range r.Segments {
		if segment.Speaker == nil {
			continue
		}

		speakerSegments = append(speakerSegments, TranscriptionSpeakerSegment{
			Text:        segment.Text,
			StartSecond: segment.Start,
			EndSecond:   segment.End,
			Speaker:     *segment.Speaker,
		})
	}

	if len(speakerSegments) == 0 && r.Languages == nil {
		return nil
	}

	return model.ProviderMetadata{
		Namespace: TranscriptionMetadata{
			Segments:  speakerSegments,
			Languages: r.Languages,
		},
	}
}

func transcriptionUsage(usage json.RawMessage) json.RawMessage {
	trimmed := strings.TrimSpace(string(usage))
	if trimmed == "" || trimmed == "null" {
		return nil
	}

	return usage
}

func transcriptionSeconds(seconds *float64) time.Duration {
	if seconds == nil {
		return 0
	}

	return transcriptionDuration(*seconds)
}

func transcriptionDuration(seconds float64) time.Duration {
	return time.Duration(seconds * float64(time.Second))
}
