package xai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"slices"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// speechPath is the text-to-speech endpoint.
const speechPath = "/tts"

// Speech defaults.
const (
	defaultVoice    = "eve"
	defaultCodec    = "mp3"
	defaultLanguage = "auto"
)

// speechCodecs are the supported output codecs.
var speechCodecs = []string{
	"mp3",
	"wav",
	"pcm",
	"mulaw",
	"alaw",
}

var (
	speechSampleRates = []int{
		8000,
		16000,
		22050,
		24000,
		44100,
		48000,
	}

	speechBitRates = []int{
		32000,
		64000,
		96000,
		128000,
		192000,
	}
)

// maxStreamingLatency is the largest streaming latency optimization level.
const maxStreamingLatency = 2

// SpeechOptions are the provider options of the speech model, read from the
// "xai" namespace.
type SpeechOptions struct {
	// SampleRate is the sample rate in Hz: 8000, 16000, 22050, 24000, 44100
	// or 48000.
	SampleRate *int `json:"sampleRate,omitempty"`

	// BitRate is the MP3 bit rate in bits per second: 32000, 64000, 96000,
	// 128000 or 192000. It applies only to mp3 output.
	BitRate *int `json:"bitRate,omitempty"`

	// OptimizeStreamingLatency, from 0 to 2, trades quality for a faster
	// first audio chunk.
	OptimizeStreamingLatency *int `json:"optimizeStreamingLatency,omitempty"`

	// TextNormalization converts written-form text into spoken form before
	// synthesis.
	TextNormalization *bool `json:"textNormalization,omitempty"`

	// WithTimestamps returns character-level timing metadata, reported in
	// SpeechMetadata.
	WithTimestamps *bool `json:"withTimestamps,omitempty"`

	// Replace maps phrases to spoken substitutions, as respellings or IPA
	// phonetics.
	Replace map[string]string `json:"replace,omitempty"`
}

// SpeechMetadata is the result provider metadata of the speech model,
// stored under "xai".
type SpeechMetadata struct {
	// TraceID is xAI's trace ID of the request.
	TraceID string `json:"traceId,omitempty"`

	// Duration, ContentType and AudioTimestamps are reported with
	// timestamps.
	Duration        *float64         `json:"duration,omitempty"`
	ContentType     string           `json:"contentType,omitempty"`
	AudioTimestamps *AudioTimestamps `json:"audioTimestamps,omitempty"`
}

// AudioTimestamps are character-level timings.
type AudioTimestamps struct {
	GraphChars []string     `json:"graphChars"`
	GraphTimes [][2]float64 `json:"graphTimes"`
}

// SpeechModel is the xAI text-to-speech model.
type SpeechModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "xai.speech" by default.
func (m *SpeechModel) Provider() string {
	return m.provider.name + ".speech"
}

// ModelID returns the model identifier given to Provider.Speech.
func (m *SpeechModel) ModelID() string {
	return m.modelID
}

type speechRequest struct {
	Text                     string            `json:"text"`
	VoiceID                  string            `json:"voice_id"`
	Language                 string            `json:"language"`
	OutputFormat             speechFormat      `json:"output_format"`
	Speed                    *float64          `json:"speed,omitempty"`
	OptimizeStreamingLatency *int              `json:"optimize_streaming_latency,omitempty"`
	TextNormalization        *bool             `json:"text_normalization,omitempty"`
	WithTimestamps           *bool             `json:"with_timestamps,omitempty"`
	Replace                  map[string]string `json:"replace,omitempty"`
}

type speechFormat struct {
	Codec      string `json:"codec"`
	SampleRate *int   `json:"sample_rate,omitempty"`
	BitRate    *int   `json:"bit_rate,omitempty"`
}

// timestampsResponse is the JSON envelope returned with timestamps.
type timestampsResponse struct {
	Audio           *string              `json:"audio"`
	ContentType     *string              `json:"content_type"`
	Duration        *float64             `json:"duration"`
	AudioTimestamps *timestampsAlignment `json:"audio_timestamps"`
}

type timestampsAlignment struct {
	GraphChars []string     `json:"graph_chars"`
	GraphTimes [][2]float64 `json:"graph_times"`
}

// Generate converts the text into audio.
func (m *SpeechModel) Generate(ctx context.Context, options model.SpeechOptions) (*model.SpeechResult, error) {
	timestamp := time.Now()

	speechOptions, err := readSpeechOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	body, warnings := speechRequestBody(options, speechOptions)

	request, err := m.provider.apiRequest(speechPath, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, err
	}

	result := &model.SpeechResult{
		Audio:    response.Body,
		Warnings: warnings,
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

	metadata := SpeechMetadata{
		TraceID: response.Headers.Get("X-Trace-Id"),
	}

	if body.WithTimestamps != nil && *body.WithTimestamps {
		if err := applyTimestamps(result, &metadata, response.Body); err != nil {
			return nil, err
		}
	}

	result.ProviderMetadata = model.ProviderMetadata{
		Namespace: metadata,
	}

	return result, nil
}

// speechRequestBody builds the request with xAI's defaults. Unsupported
// output formats fall back to mp3.
func speechRequestBody(options model.SpeechOptions, speechOptions SpeechOptions) (speechRequest, []model.Warning) {
	var warnings []model.Warning

	voice := options.Voice
	if voice == "" {
		voice = defaultVoice
	}

	language := options.Language
	if language == "" {
		language = defaultLanguage
	}

	outputFormat := options.OutputFormat
	if outputFormat == "" {
		outputFormat = defaultCodec
	}

	codec := defaultCodec
	if slices.Contains(speechCodecs, outputFormat) {
		codec = outputFormat
	} else {
		warnings = append(warnings, model.UnsupportedWarning(
			"outputFormat",
			fmt.Sprintf("Unsupported output format: %s. Using mp3 instead.", outputFormat),
		))
	}

	if options.Instructions != "" {
		warnings = append(warnings, model.UnsupportedWarning(
			"instructions",
			"xAI speech models do not support the `instructions` option. Use xAI speech tags in `text` to control delivery.",
		))
	}

	format := speechFormat{
		Codec:      codec,
		SampleRate: speechOptions.SampleRate,
	}

	if speechOptions.BitRate != nil {
		if codec == defaultCodec {
			format.BitRate = speechOptions.BitRate
		} else {
			warnings = append(warnings, model.UnsupportedWarning(
				"providerOptions",
				"xAI `bitRate` is supported only for mp3 output. It was ignored.",
			))
		}
	}

	return speechRequest{
		Text:                     options.Text,
		VoiceID:                  voice,
		Language:                 language,
		OutputFormat:             format,
		Speed:                    options.Speed,
		OptimizeStreamingLatency: speechOptions.OptimizeStreamingLatency,
		TextNormalization:        speechOptions.TextNormalization,
		WithTimestamps:           speechOptions.WithTimestamps,
		Replace:                  speechOptions.Replace,
	}, warnings
}

// applyTimestamps decodes the timestamps envelope into the audio and
// metadata. An envelope without audio yields empty audio.
func applyTimestamps(result *model.SpeechResult, metadata *SpeechMetadata, body []byte) error {
	var envelope timestampsResponse
	if err := apicall.DecodeJSON(body, &envelope); err != nil {
		return err
	}

	result.Audio = []byte{}
	if envelope.Audio != nil {
		audio, err := base64.StdEncoding.DecodeString(*envelope.Audio)
		if err != nil {
			return fmt.Errorf("%w: decode audio: %w", model.ErrInvalidResponse, err)
		}

		result.Audio = audio
	}

	result.Response.Body = json.RawMessage(body)

	metadata.Duration = envelope.Duration
	if envelope.ContentType != nil {
		metadata.ContentType = *envelope.ContentType
	}

	if envelope.AudioTimestamps != nil {
		metadata.AudioTimestamps = &AudioTimestamps{
			GraphChars: envelope.AudioTimestamps.GraphChars,
			GraphTimes: envelope.AudioTimestamps.GraphTimes,
		}
	}

	return nil
}

// readSpeechOptions decodes and validates the speech options.
func readSpeechOptions(options model.ProviderOptions) (SpeechOptions, error) {
	speechOptions, _, err := model.ProviderValue[SpeechOptions](options, Namespace)
	if err != nil {
		return SpeechOptions{}, err
	}

	if speechOptions.SampleRate != nil && !slices.Contains(speechSampleRates, *speechOptions.SampleRate) {
		return SpeechOptions{}, fmt.Errorf("%w: xai provider options: unsupported sampleRate %d", model.ErrInvalidArgument, *speechOptions.SampleRate)
	}

	if speechOptions.BitRate != nil && !slices.Contains(speechBitRates, *speechOptions.BitRate) {
		return SpeechOptions{}, fmt.Errorf("%w: xai provider options: unsupported bitRate %d", model.ErrInvalidArgument, *speechOptions.BitRate)
	}

	if latency := speechOptions.OptimizeStreamingLatency; latency != nil && (*latency < 0 || *latency > maxStreamingLatency) {
		return SpeechOptions{}, fmt.Errorf("%w: xai provider options: optimizeStreamingLatency must be between 0 and %d", model.ErrInvalidArgument, maxStreamingLatency)
	}

	return speechOptions, nil
}

var _ model.SpeechModel = (*SpeechModel)(nil)
