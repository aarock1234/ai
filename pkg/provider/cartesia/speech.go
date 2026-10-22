package cartesia

import (
	"context"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// speechPath is the text-to-speech endpoint that returns audio bytes.
	speechPath = "/tts/bytes"

	// defaultOutputFormat is the format used when the call sets none.
	defaultOutputFormat = "mp3"

	// defaultBitRate is the mp3 bit rate used when none is derived.
	defaultBitRate = 128000
)

// outputFormat is the output_format of a speech request. MP3 output has a
// bit rate; raw and WAV output have an encoding.
type outputFormat struct {
	Container  Container `json:"container"`
	Encoding   Encoding  `json:"encoding,omitempty"`
	SampleRate int       `json:"sample_rate"`
	BitRate    int       `json:"bit_rate,omitempty"`
}

// defaultMP3Format is the format used for "mp3" and unknown formats.
var defaultMP3Format = outputFormat{
	Container:  ContainerMP3,
	SampleRate: 44100,
	BitRate:    defaultBitRate,
}

// outputFormats map output format names to Cartesia formats.
var outputFormats = map[string]outputFormat{
	"alaw": {
		Container:  ContainerRaw,
		Encoding:   EncodingPCMAlaw,
		SampleRate: 8000,
	},
	"mp3": defaultMP3Format,
	"mulaw": {
		Container:  ContainerRaw,
		Encoding:   EncodingPCMMulaw,
		SampleRate: 8000,
	},
	"pcm": {
		Container:  ContainerRaw,
		Encoding:   EncodingPCMF32LE,
		SampleRate: 44100,
	},
	"raw": {
		Container:  ContainerRaw,
		Encoding:   EncodingPCMF32LE,
		SampleRate: 44100,
	},
	"wav": {
		Container:  ContainerWAV,
		Encoding:   EncodingPCMS16LE,
		SampleRate: 44100,
	},
}

// SpeechModel is a Cartesia text-to-speech model.
type SpeechModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "cartesia.speech" by default.
func (m *SpeechModel) Provider() string {
	return m.provider.name + ".speech"
}

// ModelID returns the model identifier.
func (m *SpeechModel) ModelID() string {
	return m.modelID
}

type speechRequest struct {
	ModelID          string            `json:"model_id"`
	Transcript       string            `json:"transcript"`
	Voice            voiceRequest      `json:"voice"`
	Language         string            `json:"language,omitempty"`
	OutputFormat     outputFormat      `json:"output_format"`
	GenerationConfig *generationConfig `json:"generation_config,omitempty"`
}

type voiceRequest struct {
	Mode string `json:"mode"`
	ID   string `json:"id"`
}

type generationConfig struct {
	Speed float64 `json:"speed"`
}

// Generate converts text into audio. A voice ID is required. The output
// format may carry a sample rate, such as "wav_24000".
func (m *SpeechModel) Generate(ctx context.Context, options model.SpeechOptions) (*model.SpeechResult, error) {
	timestamp := time.Now()

	body, warnings, err := m.speechRequest(options)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(speechPath, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, err
	}

	return &model.SpeechResult{
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
	}, nil
}

func (m *SpeechModel) speechRequest(options model.SpeechOptions) (speechRequest, []model.Warning, error) {
	cartesiaOptions, err := readSpeechOptions(options.ProviderOptions)
	if err != nil {
		return speechRequest{}, nil, err
	}

	if options.Voice == "" {
		return speechRequest{}, nil, fmt.Errorf("%w: cartesia speech models require a voice to be set", model.ErrInvalidArgument)
	}

	requestedFormat := options.OutputFormat
	if requestedFormat == "" {
		requestedFormat = defaultOutputFormat
	}

	format, warnings := resolveOutputFormat(requestedFormat, cartesiaOptions)

	body := speechRequest{
		ModelID:    m.modelID,
		Transcript: options.Text,
		Voice: voiceRequest{
			Mode: "id",
			ID:   options.Voice,
		},
		Language:     options.Language,
		OutputFormat: format,
	}

	if cartesiaOptions.Language != nil {
		body.Language = *cartesiaOptions.Language
	}

	speed := options.Speed
	if cartesiaOptions.Speed != nil {
		speed = cartesiaOptions.Speed
	}

	if speed != nil {
		if *speed >= minSpeed && *speed <= maxSpeed {
			body.GenerationConfig = &generationConfig{
				Speed: *speed,
			}
		} else {
			warnings = append(warnings, model.UnsupportedWarning(
				"speed",
				"Cartesia speed must be between 0.6 and 1.5. The speed option was ignored.",
			))
		}
	}

	if options.Instructions != "" {
		warnings = append(warnings, model.UnsupportedWarning(
			"instructions",
			"Cartesia speech models do not support instructions. Instructions parameter was ignored.",
		))
	}

	return body, warnings, nil
}

// resolveOutputFormat maps an output format such as "wav" or "wav_24000" to
// a Cartesia format and applies the provider options. Unknown formats fall
// back to mp3 with a warning.
func resolveOutputFormat(requested string, options SpeechOptions) (outputFormat, []model.Warning) {
	resolved, warnings := namedOutputFormat(requested)

	container := resolved.Container
	if options.Container != "" {
		container = options.Container
	}

	sampleRate := resolved.SampleRate
	if options.SampleRate != nil {
		sampleRate = *options.SampleRate
	}

	if container == ContainerMP3 {
		return mp3OutputFormat(resolved, options, sampleRate, warnings)
	}

	if options.BitRate != nil {
		warnings = append(warnings, model.UnsupportedWarning(
			"providerOptions.cartesia.bitRate",
			"Cartesia raw and WAV output do not accept a bit rate. The bitRate option was ignored.",
		))
	}

	encoding := options.Encoding
	if encoding == "" {
		encoding = resolved.Encoding
	}

	if encoding == "" && container == ContainerWAV {
		encoding = EncodingPCMS16LE
	}

	if encoding == "" {
		encoding = EncodingPCMF32LE
	}

	return outputFormat{
		Container:  container,
		Encoding:   encoding,
		SampleRate: sampleRate,
	}, warnings
}

func mp3OutputFormat(resolved outputFormat, options SpeechOptions, sampleRate int, warnings []model.Warning) (outputFormat, []model.Warning) {
	if options.Encoding != "" {
		warnings = append(warnings, model.UnsupportedWarning(
			"providerOptions.cartesia.encoding",
			"Cartesia MP3 output does not accept an encoding. The encoding option was ignored.",
		))
	}

	bitRate := defaultBitRate
	if resolved.Container == ContainerMP3 {
		bitRate = resolved.BitRate
	}

	if options.BitRate != nil {
		bitRate = *options.BitRate
	}

	return outputFormat{
		Container:  ContainerMP3,
		SampleRate: sampleRate,
		BitRate:    bitRate,
	}, warnings
}

// namedOutputFormat returns the format named by the first part of requested
// and applies a sample rate suffix.
func namedOutputFormat(requested string) (outputFormat, []model.Warning) {
	parts := strings.Split(strings.ToLower(requested), "_")

	resolved, ok := outputFormats[parts[0]]
	if !ok {
		return defaultMP3Format, []model.Warning{
			model.UnsupportedWarning(
				"outputFormat",
				fmt.Sprintf(
					"Unknown output format %q. Falling back to mp3. Use providerOptions.cartesia to configure container, encoding, and sampleRate directly.",
					requested,
				),
			),
		}
	}

	if len(parts) < 2 {
		return resolved, nil
	}

	sampleRate, err := strconv.Atoi(parts[1])
	if err == nil && len(parts) == 2 && slices.Contains(sampleRates, sampleRate) {
		resolved.SampleRate = sampleRate

		return resolved, nil
	}

	return resolved, []model.Warning{
		model.UnsupportedWarning(
			"outputFormat",
			fmt.Sprintf(
				"Unsupported Cartesia sample rate in output format %q. Using %d Hz instead.",
				requested,
				resolved.SampleRate,
			),
		),
	}
}

var _ model.SpeechModel = (*SpeechModel)(nil)
