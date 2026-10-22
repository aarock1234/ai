package ai

import (
	"context"
	"mime"
	"net/http"
	"strings"

	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/pkg/model"
)

// defaultSpeechMediaType is used when neither the audio, the response nor the
// requested format identifies the media type.
const defaultSpeechMediaType = "audio/mp3"

// GenerateSpeechOptions configures GenerateSpeech.
type GenerateSpeechOptions struct {
	Model model.SpeechModel

	// Text is the text to speak.
	Text string

	// Voice selects the voice.
	Voice string

	// OutputFormat requests an audio format, such as "mp3" or "wav".
	OutputFormat string

	// Instructions guide the delivery, for example "speak slowly".
	Instructions string

	// Speed scales the speaking rate.
	Speed *float64

	// Language is an ISO 639-1 code, or "auto" for detection. Provider
	// support varies.
	Language string

	// MaxRetries is the number of retries per model call. Nil selects 2;
	// zero disables retries.
	MaxRetries *int

	Headers         http.Header
	ProviderOptions model.ProviderOptions

	// Telemetry sends the lifecycle events of the call to telemetry
	// integrations.
	Telemetry TelemetryOptions

	// LogWarnings receives the warnings of the call. Nil discards them; they
	// are always returned on the result.
	LogWarnings LogWarningsFunc
}

// GenerateSpeechResult is the outcome of GenerateSpeech.
type GenerateSpeechResult struct {
	Audio    GeneratedAudioFile
	Warnings []model.Warning

	// Responses has one entry per model call.
	Responses []model.ResponseInfo

	ProviderMetadata model.ProviderMetadata
}

// GenerateSpeech converts text into speech audio.
func GenerateSpeech(ctx context.Context, options GenerateSpeechOptions) (*GenerateSpeechResult, error) {
	if options.Model == nil {
		return nil, &InvalidArgumentError{
			Parameter: "model",
			Message:   "model is required",
		}
	}

	retry, err := prepareRetries(options.MaxRetries)
	if err != nil {
		return nil, err
	}

	telemetry := newTelemetryDispatcher(options.Telemetry)

	start := GenerateSpeechStartEvent{
		CallID:          id.New("call"),
		OperationID:     "ai.generateSpeech",
		Provider:        options.Model.Provider(),
		ModelID:         options.Model.ModelID(),
		Text:            options.Text,
		Voice:           options.Voice,
		OutputFormat:    options.OutputFormat,
		Instructions:    options.Instructions,
		Speed:           options.Speed,
		Language:        options.Language,
		MaxRetries:      retry.maxRetries,
		Headers:         options.Headers,
		ProviderOptions: options.ProviderOptions,
	}

	telemetry.onGenerateSpeechStart(ctx, start)

	result, response, err := generateSpeech(ctx, options, retry)
	if err != nil {
		return nil, telemetry.onOperationError(ctx, start.CallID, err)
	}

	options.LogWarnings.log(ctx, WarningsEvent{
		Warnings: result.Warnings,
		Provider: start.Provider,
		ModelID:  start.ModelID,
	})

	telemetry.onGenerateSpeechEnd(ctx, GenerateSpeechEndEvent{
		CallID:           start.CallID,
		OperationID:      start.OperationID,
		Provider:         start.Provider,
		ModelID:          start.ModelID,
		Text:             start.Text,
		AudioByteLength:  len(result.Audio.Data),
		AudioMediaType:   result.Audio.MediaType,
		AudioFormat:      result.Audio.Format,
		Usage:            response.Usage,
		Warnings:         result.Warnings,
		ProviderMetadata: result.ProviderMetadata,
		Response:         speechTelemetryResponse(response.Response),
	})

	return result, nil
}

// generateSpeech calls the model with retries and returns the result
// together with the model's response.
func generateSpeech(ctx context.Context, options GenerateSpeechOptions, retry retryPolicy) (*GenerateSpeechResult, *model.SpeechResult, error) {
	response, err := retryCall(ctx, retry, func(ctx context.Context) (*model.SpeechResult, error) {
		return options.Model.Generate(ctx, model.SpeechOptions{
			Text:            options.Text,
			Voice:           options.Voice,
			OutputFormat:    options.OutputFormat,
			Instructions:    options.Instructions,
			Speed:           options.Speed,
			Language:        options.Language,
			Headers:         options.Headers,
			ProviderOptions: options.ProviderOptions,
		})
	})
	if err != nil {
		return nil, nil, err
	}

	if response == nil || len(response.Audio) == 0 {
		return nil, nil, &NoSpeechGeneratedError{
			Responses: speechResponses(response),
		}
	}

	mediaType := speechMediaType(response, options.OutputFormat)

	result := &GenerateSpeechResult{
		Audio:            newGeneratedAudioFile(response.Audio, mediaType),
		Warnings:         response.Warnings,
		Responses:        speechResponses(response),
		ProviderMetadata: response.ProviderMetadata,
	}

	return result, response, nil
}

// speechTelemetryResponse keeps the response metadata and headers, leaving
// out the body, as upstream does.
func speechTelemetryResponse(response model.ResponseInfo) model.ResponseInfo {
	return model.ResponseInfo{
		ResponseMetadata: response.ResponseMetadata,
		Headers:          response.Headers,
	}
}

func speechResponses(response *model.SpeechResult) []model.ResponseInfo {
	if response == nil {
		return nil
	}

	return []model.ResponseInfo{
		response.Response,
	}
}

// speechMediaType identifies the audio from its signature, then from an audio
// Content-Type response header, then from a raw requested output format.
func speechMediaType(response *model.SpeechResult, outputFormat string) string {
	if detected, ok := mediatype.Detect(response.Audio, "audio"); ok {
		return detected
	}

	if fromHeader := responseAudioMediaType(response.Response.Headers); fromHeader != "" {
		return fromHeader
	}

	if fromFormat := outputFormatMediaType(outputFormat); fromFormat != "" {
		return fromFormat
	}

	return defaultSpeechMediaType
}

func responseAudioMediaType(headers http.Header) string {
	contentType := headers.Get("Content-Type")
	if contentType == "" {
		return ""
	}

	mediaType, _, err := mime.ParseMediaType(contentType)
	if err != nil || !strings.HasPrefix(mediaType, "audio/") {
		return ""
	}

	return mediaType
}

// outputFormatMediaType maps the raw audio formats that have no detectable
// signature to their media types.
func outputFormatMediaType(outputFormat string) string {
	switch strings.ToLower(strings.TrimSpace(outputFormat)) {
	case "pcm", "audio/pcm":
		return "audio/pcm"
	case "audio/l16":
		return "audio/l16"
	case "mulaw", "audio/mulaw":
		return "audio/mulaw"
	case "alaw", "audio/alaw":
		return "audio/alaw"
	default:
		return ""
	}
}
