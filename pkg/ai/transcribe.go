package ai

import (
	"context"
	"net/http"
	"time"

	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/pkg/model"
)

// defaultTranscriptionMediaType is used when the audio's media type is
// neither reported by a download nor detectable.
const defaultTranscriptionMediaType = "audio/wav"

// TranscribeOptions configures Transcribe.
type TranscribeOptions struct {
	Model model.TranscriptionModel

	// Audio is the audio to transcribe: model.FileBytes, or model.FileURL
	// for an http(s) or data URL that is downloaded first.
	Audio model.FileData

	// Download fetches Audio URLs. Nil uses NewDownloader with the default
	// size limit.
	Download Downloader

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

// TranscribeResult is the outcome of Transcribe.
type TranscribeResult struct {
	Text     string
	Segments []model.TranscriptionSegment

	// Language is the detected ISO 639-1 language code, if reported.
	Language string

	// Duration is the length of the audio, or zero when unknown.
	Duration time.Duration

	Warnings []model.Warning

	// Responses has one entry per model call.
	Responses []model.ResponseInfo

	ProviderMetadata model.ProviderMetadata
}

// transcriptionAudio is audio ready to send to a transcription model.
type transcriptionAudio struct {
	data      []byte
	mediaType string
}

// Transcribe converts audio into text. StreamTranscribe transcribes live
// audio as it arrives.
func Transcribe(ctx context.Context, options TranscribeOptions) (*TranscribeResult, error) {
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

	start := TranscriptionStartEvent{
		CallID:          id.New("call"),
		OperationID:     "ai.transcribe",
		Provider:        options.Model.Provider(),
		ModelID:         options.Model.ModelID(),
		MaxRetries:      &retry.maxRetries,
		Headers:         options.Headers,
		ProviderOptions: options.ProviderOptions,
	}

	audio, err := loadTranscriptionAudio(ctx, options)
	if err != nil {
		telemetry.onTranscriptionStart(ctx, start)

		return nil, telemetry.onOperationError(ctx, start.CallID, err)
	}

	byteLength := len(audio.data)
	start.AudioByteLength = &byteLength
	start.AudioMediaType = audio.mediaType

	telemetry.onTranscriptionStart(ctx, start)

	response, err := transcribeAudio(ctx, options, retry, audio)
	if err != nil {
		return nil, telemetry.onOperationError(ctx, start.CallID, err)
	}

	options.LogWarnings.log(ctx, WarningsEvent{
		Warnings: response.Warnings,
		Provider: start.Provider,
		ModelID:  start.ModelID,
	})

	telemetry.onTranscriptionEnd(ctx, TranscriptionEndEvent{
		CallID:           start.CallID,
		OperationID:      start.OperationID,
		Provider:         start.Provider,
		ModelID:          start.ModelID,
		AudioByteLength:  byteLength,
		AudioMediaType:   audio.mediaType,
		Text:             response.Text,
		Segments:         response.Segments,
		Language:         response.Language,
		Duration:         response.Duration,
		Usage:            response.Usage,
		Warnings:         response.Warnings,
		ProviderMetadata: response.ProviderMetadata,
		Response:         response.Response,
	})

	return &TranscribeResult{
		Text:     response.Text,
		Segments: response.Segments,
		Language: response.Language,
		Duration: response.Duration,
		Warnings: response.Warnings,
		Responses: []model.ResponseInfo{
			response.Response,
		},
		ProviderMetadata: response.ProviderMetadata,
	}, nil
}

// transcribeAudio calls the model with retries.
func transcribeAudio(
	ctx context.Context,
	options TranscribeOptions,
	retry retryPolicy,
	audio transcriptionAudio,
) (*model.TranscriptionResult, error) {
	response, err := retryCall(ctx, retry, func(ctx context.Context) (*model.TranscriptionResult, error) {
		return options.Model.Transcribe(ctx, model.TranscriptionOptions{
			Audio:           audio.data,
			MediaType:       audio.mediaType,
			Headers:         options.Headers,
			ProviderOptions: options.ProviderOptions,
		})
	})
	if err != nil {
		return nil, err
	}

	if response == nil {
		return nil, &NoTranscriptGeneratedError{}
	}

	return response, nil
}

// loadTranscriptionAudio downloads URL audio and determines its media type:
// the download's Content-Type, then the audio signature, then WAV.
func loadTranscriptionAudio(ctx context.Context, options TranscribeOptions) (transcriptionAudio, error) {
	var audio transcriptionAudio

	switch typed := options.Audio.(type) {
	case model.FileBytes:
		audio.data = typed.Data

	case model.FileURL:
		download := options.Download
		if download == nil {
			download = NewDownloader(DownloadOptions{})
		}

		downloaded, err := download(ctx, typed.URL)
		if err != nil {
			return transcriptionAudio{}, err
		}

		if downloaded == nil {
			return transcriptionAudio{}, &DownloadError{
				URL:     typed.URL.String(),
				Message: "download returned no audio",
			}
		}

		audio.data = downloaded.Data
		audio.mediaType = downloaded.MediaType

	default:
		return transcriptionAudio{}, &InvalidArgumentError{
			Parameter: "audio",
			Value:     options.Audio,
			Message:   "audio must be model.FileBytes or model.FileURL",
		}
	}

	if audio.mediaType != "" {
		return audio, nil
	}

	if detected, ok := mediatype.Detect(audio.data, "audio"); ok {
		audio.mediaType = detected

		return audio, nil
	}

	audio.mediaType = defaultTranscriptionMediaType

	return audio, nil
}
