package gladia

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/download"
	"github.com/aarock1234/ai/internal/origin"
	"github.com/aarock1234/ai/internal/poll"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// uploadPath receives the audio file.
	uploadPath = "/v2/upload"

	// jobPath starts a pre-recorded transcription job.
	jobPath = "/v2/pre-recorded"

	// modelID is the identifier of Gladia's single transcription model.
	modelID = "default"

	// pollingInterval is the delay between job result checks.
	pollingInterval = time.Second

	// pollingTimeout bounds the time spent polling a job.
	pollingTimeout = time.Minute

	// maxResultBytes bounds a job result fetched from another origin.
	maxResultBytes = 64 << 20
)

// TranscriptionModel is the Gladia transcription model.
type TranscriptionModel struct {
	provider *Provider
}

// Provider returns the provider identifier, "gladia.transcription" by
// default.
func (m *TranscriptionModel) Provider() string {
	return m.provider.name + ".transcription"
}

// ModelID returns "default": Gladia has a single transcription model.
func (m *TranscriptionModel) ModelID() string {
	return modelID
}

type uploadResponse struct {
	AudioURL string `json:"audio_url"`
}

type jobResponse struct {
	ResultURL string `json:"result_url"`
}

// polledResult is a finished job and the headers of the response that
// carried it.
type polledResult struct {
	job     TranscriptionMetadata
	headers http.Header
}

// Transcribe uploads the audio, starts a job and polls its result for up to
// a minute. Cancelling ctx stops polling. A failed job returns an error
// wrapping ErrTranscriptionFailed and a slow one ErrTranscriptionTimeout.
func (m *TranscriptionModel) Transcribe(ctx context.Context, options model.TranscriptionOptions) (*model.TranscriptionResult, error) {
	timestamp := time.Now()

	transcriptionOptions, err := readTranscriptionOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	headers, err := m.provider.requestHeaders(options.Headers)
	if err != nil {
		return nil, err
	}

	audioURL, err := m.upload(ctx, options, headers)
	if err != nil {
		return nil, err
	}

	resultURL, err := m.startJob(ctx, newJobRequest(transcriptionOptions, audioURL), headers)
	if err != nil {
		return nil, err
	}

	polled, err := m.waitForResult(ctx, resultURL, headers)
	if err != nil {
		return nil, err
	}

	result, err := transcriptionResult(polled.job)
	if err != nil {
		return nil, err
	}

	result.Response = model.ResponseInfo{
		Timestamp: timestamp,
		ModelID:   modelID,
		Headers:   polled.headers,
	}

	return result, nil
}

// upload sends the audio file and returns the URL Gladia stored it at.
func (m *TranscriptionModel) upload(ctx context.Context, options model.TranscriptionOptions, headers http.Header) (string, error) {
	fields := []apicall.FormField{
		{
			Name:      "audio",
			Filename:  "audio." + audioExtension(options.MediaType),
			MediaType: options.MediaType,
			Data:      options.Audio,
		},
	}

	response, err := apicall.PostForm(ctx, m.provider.apiRequest(m.provider.baseURL+uploadPath, headers), fields)
	if err != nil {
		return "", err
	}

	var decoded uploadResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return "", err
	}

	if decoded.AudioURL == "" {
		return "", fmt.Errorf("%w: gladia upload response has no audio_url", model.ErrInvalidResponse)
	}

	return decoded.AudioURL, nil
}

// startJob starts the transcription job and returns its result URL.
func (m *TranscriptionModel) startJob(ctx context.Context, body jobRequest, headers http.Header) (string, error) {
	response, err := apicall.PostJSON(ctx, m.provider.apiRequest(m.provider.baseURL+jobPath, headers), body)
	if err != nil {
		return "", err
	}

	var decoded jobResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return "", err
	}

	if decoded.ResultURL == "" {
		return "", fmt.Errorf("%w: gladia job response has no result_url", model.ErrInvalidResponse)
	}

	return decoded.ResultURL, nil
}

// waitForResult polls the result URL until the job is done or fails, the
// polling timeout passes or ctx is cancelled.
func (m *TranscriptionModel) waitForResult(ctx context.Context, resultURL string, headers http.Header) (polledResult, error) {
	deadline := time.Now().Add(pollingTimeout)

	var delay time.Duration

	for {
		if err := poll.Wait(ctx, delay); err != nil {
			return polledResult{}, err
		}

		if time.Now().After(deadline) {
			return polledResult{}, ErrTranscriptionTimeout
		}

		polled, err := m.fetchResult(ctx, resultURL, headers)
		if err != nil {
			return polledResult{}, err
		}

		switch polled.job.Status {
		case JobDone:
			return polled, nil
		case JobError:
			return polledResult{}, ErrTranscriptionFailed
		}

		delay = pollingInterval
	}
}

// fetchResult reads the job result. The result URL comes from the provider
// response: on the API origin it is fetched with credentials, and on any
// other origin it is validated as an untrusted URL and fetched without them.
func (m *TranscriptionModel) fetchResult(ctx context.Context, resultURL string, headers http.Header) (polledResult, error) {
	var (
		body            []byte
		responseHeaders http.Header
	)

	if origin.Same(resultURL, m.provider.baseURL) {
		response, err := apicall.Get(ctx, m.provider.apiRequest(resultURL, headers))
		if err != nil {
			return polledResult{}, err
		}

		body = response.Body
		responseHeaders = response.Headers
	} else {
		file, err := download.Fetch(ctx, download.Request{
			URL:                resultURL,
			Headers:            headers,
			CredentialedOrigin: m.provider.baseURL,
			TrustedOrigin:      m.provider.baseURL,
			Client:             m.provider.httpClient,
			MaxBytes:           maxResultBytes,
		})
		if err != nil {
			return polledResult{}, fmt.Errorf("fetch gladia job result: %w", err)
		}

		body = file.Data
	}

	var job TranscriptionMetadata
	if err := apicall.DecodeJSON(body, &job); err != nil {
		return polledResult{}, err
	}

	if err := job.Status.validate(); err != nil {
		return polledResult{}, err
	}

	return polledResult{
		job:     job,
		headers: responseHeaders,
	}, nil
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

func seconds(value float64) time.Duration {
	return time.Duration(value * float64(time.Second))
}

var _ model.TranscriptionModel = (*TranscriptionModel)(nil)
