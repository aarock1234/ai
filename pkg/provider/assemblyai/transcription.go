package assemblyai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/poll"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// uploadPath receives the raw audio.
	uploadPath = "/v2/upload"

	// transcriptPath submits a transcript and, followed by its ID, reads it.
	transcriptPath = "/v2/transcript"
)

// TranscriptionModel is an AssemblyAI transcription model.
type TranscriptionModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "assemblyai.transcription" by
// default.
func (m *TranscriptionModel) Provider() string {
	return m.provider.name + ".transcription"
}

// ModelID returns the model identifier.
func (m *TranscriptionModel) ModelID() string {
	return m.modelID
}

type uploadResponse struct {
	UploadURL string `json:"upload_url"`
}

type submitResponse struct {
	ID     string           `json:"id"`
	Status transcriptStatus `json:"status"`
}

// Transcribe uploads the audio, submits a transcript and polls it until it
// completes. Cancelling ctx stops polling. A transcript that finishes with
// an error returns an error wrapping ErrTranscriptionFailed.
func (m *TranscriptionModel) Transcribe(ctx context.Context, options model.TranscriptionOptions) (*model.TranscriptionResult, error) {
	timestamp := time.Now()

	transcriptionOptions, err := readTranscriptionOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	audioURL, err := m.upload(ctx, options)
	if err != nil {
		return nil, err
	}

	body, warnings := m.transcriptRequest(transcriptionOptions, audioURL)

	transcriptID, err := m.submit(ctx, body, options.Headers)
	if err != nil {
		return nil, err
	}

	completed, err := m.waitForCompletion(ctx, transcriptID, options.Headers)
	if err != nil {
		return nil, err
	}

	result, err := transcriptionResult(completed)
	if err != nil {
		return nil, err
	}

	result.Warnings = warnings
	result.Response.Timestamp = timestamp
	result.Response.ModelID = m.modelID

	return result, nil
}

// upload sends the audio and returns the URL AssemblyAI stored it at.
func (m *TranscriptionModel) upload(ctx context.Context, options model.TranscriptionOptions) (string, error) {
	request, err := m.provider.apiRequest(uploadPath, options.Headers)
	if err != nil {
		return "", err
	}

	response, err := apicall.PostBody(ctx, request, options.Audio, "application/octet-stream")
	if err != nil {
		return "", err
	}

	var decoded uploadResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return "", err
	}

	return decoded.UploadURL, nil
}

// submit creates a transcript and returns its ID.
func (m *TranscriptionModel) submit(ctx context.Context, body transcriptRequest, headers http.Header) (string, error) {
	request, err := m.provider.apiRequest(transcriptPath, headers)
	if err != nil {
		return "", err
	}

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return "", err
	}

	var decoded submitResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return "", err
	}

	if err := decoded.Status.validate(); err != nil {
		return "", err
	}

	return decoded.ID, nil
}

// completedTranscript is a transcript with status completed and the
// response that carried it.
type completedTranscript struct {
	transcript transcriptResponse
	headers    http.Header
	body       json.RawMessage
}

// waitForCompletion polls the transcript until its status is neither queued
// nor processing. The wait between polls stops when ctx is cancelled.
func (m *TranscriptionModel) waitForCompletion(ctx context.Context, transcriptID string, headers http.Header) (completedTranscript, error) {
	request, err := m.provider.apiRequest(transcriptPath+"/"+url.PathEscape(transcriptID), headers)
	if err != nil {
		return completedTranscript{}, err
	}

	var delay time.Duration

	for {
		if err := poll.Wait(ctx, delay); err != nil {
			return completedTranscript{}, err
		}

		response, err := apicall.Get(ctx, request)
		if err != nil {
			return completedTranscript{}, err
		}

		var transcript transcriptResponse
		if err := apicall.DecodeJSON(response.Body, &transcript); err != nil {
			return completedTranscript{}, err
		}

		if err := transcript.Status.validate(); err != nil {
			return completedTranscript{}, err
		}

		switch transcript.Status {
		case statusCompleted:
			return completedTranscript{
				transcript: transcript,
				headers:    response.Headers,
				body:       response.Body,
			}, nil

		case statusFailed:
			return completedTranscript{}, transcriptError(transcript.Error)
		}

		delay = m.provider.pollInterval
	}
}

func transcriptError(message *string) error {
	if message == nil {
		return fmt.Errorf("%w: unknown error", ErrTranscriptionFailed)
	}

	return fmt.Errorf("%w: %s", ErrTranscriptionFailed, *message)
}

var _ model.TranscriptionModel = (*TranscriptionModel)(nil)
