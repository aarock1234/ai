package revai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/poll"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// jobsPath submits jobs and, followed by an ID, reads them.
	jobsPath = "/speechtotext/v1/jobs"

	// pollingInterval is the delay between job status checks.
	pollingInterval = time.Second

	// pollingTimeout bounds the time spent polling a job.
	pollingTimeout = time.Minute

	// statusTranscribed and statusFailed are the final job statuses.
	statusTranscribed = "transcribed"
	statusFailed      = "failed"
)

// TranscriptionModel is a Rev.ai transcription model.
type TranscriptionModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "revai.transcription" by
// default.
func (m *TranscriptionModel) Provider() string {
	return m.provider.name + ".transcription"
}

// ModelID returns the model identifier.
func (m *TranscriptionModel) ModelID() string {
	return m.modelID
}

type jobResponse struct {
	ID       *string `json:"id"`
	Status   *string `json:"status"`
	Language *string `json:"language"`
}

func (r jobResponse) status() string {
	if r.Status == nil {
		return ""
	}

	return *r.Status
}

// Transcribe submits a job with the audio, polls it for up to a minute until
// it is transcribed and reads the transcript. Cancelling ctx stops polling.
func (m *TranscriptionModel) Transcribe(ctx context.Context, options model.TranscriptionOptions) (*model.TranscriptionResult, error) {
	timestamp := time.Now()

	transcriptionOptions, present, err := readTranscriptionOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	submitted, err := m.submit(ctx, options, newJobConfig(m.modelID, transcriptionOptions, present))
	if err != nil {
		return nil, err
	}

	if submitted.status() == statusFailed {
		return nil, ErrJobSubmissionFailed
	}

	if submitted.ID == nil {
		return nil, fmt.Errorf("%w: revai job has no id", model.ErrInvalidResponse)
	}

	jobPath := jobsPath + "/" + url.PathEscape(*submitted.ID)

	if err := m.waitForTranscription(ctx, submitted, jobPath, options.Headers); err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(jobPath+"/transcript", options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.Get(ctx, request)
	if err != nil {
		return nil, err
	}

	var decoded transcriptResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	segments, durationSeconds := transcriptSegments(decoded.Monologues)

	result := &model.TranscriptionResult{
		Text:     transcriptText(decoded.Monologues),
		Segments: segments,
		Duration: seconds(durationSeconds),
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: timestamp,
				ModelID:   m.modelID,
			},
			Headers: response.Headers,
			Body:    response.Body,
		},
	}

	if submitted.Language != nil {
		result.Language = *submitted.Language
	}

	return result, nil
}

// submit sends the audio and the job config.
func (m *TranscriptionModel) submit(ctx context.Context, options model.TranscriptionOptions, config jobConfig) (jobResponse, error) {
	encodedConfig, err := json.Marshal(config)
	if err != nil {
		return jobResponse{}, fmt.Errorf("encode job config: %w", err)
	}

	fields := []apicall.FormField{
		{
			Name:      "media",
			Filename:  "audio." + audioExtension(options.MediaType),
			MediaType: options.MediaType,
			Data:      options.Audio,
		},
		{
			Name:  "config",
			Value: string(encodedConfig),
		},
	}

	request, err := m.provider.apiRequest(jobsPath, options.Headers)
	if err != nil {
		return jobResponse{}, err
	}

	response, err := apicall.PostForm(ctx, request, fields)
	if err != nil {
		return jobResponse{}, err
	}

	var decoded jobResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return jobResponse{}, err
	}

	return decoded, nil
}

// waitForTranscription polls the job until it is transcribed or fails, the
// polling timeout passes or ctx is cancelled.
func (m *TranscriptionModel) waitForTranscription(ctx context.Context, submitted jobResponse, jobPath string, headers http.Header) error {
	if submitted.status() == statusTranscribed {
		return nil
	}

	request, err := m.provider.apiRequest(jobPath, headers)
	if err != nil {
		return err
	}

	deadline := time.Now().Add(pollingTimeout)

	var delay time.Duration

	for {
		if err := poll.Wait(ctx, delay); err != nil {
			return err
		}

		if time.Now().After(deadline) {
			return ErrTranscriptionTimeout
		}

		response, err := apicall.Get(ctx, request)
		if err != nil {
			return err
		}

		var job jobResponse
		if err := apicall.DecodeJSON(response.Body, &job); err != nil {
			return err
		}

		switch job.status() {
		case statusTranscribed:
			return nil
		case statusFailed:
			return ErrTranscriptionFailed
		}

		delay = pollingInterval
	}
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
