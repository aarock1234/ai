package blackforestlabs

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/poll"
	"github.com/aarock1234/ai/pkg/model"
)

// Poll defaults of Generate.
const (
	defaultVideoPollInterval = 2 * time.Second
	defaultVideoPollTimeout  = 10 * time.Minute
)

// generatedVideoMediaType is the media type of generated videos.
const generatedVideoMediaType = "video/mp4"

var (
	// ErrVideoGenerationFailed reports a generation that failed, or that was
	// ready without a video.
	ErrVideoGenerationFailed = errors.New("black forest labs video generation failed")

	// ErrVideoGenerationTimeout reports a generation that Generate stopped
	// waiting for.
	ErrVideoGenerationTimeout = errors.New("black forest labs video generation timed out")
)

var (
	_ model.VideoModel     = (*VideoModel)(nil)
	_ model.VideoGenerator = (*VideoModel)(nil)
	_ model.VideoOperator  = (*VideoModel)(nil)
	_ model.VideoProvider  = (*Provider)(nil)
)

// VideoModel is a FLUX 3 video model. Generation is asynchronous: Start
// queues a generation and Status checks it. Generate does both, polling
// with the provider's poll settings.
type VideoModel struct {
	modelID  string
	provider *Provider
}

// VideoOperation identifies a queued generation. It is the JSON operation
// reference of Start and Status.
type VideoOperation struct {
	RequestID        string   `json:"requestId"`
	PollingURL       string   `json:"pollingUrl"`
	Cost             *float64 `json:"cost,omitempty"`
	InputMegapixels  *float64 `json:"inputMegapixels,omitempty"`
	OutputMegapixels *float64 `json:"outputMegapixels,omitempty"`
}

// VideoMetadata is the provider metadata of a completed generation, stored
// under the "blackForestLabs" namespace.
type VideoMetadata struct {
	Videos []VideoInfo `json:"videos"`
}

// VideoInfo describes one generated video. Nil fields were not reported.
type VideoInfo struct {
	ID        string   `json:"id"`
	VideoURL  string   `json:"videoUrl"`
	Seed      *float64 `json:"seed,omitempty"`
	StartTime *float64 `json:"start_time,omitempty"`
	EndTime   *float64 `json:"end_time,omitempty"`
	Duration  *float64 `json:"duration,omitempty"`

	// DraftCache is the bundle URL of a draft generation, which
	// VideoOptions.DraftCache accepts.
	DraftCache *string `json:"draftCache,omitempty"`

	// Cost is the settled cost, or the estimate of the submission.
	Cost             *float64 `json:"cost,omitempty"`
	InputMegapixels  *float64 `json:"inputMegapixels,omitempty"`
	OutputMegapixels *float64 `json:"outputMegapixels,omitempty"`
}

// videoPollResponse is the status of a queued video. Older responses report
// the status as "state"; Cost is the settled cost of finished generations.
type videoPollResponse struct {
	Status  *string         `json:"status"`
	State   *string         `json:"state"`
	Details json.RawMessage `json:"details"`
	Result  json.RawMessage `json:"result"`
	Cost    *float64        `json:"cost"`
}

// videoPollResult describes a ready video.
type videoPollResult struct {
	Sample     string   `json:"sample"`
	Seed       *float64 `json:"seed"`
	StartTime  *float64 `json:"start_time"`
	EndTime    *float64 `json:"end_time"`
	Duration   *float64 `json:"duration"`
	DraftCache *string  `json:"draft_cache"`
}

// Video returns a video model, such as "flux-3-video".
func (p *Provider) Video(modelID string) *VideoModel {
	return &VideoModel{
		modelID:  modelID,
		provider: p,
	}
}

// VideoModel returns Video(modelID).
func (p *Provider) VideoModel(modelID string) (model.VideoModel, error) {
	return p.Video(modelID), nil
}

// Provider returns the provider identifier, "black-forest-labs.video" by
// default.
func (m *VideoModel) Provider() string {
	return m.provider.name + ".video"
}

// ModelID returns the model identifier.
func (m *VideoModel) ModelID() string {
	return m.modelID
}

// MaxVideosPerCall returns 1.
func (m *VideoModel) MaxVideosPerCall() int {
	return 1
}

// Start queues a generation.
func (m *VideoModel) Start(ctx context.Context, options model.VideoStartOptions) (*model.VideoStartResult, error) {
	timestamp := time.Now()

	body, warnings, err := buildVideoRequest(options.VideoOptions)
	if err != nil {
		return nil, err
	}

	headers, err := m.provider.authHeaders(options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, m.provider.apiRequest(m.provider.baseURL+"/"+m.modelID, headers), body)
	if err != nil {
		return nil, classifyError(err)
	}

	var submitted submitResponse
	if err := apicall.DecodeJSON(response.Body, &submitted); err != nil {
		return nil, err
	}

	if _, err := statusURL(submitted); err != nil {
		return nil, err
	}

	operation, err := json.Marshal(VideoOperation{
		RequestID:        submitted.ID,
		PollingURL:       submitted.PollingURL,
		Cost:             submitted.Cost,
		InputMegapixels:  submitted.InputMegapixels,
		OutputMegapixels: submitted.OutputMegapixels,
	})
	if err != nil {
		return nil, fmt.Errorf("encode video operation: %w", err)
	}

	return &model.VideoStartResult{
		Operation: operation,
		Warnings:  warnings,
		Response:  m.responseInfo(timestamp, response.Headers, response.Body),
	}, nil
}

// Status checks a generation. Failed and moderated generations report an
// error status; a ready generation without a video returns an error
// wrapping ErrVideoGenerationFailed.
func (m *VideoModel) Status(ctx context.Context, options model.VideoStatusOptions) (*model.VideoStatusResult, error) {
	timestamp := time.Now()

	var operation VideoOperation
	if err := json.Unmarshal(options.Operation, &operation); err != nil {
		return nil, fmt.Errorf("%w: video operation: %w", model.ErrInvalidArgument, err)
	}

	pollURL, err := statusURL(submitResponse{
		ID:         operation.RequestID,
		PollingURL: operation.PollingURL,
	})
	if err != nil {
		return nil, err
	}

	headers, err := m.provider.authHeaders(options.Headers)
	if err != nil {
		return nil, err
	}

	body, err := m.provider.fetchStatus(ctx, pollURL, headers)
	if err != nil {
		return nil, err
	}

	var response videoPollResponse
	if err := apicall.DecodeJSON(body, &response); err != nil {
		return nil, err
	}

	status, err := response.status()
	if err != nil {
		return nil, err
	}

	result := &model.VideoStatusResult{
		Status:   model.VideoPending,
		Response: m.responseInfo(timestamp, nil, body),
	}

	switch {
	case status == statusReady:
		return completedVideo(result, operation, response)

	case slices.Contains(terminalStatuses, status):
		result.Status = model.VideoError
		result.Error = failureMessage(status, response.Details, operation.RequestID)

		return result, nil

	default:
		return result, nil
	}
}

// Generate queues a generation and polls it until it completes, fails or
// the provider's poll timeout passes.
func (m *VideoModel) Generate(ctx context.Context, options model.VideoOptions) (*model.VideoResult, error) {
	started, err := m.Start(ctx, model.VideoStartOptions{
		VideoOptions: options,
	})
	if err != nil {
		return nil, err
	}

	settings := m.pollSettings()

	pollCtx, cancel := context.WithTimeoutCause(ctx, settings.timeout, ErrVideoGenerationTimeout)
	defer cancel()

	completed, err := m.awaitCompletion(pollCtx, started.Operation, options.Headers, settings)
	if err != nil {
		return nil, timeoutCause(ctx, pollCtx, settings, err)
	}

	return &model.VideoResult{
		Videos:           completed.Videos,
		ProviderMetadata: completed.ProviderMetadata,
		Warnings:         slices.Concat(started.Warnings, completed.Warnings),
		Response:         completed.Response,
	}, nil
}

// pollSettings returns the provider's poll settings with the video
// defaults.
func (m *VideoModel) pollSettings() pollSettings {
	return pollSettings{
		interval: durationOr(m.provider.pollInterval, defaultVideoPollInterval),
		timeout:  durationOr(m.provider.pollTimeout, defaultVideoPollTimeout),
	}
}

// awaitCompletion checks the generation every poll interval until it
// completes or fails, or ctx ends.
func (m *VideoModel) awaitCompletion(
	ctx context.Context,
	operation json.RawMessage,
	headers http.Header,
	settings pollSettings,
) (*model.VideoStatusResult, error) {
	for {
		if err := poll.Wait(ctx, settings.interval); err != nil {
			return nil, err
		}

		current, err := m.Status(ctx, model.VideoStatusOptions{
			Operation: operation,
			Headers:   headers,
		})
		if err != nil {
			return nil, err
		}

		switch current.Status {
		case model.VideoPending:
			continue

		case model.VideoError:
			return nil, fmt.Errorf("%w: %s", ErrVideoGenerationFailed, current.Error)

		default:
			return current, nil
		}
	}
}

// status returns the generation status, reported as "status" or "state".
func (r videoPollResponse) status() (taskStatus, error) {
	switch {
	case r.Status != nil:
		return taskStatus(*r.Status), nil

	case r.State != nil:
		return taskStatus(*r.State), nil

	default:
		return "", fmt.Errorf("%w: missing status in Black Forest Labs poll response", model.ErrInvalidResponse)
	}
}

// completedVideo reports the ready video and its metadata.
func completedVideo(
	result *model.VideoStatusResult,
	operation VideoOperation,
	response videoPollResponse,
) (*model.VideoStatusResult, error) {
	ready, ok := decodeReadyResult(response.Result)
	if !ok {
		return nil, fmt.Errorf(
			"%w: Black Forest Labs reported the video as Ready but returned no result.sample URL. Request id: %s",
			ErrVideoGenerationFailed,
			operation.RequestID,
		)
	}

	cost := response.Cost
	if cost == nil {
		cost = operation.Cost
	}

	result.Status = model.VideoCompleted
	result.Videos = []model.VideoData{
		model.VideoDataURL{
			URL:       ready.Sample,
			MediaType: generatedVideoMediaType,
		},
	}
	result.ProviderMetadata = model.ProviderMetadata{
		Namespace: VideoMetadata{
			Videos: []VideoInfo{
				{
					ID:               operation.RequestID,
					VideoURL:         ready.Sample,
					Seed:             ready.Seed,
					StartTime:        ready.StartTime,
					EndTime:          ready.EndTime,
					Duration:         ready.Duration,
					DraftCache:       ready.DraftCache,
					Cost:             cost,
					InputMegapixels:  operation.InputMegapixels,
					OutputMegapixels: operation.OutputMegapixels,
				},
			},
		},
	}

	return result, nil
}

// decodeReadyResult decodes the result of a ready generation, which must
// carry an absolute sample URL.
func decodeReadyResult(raw json.RawMessage) (videoPollResult, bool) {
	var ready videoPollResult
	if err := json.Unmarshal(raw, &ready); err != nil {
		return videoPollResult{}, false
	}

	sample, err := url.Parse(ready.Sample)
	if err != nil || sample.Scheme == "" || sample.Host == "" {
		return videoPollResult{}, false
	}

	return ready, true
}

// failureMessage describes a failed generation. Moderation refusals carry
// their details as a string, other failures as an object.
func failureMessage(status taskStatus, details json.RawMessage, requestID string) string {
	message := fmt.Sprintf("Black Forest Labs video generation failed with status %q", status)

	if detail := describeDetails(details); detail != "" {
		message += ": " + detail
	}

	return message + ". Request id: " + requestID
}

func describeDetails(details json.RawMessage) string {
	trimmed := bytes.TrimSpace(details)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return ""
	}

	var text string
	if err := json.Unmarshal(trimmed, &text); err == nil {
		return text
	}

	var compacted bytes.Buffer
	if err := json.Compact(&compacted, trimmed); err != nil {
		return string(trimmed)
	}

	return compacted.String()
}

func (m *VideoModel) responseInfo(timestamp time.Time, headers http.Header, body []byte) model.ResponseInfo {
	return model.ResponseInfo{
		ResponseMetadata: model.ResponseMetadata{
			Timestamp: timestamp,
			ModelID:   m.modelID,
		},
		Headers: headers,
		Body:    body,
	}
}
