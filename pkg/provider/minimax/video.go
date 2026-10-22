package minimax

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/apikey"
	"github.com/aarock1234/ai/internal/optional"
	"github.com/aarock1234/ai/internal/poll"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// defaultVideoBaseURL is the API root of video generation, without a
	// version suffix.
	defaultVideoBaseURL = "https://api.minimax.io"

	// Video endpoints below the video base URL.
	videoGenerationPath = "/v2/video_generation"
	videoQueryPath      = "/v2/query/video_generation/"

	// generatedVideoMediaType is the media type of generated videos.
	generatedVideoMediaType = "video/mp4"

	// defaultVideoErrorMessage describes error responses without a message.
	defaultVideoErrorMessage = "MiniMax video generation error"
)

// Poll defaults of Generate.
const (
	defaultVideoPollInterval = 10 * time.Second
	defaultVideoPollTimeout  = 10 * time.Minute
)

// Task statuses of video generation. Others, such as queued and running,
// are pending.
const (
	taskSucceeded = "succeeded"
	taskFailed    = "failed"
	taskCancelled = "cancelled"
	taskExpired   = "expired"
)

var (
	// ErrVideoGenerationFailed reports a generation that failed, was
	// cancelled or expired, or that succeeded without a video URL.
	ErrVideoGenerationFailed = errors.New("minimax video generation failed")

	// ErrVideoGenerationTimeout reports a generation that Generate stopped
	// waiting for.
	ErrVideoGenerationTimeout = errors.New("minimax video generation timed out")
)

var (
	_ model.VideoModel     = (*VideoModel)(nil)
	_ model.VideoGenerator = (*VideoModel)(nil)
	_ model.VideoOperator  = (*VideoModel)(nil)
	_ model.VideoProvider  = (*Provider)(nil)
)

// VideoModel is a MiniMax video model, such as "MiniMax-H3". Generation is
// asynchronous: Start submits a task and Status checks it. Generate does
// both, polling with the pollIntervalMs and pollTimeoutMs options.
//
// A webhook URL given to Start is sent as the callback URL. MiniMax first
// sends a challenge that the receiver must echo within three seconds, and
// then progress notifications, so the model does not handle webhooks
// itself.
type VideoModel struct {
	modelID  string
	provider *Provider
}

// VideoOperation identifies a submitted generation task. It is the JSON
// operation reference of Start and Status.
type VideoOperation struct {
	TaskID         string              `json:"taskId"`
	ResolvedInputs VideoResolvedInputs `json:"resolvedInputs"`
}

// VideoMetadata is the provider metadata of a completed generation, stored
// under the "minimax" namespace. Nil fields were not reported.
type VideoMetadata struct {
	TaskID         string              `json:"taskId"`
	VideoURL       string              `json:"videoUrl"`
	ResolvedInputs VideoResolvedInputs `json:"resolvedInputs"`
	Duration       *float64            `json:"duration,omitempty"`
	Ratio          *string             `json:"ratio,omitempty"`
	Resolution     *string             `json:"resolution,omitempty"`
	Usage          *VideoUsage         `json:"usage,omitempty"`
}

// VideoUsage is the billed usage of a generation.
type VideoUsage struct {
	TotalSeconds  *float64 `json:"totalSeconds,omitempty"`
	InputSeconds  *float64 `json:"inputSeconds,omitempty"`
	OutputSeconds *float64 `json:"outputSeconds,omitempty"`
}

type createVideoResponse struct {
	TaskID *string `json:"task_id"`
}

type videoStatusResponse struct {
	Task *videoTask `json:"task"`
}

type videoTask struct {
	ID         *string             `json:"id"`
	Status     *string             `json:"status"`
	Content    *videoContentResult `json:"content"`
	Resolution *string             `json:"resolution"`
	Duration   *float64            `json:"duration"`
	Ratio      *string             `json:"ratio"`
	Usage      *videoTaskUsage     `json:"usage"`
	Error      *videoTaskError     `json:"error"`
}

type videoContentResult struct {
	URL *string `json:"url"`
}

type videoTaskUsage struct {
	TotalSeconds  *float64 `json:"total_seconds"`
	InputSeconds  *float64 `json:"input_seconds"`
	OutputSeconds *float64 `json:"output_seconds"`
}

type videoTaskError struct {
	// Code is a string or a number.
	Code    json.RawMessage `json:"code"`
	Message *string         `json:"message"`
}

// videoErrorResponse is the error body of the video API.
type videoErrorResponse struct {
	Type      *string           `json:"type"`
	Error     *videoErrorDetail `json:"error"`
	RequestID *string           `json:"request_id"`
}

type videoErrorDetail struct {
	Type    *string `json:"type"`
	Message *string `json:"message"`

	// HTTPCode is a string or a number.
	HTTPCode json.RawMessage `json:"http_code"`
}

// Video returns a video model, such as "MiniMax-H3" or "MiniMax-H3-Max".
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

// Provider returns the provider identifier, "minimax.video" by default.
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

// Start submits a generation task.
func (m *VideoModel) Start(ctx context.Context, options model.VideoStartOptions) (*model.VideoStartResult, error) {
	timestamp := time.Now()

	body, resolved, warnings, err := buildVideoRequest(m.modelID, options.VideoOptions)
	if err != nil {
		return nil, err
	}

	body.CallbackURL = options.WebhookURL

	request, err := m.provider.videoRequest(videoGenerationPath, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, err
	}

	var created createVideoResponse
	if err := apicall.DecodeJSON(response.Body, &created); err != nil {
		return nil, err
	}

	if created.TaskID == nil || *created.TaskID == "" {
		return nil, fmt.Errorf("%w: no task_id returned from the MiniMax API. response: %s", ErrVideoGenerationFailed, response.Body)
	}

	operation, err := json.Marshal(VideoOperation{
		TaskID:         *created.TaskID,
		ResolvedInputs: resolved,
	})
	if err != nil {
		return nil, fmt.Errorf("encode video operation: %w", err)
	}

	return &model.VideoStartResult{
		Operation: operation,
		Warnings:  warnings,
		Response:  m.responseInfo(timestamp, response),
	}, nil
}

// Status checks a generation task. Failed, cancelled and expired tasks
// report an error status; a succeeded task without a video URL returns an
// error wrapping ErrVideoGenerationFailed.
func (m *VideoModel) Status(ctx context.Context, options model.VideoStatusOptions) (*model.VideoStatusResult, error) {
	timestamp := time.Now()

	var operation VideoOperation
	if err := json.Unmarshal(options.Operation, &operation); err != nil {
		return nil, fmt.Errorf("%w: video operation: %w", model.ErrInvalidArgument, err)
	}

	request, err := m.provider.videoRequest(videoQueryPath+url.PathEscape(operation.TaskID), options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.Get(ctx, request)
	if err != nil {
		return nil, err
	}

	var status videoStatusResponse
	if err := apicall.DecodeJSON(response.Body, &status); err != nil {
		return nil, err
	}

	if status.Task == nil {
		return nil, fmt.Errorf("%w: video status response has no task", model.ErrInvalidResponse)
	}

	result := &model.VideoStatusResult{
		Status:   model.VideoPending,
		Response: m.responseInfo(timestamp, response),
	}

	task := status.Task

	switch optional.Or(task.Status, "") {
	case taskSucceeded:
		return completedVideo(result, operation, task)

	case taskFailed:
		result.Status = model.VideoError
		result.Error = failedTaskMessage(task.Error, operation.TaskID)

	case taskCancelled:
		result.Status = model.VideoError
		result.Error = "MiniMax video generation was cancelled. Task ID: " + operation.TaskID

	case taskExpired:
		result.Status = model.VideoError
		result.Error = "MiniMax video generation request expired. Task ID: " + operation.TaskID
	}

	return result, nil
}

// Generate submits a generation and polls it until it completes, fails or
// the poll timeout passes. The metadata reports the sent URL reference
// videos instead of their positions.
func (m *VideoModel) Generate(ctx context.Context, options model.VideoOptions) (*model.VideoResult, error) {
	started, err := m.Start(ctx, model.VideoStartOptions{
		VideoOptions: options,
	})
	if err != nil {
		return nil, err
	}

	videoOptions, err := readVideoOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	status, err := m.awaitCompletion(ctx, started.Operation, options.Headers, videoOptions)
	if err != nil {
		return nil, err
	}

	var operation VideoOperation
	if err := json.Unmarshal(started.Operation, &operation); err != nil {
		return nil, fmt.Errorf("decode video operation: %w", err)
	}

	metadata, ok := status.ProviderMetadata[Namespace].(VideoMetadata)
	if ok {
		metadata.ResolvedInputs = VideoResolvedInputs{
			ImageCount:         operation.ResolvedInputs.ImageCount,
			ReferenceVideoURLs: referenceVideoURLs(options.InputReferences, operation.ResolvedInputs.ReferenceVideoIndices),
		}
		status.ProviderMetadata[Namespace] = metadata
	}

	response := status.Response
	response.Timestamp = started.Response.Timestamp

	return &model.VideoResult{
		Videos:           status.Videos,
		ProviderMetadata: status.ProviderMetadata,
		Warnings:         slices.Concat(started.Warnings, status.Warnings),
		Response:         response,
	}, nil
}

// awaitCompletion polls the operation until it completes. Failures return
// an error wrapping ErrVideoGenerationFailed.
func (m *VideoModel) awaitCompletion(
	ctx context.Context,
	operation json.RawMessage,
	headers http.Header,
	videoOptions VideoOptions,
) (*model.VideoStatusResult, error) {
	interval := defaultVideoPollInterval
	if videoOptions.PollIntervalMs != nil {
		interval = time.Duration(*videoOptions.PollIntervalMs) * time.Millisecond
	}

	timeout := defaultVideoPollTimeout
	if videoOptions.PollTimeoutMs != nil {
		timeout = time.Duration(*videoOptions.PollTimeoutMs) * time.Millisecond
	}

	deadline := time.Now().Add(timeout)

	for {
		if err := poll.Wait(ctx, interval); err != nil {
			return nil, err
		}

		if time.Now().After(deadline) {
			return nil, fmt.Errorf("%w after %dms", ErrVideoGenerationTimeout, timeout.Milliseconds())
		}

		status, err := m.Status(ctx, model.VideoStatusOptions{
			Operation: operation,
			Headers:   headers,
		})
		if err != nil {
			return nil, err
		}

		switch status.Status {
		case model.VideoPending:
			continue

		case model.VideoError:
			return nil, fmt.Errorf("%w: %s", ErrVideoGenerationFailed, status.Error)

		default:
			return status, nil
		}
	}
}

// completedVideo reports the generated video and its metadata.
func completedVideo(result *model.VideoStatusResult, operation VideoOperation, task *videoTask) (*model.VideoStatusResult, error) {
	var videoURL string
	if task.Content != nil {
		videoURL = optional.Or(task.Content.URL, "")
	}

	if videoURL == "" {
		return nil, fmt.Errorf(
			"%w: MiniMax video generation completed but no video URL was returned. Task ID: %s",
			ErrVideoGenerationFailed,
			operation.TaskID,
		)
	}

	metadata := VideoMetadata{
		TaskID:         operation.TaskID,
		VideoURL:       videoURL,
		ResolvedInputs: operation.ResolvedInputs,
		Duration:       task.Duration,
		Ratio:          task.Ratio,
		Resolution:     task.Resolution,
	}

	if task.Usage != nil {
		metadata.Usage = &VideoUsage{
			TotalSeconds:  task.Usage.TotalSeconds,
			InputSeconds:  task.Usage.InputSeconds,
			OutputSeconds: task.Usage.OutputSeconds,
		}
	}

	result.Status = model.VideoCompleted
	result.Videos = []model.VideoData{
		model.VideoDataURL{
			URL:       videoURL,
			MediaType: generatedVideoMediaType,
		},
	}
	result.ProviderMetadata = model.ProviderMetadata{
		Namespace: metadata,
	}

	return result, nil
}

// failedTaskMessage describes a failed task with its error message and
// code.
func failedTaskMessage(taskError *videoTaskError, taskID string) string {
	message := "MiniMax video generation failed"

	if taskError != nil {
		if text := optional.Or(taskError.Message, ""); text != "" {
			message += ": " + text
		}

		if code := errorCode(taskError.Code); code != "" {
			message += " (" + code + ")"
		}
	}

	return message + ". Task ID: " + taskID
}

// errorCode returns a string or numeric code as text.
func errorCode(code json.RawMessage) string {
	trimmed := strings.TrimSpace(string(code))
	if trimmed == "" || trimmed == "null" {
		return ""
	}

	var text string
	if err := json.Unmarshal(code, &text); err == nil {
		return text
	}

	return trimmed
}

// referenceVideoURLs returns the URLs of the input references at indices.
func referenceVideoURLs(references []model.VideoFile, indices []int) []string {
	urls := []string{}

	for _, index := range indices {
		if index < 0 || index >= len(references) {
			continue
		}

		if reference, ok := references[index].(model.VideoURL); ok {
			urls = append(urls, reference.URL)
		}
	}

	return urls
}

func (m *VideoModel) responseInfo(timestamp time.Time, response *apicall.Response) model.ResponseInfo {
	return model.ResponseInfo{
		Timestamp: timestamp,
		ModelID:   m.modelID,
		Headers:   response.Headers,
		Body:      response.Body,
	}
}

// videoRequest describes a call to the video API, authenticated with a
// bearer token, unlike the Messages API.
func (p *Provider) videoRequest(path string, callHeaders http.Header) (apicall.Request, error) {
	apiKey, err := apikey.Load(p.apiKey, apiKeyVariable)
	if err != nil {
		return apicall.Request{}, err
	}

	authorization := http.Header{
		"Authorization": {
			"Bearer " + apiKey,
		},
	}

	baseURL := p.videoBaseURL
	if baseURL == "" {
		baseURL = defaultVideoBaseURL
	}

	return apicall.Request{
		Client:     p.httpClient,
		URL:        strings.TrimRight(baseURL, "/") + path,
		Headers:    apicall.MergeHeaders(authorization, p.headers, callHeaders),
		ParseError: parseVideoError,
	}, nil
}

// parseVideoError reads the message of a video API error response.
func parseVideoError(body []byte) string {
	var response videoErrorResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return ""
	}

	if response.Error != nil && response.Error.Message != nil {
		return *response.Error.Message
	}

	return defaultVideoErrorMessage
}
