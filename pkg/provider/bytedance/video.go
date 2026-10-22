package bytedance

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// videoTasksPath is the video generation task endpoint.
const videoTasksPath = "/contents/generations/tasks"

// generatedVideoMediaType is the media type of generated videos.
const generatedVideoMediaType = "video/mp4"

// Task statuses. ModelArk documents "cancelled"; "canceled" is handled as
// well.
const (
	taskSucceeded        = "succeeded"
	taskFailed           = "failed"
	taskExpired          = "expired"
	taskCancelled        = "cancelled"
	taskCanceledAmerican = "canceled"
)

// ErrVideoGenerationFailed reports a submission that returned no task, or a
// succeeded task without a video URL.
var ErrVideoGenerationFailed = errors.New("bytedance video generation failed")

var (
	_ model.VideoModel    = (*VideoModel)(nil)
	_ model.VideoOperator = (*VideoModel)(nil)
)

// VideoModel is a ByteDance Seedance video model. Generation is
// asynchronous: Start submits a task and Status checks it; ai.GenerateVideo
// polls until the video is ready.
//
// A webhook URL passed to Start is forwarded as the task's callback URL.
// ModelArk also notifies it of progress, so the model does not implement
// model.VideoWebhookHandler; the receiver must ignore progress
// notifications.
type VideoModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "bytedance.video" by default.
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

// VideoOperation identifies a submitted generation task. It is the JSON
// operation reference of Start and Status.
type VideoOperation struct {
	TaskID string `json:"taskId"`
}

// VideoMetadata is the provider metadata of a completed generation, stored
// under the "bytedance" namespace.
type VideoMetadata struct {
	TaskID string      `json:"taskId"`
	Usage  *VideoUsage `json:"usage,omitempty"`

	// LastFrameURL is the last frame, reported when ReturnLastFrame is set.
	LastFrameURL *string `json:"lastFrameUrl,omitempty"`
}

// VideoUsage is the billed usage of a generation.
type VideoUsage struct {
	CompletionTokens *float64 `json:"completion_tokens,omitempty"`
}

type createTaskResponse struct {
	ID *string `json:"id"`
}

type taskStatusResponse struct {
	ID      *string      `json:"id"`
	Model   *string      `json:"model"`
	Status  string       `json:"status"`
	Content *taskContent `json:"content"`
	Usage   *VideoUsage  `json:"usage"`

	// Error is reported by failed tasks, whose response status is still
	// successful.
	Error *taskError `json:"error"`
}

type taskContent struct {
	VideoURL     *string `json:"video_url"`
	LastFrameURL *string `json:"last_frame_url"`
}

type taskError struct {
	Code    *string `json:"code"`
	Message *string `json:"message"`
}

// Start submits a generation task.
func (m *VideoModel) Start(ctx context.Context, options model.VideoStartOptions) (*model.VideoStartResult, error) {
	timestamp := time.Now()

	body, warnings, err := m.buildVideoRequest(options)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(videoTasksPath, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, err
	}

	var created createTaskResponse
	if err := apicall.DecodeJSON(response.Body, &created); err != nil {
		return nil, err
	}

	if created.ID == nil || *created.ID == "" {
		return nil, fmt.Errorf("%w: no task ID returned from API", ErrVideoGenerationFailed)
	}

	operation, err := json.Marshal(VideoOperation{
		TaskID: *created.ID,
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

// Status checks a generation task. Failed, expired and cancelled tasks
// report an error status.
func (m *VideoModel) Status(ctx context.Context, options model.VideoStatusOptions) (*model.VideoStatusResult, error) {
	timestamp := time.Now()

	var operation VideoOperation
	if err := json.Unmarshal(options.Operation, &operation); err != nil {
		return nil, fmt.Errorf("%w: video operation: %w", model.ErrInvalidArgument, err)
	}

	request, err := m.provider.apiRequest(videoTasksPath+"/"+url.PathEscape(operation.TaskID), options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.Get(ctx, request)
	if err != nil {
		return nil, err
	}

	var status taskStatusResponse
	if err := apicall.DecodeJSON(response.Body, &status); err != nil {
		return nil, err
	}

	result := &model.VideoStatusResult{
		Status:   model.VideoPending,
		Response: m.responseInfo(timestamp, response),
	}

	switch status.Status {
	case taskSucceeded:
		return completedVideo(result, status, operation.TaskID)

	case taskFailed, taskExpired, taskCancelled, taskCanceledAmerican:
		result.Status = model.VideoError
		result.Error = fmt.Sprintf(
			"Video generation %s. Task ID: %s. %s",
			status.Status,
			operation.TaskID,
			failureDetails(status, response.Body),
		)

		return result, nil

	default:
		return result, nil
	}
}

// completedVideo reports the generated video and its metadata.
func completedVideo(result *model.VideoStatusResult, status taskStatusResponse, taskID string) (*model.VideoStatusResult, error) {
	if status.Content == nil || status.Content.VideoURL == nil || *status.Content.VideoURL == "" {
		return nil, fmt.Errorf("%w: no video URL in response. task ID: %s", ErrVideoGenerationFailed, taskID)
	}

	metadata := VideoMetadata{
		TaskID:       taskID,
		Usage:        status.Usage,
		LastFrameURL: status.Content.LastFrameURL,
	}

	result.Status = model.VideoCompleted
	result.Videos = []model.VideoData{
		model.VideoDataURL{
			URL:       *status.Content.VideoURL,
			MediaType: generatedVideoMediaType,
		},
	}
	result.ProviderMetadata = model.ProviderMetadata{
		Namespace: metadata,
	}

	return result, nil
}

// failureDetails is the task's error message or code, or the response body
// when the task reports no reason.
func failureDetails(status taskStatusResponse, body []byte) string {
	if status.Error != nil && status.Error.Message != nil {
		return *status.Error.Message
	}

	if status.Error != nil && status.Error.Code != nil {
		return *status.Error.Code
	}

	return string(bytes.TrimSpace(body))
}

func (m *VideoModel) responseInfo(timestamp time.Time, response *apicall.Response) model.ResponseInfo {
	return model.ResponseInfo{
		Timestamp: timestamp,
		ModelID:   m.modelID,
		Headers:   response.Headers,
		Body:      response.Body,
	}
}
