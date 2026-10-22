package klingai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// Task statuses of video generation.
const (
	taskSucceeded = "succeed"
	taskFailed    = "failed"
)

// generatedVideoMediaType is the media type of generated videos.
const generatedVideoMediaType = "video/mp4"

// ErrVideoGenerationFailed reports a submission that returned no task, or a
// succeeded task without videos.
var ErrVideoGenerationFailed = errors.New("klingai video generation failed")

var (
	_ model.VideoModel    = (*VideoModel)(nil)
	_ model.VideoOperator = (*VideoModel)(nil)
)

// VideoModel is a KlingAI video model. Generation is asynchronous: Start
// submits a task and Status checks it; ai.GenerateVideo polls until the
// video is ready.
//
// A webhook URL passed to Start is forwarded as the task's callback URL.
// KlingAI also notifies it of progress, so the model does not implement
// model.VideoWebhookHandler; the receiver must ignore progress
// notifications.
type VideoModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "klingai.video" by default.
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
	TaskID       string `json:"taskId"`
	EndpointPath string `json:"endpointPath"`
}

// VideoMetadata is the provider metadata of a completed generation, stored
// under the "klingai" namespace.
type VideoMetadata struct {
	TaskID string                   `json:"taskId"`
	Videos []GeneratedVideoMetadata `json:"videos"`
}

// GeneratedVideoMetadata describes one generated video.
type GeneratedVideoMetadata struct {
	ID           string  `json:"id"`
	URL          string  `json:"url"`
	WatermarkURL *string `json:"watermarkUrl,omitempty"`
	Duration     *string `json:"duration,omitempty"`
}

type createTaskResponse struct {
	Code      float64         `json:"code"`
	Message   string          `json:"message"`
	RequestID *string         `json:"request_id"`
	Data      *createTaskData `json:"data"`
}

type createTaskData struct {
	TaskID     string  `json:"task_id"`
	TaskStatus *string `json:"task_status"`
}

type taskStatusResponse struct {
	Code      float64   `json:"code"`
	Message   string    `json:"message"`
	RequestID *string   `json:"request_id"`
	Data      *taskData `json:"data"`
}

type taskData struct {
	TaskID             string      `json:"task_id"`
	TaskStatus         string      `json:"task_status"`
	TaskStatusMessage  *string     `json:"task_status_msg"`
	FinalUnitDeduction *string     `json:"final_unit_deduction"`
	CreatedAt          *float64    `json:"created_at"`
	UpdatedAt          *float64    `json:"updated_at"`
	TaskResult         *taskResult `json:"task_result"`
}

type taskResult struct {
	Videos []taskVideo `json:"videos"`
}

type taskVideo struct {
	ID           *string `json:"id"`
	URL          *string `json:"url"`
	WatermarkURL *string `json:"watermark_url"`
	Duration     *string `json:"duration"`
}

// Start submits a generation task to the endpoint of the model's mode, or
// of multi-image-to-video when an image-to-video call has image references.
func (m *VideoModel) Start(ctx context.Context, options model.VideoStartOptions) (*model.VideoStartResult, error) {
	timestamp := m.provider.now()

	built, warnings, err := m.buildRequest(options)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(built.endpointPath, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, built.body)
	if err != nil {
		return nil, err
	}

	var created createTaskResponse
	if err := apicall.DecodeJSON(response.Body, &created); err != nil {
		return nil, err
	}

	if created.Data == nil || created.Data.TaskID == "" {
		return nil, fmt.Errorf("%w: no task_id returned from KlingAI API. response: %s", ErrVideoGenerationFailed, response.Body)
	}

	operation, err := json.Marshal(VideoOperation{
		TaskID:       created.Data.TaskID,
		EndpointPath: built.endpointPath,
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

// Status checks a generation task. A failed task reports an error status.
func (m *VideoModel) Status(ctx context.Context, options model.VideoStatusOptions) (*model.VideoStatusResult, error) {
	timestamp := m.provider.now()

	operation, err := decodeOperation(options.Operation)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(operation.EndpointPath+"/"+url.PathEscape(operation.TaskID), options.Headers)
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

	if status.Data == nil {
		return result, nil
	}

	switch status.Data.TaskStatus {
	case taskSucceeded:
		return completedResult(result, status.Data, operation.TaskID, response.Body)

	case taskFailed:
		message := "Unknown error"
		if status.Data.TaskStatusMessage != nil {
			message = *status.Data.TaskStatusMessage
		}

		result.Status = model.VideoError
		result.Error = "Video generation failed: " + message

		return result, nil

	default:
		return result, nil
	}
}

// decodeOperation reads an operation reference. The endpoint must be one
// of the task endpoints, so that status checks stay on the API.
func decodeOperation(raw json.RawMessage) (VideoOperation, error) {
	var operation VideoOperation
	if err := json.Unmarshal(raw, &operation); err != nil {
		return VideoOperation{}, fmt.Errorf("%w: video operation: %w", model.ErrInvalidArgument, err)
	}

	for _, endpoint := range kindEndpoints {
		if operation.EndpointPath == endpoint {
			return operation, nil
		}
	}

	return VideoOperation{}, fmt.Errorf("%w: video operation endpoint %q", model.ErrInvalidArgument, operation.EndpointPath)
}

// completedResult reports the generated videos and their metadata. Videos
// without a URL are skipped.
func completedResult(
	result *model.VideoStatusResult,
	data *taskData,
	taskID string,
	body []byte,
) (*model.VideoStatusResult, error) {
	if data.TaskResult == nil || len(data.TaskResult.Videos) == 0 {
		return nil, fmt.Errorf("%w: no videos in response. response: %s", ErrVideoGenerationFailed, body)
	}

	metadata := VideoMetadata{
		TaskID: taskID,
	}

	for _, video := range data.TaskResult.Videos {
		if video.URL == nil || *video.URL == "" {
			continue
		}

		result.Videos = append(result.Videos, model.VideoDataURL{
			URL:       *video.URL,
			MediaType: generatedVideoMediaType,
		})
		metadata.Videos = append(metadata.Videos, videoMetadata(video))
	}

	if len(result.Videos) == 0 {
		return nil, fmt.Errorf("%w: no valid video URLs in response", ErrVideoGenerationFailed)
	}

	result.Status = model.VideoCompleted
	result.ProviderMetadata = model.ProviderMetadata{
		Namespace: metadata,
	}

	return result, nil
}

// videoMetadata describes a video with a URL. Empty optional members are
// omitted.
func videoMetadata(video taskVideo) GeneratedVideoMetadata {
	metadata := GeneratedVideoMetadata{
		URL: *video.URL,
	}

	if video.ID != nil {
		metadata.ID = *video.ID
	}

	if video.WatermarkURL != nil && *video.WatermarkURL != "" {
		metadata.WatermarkURL = video.WatermarkURL
	}

	if video.Duration != nil && *video.Duration != "" {
		metadata.Duration = video.Duration
	}

	return metadata
}

func (m *VideoModel) responseInfo(timestamp time.Time, response *apicall.Response) model.ResponseInfo {
	return model.ResponseInfo{
		Timestamp: timestamp,
		ModelID:   m.modelID,
		Headers:   response.Headers,
		Body:      response.Body,
	}
}
