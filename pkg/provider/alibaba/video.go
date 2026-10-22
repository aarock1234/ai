package alibaba

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// Video endpoints below the video base URL.
const (
	videoSynthesisPath = "/api/v1/services/aigc/video-generation/video-synthesis"
	videoTasksPath     = "/api/v1/tasks/"
)

// Task statuses of video generation.
const (
	taskSucceeded = "SUCCEEDED"
	taskFailed    = "FAILED"
	taskCanceled  = "CANCELED"
)

// generatedVideoMediaType is the media type of generated videos.
const generatedVideoMediaType = "video/mp4"

// ErrVideoGenerationFailed reports a submission that returned no task, or a
// succeeded task without a video URL.
var ErrVideoGenerationFailed = errors.New("alibaba video generation failed")

var (
	_ model.VideoModel    = (*VideoModel)(nil)
	_ model.VideoOperator = (*VideoModel)(nil)
	_ model.VideoProvider = (*Provider)(nil)
)

// VideoModel is an Alibaba Wan video generation model. Generation is
// asynchronous: Start submits a task and Status checks it; ai.GenerateVideo
// polls until the video is ready.
type VideoModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "alibaba.video" by default.
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
// under the "alibaba" namespace.
type VideoMetadata struct {
	TaskID   string `json:"taskId"`
	VideoURL string `json:"videoUrl"`

	// ActualPrompt is the prompt after prompt extension.
	ActualPrompt string `json:"actualPrompt,omitempty"`

	Usage *VideoUsage `json:"usage,omitempty"`
}

// VideoUsage is the billed usage of a generation. InputVideoDuration, FPS
// and Ratio are reported by wan3 models.
type VideoUsage struct {
	Duration            *float64 `json:"duration,omitempty"`
	OutputVideoDuration *float64 `json:"outputVideoDuration,omitempty"`
	Resolution          *float64 `json:"resolution,omitempty"`
	Size                *string  `json:"size,omitempty"`
	InputVideoDuration  *float64 `json:"inputVideoDuration,omitempty"`
	FPS                 *float64 `json:"fps,omitempty"`
	Ratio               *string  `json:"ratio,omitempty"`
}

type createTaskResponse struct {
	Output *struct {
		TaskStatus string `json:"task_status"`
		TaskID     string `json:"task_id"`
	} `json:"output"`
	RequestID *string `json:"request_id"`
}

type taskStatusResponse struct {
	Output    *taskOutput `json:"output"`
	Usage     *taskUsage  `json:"usage"`
	RequestID *string     `json:"request_id"`
}

type taskOutput struct {
	TaskID        string  `json:"task_id"`
	TaskStatus    string  `json:"task_status"`
	VideoURL      *string `json:"video_url"`
	SubmitTime    *string `json:"submit_time"`
	ScheduledTime *string `json:"scheduled_time"`
	EndTime       *string `json:"end_time"`
	OrigPrompt    *string `json:"orig_prompt"`
	ActualPrompt  *string `json:"actual_prompt"`
	Code          *string `json:"code"`
	Message       *string `json:"message"`
}

type taskUsage struct {
	Duration            *float64 `json:"duration"`
	OutputVideoDuration *float64 `json:"output_video_duration"`
	InputVideoDuration  *float64 `json:"input_video_duration"`
	FPS                 *float64 `json:"fps"`
	SR                  *float64 `json:"SR"`
	Size                *string  `json:"size"`
	Ratio               *string  `json:"ratio"`
}

// Start submits a generation task.
func (m *VideoModel) Start(ctx context.Context, options model.VideoStartOptions) (*model.VideoStartResult, error) {
	timestamp := time.Now()

	body, warnings, err := m.buildRequest(options.VideoOptions)
	if err != nil {
		return nil, err
	}

	asyncHeader := http.Header{
		"X-DashScope-Async": {
			"enable",
		},
	}

	request, err := m.provider.apiRequest(m.provider.videoBaseURL+videoSynthesisPath, parseNativeError, options.Headers, asyncHeader)
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

	if created.Output == nil || created.Output.TaskID == "" {
		return nil, fmt.Errorf("%w: no task_id returned from alibaba api. response: %s", ErrVideoGenerationFailed, response.Body)
	}

	operation, err := json.Marshal(VideoOperation{
		TaskID: created.Output.TaskID,
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

// Status checks a generation task. Failed and canceled tasks report an
// error status.
func (m *VideoModel) Status(ctx context.Context, options model.VideoStatusOptions) (*model.VideoStatusResult, error) {
	timestamp := time.Now()

	var operation VideoOperation
	if err := json.Unmarshal(options.Operation, &operation); err != nil {
		return nil, fmt.Errorf("%w: video operation: %w", model.ErrInvalidArgument, err)
	}

	request, err := m.provider.apiRequest(m.provider.videoBaseURL+videoTasksPath+url.PathEscape(operation.TaskID), parseNativeError, options.Headers)
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

	if status.Output == nil {
		return result, nil
	}

	switch status.Output.TaskStatus {
	case taskSucceeded:
		return completedResult(result, status)

	case taskFailed, taskCanceled:
		message := fmt.Sprintf(
			"Video generation %s. Task ID: %s. %s",
			strings.ToLower(status.Output.TaskStatus),
			operation.TaskID,
			valueOrEmpty(status.Output.Message),
		)

		result.Status = model.VideoError
		result.Error = strings.TrimSpace(message)

		return result, nil

	default:
		return result, nil
	}
}

// completedResult reports the generated video and its metadata.
func completedResult(result *model.VideoStatusResult, status taskStatusResponse) (*model.VideoStatusResult, error) {
	videoURL := valueOrEmpty(status.Output.VideoURL)
	if videoURL == "" {
		return nil, fmt.Errorf("%w: no video URL in response. task ID: %s", ErrVideoGenerationFailed, status.Output.TaskID)
	}

	metadata := VideoMetadata{
		TaskID:       status.Output.TaskID,
		VideoURL:     videoURL,
		ActualPrompt: valueOrEmpty(status.Output.ActualPrompt),
	}

	if usage := status.Usage; usage != nil {
		metadata.Usage = &VideoUsage{
			Duration:            usage.Duration,
			OutputVideoDuration: usage.OutputVideoDuration,
			Resolution:          usage.SR,
			Size:                usage.Size,
			InputVideoDuration:  usage.InputVideoDuration,
			FPS:                 usage.FPS,
			Ratio:               usage.Ratio,
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

func (m *VideoModel) responseInfo(timestamp time.Time, response *apicall.Response) model.ResponseInfo {
	return model.ResponseInfo{
		Timestamp: timestamp,
		ModelID:   m.modelID,
		Headers:   response.Headers,
		Body:      response.Body,
	}
}
