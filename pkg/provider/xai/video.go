package xai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/optional"
	"github.com/aarock1234/ai/pkg/model"
)

// Request statuses of video generation. Others, such as pending, keep the
// generation pending.
const (
	videoDone    = "done"
	videoFailed  = "failed"
	videoExpired = "expired"
)

// generatedVideoMediaType is the media type of generated videos.
const generatedVideoMediaType = "video/mp4"

// ErrVideoGenerationFailed reports a start without a request ID.
var ErrVideoGenerationFailed = errors.New("xai video generation failed")

var (
	_ model.VideoModel    = (*VideoModel)(nil)
	_ model.VideoOperator = (*VideoModel)(nil)
	_ model.VideoProvider = (*Provider)(nil)
)

// VideoModel is an xAI video model, such as "grok-imagine-video" or
// "grok-imagine-video-1.5". It generates, edits and extends videos:
// VideoOptions.Mode selects the operation. Generation is asynchronous:
// Start submits a request and Status checks it.
type VideoModel struct {
	modelID  string
	provider *Provider
}

// VideoOperation identifies a submitted request. It is the JSON operation
// reference of Start and Status.
type VideoOperation struct {
	RequestID string `json:"requestId"`
}

// VideoMetadata is the provider metadata of a completed generation, stored
// under the "xai" namespace. Nil fields were not reported.
type VideoMetadata struct {
	RequestID      string           `json:"requestId"`
	VideoURL       string           `json:"videoUrl"`
	Duration       *float64         `json:"duration,omitempty"`
	CostInUsdTicks *float64         `json:"costInUsdTicks,omitempty"`
	Progress       *float64         `json:"progress,omitempty"`
	FileOutput     *VideoFileOutput `json:"fileOutput,omitempty"`
	StorageError   *string          `json:"storageError,omitempty"`
}

// VideoFileOutput describes a video stored in the Files API with
// VideoOptions.StorageOptions.
type VideoFileOutput struct {
	FileID             string   `json:"fileId"`
	Filename           string   `json:"filename"`
	ExpiresAt          *float64 `json:"expiresAt,omitempty"`
	PublicURL          *string  `json:"publicUrl,omitempty"`
	PublicURLError     *string  `json:"publicUrlError,omitempty"`
	PublicURLExpiresAt *float64 `json:"publicUrlExpiresAt,omitempty"`
}

type createVideoResponse struct {
	RequestID *string `json:"request_id"`
}

type videoStatusResponse struct {
	Status   *string           `json:"status"`
	Video    *videoStatusVideo `json:"video"`
	Model    *string           `json:"model"`
	Usage    *videoStatusUsage `json:"usage"`
	Progress *float64          `json:"progress"`
	Error    *videoStatusError `json:"error"`
}

type videoStatusVideo struct {
	URL               *string          `json:"url"`
	Duration          *float64         `json:"duration"`
	RespectModeration *bool            `json:"respect_moderation"`
	FileOutput        *videoFileOutput `json:"file_output"`
	StorageError      *string          `json:"storage_error"`
}

type videoFileOutput struct {
	FileID             string   `json:"file_id"`
	Filename           string   `json:"filename"`
	ExpiresAt          *float64 `json:"expires_at"`
	PublicURL          *string  `json:"public_url"`
	PublicURLError     *string  `json:"public_url_error"`
	PublicURLExpiresAt *float64 `json:"public_url_expires_at"`
}

type videoStatusUsage struct {
	CostInUsdTicks *float64 `json:"cost_in_usd_ticks"`
}

type videoStatusError struct {
	Code    *string `json:"code"`
	Message *string `json:"message"`
}

// Video returns a video model.
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

// Provider returns the provider identifier, "xai.video" by default.
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

// Start submits a generation, edit or extension.
func (m *VideoModel) Start(ctx context.Context, options model.VideoStartOptions) (*model.VideoStartResult, error) {
	timestamp := time.Now()

	plan, warnings, err := buildVideoPlan(m.modelID, options.VideoOptions)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(plan.path, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, plan.body)
	if err != nil {
		return nil, err
	}

	var created createVideoResponse
	if err := apicall.DecodeJSON(response.Body, &created); err != nil {
		return nil, err
	}

	if created.RequestID == nil || *created.RequestID == "" {
		return nil, fmt.Errorf("%w: no request_id returned from xAI API", ErrVideoGenerationFailed)
	}

	operation, err := json.Marshal(VideoOperation{
		RequestID: *created.RequestID,
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

// Status checks a request. Expired, failed and moderated requests, and done
// requests without a video URL, report an error status. xAI answers
// running requests with an empty body, which is pending.
func (m *VideoModel) Status(ctx context.Context, options model.VideoStatusOptions) (*model.VideoStatusResult, error) {
	timestamp := time.Now()

	var operation VideoOperation
	if err := json.Unmarshal(options.Operation, &operation); err != nil {
		return nil, fmt.Errorf("%w: video operation: %w", model.ErrInvalidArgument, err)
	}

	request, err := m.provider.apiRequest("/videos/"+encodePathSegment(operation.RequestID), options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.Get(ctx, request)
	if err != nil {
		return nil, err
	}

	result := &model.VideoStatusResult{
		Status:   model.VideoPending,
		Response: m.responseInfo(timestamp, response),
	}

	if len(bytes.TrimSpace(response.Body)) == 0 {
		return result, nil
	}

	var status videoStatusResponse
	if err := apicall.DecodeJSON(response.Body, &status); err != nil {
		return nil, err
	}

	state := optional.Or(status.Status, "")

	switch {
	case state == videoExpired:
		return failedVideo(result, "Video generation request expired."), nil

	case state == videoFailed:
		return failedVideo(result, failureMessage(status.Error)), nil

	case state == videoDone, state == "" && status.Video != nil && optional.Or(status.Video.URL, "") != "":
		return completedVideo(result, operation, status), nil

	default:
		return result, nil
	}
}

// completedVideo reports the generated video. A moderated video, or one
// without a URL, is an error.
func completedVideo(result *model.VideoStatusResult, operation VideoOperation, status videoStatusResponse) *model.VideoStatusResult {
	video := status.Video
	if video == nil {
		video = &videoStatusVideo{}
	}

	if video.RespectModeration != nil && !*video.RespectModeration {
		return failedVideo(result, "Video generation was blocked due to a content policy violation.")
	}

	videoURL := optional.Or(video.URL, "")
	if videoURL == "" && video.FileOutput != nil {
		videoURL = optional.Or(video.FileOutput.PublicURL, "")
	}

	if videoURL == "" {
		return failedVideo(result, "Video generation completed but no video URL was returned.")
	}

	metadata := VideoMetadata{
		RequestID:    operation.RequestID,
		VideoURL:     videoURL,
		Duration:     video.Duration,
		Progress:     status.Progress,
		FileOutput:   fileOutputMetadata(video.FileOutput),
		StorageError: video.StorageError,
	}

	if status.Usage != nil {
		metadata.CostInUsdTicks = status.Usage.CostInUsdTicks
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

	return result
}

func fileOutputMetadata(output *videoFileOutput) *VideoFileOutput {
	if output == nil {
		return nil
	}

	return &VideoFileOutput{
		FileID:             output.FileID,
		Filename:           output.Filename,
		ExpiresAt:          output.ExpiresAt,
		PublicURL:          output.PublicURL,
		PublicURLError:     output.PublicURLError,
		PublicURLExpiresAt: output.PublicURLExpiresAt,
	}
}

func failedVideo(result *model.VideoStatusResult, message string) *model.VideoStatusResult {
	result.Status = model.VideoError
	result.Error = message

	return result
}

// failureMessage describes a failed generation with its message or code.
func failureMessage(statusError *videoStatusError) string {
	switch {
	case statusError != nil && statusError.Message != nil:
		return "Video generation failed: " + *statusError.Message

	case statusError != nil && statusError.Code != nil:
		return "Video generation failed: " + *statusError.Code

	default:
		return "Video generation failed."
	}
}

func (m *VideoModel) responseInfo(timestamp time.Time, response *apicall.Response) model.ResponseInfo {
	return model.ResponseInfo{
		Timestamp: timestamp,
		ModelID:   m.modelID,
		Headers:   response.Headers,
		Body:      response.Body,
	}
}
