package heygen

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// videosPath is the video generation endpoint.
const videosPath = "/v3/models/videos"

// generatedVideoMediaType is the media type of generated videos.
const generatedVideoMediaType = "video/mp4"

// Video statuses.
const (
	statusPending    = "pending"
	statusProcessing = "processing"
	statusCompleted  = "completed"
	statusFailed     = "failed"
	statusCancelled  = "cancelled"
)

var (
	_ model.VideoModel    = (*VideoModel)(nil)
	_ model.VideoOperator = (*VideoModel)(nil)
)

// VideoModel is a HeyGen video model. Generation is asynchronous: Start
// submits a video and Status checks it; ai.GenerateVideo polls until the
// video is ready.
type VideoModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "heygen.video" by default.
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

// VideoOperation identifies a submitted video. It is the JSON operation
// reference of Start and Status, and the provider metadata of Start under
// the "heygen" namespace.
type VideoOperation struct {
	VideoID    string          `json:"videoId"`
	Mode       VideoMode       `json:"mode"`
	Resolution VideoResolution `json:"resolution"`
}

// GeneratedVideoMetadata describes one video in the provider metadata of
// Status. A pending or failed status stores it under the "heygen"
// namespace; a completed status stores VideoMetadata.
type GeneratedVideoMetadata struct {
	VideoOperation

	Duration    *float64      `json:"duration,omitempty"`
	Width       *float64      `json:"width,omitempty"`
	Height      *float64      `json:"height,omitempty"`
	AspectRatio *string       `json:"aspectRatio,omitempty"`
	Seed        *float64      `json:"seed,omitempty"`
	Timings     *VideoTimings `json:"timings,omitempty"`
	FailureCode *string       `json:"failureCode,omitempty"`
}

// VideoTimings are the reported generation times.
type VideoTimings struct {
	Inference float64 `json:"inference"`
}

// VideoMetadata is the provider metadata of a completed video, stored under
// the "heygen" namespace. ai.GenerateVideo concatenates the videos of
// several calls.
type VideoMetadata struct {
	Videos []GeneratedVideoMetadata `json:"videos"`
}

type createVideoResponse struct {
	Data struct {
		VideoID string `json:"video_id"`
	} `json:"data"`
}

type videoStatusResponse struct {
	Data *videoStatusData `json:"data"`
}

type videoStatusData struct {
	Status         string   `json:"status"`
	VideoURL       *string  `json:"video_url"`
	Duration       *float64 `json:"duration"`
	Width          *float64 `json:"width"`
	Height         *float64 `json:"height"`
	AspectRatio    *string  `json:"aspect_ratio"`
	Seed           *float64 `json:"seed"`
	FailureCode    *string  `json:"failure_code"`
	FailureMessage *string  `json:"failure_message"`
	Timings        *struct {
		Inference *float64 `json:"inference"`
	} `json:"timings"`
}

// Start validates the call and submits a video.
func (m *VideoModel) Start(ctx context.Context, options model.VideoStartOptions) (*model.VideoStartResult, error) {
	timestamp := time.Now()

	body, warnings, err := m.buildRequest(options.VideoOptions)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(videosPath, options.Headers)
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

	if created.Data.VideoID == "" {
		return nil, fmt.Errorf("%w: heygen response has no video_id", model.ErrInvalidResponse)
	}

	operation := VideoOperation{
		VideoID:    created.Data.VideoID,
		Mode:       body.Mode,
		Resolution: body.Resolution,
	}

	reference, err := json.Marshal(operation)
	if err != nil {
		return nil, fmt.Errorf("encode video operation: %w", err)
	}

	return &model.VideoStartResult{
		Operation: reference,
		Warnings:  warnings,
		ProviderMetadata: model.ProviderMetadata{
			Namespace: operation,
		},
		Response: m.responseInfo(timestamp, response),
	}, nil
}

// Status checks a video. Failed and cancelled videos report an error
// status; an unknown status is an error wrapping model.ErrInvalidResponse.
func (m *VideoModel) Status(ctx context.Context, options model.VideoStatusOptions) (*model.VideoStatusResult, error) {
	operation, err := decodeOperation(options.Operation)
	if err != nil {
		return nil, err
	}

	timestamp := time.Now()

	request, err := m.provider.apiRequest(videosPath+"/"+url.PathEscape(operation.VideoID), options.Headers)
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

	if status.Data == nil {
		return nil, fmt.Errorf("%w: heygen status response has no data", model.ErrInvalidResponse)
	}

	return statusResult(operation, *status.Data, m.responseInfo(timestamp, response))
}

// statusResult converts the reported status.
func statusResult(operation VideoOperation, data videoStatusData, response model.ResponseInfo) (*model.VideoStatusResult, error) {
	metadata := videoMetadata(operation, data)

	switch data.Status {
	case statusPending, statusProcessing:
		return &model.VideoStatusResult{
			Status: model.VideoPending,
			ProviderMetadata: model.ProviderMetadata{
				Namespace: metadata,
			},
			Response: response,
		}, nil

	case statusFailed, statusCancelled:
		return &model.VideoStatusResult{
			Status: model.VideoError,
			Error:  failureMessage(data),
			ProviderMetadata: model.ProviderMetadata{
				Namespace: metadata,
			},
			Response: response,
		}, nil
	}

	if data.Status != statusCompleted || data.VideoURL == nil || *data.VideoURL == "" {
		return nil, fmt.Errorf(
			"%w: HeyGen returned an unexpected video status or a completed video without a URL: %s",
			model.ErrInvalidResponse,
			response.Body,
		)
	}

	return &model.VideoStatusResult{
		Status: model.VideoCompleted,
		Videos: []model.VideoData{
			model.VideoDataURL{
				URL:       *data.VideoURL,
				MediaType: generatedVideoMediaType,
			},
		},
		ProviderMetadata: model.ProviderMetadata{
			Namespace: VideoMetadata{
				Videos: []GeneratedVideoMetadata{
					metadata,
				},
			},
		},
		Response: response,
	}, nil
}

func videoMetadata(operation VideoOperation, data videoStatusData) GeneratedVideoMetadata {
	metadata := GeneratedVideoMetadata{
		VideoOperation: operation,
		Duration:       data.Duration,
		Width:          data.Width,
		Height:         data.Height,
		AspectRatio:    data.AspectRatio,
		Seed:           data.Seed,
		FailureCode:    data.FailureCode,
	}

	if data.Timings != nil && data.Timings.Inference != nil {
		metadata.Timings = &VideoTimings{
			Inference: *data.Timings.Inference,
		}
	}

	return metadata
}

// failureMessage returns the reported failure message, or a description of
// the status, followed by the failure code.
func failureMessage(data videoStatusData) string {
	message := "HeyGen video generation " + data.Status
	if data.FailureMessage != nil {
		message = *data.FailureMessage
	}

	if data.FailureCode != nil {
		message += " (" + *data.FailureCode + ")"
	}

	return message
}

// decodeOperation reads and validates an operation reference.
func decodeOperation(raw json.RawMessage) (VideoOperation, error) {
	var operation VideoOperation
	if err := json.Unmarshal(raw, &operation); err != nil {
		return VideoOperation{}, fmt.Errorf("%w: video operation: %w", model.ErrInvalidArgument, err)
	}

	if operation.VideoID == "" || !validMode(operation.Mode) || !validResolution(operation.Resolution) {
		return VideoOperation{}, fmt.Errorf("%w: video operation requires a videoId, mode and resolution", model.ErrInvalidArgument)
	}

	return operation, nil
}

func (m *VideoModel) responseInfo(timestamp time.Time, response *apicall.Response) model.ResponseInfo {
	return model.ResponseInfo{
		Timestamp: timestamp,
		ModelID:   m.modelID,
		Headers:   response.Headers,
		Body:      response.Body,
	}
}
