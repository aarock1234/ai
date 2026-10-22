package gateway

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// Video model endpoints.
const (
	videoModelPath       = "/video-model"
	videoModelStartPath  = "/video-model/start"
	videoModelStatusPath = "/video-model/status"
)

// maxVideosPerCall is large so that ai.GenerateVideo sends every video in
// one call; the gateway applies the underlying model's limit itself.
const maxVideosPerCall = 1 << 30

// Generation states of a status response.
const (
	videoStatusPending   = "pending"
	videoStatusCompleted = "completed"
	videoStatusError     = "error"
	videoStatusCancelled = "cancelled"
)

// Generated video data types.
const (
	videoDataURL    = "url"
	videoDataBase64 = "base64"
)

// Input file types.
const (
	videoFileBytes = "file"
	videoFileURL   = "url"
)

var (
	_ model.VideoModel          = (*VideoModel)(nil)
	_ model.VideoGenerator      = (*VideoModel)(nil)
	_ model.VideoOperator       = (*VideoModel)(nil)
	_ model.VideoWebhookHandler = (*VideoModel)(nil)
)

// VideoModel is an AI Gateway video model, such as "google/veo-3.1-generate-001".
// It generates in one streaming call, or through the start/status flow, in
// which the gateway notifies a webhook on completion. It is safe for
// concurrent use.
type VideoModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "gateway" by default.
func (m *VideoModel) Provider() string {
	return m.provider.name
}

// ModelID returns the model identifier.
func (m *VideoModel) ModelID() string {
	return m.modelID
}

// MaxVideosPerCall returns a very large limit, so that requests are not
// split before they reach the gateway.
func (m *VideoModel) MaxVideosPerCall() int {
	return maxVideosPerCall
}

type videoRequest struct {
	Prompt          string                `json:"prompt,omitempty"`
	N               int                   `json:"n"`
	AspectRatio     string                `json:"aspectRatio,omitempty"`
	Resolution      string                `json:"resolution,omitempty"`
	Duration        *float64              `json:"duration,omitempty"`
	FPS             int                   `json:"fps,omitempty"`
	Seed            *int                  `json:"seed,omitempty"`
	GenerateAudio   *bool                 `json:"generateAudio,omitempty"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
	Image           *videoFile            `json:"image,omitempty"`
	FrameImages     []videoFrameImage     `json:"frameImages,omitzero"`
	InputReferences []videoFile           `json:"inputReferences,omitzero"`

	// CallbackURL is the completion webhook of a start request.
	CallbackURL string `json:"callbackUrl,omitempty"`
}

// videoFile is an input file on the wire: base64 data or a URL.
type videoFile struct {
	Type            string                `json:"type"`
	MediaType       string                `json:"mediaType,omitempty"`
	Data            string                `json:"data,omitempty"`
	URL             string                `json:"url,omitempty"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

type videoFrameImage struct {
	FrameType model.VideoFrameType `json:"frameType"`
	Image     videoFile            `json:"image"`
}

type videoStatusRequest struct {
	Operation json.RawMessage `json:"operation"`
}

// wireVideo is a generated video on the wire: a URL or base64 data.
type wireVideo struct {
	Type      string `json:"type"`
	URL       string `json:"url"`
	Data      string `json:"data"`
	MediaType string `json:"mediaType"`
}

type videoStartResponse struct {
	Operation        json.RawMessage `json:"operation"`
	Warnings         []wireWarning   `json:"warnings"`
	ProviderMetadata wireMetadata    `json:"providerMetadata"`
}

type videoStatusResponse struct {
	Status           string        `json:"status"`
	Videos           []wireVideo   `json:"videos"`
	Error            *string       `json:"error"`
	Warnings         []wireWarning `json:"warnings"`
	ProviderMetadata wireMetadata  `json:"providerMetadata"`
}

// HandleWebhook creates the endpoint that the gateway notifies on
// completion.
func (m *VideoModel) HandleWebhook(ctx context.Context, factory model.VideoWebhookFactory) (*model.VideoWebhook, error) {
	if factory == nil {
		return nil, fmt.Errorf("%w: video webhook factory is nil", model.ErrInvalidArgument)
	}

	return factory(ctx)
}

// Start submits a generation, with the completion webhook when one is
// given. The operation reference is the gateway's.
func (m *VideoModel) Start(ctx context.Context, options model.VideoStartOptions) (*model.VideoStartResult, error) {
	body, err := videoBody(options.VideoOptions)
	if err != nil {
		return nil, err
	}

	body.CallbackURL = options.WebhookURL

	request, method, err := m.provider.modelRequest(videoModelStartPath, options.Headers, modelConfigHeaders("video", m.modelID))
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, asGatewayError(err, method)
	}

	var started videoStartResponse
	if err := apicall.DecodeJSON(response.Body, &started); err != nil {
		return nil, err
	}

	return &model.VideoStartResult{
		Operation:        started.Operation,
		ProviderMetadata: started.ProviderMetadata.providerMetadata(),
		Warnings:         decodeWarnings(started.Warnings),
		Response:         m.responseInfo(response),
	}, nil
}

// Status checks a generation. A cancelled generation reports an error
// status.
func (m *VideoModel) Status(ctx context.Context, options model.VideoStatusOptions) (*model.VideoStatusResult, error) {
	request, method, err := m.provider.modelRequest(videoModelStatusPath, options.Headers, modelConfigHeaders("video", m.modelID))
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, videoStatusRequest{
		Operation: options.Operation,
	})
	if err != nil {
		return nil, asGatewayError(err, method)
	}

	var status videoStatusResponse
	if err := apicall.DecodeJSON(response.Body, &status); err != nil {
		return nil, err
	}

	return m.statusResult(status, m.responseInfo(response))
}

// statusResult converts a status response by its state.
func (m *VideoModel) statusResult(status videoStatusResponse, response model.ResponseInfo) (*model.VideoStatusResult, error) {
	result := &model.VideoStatusResult{
		ProviderMetadata: status.ProviderMetadata.providerMetadata(),
		Response:         response,
	}

	switch status.Status {
	case videoStatusPending:
		result.Status = model.VideoPending
		result.Warnings = decodeWarnings(status.Warnings)

	case videoStatusCompleted:
		videos, err := decodeVideos(status.Videos)
		if err != nil {
			return nil, err
		}

		result.Status = model.VideoCompleted
		result.Videos = videos
		result.Warnings = decodeWarnings(status.Warnings)

	case videoStatusError:
		if status.Error == nil {
			return nil, fmt.Errorf("%w: gateway video error status has no error", model.ErrInvalidResponse)
		}

		result.Status = model.VideoError
		result.Error = *status.Error

	case videoStatusCancelled:
		result.Status = model.VideoError
		result.Error = "Video generation was cancelled."

	default:
		return nil, fmt.Errorf("%w: unknown gateway video status %q", model.ErrInvalidResponse, status.Status)
	}

	return result, nil
}

// videoBody builds the request. A zero duration or seed is omitted, as
// upstream does; inline files travel as base64.
func videoBody(options model.VideoOptions) (videoRequest, error) {
	body := videoRequest{
		Prompt:          options.Prompt,
		N:               options.N,
		AspectRatio:     options.AspectRatio,
		Resolution:      options.Resolution,
		FPS:             options.FPS,
		GenerateAudio:   options.GenerateAudio,
		ProviderOptions: options.ProviderOptions,
	}

	if options.Duration != nil && *options.Duration != 0 {
		body.Duration = options.Duration
	}

	if options.Seed != nil && *options.Seed != 0 {
		body.Seed = options.Seed
	}

	if options.Image != nil {
		image, err := encodeVideoFile(options.Image)
		if err != nil {
			return videoRequest{}, err
		}

		body.Image = &image
	}

	frameImages, err := encodeFrameImages(options.FrameImages)
	if err != nil {
		return videoRequest{}, err
	}

	inputReferences, err := encodeVideoFiles(options.InputReferences)
	if err != nil {
		return videoRequest{}, err
	}

	body.FrameImages = frameImages
	body.InputReferences = inputReferences

	return body, nil
}

func encodeFrameImages(frames []model.VideoFrameImage) ([]videoFrameImage, error) {
	if frames == nil {
		return nil, nil
	}

	encoded := make([]videoFrameImage, 0, len(frames))

	for _, frame := range frames {
		image, err := encodeVideoFile(frame.Image)
		if err != nil {
			return nil, err
		}

		encoded = append(encoded, videoFrameImage{
			FrameType: frame.FrameType,
			Image:     image,
		})
	}

	return encoded, nil
}

func encodeVideoFiles(files []model.VideoFile) ([]videoFile, error) {
	if files == nil {
		return nil, nil
	}

	encoded := make([]videoFile, 0, len(files))

	for _, file := range files {
		converted, err := encodeVideoFile(file)
		if err != nil {
			return nil, err
		}

		encoded = append(encoded, converted)
	}

	return encoded, nil
}

func encodeVideoFile(file model.VideoFile) (videoFile, error) {
	switch file := file.(type) {
	case model.VideoBytes:
		return videoFile{
			Type:            videoFileBytes,
			MediaType:       file.MediaType,
			Data:            base64.StdEncoding.EncodeToString(file.Data),
			ProviderOptions: file.ProviderOptions,
		}, nil

	case model.VideoURL:
		return videoFile{
			Type:            videoFileURL,
			URL:             file.URL,
			MediaType:       file.MediaType,
			ProviderOptions: file.ProviderOptions,
		}, nil

	default:
		return videoFile{}, fmt.Errorf("%w: video file %T", model.ErrUnsupported, file)
	}
}

// decodeVideos converts generated videos. Base64 videos are decoded.
func decodeVideos(videos []wireVideo) ([]model.VideoData, error) {
	decoded := make([]model.VideoData, 0, len(videos))

	for _, video := range videos {
		switch video.Type {
		case videoDataURL:
			decoded = append(decoded, model.VideoDataURL{
				URL:       video.URL,
				MediaType: video.MediaType,
			})

		case videoDataBase64:
			data, err := base64.StdEncoding.DecodeString(video.Data)
			if err != nil {
				return nil, fmt.Errorf("%w: decode base64 video: %w", model.ErrInvalidResponse, err)
			}

			decoded = append(decoded, model.VideoDataBytes{
				Data:      data,
				MediaType: video.MediaType,
			})

		default:
			return nil, fmt.Errorf("%w: unsupported gateway video data type %q", model.ErrInvalidResponse, video.Type)
		}
	}

	return decoded, nil
}

func (m *VideoModel) responseInfo(response *apicall.Response) model.ResponseInfo {
	return model.ResponseInfo{
		Timestamp: time.Now(),
		ModelID:   m.modelID,
		Headers:   response.Headers,
		Body:      response.Body,
	}
}
