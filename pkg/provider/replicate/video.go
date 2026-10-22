package replicate

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// generatedVideoMediaType is the media type of generated videos.
const generatedVideoMediaType = "video/mp4"

// webhookCompletedEvent asks Replicate to notify the webhook only when the
// prediction completes.
const webhookCompletedEvent = "completed"

// ErrVideoGenerationFailed reports a succeeded prediction without a video
// URL.
var ErrVideoGenerationFailed = errors.New("replicate video generation failed")

// videoOptionKeys are the members that are not passed through: the polling
// settings, and the VideoOptions members mapped to input fields.
var videoOptionKeys = map[string]bool{
	"pollIntervalMs":       true,
	"pollTimeoutMs":        true,
	"maxWaitTimeInSeconds": true,
	"guidance_scale":       true,
	"num_inference_steps":  true,
	"motion_bucket_id":     true,
	"cond_aug":             true,
	"decoding_t":           true,
	"video_length":         true,
	"sizing_strategy":      true,
	"frames_per_second":    true,
	"prompt_optimizer":     true,
}

var (
	_ model.VideoModel          = (*VideoModel)(nil)
	_ model.VideoOperator       = (*VideoModel)(nil)
	_ model.VideoWebhookHandler = (*VideoModel)(nil)
)

// VideoOptions are the provider options of video models, read from the
// "replicate" namespace. Models accept different inputs: members other than
// the polling settings are sent in the prediction input, including
// model-specific members that are not fields here.
type VideoOptions struct {
	// PollIntervalMs, PollTimeoutMs and MaxWaitTimeInSeconds are accepted
	// for compatibility and ignored; ai.VideoPollOptions controls polling.
	PollIntervalMs       *float64 `json:"pollIntervalMs,omitempty"`
	PollTimeoutMs        *float64 `json:"pollTimeoutMs,omitempty"`
	MaxWaitTimeInSeconds *float64 `json:"maxWaitTimeInSeconds,omitempty"`

	GuidanceScale     *float64 `json:"guidance_scale,omitempty"`
	NumInferenceSteps *float64 `json:"num_inference_steps,omitempty"`

	// MotionBucketID, CondAug, DecodingT, VideoLength, SizingStrategy and
	// FramesPerSecond are Stable Video Diffusion inputs.
	MotionBucketID  *float64 `json:"motion_bucket_id,omitempty"`
	CondAug         *float64 `json:"cond_aug,omitempty"`
	DecodingT       *float64 `json:"decoding_t,omitempty"`
	VideoLength     *string  `json:"video_length,omitempty"`
	SizingStrategy  *string  `json:"sizing_strategy,omitempty"`
	FramesPerSecond *float64 `json:"frames_per_second,omitempty"`

	// PromptOptimizer is a MiniMax input.
	PromptOptimizer *bool `json:"prompt_optimizer,omitempty"`
}

// VideoModel is a Replicate video model. Generation is asynchronous: Start
// creates a prediction and Status checks it. Replicate notifies a webhook
// when the prediction completes.
type VideoModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "replicate.video" by default.
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

// VideoOperation identifies a prediction. It is the JSON operation
// reference of Start and Status.
type VideoOperation struct {
	GetURL string `json:"getUrl"`
}

// VideoMetadata is the provider metadata of a completed prediction, stored
// under the "replicate" namespace.
type VideoMetadata struct {
	Videos       []GeneratedVideoMetadata `json:"videos"`
	PredictionID string                   `json:"predictionId"`
	Metrics      *PredictionMetrics       `json:"metrics,omitempty"`
}

// GeneratedVideoMetadata describes one generated video.
type GeneratedVideoMetadata struct {
	URL string `json:"url"`
}

// PredictionMetrics are the reported metrics of a prediction.
type PredictionMetrics struct {
	PredictTime *float64 `json:"predict_time,omitempty"`
}

type videoPredictionRequest struct {
	Input               jsonobject.Object `json:"input"`
	Version             string            `json:"version,omitempty"`
	Webhook             string            `json:"webhook,omitempty"`
	WebhookEventsFilter []string          `json:"webhook_events_filter,omitempty"`
}

// videoInput holds the standard input fields and the mapped options. A zero
// duration, FPS or seed is omitted.
type videoInput struct {
	Prompt            string   `json:"prompt,omitempty"`
	Image             string   `json:"image,omitempty"`
	AspectRatio       string   `json:"aspect_ratio,omitempty"`
	Size              string   `json:"size,omitempty"`
	Duration          *float64 `json:"duration,omitempty"`
	FPS               int      `json:"fps,omitempty"`
	Seed              *int     `json:"seed,omitempty"`
	GuidanceScale     *float64 `json:"guidance_scale,omitempty"`
	NumInferenceSteps *float64 `json:"num_inference_steps,omitempty"`
	MotionBucketID    *float64 `json:"motion_bucket_id,omitempty"`
	CondAug           *float64 `json:"cond_aug,omitempty"`
	DecodingT         *float64 `json:"decoding_t,omitempty"`
	VideoLength       *string  `json:"video_length,omitempty"`
	SizingStrategy    *string  `json:"sizing_strategy,omitempty"`
	FramesPerSecond   *float64 `json:"frames_per_second,omitempty"`
	PromptOptimizer   *bool    `json:"prompt_optimizer,omitempty"`
}

type videoPrediction struct {
	ID      string             `json:"id"`
	Status  predictionStatus   `json:"status"`
	Output  *string            `json:"output"`
	Error   *string            `json:"error"`
	URLs    predictionURLs     `json:"urls"`
	Metrics *PredictionMetrics `json:"metrics"`
}

// HandleWebhook creates the endpoint that Replicate notifies on completion.
func (m *VideoModel) HandleWebhook(ctx context.Context, factory model.VideoWebhookFactory) (*model.VideoWebhook, error) {
	if factory == nil {
		return nil, fmt.Errorf("%w: video webhook factory is nil", model.ErrInvalidArgument)
	}

	return factory(ctx)
}

// Start creates a prediction, with the webhook URL when one is given.
func (m *VideoModel) Start(ctx context.Context, options model.VideoStartOptions) (*model.VideoStartResult, error) {
	timestamp := time.Now()

	input, err := videoPredictionInput(options.VideoOptions)
	if err != nil {
		return nil, err
	}

	headers, err := m.provider.authHeaders(options.Headers)
	if err != nil {
		return nil, err
	}

	modelID, version, _ := strings.Cut(m.modelID, ":")

	endpoint := m.provider.baseURL + "/models/" + modelID + "/predictions"
	if version != "" {
		endpoint = m.provider.baseURL + "/predictions"
	}

	body := videoPredictionRequest{
		Input:   input,
		Version: version,
	}

	if options.WebhookURL != "" {
		body.Webhook = options.WebhookURL
		body.WebhookEventsFilter = []string{
			webhookCompletedEvent,
		}
	}

	response, err := apicall.PostJSON(ctx, m.provider.apiRequest(endpoint, headers), body)
	if err != nil {
		return nil, err
	}

	var created videoPrediction
	if err := apicall.DecodeJSON(response.Body, &created); err != nil {
		return nil, err
	}

	if created.URLs.Get == "" {
		return nil, fmt.Errorf("%w: replicate prediction has no get URL", model.ErrInvalidResponse)
	}

	operation, err := json.Marshal(VideoOperation{
		GetURL: created.URLs.Get,
	})
	if err != nil {
		return nil, fmt.Errorf("encode video operation: %w", err)
	}

	return &model.VideoStartResult{
		Operation: operation,
		Response:  m.responseInfo(timestamp, response.Headers, response.Body),
	}, nil
}

// Status checks a prediction. Failed and canceled predictions report an
// error status.
func (m *VideoModel) Status(ctx context.Context, options model.VideoStatusOptions) (*model.VideoStatusResult, error) {
	timestamp := time.Now()

	var operation VideoOperation
	if err := json.Unmarshal(options.Operation, &operation); err != nil {
		return nil, fmt.Errorf("%w: video operation: %w", model.ErrInvalidArgument, err)
	}

	headers, err := m.provider.authHeaders(options.Headers)
	if err != nil {
		return nil, err
	}

	responseHeaders, body, err := m.provider.fetchStatus(ctx, operation.GetURL, headers)
	if err != nil {
		return nil, err
	}

	var current videoPrediction
	if err := apicall.DecodeJSON(body, &current); err != nil {
		return nil, err
	}

	result := &model.VideoStatusResult{
		Status:   model.VideoPending,
		Response: m.responseInfo(timestamp, responseHeaders, body),
	}

	switch current.Status {
	case statusFailed:
		reason := "Unknown error"
		if current.Error != nil {
			reason = *current.Error
		}

		result.Status = model.VideoError
		result.Error = "Video generation failed: " + reason

		return result, nil

	case statusCanceled:
		result.Status = model.VideoError
		result.Error = "Video generation was canceled"

		return result, nil

	case statusSucceeded:
		return completedVideo(result, current)

	case statusStarting, statusProcessing:
		return result, nil

	default:
		return nil, fmt.Errorf("%w: unknown prediction status %q", model.ErrInvalidResponse, current.Status)
	}
}

// completedVideo reports the output video and the prediction metadata.
func completedVideo(result *model.VideoStatusResult, current videoPrediction) (*model.VideoStatusResult, error) {
	if current.Output == nil || *current.Output == "" {
		return nil, fmt.Errorf("%w: no video URL in response", ErrVideoGenerationFailed)
	}

	result.Status = model.VideoCompleted
	result.Videos = []model.VideoData{
		model.VideoDataURL{
			URL:       *current.Output,
			MediaType: generatedVideoMediaType,
		},
	}
	result.ProviderMetadata = model.ProviderMetadata{
		Namespace: VideoMetadata{
			Videos: []GeneratedVideoMetadata{
				{
					URL: *current.Output,
				},
			},
			PredictionID: current.ID,
			Metrics:      current.Metrics,
		},
	}

	return result, nil
}

// videoPredictionInput combines the standard fields and the provider
// options, whose model-specific members are passed through.
func videoPredictionInput(options model.VideoOptions) (jsonobject.Object, error) {
	videoOptions, passthrough, err := readVideoOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	input := videoInput{
		Prompt:            options.Prompt,
		AspectRatio:       options.AspectRatio,
		Size:              options.Resolution,
		FPS:               options.FPS,
		GuidanceScale:     videoOptions.GuidanceScale,
		NumInferenceSteps: videoOptions.NumInferenceSteps,
		MotionBucketID:    videoOptions.MotionBucketID,
		CondAug:           videoOptions.CondAug,
		DecodingT:         videoOptions.DecodingT,
		VideoLength:       videoOptions.VideoLength,
		SizingStrategy:    videoOptions.SizingStrategy,
		FramesPerSecond:   videoOptions.FramesPerSecond,
		PromptOptimizer:   videoOptions.PromptOptimizer,
	}

	if options.Image != nil {
		image, err := dataurl.VideoFileURL(options.Image)
		if err != nil {
			return nil, err
		}

		input.Image = image
	}

	if options.Duration != nil && *options.Duration != 0 {
		input.Duration = options.Duration
	}

	if options.Seed != nil && *options.Seed != 0 {
		input.Seed = options.Seed
	}

	encoded, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("encode prediction input: %w", err)
	}

	object, err := jsonobject.Parse(encoded)
	if err != nil {
		return nil, fmt.Errorf("encode prediction input: %w", err)
	}

	return object.Merge(passthrough), nil
}

// readVideoOptions reads and validates the call's Replicate options and
// returns the members to pass through.
func readVideoOptions(options model.ProviderOptions) (VideoOptions, jsonobject.Object, error) {
	videoOptions, _, err := model.ProviderValue[VideoOptions](options, Namespace)
	if err != nil {
		return VideoOptions{}, nil, err
	}

	if err := videoOptions.validate(); err != nil {
		return VideoOptions{}, nil, err
	}

	raw, _, err := model.ProviderValue[jsonobject.Object](options, Namespace)
	if err != nil {
		return VideoOptions{}, nil, err
	}

	var passthrough jsonobject.Object
	for _, member := range raw {
		if !videoOptionKeys[member.Key] {
			passthrough = append(passthrough, member)
		}
	}

	return videoOptions, passthrough, nil
}

func (o VideoOptions) validate() error {
	settings := []struct {
		name  string
		value *float64
	}{
		{
			name:  "pollIntervalMs",
			value: o.PollIntervalMs,
		},
		{
			name:  "pollTimeoutMs",
			value: o.PollTimeoutMs,
		},
		{
			name:  "maxWaitTimeInSeconds",
			value: o.MaxWaitTimeInSeconds,
		},
	}

	for _, setting := range settings {
		if setting.value != nil && *setting.value <= 0 {
			return fmt.Errorf("%w: replicate %s must be positive", model.ErrInvalidArgument, setting.name)
		}
	}

	return nil
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
