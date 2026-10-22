package fal

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/internal/download"
	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/internal/origin"
	"github.com/aarock1234/ai/pkg/model"
)

// generatedVideoMediaType is the media type of videos whose content type is
// not reported.
const generatedVideoMediaType = "video/mp4"

// webhookQueryParameter carries the webhook URL of a queued request.
const webhookQueryParameter = "fal_webhook"

// ErrVideoGenerationFailed reports a queue submission without a response
// URL, or a completed request without a video.
var ErrVideoGenerationFailed = errors.New("fal video generation failed")

var (
	_ model.VideoModel          = (*VideoModel)(nil)
	_ model.VideoOperator       = (*VideoModel)(nil)
	_ model.VideoWebhookHandler = (*VideoModel)(nil)
)

// VideoModel is a fal.ai video model. Generation is asynchronous: Start
// queues a request and Status checks it. The queue notifies a webhook when
// the request completes.
type VideoModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "fal.video" by default.
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

// VideoOperation identifies a queued request. It is the JSON operation
// reference of Start and Status.
type VideoOperation struct {
	ResponseURL string `json:"responseUrl"`
	SubmitURL   string `json:"submitUrl"`
}

// VideoMetadata is the provider metadata of a completed generation, stored
// under the "fal" namespace.
type VideoMetadata struct {
	Videos []VideoInfo `json:"videos"`

	Seed            *float64      `json:"seed,omitempty"`
	Timings         *VideoTimings `json:"timings,omitempty"`
	HasNSFWConcepts []bool        `json:"has_nsfw_concepts,omitempty"`
	Prompt          *string       `json:"prompt,omitempty"`
}

// VideoInfo describes one generated video. Nil fields were not reported.
type VideoInfo struct {
	URL         string   `json:"url"`
	Width       *float64 `json:"width,omitempty"`
	Height      *float64 `json:"height,omitempty"`
	Duration    *float64 `json:"duration,omitempty"`
	FPS         *float64 `json:"fps,omitempty"`
	ContentType *string  `json:"contentType,omitempty"`
}

// VideoTimings are the reported generation times.
type VideoTimings struct {
	Inference *float64 `json:"inference,omitempty"`
}

type videoRequest struct {
	Prompt          string   `json:"prompt,omitempty"`
	ImageURL        string   `json:"image_url,omitempty"`
	AspectRatio     string   `json:"aspect_ratio,omitempty"`
	Duration        string   `json:"duration,omitempty"`
	Seed            *int     `json:"seed,omitempty"`
	Loop            *bool    `json:"loop,omitempty"`
	MotionStrength  *float64 `json:"motion_strength,omitempty"`
	Resolution      *string  `json:"resolution,omitempty"`
	NegativePrompt  *string  `json:"negative_prompt,omitempty"`
	PromptOptimizer *bool    `json:"prompt_optimizer,omitempty"`
}

type videoQueueResponse struct {
	RequestID   *string `json:"request_id"`
	ResponseURL *string `json:"response_url"`
}

type videoResponse struct {
	Video           *falVideo     `json:"video"`
	Seed            *float64      `json:"seed"`
	Timings         *VideoTimings `json:"timings"`
	HasNSFWConcepts []bool        `json:"has_nsfw_concepts"`
	Prompt          *string       `json:"prompt"`
}

type falVideo struct {
	URL         string   `json:"url"`
	Width       *float64 `json:"width"`
	Height      *float64 `json:"height"`
	Duration    *float64 `json:"duration"`
	FPS         *float64 `json:"fps"`
	ContentType *string  `json:"content_type"`
}

// HandleWebhook creates the endpoint that the queue notifies on completion.
func (m *VideoModel) HandleWebhook(ctx context.Context, factory model.VideoWebhookFactory) (*model.VideoWebhook, error) {
	if factory == nil {
		return nil, fmt.Errorf("%w: video webhook factory is nil", model.ErrInvalidArgument)
	}

	return factory(ctx)
}

// Start queues a generation, with the webhook URL when one is given.
func (m *VideoModel) Start(ctx context.Context, options model.VideoStartOptions) (*model.VideoStartResult, error) {
	timestamp := time.Now()

	body, err := videoBody(options.VideoOptions)
	if err != nil {
		return nil, err
	}

	submitURL := m.submitURL(options.WebhookURL)

	request, err := m.provider.apiRequest(submitURL, options.Headers, parseError)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, err
	}

	var queued videoQueueResponse
	if err := apicall.DecodeJSON(response.Body, &queued); err != nil {
		return nil, err
	}

	if queued.ResponseURL == nil || *queued.ResponseURL == "" {
		return nil, fmt.Errorf("%w: no response URL returned from queue endpoint", ErrVideoGenerationFailed)
	}

	operation, err := json.Marshal(VideoOperation{
		ResponseURL: *queued.ResponseURL,
		SubmitURL:   submitURL,
	})
	if err != nil {
		return nil, fmt.Errorf("encode video operation: %w", err)
	}

	return &model.VideoStartResult{
		Operation: operation,
		Response:  m.responseInfo(timestamp, response.Headers, response.Body),
	}, nil
}

// Status checks a queued request. A request that is still running is
// pending; a failed response from the API reports an error status.
func (m *VideoModel) Status(ctx context.Context, options model.VideoStatusOptions) (*model.VideoStatusResult, error) {
	timestamp := time.Now()

	var operation VideoOperation
	if err := json.Unmarshal(options.Operation, &operation); err != nil {
		return nil, fmt.Errorf("%w: video operation: %w", model.ErrInvalidArgument, err)
	}

	headers, body, err := m.fetchStatus(ctx, operation.ResponseURL, options.Headers)
	if err != nil {
		return failedStatus(err, m.responseInfo(timestamp, nil, nil))
	}

	var decoded videoResponse
	if err := apicall.DecodeJSON(body, &decoded); err != nil {
		return nil, err
	}

	return completedVideo(decoded, m.responseInfo(timestamp, headers, body))
}

// submitURL is the queue endpoint of the model, with the webhook URL.
func (m *VideoModel) submitURL(webhookURL string) string {
	modelID := strings.TrimPrefix(m.modelID, "fal-ai/")
	modelID = strings.TrimPrefix(modelID, "fal/")

	submitURL := queueBaseURL + "/" + modelID
	if webhookURL != "" {
		submitURL += "?" + webhookQueryParameter + "=" + url.QueryEscape(webhookURL)
	}

	return submitURL
}

// fetchStatus reads the response URL, which comes from the queue response.
// Credentials are sent only to the queue origin; other URLs are validated
// and receive only non-credential headers.
func (m *VideoModel) fetchStatus(ctx context.Context, location string, callHeaders http.Header) (http.Header, []byte, error) {
	if origin.Same(location, queueBaseURL) {
		request, err := m.provider.apiRequest(location, callHeaders, parseError)
		if err != nil {
			return nil, nil, err
		}

		response, err := apicall.Get(ctx, request)
		if err != nil {
			return nil, nil, err
		}

		return response.Headers, response.Body, nil
	}

	downloaded, err := download.Fetch(ctx, download.Request{
		URL:                location,
		Headers:            apicall.MergeHeaders(m.provider.headers, callHeaders),
		CredentialedOrigin: queueBaseURL,
		TrustedOrigin:      queueBaseURL,
		Client:             m.provider.httpClient,
	})
	if err != nil {
		return nil, nil, err
	}

	return nil, downloaded.Data, nil
}

// failedStatus converts a failed status check. A request in progress is
// pending and an error response is an error status; other failures are
// returned.
func failedStatus(err error, response model.ResponseInfo) (*model.VideoStatusResult, error) {
	if isInProgress(err) {
		return &model.VideoStatusResult{
			Status:   model.VideoPending,
			Response: response,
		}, nil
	}

	apiErr, ok := errors.AsType[*model.APICallError](err)
	if !ok || apiErr.StatusCode == 0 {
		return nil, err
	}

	response.Headers = apiErr.ResponseHeaders
	response.Body = []byte(apiErr.ResponseBody)

	return &model.VideoStatusResult{
		Status:   model.VideoError,
		Error:    apiErr.Message,
		Response: response,
	}, nil
}

// completedVideo reports the generated video and its metadata.
func completedVideo(decoded videoResponse, response model.ResponseInfo) (*model.VideoStatusResult, error) {
	if decoded.Video == nil || decoded.Video.URL == "" {
		return nil, fmt.Errorf("%w: no video URL in response", ErrVideoGenerationFailed)
	}

	video := decoded.Video

	mediaType := generatedVideoMediaType
	if video.ContentType != nil && *video.ContentType != "" {
		mediaType = *video.ContentType
	}

	metadata := VideoMetadata{
		Videos: []VideoInfo{
			{
				URL:         video.URL,
				Width:       video.Width,
				Height:      video.Height,
				Duration:    video.Duration,
				FPS:         video.FPS,
				ContentType: video.ContentType,
			},
		},
		Seed:            decoded.Seed,
		Timings:         decoded.Timings,
		HasNSFWConcepts: decoded.HasNSFWConcepts,
	}

	if decoded.Prompt != nil && *decoded.Prompt != "" {
		metadata.Prompt = decoded.Prompt
	}

	return &model.VideoStatusResult{
		Status: model.VideoCompleted,
		Videos: []model.VideoData{
			model.VideoDataURL{
				URL:       video.URL,
				MediaType: mediaType,
			},
		},
		ProviderMetadata: model.ProviderMetadata{
			Namespace: metadata,
		},
		Response: response,
	}, nil
}

// videoBody builds the queued request: the standard settings, the mapped
// provider options, then the passthrough options, which replace fields of
// the same name. A zero duration or seed is omitted.
func videoBody(options model.VideoOptions) (jsonobject.Object, error) {
	videoOptions, passthrough, err := readVideoOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	request := videoRequest{
		Prompt:          options.Prompt,
		AspectRatio:     options.AspectRatio,
		Loop:            videoOptions.Loop,
		MotionStrength:  videoOptions.MotionStrength,
		Resolution:      videoOptions.Resolution,
		NegativePrompt:  videoOptions.NegativePrompt,
		PromptOptimizer: videoOptions.PromptOptimizer,
	}

	if options.Image != nil {
		image, err := dataurl.VideoFileURL(options.Image)
		if err != nil {
			return nil, err
		}

		request.ImageURL = image
	}

	if options.Duration != nil && *options.Duration != 0 {
		request.Duration = strconv.FormatFloat(*options.Duration, 'f', -1, 64) + "s"
	}

	if options.Seed != nil && *options.Seed != 0 {
		request.Seed = options.Seed
	}

	encoded, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("encode request body: %w", err)
	}

	body, err := jsonobject.Parse(encoded)
	if err != nil {
		return nil, fmt.Errorf("encode request body: %w", err)
	}

	return body.Merge(passthrough), nil
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
