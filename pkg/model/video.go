package model

import (
	"context"
	"encoding/json"
	"net/http"
)

// VideoModelKind names video models in errors and registries.
const VideoModelKind Kind = "video model"

// VideoModel generates videos. The video contract is experimental and may
// change.
//
// A video model implements VideoGenerator, VideoOperator or both.
// VideoGenerator waits for the videos in one call; VideoOperator starts an
// asynchronous generation and checks its status, so that the caller
// controls polling or webhooks. Models whose API notifies webhooks also
// implement VideoWebhookHandler.
type VideoModel interface {
	Provider() string
	ModelID() string

	// MaxVideosPerCall returns the provider's per-call limit, or zero when
	// it is unknown.
	MaxVideosPerCall() int
}

// VideoGenerator generates videos in one blocking call.
type VideoGenerator interface {
	Generate(ctx context.Context, options VideoOptions) (*VideoResult, error)
}

// VideoOperator starts asynchronous video generations and checks their
// status.
type VideoOperator interface {
	// Start submits a generation and returns an opaque operation reference.
	Start(ctx context.Context, options VideoStartOptions) (*VideoStartResult, error)

	// Status checks a generation started with Start.
	Status(ctx context.Context, options VideoStatusOptions) (*VideoStatusResult, error)
}

// VideoWebhookHandler is implemented by video operators whose API notifies a
// webhook when a generation completes. Its presence tells the caller to
// create a webhook endpoint instead of polling; models without it never
// call the factory.
type VideoWebhookHandler interface {
	// HandleWebhook calls factory to create the endpoint and returns the URL
	// to pass to Start, with the wait for its notification.
	HandleWebhook(ctx context.Context, factory VideoWebhookFactory) (*VideoWebhook, error)
}

// VideoProvider is implemented by providers that offer video models. It is
// separate from Provider because most providers have none.
type VideoProvider interface {
	VideoModel(modelID string) (VideoModel, error)
}

// VideoWebhookFactory creates a webhook endpoint for one generation.
type VideoWebhookFactory func(ctx context.Context) (*VideoWebhook, error)

// VideoWebhook is a webhook endpoint that receives one notification.
type VideoWebhook struct {
	// URL is where the provider sends the notification.
	URL string

	// Wait blocks until the notification arrives or ctx ends.
	Wait func(ctx context.Context) (*VideoOperationWebhook, error)
}

// VideoOperationWebhook is a received webhook notification.
type VideoOperationWebhook struct {
	Headers http.Header
	Body    json.RawMessage
}

// VideoOptions are the inputs of one video generation. Empty values leave
// the provider default.
type VideoOptions struct {
	Prompt string

	// N is the number of videos to generate.
	N int

	// AspectRatio has the form "16:9", or is "adaptive" to follow the input
	// media.
	AspectRatio string

	// Resolution has the form "1280x720".
	Resolution string

	// Duration is the video length in seconds.
	Duration *float64

	// FPS is the frame rate.
	FPS int

	Seed *int

	// Image is the start frame of image-to-video generation.
	Image VideoFile

	// FrameImages are role-tagged first and last frame images.
	FrameImages []VideoFrameImage

	// InputReferences are the images and videos of reference-to-video
	// generation. Providers route each by its media type.
	InputReferences []VideoFile

	// GenerateAudio asks for audio alongside the video.
	GenerateAudio *bool

	Headers         http.Header
	ProviderOptions ProviderOptions
}

// VideoStartOptions are the inputs of Start.
type VideoStartOptions struct {
	VideoOptions

	// WebhookURL is where the provider should send a notification when the
	// generation completes. It is empty when the caller polls.
	WebhookURL string
}

// VideoStatusOptions are the inputs of Status.
type VideoStatusOptions struct {
	// Operation is the reference returned by Start.
	Operation json.RawMessage

	Headers http.Header
}

// VideoFile is an input image or video: VideoBytes or VideoURL.
type VideoFile interface {
	isVideoFile()
}

// VideoBytes is inline file content.
type VideoBytes struct {
	// MediaType is a video type such as "video/mp4" or an image type such as
	// "image/png".
	MediaType string
	Data      []byte

	ProviderOptions ProviderOptions
}

// VideoURL is a file the provider downloads.
type VideoURL struct {
	URL string

	// MediaType is empty when unknown.
	MediaType string

	ProviderOptions ProviderOptions
}

func (VideoBytes) isVideoFile() {}
func (VideoURL) isVideoFile()   {}

// VideoFrameType is the role of a frame image.
type VideoFrameType string

// Frame types.
const (
	// VideoFirstFrame is the frame the model animates from.
	VideoFirstFrame VideoFrameType = "first_frame"

	// VideoLastFrame is the frame the model animates towards.
	VideoLastFrame VideoFrameType = "last_frame"
)

// VideoFrameImage is a role-tagged frame image.
type VideoFrameImage struct {
	Image     VideoFile
	FrameType VideoFrameType
}

// VideoData is a generated video: VideoDataURL or VideoDataBytes. Providers
// decode base64 videos into VideoDataBytes.
type VideoData interface {
	isVideoData()
}

// VideoDataURL is a generated video served from a URL.
type VideoDataURL struct {
	URL       string
	MediaType string
}

// VideoDataBytes is a generated video returned inline.
type VideoDataBytes struct {
	Data      []byte
	MediaType string
}

func (VideoDataURL) isVideoData()   {}
func (VideoDataBytes) isVideoData() {}

// VideoResult is the outcome of Generate.
type VideoResult struct {
	Videos []VideoData

	ProviderMetadata ProviderMetadata
	Warnings         []Warning
	Response         ResponseInfo
}

// VideoStartResult is the outcome of Start.
type VideoStartResult struct {
	// Operation is a JSON reference, such as a task ID, that Status accepts.
	// It may be persisted and checked from another process.
	Operation json.RawMessage

	ProviderMetadata ProviderMetadata
	Warnings         []Warning
	Response         ResponseInfo
}

// VideoStatus is the state of an asynchronous generation.
type VideoStatus string

// Generation states.
const (
	VideoPending   VideoStatus = "pending"
	VideoCompleted VideoStatus = "completed"
	VideoError     VideoStatus = "error"
)

// VideoStatusResult is the state of an asynchronous generation. Videos are
// set when it completed; Error describes a failure.
type VideoStatusResult struct {
	Status VideoStatus
	Videos []VideoData
	Error  string

	ProviderMetadata ProviderMetadata
	Warnings         []Warning
	Response         ResponseInfo
}
