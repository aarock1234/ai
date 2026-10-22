package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"slices"

	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/internal/merge"
	"github.com/aarock1234/ai/pkg/model"
)

// Media types of generated videos and input images whose type is unknown.
const (
	defaultVideoMediaType      = "video/mp4"
	defaultVideoImageMediaType = "image/png"
	genericBinaryMediaType     = "application/octet-stream"
)

var (
	// ErrVideoGenerationFailed reports an asynchronous generation whose
	// status is an error, or that had not completed after its webhook
	// notification.
	ErrVideoGenerationFailed = errors.New("video generation failed")

	// ErrVideoGenerationTimeout reports an asynchronous generation that did
	// not complete within the poll timeout.
	ErrVideoGenerationTimeout = errors.New("video generation timed out")
)

// VideoSettings are the generation settings shared by GenerateVideoOptions
// and StartVideoOptions.
type VideoSettings struct {
	Model model.VideoModel

	// Prompt describes the video to generate.
	Prompt string

	// Image is the start frame of image-to-video generation: inline bytes,
	// an http(s) URL that the provider downloads, or a data URL. A
	// first-frame FrameImages entry replaces it.
	Image model.FileData

	// FrameImages are role-tagged first and last frame images.
	FrameImages []VideoFrameImage

	// InputReferences are the images and videos of reference-to-video
	// generation. They are ignored when FrameImages are set.
	InputReferences []VideoInputReference

	// N is the number of videos to generate. Zero means one.
	N int

	// MaxVideosPerCall overrides the model's per-call limit. Zero uses the
	// model's limit, and one video per call when the model has none.
	MaxVideosPerCall int

	// AspectRatio has the form "{width}:{height}", or is "adaptive" to
	// follow the input media.
	AspectRatio string

	// Resolution has the form "{width}x{height}".
	Resolution string

	// Duration is the video length in seconds.
	Duration *float64

	// FPS is the frame rate.
	FPS int

	Seed *int

	// GenerateAudio asks for audio alongside the video.
	GenerateAudio *bool

	// MaxRetries is the number of retries per model call. Nil selects 2;
	// zero disables retries.
	MaxRetries *int

	Headers         http.Header
	ProviderOptions model.ProviderOptions
}

// VideoFrameImage is a role-tagged frame image.
type VideoFrameImage struct {
	// Image is inline bytes, a URL or a data URL.
	Image model.FileData

	FrameType model.VideoFrameType
}

// VideoInputReference is a reference image or video.
type VideoInputReference struct {
	// Data is inline bytes, a URL or a data URL.
	Data model.FileData

	// MediaType is optional; inline data is detected without it.
	MediaType string
}

// GenerateVideoOptions configures GenerateVideo.
type GenerateVideoOptions struct {
	VideoSettings

	// Download fetches videos the provider returns as URLs. Nil uses
	// NewDownloader with the default size limit.
	Download Downloader

	// Poll configures the asynchronous start/status flow. Setting it makes
	// models that implement model.VideoOperator use that flow even when
	// they can also generate in one call.
	Poll *VideoPollOptions

	// Webhook creates a webhook endpoint for models that implement
	// model.VideoWebhookHandler; the call then waits for the notification
	// instead of polling. Other operator models poll and warn.
	Webhook model.VideoWebhookFactory

	// LogWarnings receives the warnings of the call.
	LogWarnings LogWarningsFunc
}

// GenerateVideoResult is the outcome of GenerateVideo.
type GenerateVideoResult struct {
	// Video is the first generated video.
	Video GeneratedFile

	// Videos are the generated videos of every call, in call order.
	Videos []GeneratedFile

	Warnings []model.Warning

	// Responses has one entry per model call.
	Responses []VideoResponse

	// ProviderMetadata merges the metadata of every call. Per-video
	// "videos" arrays are concatenated.
	ProviderMetadata model.ProviderMetadata
}

// VideoResponse describes one video model call.
type VideoResponse struct {
	model.ResponseInfo

	// ProviderMetadata is the metadata of this call only.
	ProviderMetadata model.ProviderMetadata
}

// GenerateVideo generates videos. Requests for more videos than the model
// allows per call are split into concurrent calls. The video API is
// experimental and may change.
//
// Models that implement model.VideoGenerator are called once per batch.
// Models that implement model.VideoOperator are started and then polled, or
// notified through Webhook, until the videos are ready; Poll selects that
// flow for models that implement both.
func GenerateVideo(ctx context.Context, options GenerateVideoOptions) (*GenerateVideoResult, error) {
	if err := options.validate(); err != nil {
		return nil, err
	}

	retry, err := prepareRetries(options.MaxRetries)
	if err != nil {
		return nil, err
	}

	inputs, err := normalizeVideoInputs(options.VideoSettings)
	if err != nil {
		return nil, err
	}

	caller, err := newVideoCaller(ctx, options, retry)
	if err != nil {
		return nil, err
	}

	request := inputs.callOptions(options.VideoSettings)
	counts := callCounts(max(options.N, 1), videosPerCall(options.VideoSettings))

	results, err := mapConcurrently(ctx, counts, 0, func(ctx context.Context, count int) (*model.VideoResult, error) {
		callRequest := request
		callRequest.N = count

		return caller.call(ctx, callRequest)
	})
	if err != nil {
		return nil, err
	}

	download := options.Download
	if download == nil {
		download = NewDownloader(DownloadOptions{})
	}

	result, err := collectVideoResults(ctx, results, download)
	if err != nil {
		return nil, err
	}

	result.Warnings = slices.Concat(inputs.warnings, result.Warnings)

	if len(result.Videos) == 0 {
		return nil, &NoVideoGeneratedError{
			Responses: result.Responses,
		}
	}

	result.Video = result.Videos[0]

	options.LogWarnings.log(ctx, WarningsEvent{
		Warnings: result.Warnings,
		Provider: options.Model.Provider(),
		ModelID:  options.Model.ModelID(),
	})

	return result, nil
}

// videoCaller makes the model calls of GenerateVideo through the model's
// blocking Generate method or its start/status flow.
type videoCaller struct {
	options      GenerateVideoOptions
	retry        retryPolicy
	generator    model.VideoGenerator
	operator     model.VideoOperator
	useOperation bool
}

// newVideoCaller selects the call flow: the start/status flow when the
// model supports it and either Poll or Webhook is set or it cannot generate
// in one call. Poll and Webhook settings that the model cannot use are
// logged as a warning.
func newVideoCaller(ctx context.Context, options GenerateVideoOptions, retry retryPolicy) (videoCaller, error) {
	generator, canGenerate := options.Model.(model.VideoGenerator)
	operator, canOperate := options.Model.(model.VideoOperator)

	if !canGenerate && !canOperate {
		return videoCaller{}, fmt.Errorf(
			"%w: video model %s implements neither Generate nor Start and Status",
			model.ErrUnsupported,
			options.Model.ModelID(),
		)
	}

	wantsOperation := options.Poll != nil || options.Webhook != nil
	if wantsOperation && !canOperate {
		options.LogWarnings.log(ctx, WarningsEvent{
			Warnings: []model.Warning{
				{
					Type:    model.WarningOther,
					Message: "poll/webhook options were provided but the model does not support Start/Status. Falling back to Generate.",
				},
			},
			Provider: options.Model.Provider(),
			ModelID:  options.Model.ModelID(),
		})
	}

	return videoCaller{
		options:      options,
		retry:        retry,
		generator:    generator,
		operator:     operator,
		useOperation: canOperate && (wantsOperation || !canGenerate),
	}, nil
}

// call generates the videos of one request.
func (c videoCaller) call(ctx context.Context, request model.VideoOptions) (*model.VideoResult, error) {
	if !c.useOperation {
		return generateVideosWithRetry(ctx, c.generator, c.retry, request)
	}

	operation := videoOperation{
		model:    c.options.Model,
		operator: c.operator,
		options:  request,
		poll:     resolvePollOptions(c.options.Poll),
		webhook:  c.options.Webhook,
		retry:    c.retry,
	}

	return operation.run(ctx)
}

// validate checks the settings shared by every video function.
func (s VideoSettings) validate() error {
	if s.Model == nil {
		return &InvalidArgumentError{
			Parameter: "model",
			Message:   "model is required",
		}
	}

	if s.N < 0 {
		return &InvalidArgumentError{
			Parameter: "n",
			Value:     s.N,
			Message:   "n must be >= 0",
		}
	}

	if s.MaxVideosPerCall < 0 {
		return &InvalidArgumentError{
			Parameter: "maxVideosPerCall",
			Value:     s.MaxVideosPerCall,
			Message:   "maxVideosPerCall must be >= 0",
		}
	}

	return nil
}

// videosPerCall returns the explicit limit, the model's limit, or one.
func videosPerCall(settings VideoSettings) int {
	if limit := knownVideosPerCall(settings); limit > 0 {
		return limit
	}

	return 1
}

// knownVideosPerCall returns the explicit limit or the model's limit, or
// zero when neither is known.
func knownVideosPerCall(settings VideoSettings) int {
	if settings.MaxVideosPerCall > 0 {
		return settings.MaxVideosPerCall
	}

	return settings.Model.MaxVideosPerCall()
}

func generateVideosWithRetry(
	ctx context.Context,
	generator model.VideoGenerator,
	retry retryPolicy,
	request model.VideoOptions,
) (*model.VideoResult, error) {
	return retryCall(ctx, retry, func(ctx context.Context) (*model.VideoResult, error) {
		result, err := generator.Generate(ctx, request)
		if err != nil {
			return nil, err
		}

		if result == nil {
			return nil, fmt.Errorf("%w: video model returned no result", model.ErrInvalidResponse)
		}

		return result, nil
	})
}

// videoInputs are the normalized prompt media of a video call.
type videoInputs struct {
	image           model.VideoFile
	frameImages     []model.VideoFrameImage
	inputReferences []model.VideoFile
	warnings        []model.Warning
}

// normalizeVideoInputs converts the prompt media. A first-frame image
// replaces Image, and frame images exclude input references; both
// precedences produce warnings.
func normalizeVideoInputs(settings VideoSettings) (videoInputs, error) {
	var inputs videoInputs

	image, err := optionalVideoFile(settings.Image, true)
	if err != nil {
		return videoInputs{}, err
	}

	frameImages, err := videoFrameImages(settings.FrameImages)
	if err != nil {
		return videoInputs{}, err
	}

	references, err := videoReferences(settings.InputReferences)
	if err != nil {
		return videoInputs{}, err
	}

	inputs.frameImages = frameImages
	inputs.inputReferences = references

	if len(frameImages) > 0 && len(references) > 0 {
		inputs.inputReferences = nil
		inputs.warnings = append(inputs.warnings, model.Warning{
			Type:    model.WarningOther,
			Message: "inputReferences were ignored because frameImages were provided; frameImages and inputReferences cannot be combined.",
		})
	}

	firstFrame := firstFrameImage(frameImages)
	if image != nil && firstFrame != nil {
		inputs.warnings = append(inputs.warnings, model.Warning{
			Type:    model.WarningOther,
			Message: "prompt.image was ignored because a first_frame frameImage was provided; the first_frame frameImage takes precedence as the start image.",
		})
	}

	inputs.image = image
	if firstFrame != nil {
		inputs.image = firstFrame
	}

	return inputs, nil
}

// callOptions returns the model call options without the video count.
func (i videoInputs) callOptions(settings VideoSettings) model.VideoOptions {
	return model.VideoOptions{
		Prompt:          settings.Prompt,
		AspectRatio:     settings.AspectRatio,
		Resolution:      settings.Resolution,
		Duration:        settings.Duration,
		FPS:             settings.FPS,
		Seed:            settings.Seed,
		Image:           i.image,
		FrameImages:     i.frameImages,
		InputReferences: i.inputReferences,
		GenerateAudio:   settings.GenerateAudio,
		Headers:         settings.Headers,
		ProviderOptions: settings.ProviderOptions,
	}
}

func firstFrameImage(frameImages []model.VideoFrameImage) model.VideoFile {
	for _, frame := range frameImages {
		if frame.FrameType == model.VideoFirstFrame {
			return frame.Image
		}
	}

	return nil
}

func videoFrameImages(frames []VideoFrameImage) ([]model.VideoFrameImage, error) {
	if frames == nil {
		return nil, nil
	}

	converted := make([]model.VideoFrameImage, 0, len(frames))
	for _, frame := range frames {
		image, err := optionalVideoFile(frame.Image, true)
		if err != nil {
			return nil, err
		}

		if image == nil {
			continue
		}

		converted = append(converted, model.VideoFrameImage{
			Image:     image,
			FrameType: frame.FrameType,
		})
	}

	return converted, nil
}

// videoReferences converts the input references. An explicit media type
// replaces the detected one.
func videoReferences(references []VideoInputReference) ([]model.VideoFile, error) {
	if references == nil {
		return nil, nil
	}

	converted := make([]model.VideoFile, 0, len(references))
	for _, reference := range references {
		file, err := optionalVideoFile(reference.Data, false)
		if err != nil {
			return nil, err
		}

		if file == nil {
			continue
		}

		converted = append(converted, withVideoMediaType(file, reference.MediaType))
	}

	return converted, nil
}

func withVideoMediaType(file model.VideoFile, mediaType string) model.VideoFile {
	if mediaType == "" {
		return file
	}

	switch typed := file.(type) {
	case model.VideoBytes:
		typed.MediaType = mediaType

		return typed

	case model.VideoURL:
		typed.MediaType = mediaType

		return typed

	default:
		return file
	}
}

// optionalVideoFile converts prompt media into a video model file. URLs are
// passed to the provider; data URLs and bytes are sent inline with a
// detected media type, restricted to images for start and frame images. It
// returns nil for missing data.
func optionalVideoFile(data model.FileData, restrictToImages bool) (model.VideoFile, error) {
	if data == nil {
		return nil, nil
	}

	normalized, mediaType, err := normalizeFileData(data, "")
	if err != nil {
		return nil, err
	}

	switch typed := normalized.(type) {
	case model.FileURL:
		return model.VideoURL{
			URL: typed.URL.String(),
		}, nil

	case model.FileBytes:
		if mediaType == "" {
			mediaType = detectVideoInputMediaType(typed.Data, restrictToImages)
		}

		return model.VideoBytes{
			MediaType: mediaType,
			Data:      typed.Data,
		}, nil

	default:
		return nil, &InvalidArgumentError{
			Parameter: "prompt",
			Message:   "video inputs must be inline bytes or URLs",
		}
	}
}

func detectVideoInputMediaType(data []byte, restrictToImages bool) string {
	topLevel := ""
	if restrictToImages {
		topLevel = "image"
	}

	if detected, ok := mediatype.Detect(data, topLevel); ok {
		return detected
	}

	return defaultVideoImageMediaType
}

// collectVideoResults downloads URL videos and gathers the warnings,
// responses and metadata of every call.
func collectVideoResults(
	ctx context.Context,
	results []*model.VideoResult,
	download Downloader,
) (*GenerateVideoResult, error) {
	collected := &GenerateVideoResult{
		ProviderMetadata: model.ProviderMetadata{},
	}

	for _, result := range results {
		for _, data := range result.Videos {
			video, err := generatedVideo(ctx, data, download)
			if err != nil {
				return nil, err
			}

			collected.Videos = append(collected.Videos, video)
		}

		collected.Warnings = append(collected.Warnings, result.Warnings...)
		collected.Responses = append(collected.Responses, VideoResponse{
			ResponseInfo:     result.Response,
			ProviderMetadata: result.ProviderMetadata,
		})

		mergeVideoProviderMetadata(collected.ProviderMetadata, result.ProviderMetadata)
	}

	return collected, nil
}

// generatedVideo returns the video content, downloading URL videos. The
// media type is the provider's, then the download's, then the detected one,
// and "video/mp4" otherwise.
func generatedVideo(ctx context.Context, data model.VideoData, download Downloader) (GeneratedFile, error) {
	switch typed := data.(type) {
	case model.VideoDataURL:
		videoURL, err := url.Parse(typed.URL)
		if err != nil {
			return GeneratedFile{}, fmt.Errorf("%w: video URL: %w", model.ErrInvalidResponse, err)
		}

		downloaded, err := download(ctx, videoURL)
		if err != nil {
			return GeneratedFile{}, err
		}

		return GeneratedFile{
			Data:      downloaded.Data,
			MediaType: videoMediaType(downloaded.Data, typed.MediaType, downloaded.MediaType),
		}, nil

	case model.VideoDataBytes:
		return GeneratedFile{
			Data:      typed.Data,
			MediaType: videoMediaType(typed.Data, typed.MediaType),
		}, nil

	default:
		return GeneratedFile{}, fmt.Errorf("%w: video data %T", model.ErrInvalidResponse, data)
	}
}

// videoMediaType returns the first specific candidate, or the type detected
// from the content.
func videoMediaType(content []byte, candidates ...string) string {
	for _, candidate := range candidates {
		if candidate != "" && candidate != genericBinaryMediaType {
			return candidate
		}
	}

	if detected, ok := mediatype.Detect(content, "video"); ok {
		return detected
	}

	return defaultVideoMediaType
}

// mergeVideoProviderMetadata adds one call's metadata to the combined
// metadata: provider objects are merged key by key, and their "videos"
// arrays are concatenated.
func mergeVideoProviderMetadata(combined, metadata model.ProviderMetadata) {
	for provider, value := range metadata {
		existing, hasExisting := combined[provider]
		if !hasExisting {
			combined[provider] = value

			continue
		}

		merged, ok := merge.Shallow(existing, value)
		if !ok {
			combined[provider] = value

			continue
		}

		existingVideos, existingOK := metadataVideos(existing)
		nextVideos, nextOK := metadataVideos(value)

		if existingOK && nextOK {
			merged["videos"] = slices.Concat(existingVideos, nextVideos)
		}

		combined[provider] = merged
	}
}

// metadataVideos returns the "videos" array of a provider metadata object.
func metadataVideos(value any) ([]any, bool) {
	object, ok := merge.Object(value)
	if !ok {
		return nil, false
	}

	videos, present := object["videos"]
	if !present {
		return nil, false
	}

	encoded, err := json.Marshal(videos)
	if err != nil {
		return nil, false
	}

	var entries []any
	if err := json.Unmarshal(encoded, &entries); err != nil || entries == nil {
		return nil, false
	}

	return entries, true
}
