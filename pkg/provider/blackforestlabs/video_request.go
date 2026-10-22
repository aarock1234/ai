package blackforestlabs

import (
	"encoding/base64"
	"fmt"
	"math"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/pkg/model"
)

// Duration limits of FLUX 3 video, in whole seconds.
const (
	minVideoDurationSeconds = 5
	maxVideoDurationSeconds = 20
)

// untimedKeyframesNeedingDuration is the number of untimed keyframes from
// which the API needs an explicit duration.
const untimedKeyframesNeedingDuration = 3

// Shorter sides of the resolution tiers, in pixels. Shorter sides up to
// hdShorterSide map onto hd.
const (
	hdShorterSide  = 720
	fhdShorterSide = 1080
)

// Generation modes of FLUX 3 video.
const (
	modeTextToVideo  = "t2v"
	modeImageToVideo = "i2v"
	modeVideoToVideo = "v2v"
	modeDraftEnhance = "draft_enhance"
)

// pixelResolutionPattern matches "{width}x{height}" resolutions.
var pixelResolutionPattern = regexp.MustCompile(`^(\d+)x(\d+)$`)

// videoRequest is the body of a FLUX 3 video generation.
type videoRequest struct {
	Mode            string          `json:"mode"`
	Prompt          *string         `json:"prompt,omitempty"`
	AspectRatio     string          `json:"aspect_ratio,omitempty"`
	Duration        *int            `json:"duration,omitempty"`
	Resolution      VideoResolution `json:"resolution,omitempty"`
	Version         *string         `json:"version,omitempty"`
	GenerateAudio   *bool           `json:"generate_audio,omitempty"`
	SafetyTolerance *int            `json:"safety_tolerance,omitempty"`
	Draft           *bool           `json:"draft,omitempty"`
	Keyframes       VideoKeyframes  `json:"keyframes,omitempty"`
	StartVideo      string          `json:"start_video,omitempty"`
	DraftCache      *string         `json:"draft_cache,omitempty"`
}

// videoRequestBuilder converts the call options into a request body and
// collects the warnings of ignored or adjusted inputs.
type videoRequestBuilder struct {
	options      model.VideoOptions
	videoOptions VideoOptions
	warnings     []model.Warning
}

// buildVideoRequest returns the request body of a generation.
func buildVideoRequest(options model.VideoOptions) (videoRequest, []model.Warning, error) {
	videoOptions, err := readVideoOptions(options.ProviderOptions)
	if err != nil {
		return videoRequest{}, nil, err
	}

	builder := &videoRequestBuilder{
		options:      options,
		videoOptions: videoOptions,
	}

	if videoOptions.DraftCache != nil {
		request := builder.draftEnhanceRequest()

		return request, builder.warnings, nil
	}

	request, err := builder.generationRequest()
	if err != nil {
		return videoRequest{}, nil, err
	}

	return request, builder.warnings, nil
}

func (b *videoRequestBuilder) warn(feature string, details string) {
	b.warnings = append(b.warnings, model.UnsupportedWarning(feature, details))
}

// draftEnhanceRequest replays a draft bundle at full quality. The bundle
// pins the original mode, prompt, seed and media, and the API accepts only
// the safety tolerance with it, so every other setting is reported as
// ignored.
func (b *videoRequestBuilder) draftEnhanceRequest() videoRequest {
	options := b.options
	videoOptions := b.videoOptions

	pinnedByBundle := []struct {
		feature string
		set     bool
	}{
		{
			feature: "prompt",
			set:     strings.TrimSpace(options.Prompt) != "",
		},
		{
			feature: "aspectRatio",
			set:     options.AspectRatio != "" || videoOptions.AspectRatio != "",
		},
		{
			feature: "resolution",
			set:     options.Resolution != "" || videoOptions.Resolution != "",
		},
		{
			feature: "duration",
			set:     options.Duration != nil,
		},
		{
			feature: "fps",
			set:     options.FPS != 0,
		},
		{
			feature: "seed",
			set:     options.Seed != nil,
		},
		{
			feature: "generateAudio",
			set:     options.GenerateAudio != nil,
		},
		{
			feature: "image",
			set:     options.Image != nil,
		},
		{
			feature: "frameImages",
			set:     len(options.FrameImages) > 0,
		},
		{
			feature: "inputReferences",
			set:     len(options.InputReferences) > 0,
		},
		{
			feature: "keyframes",
			set:     len(videoOptions.Keyframes) > 0,
		},
		{
			feature: "version",
			set:     videoOptions.Version != nil,
		},
	}

	for _, pinned := range pinnedByBundle {
		if !pinned.set {
			continue
		}

		b.warn(pinned.feature, fmt.Sprintf(
			"FLUX 3 draft enhance replays the draft bundle as it was generated, so %q was ignored. Set it on the original draft request instead.",
			pinned.feature,
		))
	}

	if videoOptions.Draft != nil {
		b.warn("draft", "FLUX 3 draft enhance always renders at full quality. The draft option was ignored.")
	}

	b.warnMultipleVideos()

	return videoRequest{
		Mode:            modeDraftEnhance,
		DraftCache:      videoOptions.DraftCache,
		SafetyTolerance: videoOptions.SafetyTolerance,
	}
}

// generationRequest builds a text-, image- or video-to-video request.
func (b *videoRequestBuilder) generationRequest() (videoRequest, error) {
	if b.options.FPS != 0 {
		b.warn("fps", "FLUX 3 video does not support a custom frame rate.")
	}

	if b.options.Seed != nil {
		b.warn("seed", "FLUX 3 video does not accept a seed.")
	}

	b.warnMultipleVideos()

	resolution := b.resolution()
	aspectRatio := b.aspectRatio()
	keyframes := b.keyframes()
	startVideo := b.startVideo(keyframes != nil)
	duration := b.duration()

	if duration == nil && keyframes.untimed() >= untimedKeyframesNeedingDuration {
		return videoRequest{}, fmt.Errorf(
			"%w: duration: FLUX 3 video requires an explicit duration when %d or more keyframes are sent without a timestamp",
			model.ErrInvalidArgument,
			untimedKeyframesNeedingDuration,
		)
	}

	request := videoRequest{
		Mode:            modeTextToVideo,
		Prompt:          &b.options.Prompt,
		AspectRatio:     aspectRatio,
		Duration:        duration,
		Resolution:      resolution,
		Version:         b.videoOptions.Version,
		GenerateAudio:   b.options.GenerateAudio,
		SafetyTolerance: b.videoOptions.SafetyTolerance,
		Draft:           b.videoOptions.Draft,
	}

	switch {
	case keyframes != nil:
		request.Mode = modeImageToVideo
		request.Keyframes = keyframes

	case startVideo != "":
		request.Mode = modeVideoToVideo
		request.StartVideo = startVideo
	}

	return request, nil
}

func (b *videoRequestBuilder) warnMultipleVideos() {
	if b.options.N > 1 {
		b.warn("n", "FLUX 3 video generates a single video per call. Only 1 video will be generated.")
	}
}

// resolution returns the resolution tier. The provider option is already a
// tier and wins; a call option is resolved onto a tier only when it is the
// sole source.
func (b *videoRequestBuilder) resolution() VideoResolution {
	resolution := b.videoOptions.Resolution
	requested := b.options.Resolution

	if requested == "" {
		return resolution
	}

	tier, derived, ok := resolveResolutionTier(requested)

	switch {
	case resolution != "":
		if !ok {
			b.warn("resolution", fmt.Sprintf(
				"Unrecognized resolution %q. FLUX 3 video supports \"hd\" and \"fhd\", so providerOptions.blackForestLabs.resolution (%q) was used instead.",
				requested,
				resolution,
			))
		}

		return resolution

	case !ok:
		b.warn("resolution", fmt.Sprintf(
			"Unrecognized resolution %q. FLUX 3 video supports \"hd\" and \"fhd\", or a {width}x{height} value to map onto one.",
			requested,
		))

		return ""

	default:
		if derived {
			b.warnings = append(b.warnings, model.CompatibilityWarning("resolution", fmt.Sprintf(
				"FLUX 3 video renders at \"hd\" or \"fhd\"; the requested resolution %q was mapped to %q.",
				requested,
				tier,
			)))
		}

		return tier
	}
}

// resolveResolutionTier maps a named tier or a "{width}x{height}"
// resolution onto a tier. It reports whether the tier was derived from
// dimensions other than 720 or 1080 pixels on the shorter side.
func resolveResolutionTier(resolution string) (VideoResolution, bool, bool) {
	named := VideoResolution(strings.ToLower(resolution))
	if slices.Contains(videoResolutions, named) {
		return named, false, true
	}

	match := pixelResolutionPattern.FindStringSubmatch(resolution)
	if match == nil {
		return "", false, false
	}

	width, widthErr := strconv.Atoi(match[1])
	height, heightErr := strconv.Atoi(match[2])
	if widthErr != nil || heightErr != nil {
		return "", false, false
	}

	shorterSide := min(width, height)

	tier := VideoResolutionFHD
	if shorterSide <= hdShorterSide {
		tier = VideoResolutionHD
	}

	return tier, shorterSide != hdShorterSide && shorterSide != fhdShorterSide, true
}

// aspectRatio returns the aspect ratio. The provider option can also be
// "auto", so it wins.
func (b *videoRequestBuilder) aspectRatio() string {
	if b.videoOptions.AspectRatio != "" || b.options.AspectRatio == "" {
		return b.videoOptions.AspectRatio
	}

	if slices.Contains(videoAspectRatios, b.options.AspectRatio) {
		return b.options.AspectRatio
	}

	b.warn("aspectRatio", fmt.Sprintf(
		"FLUX 3 video does not support the aspect ratio %q. Using the provider default (auto).",
		b.options.AspectRatio,
	))

	return ""
}

// keyframes returns the keyframes of image-to-video generation, or nil for
// other modes. The provider option covers shapes the frame images cannot
// express, so it wins over them.
func (b *videoRequestBuilder) keyframes() VideoKeyframes {
	first, last := b.frames()

	if len(b.videoOptions.Keyframes) > 0 {
		if first != nil || last != nil {
			feature := "image"
			if len(b.options.FrameImages) > 0 {
				feature = "frameImages"
			}

			b.warn(feature, "FLUX 3 video takes a single keyframe list. providerOptions.blackForestLabs.keyframes was used and the top-level frame images were ignored.")
		}

		return slices.Clone(b.videoOptions.Keyframes)
	}

	if first == nil {
		return nil
	}

	keyframes := VideoKeyframes{
		{
			Image: videoInput(first),
		},
	}

	if last != nil {
		keyframes = append(keyframes, VideoKeyframe{
			Image: videoInput(last),
		})
	}

	return keyframes
}

// frames returns the usable first and last frames. A standalone image is
// the first frame. Frames that are not images are ignored, and so is a last
// frame without a first frame: keyframes are positional.
func (b *videoRequestBuilder) frames() (model.VideoFile, model.VideoFile) {
	firstFrameImage := frameImage(b.options.FrameImages, model.VideoFirstFrame)

	first := firstFrameImage
	if first == nil {
		first = b.options.Image
	}

	last := frameImage(b.options.FrameImages, model.VideoLastFrame)

	if first != nil {
		if topLevel := nonImageMediaType(first); topLevel != "" {
			feature := "image"
			if firstFrameImage != nil {
				feature = "frameImages"
			}

			details := fmt.Sprintf("FLUX 3 video only accepts an image as a keyframe; the %q file was ignored.", videoFileMediaType(first))
			if topLevel == "video" {
				details = "FLUX 3 video does not accept a video as a keyframe. Pass it as an inputReference to continue from it instead."
			}

			b.warn(feature, details)
			first = nil
		}
	}

	if last == nil {
		return first, nil
	}

	if first == nil {
		b.warn("frameImages", "FLUX 3 video requires a first_frame when a last_frame is provided. The last_frame was ignored.")

		return nil, nil
	}

	if topLevel := nonImageMediaType(last); topLevel != "" {
		details := fmt.Sprintf("FLUX 3 video only accepts an image as a keyframe; the %q last_frame was ignored.", videoFileMediaType(last))
		if topLevel == "video" {
			details = "FLUX 3 video does not accept a video as a keyframe. The last_frame video was ignored."
		}

		b.warn("frameImages", details)

		return first, nil
	}

	return first, last
}

// startVideo returns the video to continue from. FLUX 3 takes a single
// video and has no reference images, so image references are ignored.
func (b *videoRequestBuilder) startVideo(hasKeyframes bool) string {
	var videos []model.VideoFile

	for _, file := range b.options.InputReferences {
		mediaType := videoFileMediaType(file)

		switch topLevel := mediatype.TopLevel(mediaType); {
		case mediaType == "":
			b.warnings = append(b.warnings, model.CompatibilityWarning(
				"inputReferences",
				"FLUX 3 video only accepts a video reference, so the reference with no mediaType was treated as the video to continue from. Pass { url, mediaType: \"video/mp4\" } to be explicit.",
			))
			videos = append(videos, file)

		case topLevel == "video":
			videos = append(videos, file)

		case topLevel == "image":
			b.warn("inputReferences", "FLUX 3 video has no reference-image input. Pass images as `image`, `frameImages`, or providerOptions.blackForestLabs.keyframes instead. The reference was ignored.")

		default:
			b.warn("inputReferences", fmt.Sprintf("FLUX 3 video only accepts a video reference; the %q reference was ignored.", mediaType))
		}
	}

	if len(videos) == 0 {
		return ""
	}

	if hasKeyframes {
		b.warn("inputReferences", "FLUX 3 video cannot combine keyframes with a video to continue from. The video reference was ignored.")

		return ""
	}

	if len(videos) > 1 {
		b.warn("inputReferences", "FLUX 3 video continues from a single video. Only the first video reference was used.")
	}

	return videoInput(videos[0])
}

// duration returns the duration in whole seconds between 5 and 20, or nil
// to let the API choose one that matches the content.
func (b *videoRequestBuilder) duration() *int {
	if b.options.Duration == nil {
		return nil
	}

	requested := *b.options.Duration
	duration := requested

	if duration != math.Trunc(duration) {
		duration = math.Round(duration)

		b.warn("duration", fmt.Sprintf(
			"FLUX 3 video requires a whole number of seconds. The requested duration of %s was rounded to %s.",
			formatNumber(requested),
			formatNumber(duration),
		))
	}

	switch {
	case duration > maxVideoDurationSeconds:
		b.warn("duration", fmt.Sprintf(
			"FLUX 3 video supports at most %d seconds. The requested duration of %s was clamped to %d.",
			maxVideoDurationSeconds,
			formatNumber(requested),
			maxVideoDurationSeconds,
		))

		duration = maxVideoDurationSeconds

	case duration < minVideoDurationSeconds:
		b.warn("duration", fmt.Sprintf(
			"FLUX 3 video requires at least %d seconds. The requested duration of %s was clamped to %d.",
			minVideoDurationSeconds,
			formatNumber(requested),
			minVideoDurationSeconds,
		))

		duration = minVideoDurationSeconds
	}

	return new(int(duration))
}

// frameImage returns the frame image of frameType, or nil.
func frameImage(frames []model.VideoFrameImage, frameType model.VideoFrameType) model.VideoFile {
	for _, frame := range frames {
		if frame.FrameType == frameType {
			return frame.Image
		}
	}

	return nil
}

// videoFileMediaType returns the declared media type of an input file,
// which is empty for URLs without one.
func videoFileMediaType(file model.VideoFile) string {
	switch typed := file.(type) {
	case model.VideoBytes:
		return typed.MediaType

	case model.VideoURL:
		return typed.MediaType

	default:
		return ""
	}
}

// nonImageMediaType returns the top-level media type of a file that is
// declared as something other than an image, or empty.
func nonImageMediaType(file model.VideoFile) string {
	mediaType := videoFileMediaType(file)
	if mediaType == "" {
		return ""
	}

	topLevel := mediatype.TopLevel(mediaType)
	if topLevel == "image" {
		return ""
	}

	return topLevel
}

// videoInput returns a conditioning file as the API takes it: an http(s)
// URL or bare base64, not a data URL.
func videoInput(file model.VideoFile) string {
	switch typed := file.(type) {
	case model.VideoURL:
		return typed.URL

	case model.VideoBytes:
		return base64.StdEncoding.EncodeToString(typed.Data)

	default:
		return ""
	}
}

func formatNumber(value float64) string {
	return strconv.FormatFloat(value, 'f', -1, 64)
}
