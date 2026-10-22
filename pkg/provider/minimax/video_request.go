package minimax

import (
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/pkg/model"
)

// Defaults and limits of MiniMax video generation.
const (
	defaultVideoResolution      = VideoResolution2K
	defaultVideoRatio           = VideoRatio16x9
	defaultVideoDurationSeconds = 5
	maxVideoDurationSeconds     = 15
	maxReferenceImages          = 9
	maxReferenceVideos          = 3
	maxReferenceAudios          = 3
)

// Models with their own video rules.
const (
	modelH3    = "MiniMax-H3"
	modelH3Max = "MiniMax-H3-Max"
)

// Content part types and roles of a video request.
const (
	videoContentText  = "text"
	videoContentImage = "image_url"
	videoContentVideo = "video_url"
	videoContentAudio = "audio_url"

	roleFirstFrame     = "first_frame"
	roleLastFrame      = "last_frame"
	roleReferenceImage = "reference_image"
	roleReferenceVideo = "reference_video"
	roleReferenceAudio = "reference_audio"
)

// modelVideoResolutions are the resolution tiers and default of models that do
// not support every tier.
var modelVideoResolutions = map[string]modelVideoResolutionSettings{
	modelH3: {
		supported: []VideoResolution{
			VideoResolution768P,
			VideoResolution2K,
		},
		fallback: VideoResolution2K,
	},
	modelH3Max: {
		supported: []VideoResolution{
			VideoResolution480P,
			VideoResolution768P,
		},
		fallback: VideoResolution768P,
	},
}

// videoResolutionTiers maps canonical frame sizes onto resolution tiers, one per
// aspect ratio and tier. MiniMax does not publish the tier dimensions: the
// 480P and 768P rows are named for the shorter side, the 2K rows for the
// longer one.
var videoResolutionTiers = map[string]VideoResolution{
	"480x480":   VideoResolution480P,
	"1120x480":  VideoResolution480P,
	"854x480":   VideoResolution480P,
	"640x480":   VideoResolution480P,
	"480x854":   VideoResolution480P,
	"480x640":   VideoResolution480P,
	"768x768":   VideoResolution768P,
	"1792x768":  VideoResolution768P,
	"1366x768":  VideoResolution768P,
	"1024x768":  VideoResolution768P,
	"768x1366":  VideoResolution768P,
	"768x1024":  VideoResolution768P,
	"2048x2048": VideoResolution2K,
	"2560x1080": VideoResolution2K,
	"2560x1440": VideoResolution2K,
	"2048x1536": VideoResolution2K,
	"1440x2560": VideoResolution2K,
	"1536x2048": VideoResolution2K,
}

type modelVideoResolutionSettings struct {
	supported []VideoResolution
	fallback  VideoResolution
}

// videoRequest is the body of a video generation.
type videoRequest struct {
	Model         string          `json:"model"`
	Content       []videoContent  `json:"content"`
	Resolution    VideoResolution `json:"resolution"`
	Duration      int             `json:"duration"`
	Ratio         VideoRatio      `json:"ratio,omitempty"`
	AIGCWatermark *bool           `json:"aigc_watermark,omitempty"`
	CallbackURL   string          `json:"callback_url,omitempty"`
}

// videoContent is one content part: text, or an image, video or audio URL
// with its role.
type videoContent struct {
	Type     string          `json:"type"`
	Text     *string         `json:"text,omitempty"`
	ImageURL *videoMediaLink `json:"image_url,omitempty"`
	VideoURL *videoMediaLink `json:"video_url,omitempty"`
	AudioURL *videoMediaLink `json:"audio_url,omitempty"`
	Role     string          `json:"role,omitempty"`
}

type videoMediaLink struct {
	URL string `json:"url"`
}

// VideoResolvedInputs counts the inputs that were sent after the limits and
// rejections of the model, which warnings alone do not report.
type VideoResolvedInputs struct {
	ImageCount int `json:"imageCount"`

	// ReferenceVideoIndices are the input reference positions of the sent
	// URL reference videos. Status reports them.
	ReferenceVideoIndices []int `json:"referenceVideoIndices,omitzero"`

	// ReferenceVideoURLs are the sent URL reference videos. Generate
	// reports them instead of the indices.
	ReferenceVideoURLs []string `json:"referenceVideoUrls,omitzero"`
}

// videoRequestBuilder converts the call options into a request body and
// collects the warnings of ignored or adjusted inputs.
type videoRequestBuilder struct {
	modelID      string
	options      model.VideoOptions
	videoOptions VideoOptions
	warnings     []model.Warning

	content  []videoContent
	resolved VideoResolvedInputs
}

// buildVideoRequest returns the request body of a generation and the inputs
// it carries.
func buildVideoRequest(modelID string, options model.VideoOptions) (videoRequest, VideoResolvedInputs, []model.Warning, error) {
	videoOptions, err := readVideoOptions(options.ProviderOptions)
	if err != nil {
		return videoRequest{}, VideoResolvedInputs{}, nil, err
	}

	builder := &videoRequestBuilder{
		modelID:      modelID,
		options:      options,
		videoOptions: videoOptions,
		resolved: VideoResolvedInputs{
			ReferenceVideoIndices: []int{},
		},
	}

	request := builder.build()

	return request, builder.resolved, builder.warnings, nil
}

func (b *videoRequestBuilder) warn(feature string, details string) {
	b.warnings = append(b.warnings, model.UnsupportedWarning(feature, details))
}

func (b *videoRequestBuilder) build() videoRequest {
	b.warnUnsupportedSettings()

	resolution := b.resolution()

	b.content = []videoContent{
		{
			Type: videoContentText,
			Text: &b.options.Prompt,
		},
	}

	usesFrameImages := b.addFrameImages()
	b.addReferences(usesFrameImages)

	request := videoRequest{
		Model:         b.modelID,
		Content:       b.content,
		Resolution:    resolution,
		Duration:      b.duration(),
		Ratio:         b.ratio(usesFrameImages),
		AIGCWatermark: b.videoOptions.AIGCWatermark,
	}

	return request
}

func (b *videoRequestBuilder) warnUnsupportedSettings() {
	if b.options.FPS != 0 {
		b.warn("fps", b.modelID+" does not support a custom frame rate.")
	}

	if b.options.Seed != nil {
		b.warn("seed", b.modelID+" does not support a seed.")
	}

	if b.options.N > 1 {
		b.warn("n", b.modelID+" generates a single video per call. Only 1 video will be generated.")
	}

	if b.options.GenerateAudio != nil {
		b.warn("generateAudio", "The "+b.modelID+" API does not expose an audio parameter. The generateAudio option was ignored.")
	}
}

// supportedResolutions returns the tiers of the model and its default.
func (b *videoRequestBuilder) supportedResolutions() ([]VideoResolution, VideoResolution) {
	if settings, ok := modelVideoResolutions[b.modelID]; ok {
		return settings.supported, settings.fallback
	}

	return videoResolutions, defaultVideoResolution
}

// resolution returns the resolution tier. The provider option is already a
// tier and wins; a call option is resolved onto a tier.
func (b *videoRequestBuilder) resolution() VideoResolution {
	supported, fallback := b.supportedResolutions()
	supportedNames := videoResolutionNames(supported)

	resolution := b.videoOptions.Resolution
	if resolution != "" && !slices.Contains(supported, resolution) {
		b.warn("resolution", fmt.Sprintf(
			"%s supports %s. The provider resolution %q was ignored.",
			b.modelID,
			supportedNames,
			resolution,
		))

		resolution = ""
	}

	requested := b.options.Resolution
	if requested != "" {
		mapped, ok := resolveVideoResolutionTier(requested)

		switch {
		case resolution != "" && !ok:
			b.warn("resolution", fmt.Sprintf(
				"Unrecognized resolution %q. %s supports %s, so providerOptions.minimax.resolution (%q) was used instead.",
				requested,
				b.modelID,
				supportedNames,
				resolution,
			))

		case resolution != "" && mapped != resolution:
			b.warn("resolution", fmt.Sprintf(
				"The resolution %q selects %s, but providerOptions.minimax.resolution (%q) was used instead.",
				requested,
				mapped,
				resolution,
			))

		case resolution != "":

		case ok && slices.Contains(supported, mapped):
			resolution = mapped

		case !ok:
			b.warn("resolution", fmt.Sprintf("Unrecognized resolution %q. %s supports %s.", requested, b.modelID, supportedNames))

		default:
			b.warn("resolution", fmt.Sprintf("%s does not support the resolution %q. It supports %s.", b.modelID, mapped, supportedNames))
		}
	}

	if resolution == "" {
		return fallback
	}

	return resolution
}

// resolveVideoResolutionTier maps a named tier, in any case, or a canonical
// frame size onto a tier.
func resolveVideoResolutionTier(resolution string) (VideoResolution, bool) {
	named := VideoResolution(strings.ToUpper(resolution))
	if slices.Contains(videoResolutions, named) {
		return named, true
	}

	tier, ok := videoResolutionTiers[resolution]

	return tier, ok
}

func videoResolutionNames(resolutions []VideoResolution) string {
	names := make([]string, 0, len(resolutions))
	for _, resolution := range resolutions {
		names = append(names, strconv.Quote(string(resolution)))
	}

	return strings.Join(names, " or ")
}

// addFrameImages adds the first and last frames and reports whether there
// were any. A standalone image is the first frame; frames that are not
// images, and a last frame without a first frame, are ignored.
func (b *videoRequestBuilder) addFrameImages() bool {
	firstFrameImage := videoFrameImage(b.options.FrameImages, model.VideoFirstFrame)

	first := firstFrameImage
	if first == nil {
		first = b.options.Image
	}

	if first != nil {
		if topLevel := nonImageVideoMediaType(first); topLevel != "" {
			feature := "image"
			if firstFrameImage != nil {
				feature = "frameImages"
			}

			details := fmt.Sprintf("%s only accepts an image as a frame image; the %q file was ignored.", b.modelID, videoFileMediaType(first))
			if topLevel == "video" {
				details = b.modelID + " does not accept a video as a frame image. The video was ignored."
			}

			b.warn(feature, details)
			first = nil
		}
	}

	last := b.lastFrame(first)

	if first != nil {
		b.addImage(first, roleFirstFrame)
	}

	if last != nil {
		b.addImage(last, roleLastFrame)
	}

	return first != nil || last != nil
}

// lastFrame returns the usable last frame.
func (b *videoRequestBuilder) lastFrame(first model.VideoFile) model.VideoFile {
	last := videoFrameImage(b.options.FrameImages, model.VideoLastFrame)
	if last == nil {
		return nil
	}

	if first == nil {
		b.warn("frameImages", b.modelID+" requires a first_frame when a last_frame is provided. The last_frame was ignored.")

		return nil
	}

	topLevel := nonImageVideoMediaType(last)
	if topLevel == "" {
		return last
	}

	details := fmt.Sprintf("%s only accepts an image as a frame image; the %q last_frame was ignored.", b.modelID, videoFileMediaType(last))
	if topLevel == "video" {
		details = b.modelID + " does not accept a video as a frame image. The last_frame video was ignored."
	}

	b.warn("frameImages", details)

	return nil
}

// addReferences adds the reference images, videos and audios.
// MiniMax-H3-Max has no references, and they cannot be combined with frame
// images.
func (b *videoRequestBuilder) addReferences(usesFrameImages bool) {
	references := b.options.InputReferences
	audioURLs := b.videoOptions.ReferenceAudioURLs
	supportsReferences := b.modelID != modelH3Max

	if !supportsReferences && len(references) > 0 {
		b.warn("inputReferences", "MiniMax-H3-Max does not support reference-to-video inputs. The references were ignored.")
	}

	if !supportsReferences && len(audioURLs) > 0 {
		b.warn("referenceAudioUrls", "MiniMax-H3-Max does not support reference audio. The audio was ignored.")
	}

	usesReferences := supportsReferences && (len(references) > 0 || len(audioURLs) > 0)
	if !usesReferences {
		return
	}

	if usesFrameImages {
		b.warn("inputReferences", "MiniMax-H3 cannot combine frame images with reference inputs. The references were ignored.")

		return
	}

	images, videos := b.routeReferences(references)

	for _, image := range images[:min(len(images), maxReferenceImages)] {
		b.addImage(image, roleReferenceImage)
	}

	if len(images) > maxReferenceImages {
		b.warn("inputReferences", fmt.Sprintf("MiniMax-H3 accepts at most %d reference images. Extra images were ignored.", maxReferenceImages))
	}

	for _, index := range videos[:min(len(videos), maxReferenceVideos)] {
		b.addReferenceVideo(index)
	}

	if len(videos) > maxReferenceVideos {
		b.warn("inputReferences", fmt.Sprintf("MiniMax-H3 accepts at most %d reference videos. Extra videos were ignored.", maxReferenceVideos))
	}

	b.addReferenceAudios(len(images) > 0 || len(videos) > 0)
}

// routeReferences splits the references into images and the positions of
// videos by media type. References without one are treated as images.
func (b *videoRequestBuilder) routeReferences(references []model.VideoFile) ([]model.VideoFile, []int) {
	var images []model.VideoFile
	var videos []int

	for index, file := range references {
		mediaType := videoFileMediaType(file)

		switch topLevel := mediatype.TopLevel(mediaType); {
		case mediaType == "":
			b.warn("inputReferences", "MiniMax-H3 requires an explicit mediaType to route URL references as video or image. Pass { data: url, mediaType: \"video/mp4\" } for video references. The reference was treated as an image.")
			images = append(images, file)

		case topLevel == "video":
			videos = append(videos, index)

		case topLevel == "image":
			images = append(images, file)

		default:
			b.warn("inputReferences", fmt.Sprintf(
				"MiniMax-H3 only accepts image and video references; the %q reference was ignored. Pass reference audio via providerOptions.minimax.referenceAudioUrls.",
				mediaType,
			))
		}
	}

	return images, videos
}

// addReferenceVideo adds the reference video at index. The positions of
// URL videos, not their URLs or data, are kept for the operation.
func (b *videoRequestBuilder) addReferenceVideo(index int) {
	video := b.options.InputReferences[index]

	b.content = append(b.content, videoContent{
		Type: videoContentVideo,
		VideoURL: &videoMediaLink{
			URL: videoMediaURL(video),
		},
		Role: roleReferenceVideo,
	})

	if _, isURL := video.(model.VideoURL); isURL {
		b.resolved.ReferenceVideoIndices = append(b.resolved.ReferenceVideoIndices, index)
	}
}

// addReferenceAudios adds the reference audios, which must accompany a
// reference image or video.
func (b *videoRequestBuilder) addReferenceAudios(hasVisualReferences bool) {
	audioURLs := b.videoOptions.ReferenceAudioURLs
	if len(audioURLs) == 0 {
		return
	}

	if !hasVisualReferences {
		b.warn("referenceAudioUrls", "MiniMax-H3 reference audio must be paired with at least one reference image or video. The audio was ignored.")

		return
	}

	for _, audioURL := range audioURLs[:min(len(audioURLs), maxReferenceAudios)] {
		b.content = append(b.content, videoContent{
			Type: videoContentAudio,
			AudioURL: &videoMediaLink{
				URL: audioURL,
			},
			Role: roleReferenceAudio,
		})
	}

	if len(audioURLs) > maxReferenceAudios {
		b.warn("referenceAudioUrls", fmt.Sprintf("MiniMax-H3 accepts at most %d reference audios. Extra audios were ignored.", maxReferenceAudios))
	}
}

func (b *videoRequestBuilder) addImage(image model.VideoFile, role string) {
	b.content = append(b.content, videoContent{
		Type: videoContentImage,
		ImageURL: &videoMediaLink{
			URL: videoMediaURL(image),
		},
		Role: role,
	})

	b.resolved.ImageCount++
}

// ratio returns the aspect ratio. Text-to-video falls back to 16:9 and
// does not accept "adaptive"; with frame images the ratio follows the
// image, so a requested ratio is ignored.
func (b *videoRequestBuilder) ratio(usesFrameImages bool) VideoRatio {
	isTextToVideo := len(b.content) == 1
	ratio := b.videoOptions.Ratio
	requested := VideoRatio(b.options.AspectRatio)

	if ratio == "" && requested != "" {
		switch {
		case slices.Contains(videoRatios, requested):
			ratio = requested

		case isTextToVideo:
			b.warn("aspectRatio", fmt.Sprintf("%s does not support the aspect ratio %q. Using the default (%s).", b.modelID, requested, defaultVideoRatio))
			ratio = defaultVideoRatio

		default:
			b.warn("aspectRatio", fmt.Sprintf("%s does not support the aspect ratio %q. Using the provider default (adaptive).", b.modelID, requested))
		}
	}

	if ratio == VideoRatioAdaptive && isTextToVideo {
		b.warn("aspectRatio", fmt.Sprintf("%s text-to-video does not support the adaptive aspect ratio. Using the default (%s).", b.modelID, defaultVideoRatio))
		ratio = defaultVideoRatio
	}

	if usesFrameImages && ratio != "" {
		b.warn("aspectRatio", b.modelID+" derives the aspect ratio from the frame image; the requested ratio was ignored.")
		ratio = ""
	}

	if ratio == "" && isTextToVideo {
		return defaultVideoRatio
	}

	return ratio
}

// duration returns a whole number of seconds within the model's range.
// MiniMax-H3 starts at 4 seconds and other models at 5; fractional values
// are rounded before clamping.
func (b *videoRequestBuilder) duration() int {
	if b.options.Duration == nil {
		return defaultVideoDurationSeconds
	}

	minimum := 5
	if b.modelID == modelH3 {
		minimum = 4
	}

	requested := *b.options.Duration
	duration := requested

	if duration != math.Trunc(duration) {
		duration = math.Round(duration)

		b.warn("duration", fmt.Sprintf(
			"%s requires a whole number of seconds. The requested duration of %s was rounded to %s.",
			b.modelID,
			formatVideoNumber(requested),
			formatVideoNumber(duration),
		))
	}

	switch {
	case duration > maxVideoDurationSeconds:
		b.warn("duration", fmt.Sprintf(
			"%s supports at most %d seconds. The requested duration of %s was clamped to %d.",
			b.modelID,
			maxVideoDurationSeconds,
			formatVideoNumber(requested),
			maxVideoDurationSeconds,
		))

		return maxVideoDurationSeconds

	case duration < float64(minimum):
		b.warn("duration", fmt.Sprintf(
			"%s requires at least %d seconds. The requested duration of %s was clamped to %d.",
			b.modelID,
			minimum,
			formatVideoNumber(requested),
			minimum,
		))

		return minimum

	default:
		return int(duration)
	}
}

// videoFrameImage returns the frame image of frameType, or nil.
func videoFrameImage(frames []model.VideoFrameImage, frameType model.VideoFrameType) model.VideoFile {
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

// nonImageVideoMediaType returns the top-level media type of a file declared as
// something other than an image, or empty.
func nonImageVideoMediaType(file model.VideoFile) string {
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

// videoMediaURL passes URLs through and encodes inline data as a data URL.
func videoMediaURL(file model.VideoFile) string {
	switch typed := file.(type) {
	case model.VideoURL:
		return typed.URL

	case model.VideoBytes:
		return dataurl.Encode(typed.MediaType, typed.Data)

	default:
		return ""
	}
}

func formatVideoNumber(value float64) string {
	return strconv.FormatFloat(value, 'f', -1, 64)
}
