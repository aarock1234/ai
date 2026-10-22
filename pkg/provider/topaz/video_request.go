package topaz

import (
	"encoding/json"
	"fmt"
	"math"
	"path"
	"slices"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// Defaults of the enhanced video.
const (
	defaultAudioTransfer   = "Copy"
	defaultAudioCodec      = "AAC"
	defaultOutputContainer = "mp4"
	noAudioTransfer        = "None"
	defaultUploadMediaType = "application/octet-stream"
	externalSourceProvider = "s3"
)

// videoAPIModelIDs maps readable video model IDs onto Topaz API model
// names. Other IDs, such as "slp-2.6", are sent unchanged.
var videoAPIModelIDs = map[string]string{
	"proteus":               "prob-4",
	"starlight-precise-2.6": "slp-2.6",
}

// mediaTypeContainers maps input media types onto source containers.
var mediaTypeContainers = map[string]string{
	"video/mp4":        "mp4",
	"video/quicktime":  "mov",
	"video/mov":        "mov",
	"video/x-matroska": "mkv",
	"video/matroska":   "mkv",
	"video/webm":       "webm",
	"video/x-msvideo":  "avi",
	"video/avi":        "avi",
	"video/mpeg":       "mpeg",
	"video/mp2t":       "ts",
	"video/x-ms-wmv":   "wmv",
	"video/x-flv":      "flv",
	"video/3gpp":       "3gp",
	"video/x-m4v":      "m4v",
	"application/mxf":  "mxf",
}

// containerMediaTypes maps containers onto media types.
var containerMediaTypes = map[string]string{
	"mp4":  "video/mp4",
	"m4v":  "video/mp4",
	"mov":  "video/quicktime",
	"mkv":  "video/x-matroska",
	"webm": "video/webm",
	"avi":  "video/x-msvideo",
	"mpeg": "video/mpeg",
	"mpg":  "video/mpeg",
	"ts":   "video/mp2t",
	"wmv":  "video/x-ms-wmv",
	"flv":  "video/x-flv",
	"3gp":  "video/3gpp",
	"mxf":  "application/mxf",
}

// expressRequest creates an express video request.
type expressRequest struct {
	Source  expressSource       `json:"source"`
	Output  expressOutput       `json:"output"`
	Filters []jsonobject.Object `json:"filters"`
}

type expressSource struct {
	Container  string          `json:"container"`
	Duration   *float64        `json:"duration,omitempty"`
	FrameCount *int            `json:"frameCount,omitempty"`
	FrameRate  *float64        `json:"frameRate,omitempty"`
	Resolution *videoDimension `json:"resolution,omitempty"`

	// External is the URL Topaz fetches the input from; Size is the byte
	// size of an uploaded input.
	External *externalSource `json:"external,omitempty"`
	Size     *int            `json:"size,omitempty"`
}

type externalSource struct {
	Provider     string `json:"provider"`
	PresignedURL string `json:"presignedUrl"`
}

type videoDimension struct {
	Width  int `json:"width"`
	Height int `json:"height"`
}

type expressOutput struct {
	Resolution              videoDimension `json:"resolution"`
	FrameRate               *float64       `json:"frameRate,omitempty"`
	AudioTransfer           string         `json:"audioTransfer"`
	AudioCodec              string         `json:"audioCodec,omitempty"`
	AudioBitrate            *string        `json:"audioBitrate,omitempty"`
	VideoEncoder            string         `json:"videoEncoder,omitempty"`
	VideoProfile            *string        `json:"videoProfile,omitempty"`
	VideoBitrate            *string        `json:"videoBitrate,omitempty"`
	DynamicCompressionLevel string         `json:"dynamicCompressionLevel,omitempty"`
	CropToFit               *bool          `json:"cropToFit,omitempty"`
	Container               string         `json:"container"`
}

// sourceMetadata is the complete metadata of the input video.
type sourceMetadata struct {
	width      int
	height     int
	duration   float64
	frameRate  float64
	frameCount int
}

// expressPlan is an express request with the input to upload, if any.
type expressPlan struct {
	request expressRequest

	// upload is the inline input, uploaded after the request is created.
	upload []byte

	// uploadMediaType is the content type of the upload.
	uploadMediaType string
}

// buildExpressPlan converts the call options into an express request.
func (m *VideoModel) buildExpressPlan(options model.VideoOptions, videoOptions VideoOptions) (expressPlan, []model.Warning, error) {
	warnings := unsupportedVideoSettings(options)

	input, inputWarnings, err := selectInputVideo(options)
	if err != nil {
		return expressPlan{}, nil, err
	}

	warnings = append(warnings, inputWarnings...)

	declaredContainer := ""
	if videoOptions.Source != nil {
		declaredContainer = videoOptions.Source.Container
	}

	container, err := resolveContainer(input, declaredContainer)
	if err != nil {
		return expressPlan{}, nil, err
	}

	metadata, err := resolveSourceMetadata(videoOptions.Source)
	if err != nil {
		return expressPlan{}, nil, err
	}

	output, err := buildOutput(options, videoOptions.Output, container, metadata)
	if err != nil {
		return expressPlan{}, nil, err
	}

	filter, err := m.filter(videoOptions)
	if err != nil {
		return expressPlan{}, nil, err
	}

	plan := expressPlan{
		request: expressRequest{
			Source:  expressSourceOf(input, container, metadata),
			Output:  output,
			Filters: append([]jsonobject.Object{filter}, videoOptions.AdditionalFilters...),
		},
		uploadMediaType: defaultUploadMediaType,
	}

	if mediaType, ok := containerMediaTypes[container]; ok {
		plan.uploadMediaType = mediaType
	}

	if bytes, ok := input.(model.VideoBytes); ok {
		plan.upload = bytes.Data
	}

	return plan, warnings, nil
}

// unsupportedVideoSettings reports the call settings that enhancement does
// not use. An empty prompt is expected and not reported.
func unsupportedVideoSettings(options model.VideoOptions) []model.Warning {
	var warnings []model.Warning

	warn := func(feature string, details string) {
		warnings = append(warnings, model.UnsupportedWarning(feature, details))
	}

	if strings.TrimSpace(options.Prompt) != "" {
		warn("prompt", "Topaz video models enhance an existing video and do not take a text prompt. The prompt was ignored.")
	}

	if options.AspectRatio != "" {
		warn("aspectRatio", "Topaz video models do not support aspectRatio. Use `resolution`, or the `output` provider option, to set the output dimensions.")
	}

	if options.Seed != nil {
		warn("seed", "Topaz video models do not support seed.")
	}

	if options.Duration != nil {
		warn("duration", "Topaz video models enhance the whole input video, so duration was ignored. Pass the input duration via the `source.duration` provider option instead.")
	}

	if options.GenerateAudio != nil {
		warn("generateAudio", "Topaz video models do not generate audio. Use the `output.audioTransfer` provider option to control how the input audio track is carried over.")
	}

	if len(options.FrameImages) > 0 {
		warn("frameImages", "Topaz video models do not support first/last frame generation. The frame images were ignored.")
	}

	if options.N > 1 {
		warn("n", "Topaz video models enhance one video per call. Only 1 video will be produced.")
	}

	return warnings
}

// selectInputVideo returns the first video reference.
func selectInputVideo(options model.VideoOptions) (model.VideoFile, []model.Warning, error) {
	var videos []model.VideoFile

	for _, reference := range options.InputReferences {
		if isVideoReference(reference) {
			videos = append(videos, reference)
		}
	}

	if len(videos) == 0 {
		if options.Image != nil {
			return nil, nil, fmt.Errorf(
				"%w: image: Topaz video models enhance an existing video, not a still image. Pass the input video via inputReferences",
				model.ErrInvalidArgument,
			)
		}

		return nil, nil, fmt.Errorf(
			"%w: inputReferences: Topaz video models require an input video. Pass it via inputReferences; for URL references, include the media type so the reference is recognized as a video",
			model.ErrInvalidArgument,
		)
	}

	var warnings []model.Warning
	if len(options.InputReferences) > 1 {
		warnings = append(warnings, model.UnsupportedWarning(
			"inputReferences",
			"Topaz video models enhance a single video. Only the first video reference was used.",
		))
	}

	return videos[0], warnings, nil
}

// isVideoReference reports whether a reference is a video, by its media
// type or, for URLs without one, its file extension.
func isVideoReference(reference model.VideoFile) bool {
	switch typed := reference.(type) {
	case model.VideoBytes:
		return isVideoMediaType(typed.MediaType)

	case model.VideoURL:
		if typed.MediaType != "" {
			return isVideoMediaType(typed.MediaType)
		}

		_, ok := containerFromURL(typed.URL)

		return ok

	default:
		return false
	}
}

func isVideoMediaType(mediaType string) bool {
	return strings.HasPrefix(strings.ToLower(mediaType), "video/")
}

// resolveContainer returns the declared container, or the one of the input
// media type or URL extension.
func resolveContainer(input model.VideoFile, declared string) (string, error) {
	if declared != "" {
		return declared, nil
	}

	switch typed := input.(type) {
	case model.VideoBytes:
		if container, ok := mediaTypeContainers[strings.ToLower(typed.MediaType)]; ok {
			return container, nil
		}

		return "", fmt.Errorf(
			"%w: inputReferences: could not map the media type %q onto a Topaz container. Set the source.container provider option explicitly",
			model.ErrInvalidArgument,
			typed.MediaType,
		)

	case model.VideoURL:
		if container, ok := mediaTypeContainers[strings.ToLower(typed.MediaType)]; ok {
			return container, nil
		}

		if container, ok := containerFromURL(typed.URL); ok {
			return container, nil
		}

		return "", fmt.Errorf(
			"%w: inputReferences: could not determine the container of the input video at %q. Set the source.container provider option, or pass the media type on the reference",
			model.ErrInvalidArgument,
			typed.URL,
		)

	default:
		return "", fmt.Errorf("%w: video input %T", model.ErrUnsupported, input)
	}
}

// containerFromURL returns the container of a URL's file extension; "qt"
// is mov.
func containerFromURL(location string) (string, bool) {
	withoutQuery, _, _ := strings.Cut(location, "?")
	withoutFragment, _, _ := strings.Cut(withoutQuery, "#")

	extension := strings.ToLower(strings.TrimPrefix(path.Ext(withoutFragment), "."))
	if extension == "qt" {
		return "mov", true
	}

	if slices.Contains(videoSourceContainers, extension) {
		return extension, true
	}

	return "", false
}

// resolveSourceMetadata returns the input metadata, which is all or
// nothing. The frame count is derived when only the duration and frame rate
// are known.
func resolveSourceMetadata(source *VideoSource) (*sourceMetadata, error) {
	if source == nil || source.metadataIsEmpty() {
		return nil, nil
	}

	var missing []string

	if source.Width == nil {
		missing = append(missing, "source.width")
	}

	if source.Height == nil {
		missing = append(missing, "source.height")
	}

	if source.Duration == nil {
		missing = append(missing, "source.duration")
	}

	if source.FrameRate == nil {
		missing = append(missing, "source.frameRate")
	}

	if len(missing) > 0 {
		return nil, fmt.Errorf(
			"%w: providerOptions.topaz.source: source metadata must be complete. Missing: %s",
			model.ErrInvalidArgument,
			strings.Join(missing, ", "),
		)
	}

	frameCount := int(math.Round(*source.Duration * *source.FrameRate))
	if source.FrameCount != nil {
		frameCount = *source.FrameCount
	}

	return &sourceMetadata{
		width:      *source.Width,
		height:     *source.Height,
		duration:   *source.Duration,
		frameRate:  *source.FrameRate,
		frameCount: frameCount,
	}, nil
}

// expressSourceOf describes the input. Topaz fetches URL inputs itself, so
// they are never downloaded; inline inputs are uploaded afterwards.
func expressSourceOf(input model.VideoFile, container string, metadata *sourceMetadata) expressSource {
	source := expressSource{
		Container: container,
	}

	if metadata != nil {
		source.Duration = &metadata.duration
		source.FrameCount = &metadata.frameCount
		source.FrameRate = &metadata.frameRate
		source.Resolution = &videoDimension{
			Width:  metadata.width,
			Height: metadata.height,
		}
	}

	switch typed := input.(type) {
	case model.VideoURL:
		source.External = &externalSource{
			Provider:     externalSourceProvider,
			PresignedURL: typed.URL,
		}

	case model.VideoBytes:
		source.Size = new(len(typed.Data))
	}

	return source
}

// buildOutput resolves the output settings. The output dimensions come from
// the output option, the resolution call option or the source metadata.
func buildOutput(
	options model.VideoOptions,
	output *VideoOutput,
	sourceContainer string,
	metadata *sourceMetadata,
) (expressOutput, error) {
	if output == nil {
		output = &VideoOutput{}
	}

	width, height := parseResolution(options.Resolution)
	width = firstInt(output.Width, width)
	height = firstInt(output.Height, height)

	if metadata != nil {
		width = firstInt(width, &metadata.width)
		height = firstInt(height, &metadata.height)
	}

	if width == nil || height == nil {
		return expressOutput{}, fmt.Errorf(
			"%w: resolution: Topaz needs the output resolution. Set the resolution call option or the output.width and output.height provider options",
			model.ErrInvalidArgument,
		)
	}

	audioTransfer := output.AudioTransfer
	if audioTransfer == "" {
		audioTransfer = defaultAudioTransfer
	}

	audioCodec := output.AudioCodec
	if audioCodec == "" && audioTransfer != noAudioTransfer {
		audioCodec = defaultAudioCodec
	}

	return expressOutput{
		Resolution: videoDimension{
			Width:  *width,
			Height: *height,
		},
		FrameRate:               outputFrameRate(options, output, metadata),
		AudioTransfer:           audioTransfer,
		AudioCodec:              audioCodec,
		AudioBitrate:            output.AudioBitrate,
		VideoEncoder:            output.VideoEncoder,
		VideoProfile:            output.VideoProfile,
		VideoBitrate:            output.VideoBitrate,
		DynamicCompressionLevel: output.DynamicCompressionLevel,
		CropToFit:               output.CropToFit,
		Container:               outputContainer(output, sourceContainer),
	}, nil
}

// outputFrameRate prefers the output option, then the fps call option, then
// the source frame rate.
func outputFrameRate(options model.VideoOptions, output *VideoOutput, metadata *sourceMetadata) *float64 {
	switch {
	case output.FrameRate != nil:
		return output.FrameRate

	case options.FPS != 0:
		return new(float64(options.FPS))

	case metadata != nil:
		return &metadata.frameRate

	default:
		return nil
	}
}

// outputContainer mirrors the Topaz container rules, so that Status reports
// the media type of the file Topaz produces.
func outputContainer(output *VideoOutput, sourceContainer string) string {
	switch output.VideoEncoder {
	case "ProRes":
		return "mov"

	case "AV1", "VP9":
		return "mp4"
	}

	if output.Container != "" {
		return output.Container
	}

	// The default H265 encoder only writes mp4, mov and mkv.
	if sourceContainer == "mov" || sourceContainer == "mkv" {
		return sourceContainer
	}

	return defaultOutputContainer
}

// parseResolution reads a "WIDTHxHEIGHT" resolution. Unreadable sides are
// nil.
func parseResolution(resolution string) (*int, *int) {
	if resolution == "" {
		return nil, nil
	}

	widthText, heightText, _ := strings.Cut(resolution, "x")

	return parseSide(widthText), parseSide(heightText)
}

func parseSide(text string) *int {
	value, err := strconv.Atoi(text)
	if err != nil {
		return nil
	}

	return &value
}

// filter returns the model's filter: the API model name, the typed
// settings and the untyped filter overrides, in that order.
func (m *VideoModel) filter(videoOptions VideoOptions) (jsonobject.Object, error) {
	encoded, err := json.Marshal(videoOptions.VideoFilterSettings)
	if err != nil {
		return nil, fmt.Errorf("encode video filter: %w", err)
	}

	settings, err := jsonobject.Parse(encoded)
	if err != nil {
		return nil, fmt.Errorf("encode video filter: %w", err)
	}

	filter := jsonobject.Object{
		{
			Key:   "model",
			Value: jsonobject.EncodeString(m.apiModelID()),
		},
	}

	return filter.Merge(settings).Merge(videoOptions.Filter), nil
}

// apiModelID returns the Topaz API model name of the model.
func (m *VideoModel) apiModelID() string {
	if name, ok := videoAPIModelIDs[m.modelID]; ok {
		return name
	}

	return m.modelID
}
