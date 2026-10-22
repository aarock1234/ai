package klingai

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// generationKind is the endpoint a generation uses.
type generationKind string

const (
	kindTextToVideo       generationKind = "t2v"
	kindImageToVideo      generationKind = "i2v"
	kindMultiImageToVideo generationKind = "mi2v"
	kindMotionControl     generationKind = "motion-control"
)

// kindEndpoints are the task endpoints of each generation kind.
var kindEndpoints = map[generationKind]string{
	kindTextToVideo:       "/v1/videos/text2video",
	kindImageToVideo:      "/v1/videos/image2video",
	kindMultiImageToVideo: "/v1/videos/multi-image2video",
	kindMotionControl:     "/v1/videos/motion-control",
}

// kindSuffixes are the model ID suffixes of each kind. Multi-image models
// use the image-to-video IDs.
var kindSuffixes = map[generationKind]string{
	kindTextToVideo:       "-t2v",
	kindImageToVideo:      "-i2v",
	kindMultiImageToVideo: "-i2v",
	kindMotionControl:     "-motion-control",
}

// trailingZeroVersion matches the ".0" of versions such as "v3.0".
var trailingZeroVersion = regexp.MustCompile(`\.0$`)

// videoRequest is the body of every task endpoint. Each kind sets its own
// subset of the fields.
type videoRequest struct {
	ModelName            string                `json:"model_name"`
	Prompt               string                `json:"prompt,omitempty"`
	Image                string                `json:"image,omitempty"`
	ImageURL             string                `json:"image_url,omitempty"`
	ImageList            []imageListItem       `json:"image_list,omitempty"`
	ImageTail            string                `json:"image_tail,omitempty"`
	VideoURL             string                `json:"video_url,omitempty"`
	CharacterOrientation *CharacterOrientation `json:"character_orientation,omitempty"`
	Mode                 *VideoMode            `json:"mode,omitempty"`
	NegativePrompt       *string               `json:"negative_prompt,omitempty"`
	Sound                *Sound                `json:"sound,omitempty"`
	CFGScale             *float64              `json:"cfg_scale,omitempty"`
	CameraControl        *CameraControl        `json:"camera_control,omitempty"`
	StaticMask           *string               `json:"static_mask,omitempty"`
	DynamicMasks         []DynamicMask         `json:"dynamic_masks,omitempty"`
	AspectRatio          string                `json:"aspect_ratio,omitempty"`
	Duration             string                `json:"duration,omitempty"`
	MultiShot            *bool                 `json:"multi_shot,omitempty"`
	ShotType             *ShotType             `json:"shot_type,omitempty"`
	MultiPrompt          []MultiPromptShot     `json:"multi_prompt,omitempty"`
	ElementList          []Element             `json:"element_list,omitempty"`
	VoiceList            []Voice               `json:"voice_list,omitempty"`
	KeepOriginalSound    *KeepOriginalSound    `json:"keep_original_sound,omitempty"`
	WatermarkInfo        *watermarkInfo        `json:"watermark_info,omitempty"`
}

type imageListItem struct {
	Image string `json:"image"`
}

type watermarkInfo struct {
	Enabled bool `json:"enabled"`
}

// videoRequestBuilder builds one request and collects its warnings.
type videoRequestBuilder struct {
	modelID      string
	options      model.VideoStartOptions
	videoOptions VideoOptions
	warnings     []model.Warning
}

// builtRequest is a request body and the endpoint it is sent to.
type builtRequest struct {
	body         jsonobject.Object
	endpointPath string
}

// buildRequest converts the call options into the body of the endpoint the
// model ID and the references select.
func (m *VideoModel) buildRequest(options model.VideoStartOptions) (builtRequest, []model.Warning, error) {
	kind, err := m.detectKind()
	if err != nil {
		return builtRequest{}, nil, err
	}

	videoOptions, passthrough, err := readVideoOptions(options.ProviderOptions)
	if err != nil {
		return builtRequest{}, nil, err
	}

	builder := &videoRequestBuilder{
		modelID:      m.modelID,
		options:      options,
		videoOptions: videoOptions,
	}

	references := builder.referenceImages()
	if kind == kindImageToVideo && references != nil {
		kind = kindMultiImageToVideo
	}

	request, err := builder.request(kind, references)
	if err != nil {
		return builtRequest{}, nil, err
	}

	if references != nil && kind != kindMultiImageToVideo {
		builder.warn(
			"inputReferences",
			"KlingAI only supports inputReferences (reference-to-video) on image-to-video models. The reference images were ignored.",
		)
	}

	builder.addUniversalWarnings()

	body, err := requestObject(request, passthrough, options.WebhookURL)
	if err != nil {
		return builtRequest{}, nil, err
	}

	return builtRequest{
		body:         body,
		endpointPath: kindEndpoints[kind],
	}, builder.warnings, nil
}

// detectKind reads the generation kind from the model ID suffix.
func (m *VideoModel) detectKind() (generationKind, error) {
	switch {
	case strings.HasSuffix(m.modelID, "-t2v"):
		return kindTextToVideo, nil
	case strings.HasSuffix(m.modelID, "-i2v"):
		return kindImageToVideo, nil
	case strings.HasSuffix(m.modelID, "-motion-control"):
		return kindMotionControl, nil
	default:
		return "", model.NoSuchModelError(m.provider.name, model.VideoModelKind, m.modelID)
	}
}

// request builds the body of one generation kind.
func (b *videoRequestBuilder) request(kind generationKind, references []model.VideoFile) (videoRequest, error) {
	switch kind {
	case kindMotionControl:
		return b.motionControlRequest()
	case kindTextToVideo:
		return b.textToVideoRequest(), nil
	case kindMultiImageToVideo:
		return b.multiImageRequest(references)
	default:
		return b.imageToVideoRequest()
	}
}

func (b *videoRequestBuilder) textToVideoRequest() videoRequest {
	request := b.baseRequest(kindTextToVideo)
	request.Sound = b.sound()
	request.CameraControl = b.videoOptions.CameraControl
	request.AspectRatio = b.options.AspectRatio
	request.Duration = b.duration()
	request.MultiShot = b.videoOptions.MultiShot
	request.ShotType = b.videoOptions.ShotType
	request.MultiPrompt = b.videoOptions.MultiPrompt
	request.VoiceList = b.videoOptions.VoiceList

	if b.startImage() != nil {
		b.warn(
			"image",
			"KlingAI text-to-video does not support image input. Use an image-to-video model instead.",
		)
	}

	return request
}

func (b *videoRequestBuilder) imageToVideoRequest() (videoRequest, error) {
	request := b.baseRequest(kindImageToVideo)

	if startImage := b.startImage(); startImage != nil {
		image, err := imageString(startImage)
		if err != nil {
			return videoRequest{}, err
		}

		request.Image = image
	}

	imageTail, err := b.imageTail()
	if err != nil {
		return videoRequest{}, err
	}

	request.ImageTail = imageTail
	request.Sound = b.sound()
	request.CameraControl = b.videoOptions.CameraControl
	request.StaticMask = b.videoOptions.StaticMask
	request.DynamicMasks = b.videoOptions.DynamicMasks
	request.MultiShot = b.videoOptions.MultiShot
	request.ShotType = b.videoOptions.ShotType
	request.MultiPrompt = b.videoOptions.MultiPrompt
	request.ElementList = b.videoOptions.ElementList
	request.VoiceList = b.videoOptions.VoiceList
	request.Duration = b.duration()

	if b.options.AspectRatio != "" {
		b.warn(
			"aspectRatio",
			"KlingAI image-to-video does not support aspectRatio. The output dimensions are determined by the input image.",
		)
	}

	return request, nil
}

func (b *videoRequestBuilder) multiImageRequest(references []model.VideoFile) (videoRequest, error) {
	request := b.baseRequest(kindMultiImageToVideo)

	for _, reference := range references {
		image, err := imageString(reference)
		if err != nil {
			return videoRequest{}, err
		}

		request.ImageList = append(request.ImageList, imageListItem{
			Image: image,
		})
	}

	request.AspectRatio = b.options.AspectRatio
	request.Duration = b.duration()

	if b.startImage() != nil {
		b.warn(
			"image",
			"KlingAI reference-to-video does not support a separate start frame. Provide all guidance images via inputReferences instead.",
		)
	}

	imageTail, err := b.imageTail()
	if err != nil {
		return videoRequest{}, err
	}

	if imageTail != "" {
		b.warn(
			"frameImages",
			"KlingAI reference-to-video does not support a last frame (image_tail). Provide all guidance images via inputReferences instead.",
		)
	}

	return request, nil
}

func (b *videoRequestBuilder) motionControlRequest() (videoRequest, error) {
	videoOptions := b.videoOptions

	if videoOptions.VideoURL == nil || *videoOptions.VideoURL == "" ||
		videoOptions.CharacterOrientation == nil || *videoOptions.CharacterOrientation == "" ||
		videoOptions.Mode == nil || *videoOptions.Mode == "" {
		return videoRequest{}, fmt.Errorf(
			"%w: KlingAI Motion Control requires providerOptions.klingai with videoUrl, characterOrientation, and mode",
			model.ErrInvalidArgument,
		)
	}

	request := videoRequest{
		ModelName:            apiModelName(b.modelID, kindMotionControl),
		Prompt:               b.options.Prompt,
		VideoURL:             *videoOptions.VideoURL,
		CharacterOrientation: videoOptions.CharacterOrientation,
		Mode:                 videoOptions.Mode,
		KeepOriginalSound:    videoOptions.KeepOriginalSound,
		ElementList:          videoOptions.ElementList,
		WatermarkInfo:        b.watermarkInfo(),
	}

	if startImage := b.startImage(); startImage != nil {
		image, err := imageString(startImage)
		if err != nil {
			return videoRequest{}, err
		}

		request.ImageURL = image
	}

	if b.options.AspectRatio != "" {
		b.warn(
			"aspectRatio",
			"KlingAI Motion Control does not support aspectRatio. The output dimensions are determined by the reference image/video.",
		)
	}

	if b.options.Duration != nil && *b.options.Duration != 0 {
		b.warn(
			"duration",
			"KlingAI Motion Control does not support custom duration. The output duration matches the reference video duration.",
		)
	}

	return request, nil
}

// baseRequest holds the fields that text-, image- and multi-image-to-video
// requests share.
func (b *videoRequestBuilder) baseRequest(kind generationKind) videoRequest {
	return videoRequest{
		ModelName:      apiModelName(b.modelID, kind),
		Prompt:         b.options.Prompt,
		NegativePrompt: b.videoOptions.NegativePrompt,
		CFGScale:       b.videoOptions.CFGScale,
		Mode:           b.videoOptions.Mode,
		WatermarkInfo:  b.watermarkInfo(),
	}
}

// sound prefers the standard GenerateAudio option over the sound option.
func (b *videoRequestBuilder) sound() *Sound {
	if b.options.GenerateAudio == nil {
		return b.videoOptions.Sound
	}

	if *b.options.GenerateAudio {
		return new(SoundOn)
	}

	return new(SoundOff)
}

func (b *videoRequestBuilder) duration() string {
	if b.options.Duration == nil {
		return ""
	}

	return strconv.FormatFloat(*b.options.Duration, 'f', -1, 64)
}

func (b *videoRequestBuilder) watermarkInfo() *watermarkInfo {
	if b.videoOptions.WatermarkEnabled == nil {
		return nil
	}

	return &watermarkInfo{
		Enabled: *b.videoOptions.WatermarkEnabled,
	}
}

// startImage is the first frame or the image. KlingAI does not accept
// videos as frames; a video is ignored with a warning.
func (b *videoRequestBuilder) startImage() model.VideoFile {
	startImage := frameImage(b.options.FrameImages, model.VideoFirstFrame)
	if startImage == nil {
		startImage = b.options.Image
	}

	if startImage != nil && isVideoFile(startImage) {
		b.warnVideoFrame()

		return nil
	}

	return startImage
}

// imageTail is the last frame, or the imageTail option.
func (b *videoRequestBuilder) imageTail() (string, error) {
	lastFrame := frameImage(b.options.FrameImages, model.VideoLastFrame)

	if lastFrame == nil {
		if b.videoOptions.ImageTail == nil {
			return "", nil
		}

		return *b.videoOptions.ImageTail, nil
	}

	if isVideoFile(lastFrame) {
		b.warnVideoFrame()

		return "", nil
	}

	return imageString(lastFrame)
}

// referenceImages returns the image references, or nil when there are none
// or frame images exclude them. Video references are ignored with a
// warning.
func (b *videoRequestBuilder) referenceImages() []model.VideoFile {
	if len(b.options.FrameImages) > 0 || len(b.options.InputReferences) == 0 {
		return nil
	}

	var images []model.VideoFile

	for _, reference := range b.options.InputReferences {
		if isVideoFile(reference) {
			b.warn(
				"inputReferences",
				"KlingAI does not support video reference inputs; the video reference was ignored.",
			)

			continue
		}

		images = append(images, reference)
	}

	return images
}

// addUniversalWarnings warns about the standard settings no KlingAI model
// supports.
func (b *videoRequestBuilder) addUniversalWarnings() {
	if b.options.Resolution != "" {
		b.warn("resolution", "KlingAI video models do not support the resolution option.")
	}

	if b.options.Seed != nil && *b.options.Seed != 0 {
		b.warn("seed", "KlingAI video models do not support seed for deterministic generation.")
	}

	if b.options.FPS != 0 {
		b.warn("fps", "KlingAI video models do not support custom FPS.")
	}

	if b.options.N > 1 {
		b.warn(
			"n",
			"KlingAI video models do not support generating multiple videos per call. Only 1 video will be generated.",
		)
	}
}

func (b *videoRequestBuilder) warnVideoFrame() {
	b.warn("frameImages", "KlingAI does not accept video as a frame image; it was ignored.")
}

func (b *videoRequestBuilder) warn(feature string, details string) {
	b.warnings = append(b.warnings, model.UnsupportedWarning(feature, details))
}

// requestObject encodes the request followed by the passthrough options,
// which replace fields of the same name, and the webhook callback URL.
func requestObject(request videoRequest, passthrough jsonobject.Object, webhookURL string) (jsonobject.Object, error) {
	encoded, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("encode request body: %w", err)
	}

	body, err := jsonobject.Parse(encoded)
	if err != nil {
		return nil, fmt.Errorf("encode request body: %w", err)
	}

	body = body.Merge(passthrough)

	if webhookURL != "" {
		body = body.With("callback_url", jsonobject.EncodeString(webhookURL))
	}

	return body, nil
}

// apiModelName derives the API model name from the model ID: the mode
// suffix is removed, a trailing ".0" version is dropped, and other dots
// become hyphens. "kling-v2.6-t2v" becomes "kling-v2-6" and
// "kling-v3.0-t2v" becomes "kling-v3".
func apiModelName(modelID string, kind generationKind) string {
	baseName := strings.TrimSuffix(modelID, kindSuffixes[kind])
	baseName = trailingZeroVersion.ReplaceAllString(baseName, "")

	return strings.ReplaceAll(baseName, ".", "-")
}

// imageString returns the URL of a URL image or the bare base64 of inline
// bytes.
func imageString(file model.VideoFile) (string, error) {
	switch file := file.(type) {
	case model.VideoURL:
		return file.URL, nil
	case model.VideoBytes:
		return base64.StdEncoding.EncodeToString(file.Data), nil
	default:
		return "", fmt.Errorf("%w: video file %T", model.ErrUnsupported, file)
	}
}

// isVideoFile reports whether the file's media type is a video type.
func isVideoFile(file model.VideoFile) bool {
	var mediaType string

	switch file := file.(type) {
	case model.VideoURL:
		mediaType = file.MediaType
	case model.VideoBytes:
		mediaType = file.MediaType
	}

	topLevel, _, _ := strings.Cut(mediaType, "/")

	return strings.EqualFold(topLevel, "video")
}

func frameImage(frames []model.VideoFrameImage, frameType model.VideoFrameType) model.VideoFile {
	for _, frame := range frames {
		if frame.FrameType == frameType {
			return frame.Image
		}
	}

	return nil
}
