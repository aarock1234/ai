package heygen

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"math"
	"slices"
	"strings"
	"unicode/utf16"

	"github.com/aarock1234/ai/pkg/model"
)

// Limits and defaults of the standard settings.
const (
	maxPromptLength = 32000
	minDuration     = 5
	maxDuration     = 15
	defaultDuration = 5
	maxSeed         = math.MaxUint32
	nativeFPS       = 24
)

// assetKind is the media kind an input asset must have.
type assetKind string

const (
	kindImage assetKind = "image"
	kindVideo assetKind = "video"
	kindAudio assetKind = "audio"
)

// frameSize is the size class and aspect ratio of a documented frame size.
type frameSize struct {
	resolution  VideoResolution
	aspectRatio string
}

// frameSizes maps "WIDTHxHEIGHT" frame sizes onto size classes, whose names
// do not consistently equal the short edge.
var frameSizes = map[string]frameSize{
	"960x416": {
		resolution:  Resolution480p,
		aspectRatio: "21:9",
	},
	"832x480": {
		resolution:  Resolution480p,
		aspectRatio: "16:9",
	},
	"640x480": {
		resolution:  Resolution480p,
		aspectRatio: "4:3",
	},
	"480x480": {
		resolution:  Resolution480p,
		aspectRatio: "1:1",
	},
	"480x640": {
		resolution:  Resolution480p,
		aspectRatio: "3:4",
	},
	"480x832": {
		resolution:  Resolution480p,
		aspectRatio: "9:16",
	},
	"1536x672": {
		resolution:  Resolution768p,
		aspectRatio: "21:9",
	},
	"1344x768": {
		resolution:  Resolution768p,
		aspectRatio: "16:9",
	},
	"1024x768": {
		resolution:  Resolution768p,
		aspectRatio: "4:3",
	},
	"768x768": {
		resolution:  Resolution768p,
		aspectRatio: "1:1",
	},
	"768x1024": {
		resolution:  Resolution768p,
		aspectRatio: "3:4",
	},
	"768x1344": {
		resolution:  Resolution768p,
		aspectRatio: "9:16",
	},
	"1890x1080": {
		resolution:  Resolution1080p,
		aspectRatio: "16:9",
	},
	"1080x1890": {
		resolution:  Resolution1080p,
		aspectRatio: "9:16",
	},
	"2688x1536": {
		resolution:  Resolution2K,
		aspectRatio: "16:9",
	},
	"1536x2688": {
		resolution:  Resolution2K,
		aspectRatio: "9:16",
	},
}

// aspectRatios are the explicit aspect ratios of text- and
// reference-to-video generation.
var aspectRatios = []string{
	"21:9",
	"16:9",
	"4:3",
	"1:1",
	"3:4",
	"9:16",
}

// adaptiveAspectRatio follows the input media; only reference-to-video
// accepts it.
const adaptiveAspectRatio = "adaptive"

type videoRequest struct {
	Model             string             `json:"model"`
	Mode              VideoMode          `json:"mode"`
	Prompt            string             `json:"prompt"`
	Duration          float64            `json:"duration"`
	Resolution        VideoResolution    `json:"resolution"`
	AspectRatio       string             `json:"aspect_ratio,omitempty"`
	Seed              *int               `json:"seed,omitempty"`
	PromptEnhancement *PromptEnhancement `json:"prompt_enhancement,omitempty"`
	Image             *assetBody         `json:"image,omitempty"`
	ReferenceImages   []assetBody        `json:"reference_images,omitempty"`
	ReferenceVideos   []assetBody        `json:"reference_videos,omitempty"`
	ReferenceAudio    []assetBody        `json:"reference_audio,omitempty"`
}

// assetBody is an input asset on the wire.
type assetBody struct {
	Type      VideoAssetType `json:"type"`
	URL       string         `json:"url,omitempty"`
	AssetID   string         `json:"asset_id,omitempty"`
	MediaType string         `json:"media_type,omitempty"`
	Data      string         `json:"data,omitempty"`
}

// references are the input references of one generation, by kind.
type references struct {
	images []VideoAsset
	videos []VideoAsset
	audio  []VideoAsset
}

func (r references) count() int {
	return len(r.images) + len(r.videos) + len(r.audio)
}

// videoRequestBuilder builds one request and collects its warnings.
type videoRequestBuilder struct {
	modelID      string
	options      model.VideoOptions
	videoOptions VideoOptions
	warnings     []model.Warning
}

// buildRequest validates the call and converts it into a generation
// request.
func (m *VideoModel) buildRequest(options model.VideoOptions) (videoRequest, []model.Warning, error) {
	videoOptions, err := readVideoOptions(options.ProviderOptions)
	if err != nil {
		return videoRequest{}, nil, err
	}

	builder := &videoRequestBuilder{
		modelID:      m.modelID,
		options:      options,
		videoOptions: videoOptions,
	}

	request, err := builder.request()
	if err != nil {
		return videoRequest{}, nil, err
	}

	return request, builder.warnings, nil
}

func (b *videoRequestBuilder) request() (videoRequest, error) {
	if err := b.validateSettings(); err != nil {
		return videoRequest{}, err
	}

	b.addUnsupportedWarnings()

	image, err := b.firstFrame()
	if err != nil {
		return videoRequest{}, err
	}

	inputs, err := b.references()
	if err != nil {
		return videoRequest{}, err
	}

	mode, err := b.mode(image, inputs)
	if err != nil {
		return videoRequest{}, err
	}

	resolution, aspectRatio, err := b.dimensions(mode)
	if err != nil {
		return videoRequest{}, err
	}

	return b.encode(mode, resolution, aspectRatio, image, inputs)
}

// validateSettings checks the prompt, duration and seed.
func (b *videoRequestBuilder) validateSettings() error {
	promptLength := len(utf16.Encode([]rune(b.options.Prompt)))
	if promptLength == 0 || promptLength > maxPromptLength {
		return fmt.Errorf("%w: prompt: HeyGen requires a prompt between 1 and 32,000 characters", model.ErrInvalidArgument)
	}

	if duration := b.options.Duration; duration != nil &&
		(*duration != math.Trunc(*duration) || *duration < minDuration || *duration > maxDuration) {
		return fmt.Errorf("%w: duration: HeyGen duration must be an integer between 5 and 15 seconds", model.ErrInvalidArgument)
	}

	if seed := b.options.Seed; seed != nil && (*seed < 0 || int64(*seed) > maxSeed) {
		return fmt.Errorf("%w: seed: HeyGen seed must be an unsigned 32-bit integer", model.ErrInvalidArgument)
	}

	return nil
}

func (b *videoRequestBuilder) addUnsupportedWarnings() {
	if b.options.N > 1 {
		b.warn("n", "HeyGen generates one video per call.")
	}

	if b.options.FPS != 0 && b.options.FPS != nativeFPS {
		b.warn("fps", "HeyGen generates video at 24 fps.")
	}

	if b.options.GenerateAudio != nil && !*b.options.GenerateAudio {
		b.warn("generateAudio", "HeyGen always generates audio alongside the video.")
	}
}

// firstFrame returns the first frame asset: the first frame image, the
// standard image, or the image option. Only one of them may be supplied.
// The start image that ai.GenerateVideo derives from the first frame is the
// same file and is accepted.
func (b *videoRequestBuilder) firstFrame() (*VideoAsset, error) {
	frames := b.options.FrameImages

	if len(frames) > 1 || slices.ContainsFunc(frames, isNotFirstFrame) {
		return nil, fmt.Errorf("%w: frameImages: HeyGen supports a single first frame, not last-frame conditioning", model.ErrInvalidArgument)
	}

	file := b.options.Image

	if len(frames) > 0 {
		if file != nil && !sameVideoFile(file, frames[0].Image) {
			return nil, fmt.Errorf("%w: image: supply either image or a first frame, not both", model.ErrInvalidArgument)
		}

		file = frames[0].Image
	}

	if file == nil {
		return b.videoOptions.Image, nil
	}

	if b.videoOptions.Image != nil {
		return nil, fmt.Errorf("%w: image: supply either a standard image or providerOptions.heygen.image, not both", model.ErrInvalidArgument)
	}

	asset, err := toAsset(file, kindImage)
	if err != nil {
		return nil, err
	}

	return &asset, nil
}

// references routes the input references by media type and appends the
// references of the provider options.
func (b *videoRequestBuilder) references() (references, error) {
	var inputs references

	for _, reference := range b.options.InputReferences {
		if err := inputs.add(reference); err != nil {
			return references{}, err
		}
	}

	inputs.images = append(inputs.images, b.videoOptions.ReferenceImages...)
	inputs.videos = append(inputs.videos, b.videoOptions.ReferenceVideos...)
	inputs.audio = append(inputs.audio, b.videoOptions.ReferenceAudio...)

	if len(inputs.images) > maxReferenceImages ||
		len(inputs.videos) > maxReferenceVideos ||
		len(inputs.audio) > maxReferenceAudio ||
		inputs.count() > maxReferences {
		return references{}, fmt.Errorf(
			"%w: inputReferences: HeyGen accepts at most 9 images, 3 videos, 3 audio references, and 12 references in total",
			model.ErrInvalidArgument,
		)
	}

	return inputs, nil
}

// mode returns the mode option, or the mode the inputs imply, and checks
// that the inputs fit it.
func (b *videoRequestBuilder) mode(image *VideoAsset, inputs references) (VideoMode, error) {
	mode := ModeTextToVideo

	switch {
	case b.videoOptions.Mode != nil:
		mode = *b.videoOptions.Mode
	case image != nil:
		mode = ModeImageToVideo
	case inputs.count() > 0:
		mode = ModeReferenceToVideo
	}

	if (mode == ModeImageToVideo) != (image != nil) {
		return "", fmt.Errorf(
			"%w: image: a first-frame image is required for image_to_video and cannot be used in other modes",
			model.ErrInvalidArgument,
		)
	}

	isReferenceMode := mode == ModeReferenceToVideo
	hasReferences := inputs.count() > 0
	hasVisualReferences := len(inputs.images)+len(inputs.videos) > 0

	if !isReferenceMode && hasReferences {
		return "", fmt.Errorf(
			"%w: inputReferences: references are only supported in reference_to_video",
			model.ErrInvalidArgument,
		)
	}

	if isReferenceMode && !hasVisualReferences {
		return "", fmt.Errorf(
			"%w: inputReferences: reference_to_video requires at least one image or video reference",
			model.ErrInvalidArgument,
		)
	}

	return mode, nil
}

// dimensions returns the size class and aspect ratio. A documented frame
// size sets both unless the resolution option selects the size class.
func (b *videoRequestBuilder) dimensions(mode VideoMode) (VideoResolution, string, error) {
	var size *frameSize

	if b.options.Resolution != "" && b.videoOptions.Resolution == nil {
		documented, ok := frameSizes[b.options.Resolution]
		if !ok {
			return "", "", fmt.Errorf(
				"%w: resolution: unsupported HeyGen frame size. Use a documented frame size or providerOptions.heygen.resolution to select 480p, 768p, 1080p, or 2k",
				model.ErrInvalidArgument,
			)
		}

		size = &documented
	}

	resolution := Resolution768p
	aspectRatio := b.options.AspectRatio

	switch {
	case b.videoOptions.Resolution != nil:
		resolution = *b.videoOptions.Resolution
	case size != nil:
		resolution = size.resolution
	}

	switch {
	case aspectRatio != "":
	case size != nil:
		aspectRatio = size.aspectRatio
	case mode == ModeTextToVideo:
		aspectRatio = "16:9"
	default:
		aspectRatio = adaptiveAspectRatio
	}

	if mode == ModeImageToVideo {
		if b.options.AspectRatio != "" || size != nil {
			b.warn(
				"aspectRatio",
				"HeyGen image-to-video follows the first frame aspect ratio; only the resolution size class is applied.",
			)
		}

		return resolution, aspectRatio, nil
	}

	if err := validateAspectRatio(mode, resolution, aspectRatio); err != nil {
		return "", "", err
	}

	return resolution, aspectRatio, nil
}

func validateAspectRatio(mode VideoMode, resolution VideoResolution, aspectRatio string) error {
	supported := slices.Contains(aspectRatios, aspectRatio) ||
		(mode == ModeReferenceToVideo && aspectRatio == adaptiveAspectRatio)

	if !supported {
		return fmt.Errorf("%w: aspectRatio: unsupported HeyGen aspect ratio for %s: %s", model.ErrInvalidArgument, mode, aspectRatio)
	}

	highResolution := resolution == Resolution1080p || resolution == Resolution2K

	if highResolution && aspectRatio != "16:9" && aspectRatio != "9:16" {
		return fmt.Errorf(
			"%w: aspectRatio: HeyGen 1080p and 2k require an explicit 16:9 or 9:16 aspect ratio for text-to-video and reference-to-video",
			model.ErrInvalidArgument,
		)
	}

	return nil
}

// encode builds the request body. Image-to-video requests have no aspect
// ratio.
func (b *videoRequestBuilder) encode(
	mode VideoMode,
	resolution VideoResolution,
	aspectRatio string,
	image *VideoAsset,
	inputs references,
) (videoRequest, error) {
	request := videoRequest{
		Model:             b.modelID,
		Mode:              mode,
		Prompt:            b.options.Prompt,
		Duration:          defaultDuration,
		Resolution:        resolution,
		Seed:              b.options.Seed,
		PromptEnhancement: b.videoOptions.PromptEnhancement,
	}

	if b.options.Duration != nil {
		request.Duration = *b.options.Duration
	}

	if mode != ModeImageToVideo {
		request.AspectRatio = aspectRatio
	}

	if image != nil {
		body, err := encodeAsset(*image, kindImage)
		if err != nil {
			return videoRequest{}, err
		}

		request.Image = &body
	}

	referenceImages, err := encodeAssets(inputs.images, kindImage)
	if err != nil {
		return videoRequest{}, err
	}

	referenceVideos, err := encodeAssets(inputs.videos, kindVideo)
	if err != nil {
		return videoRequest{}, err
	}

	referenceAudio, err := encodeAssets(inputs.audio, kindAudio)
	if err != nil {
		return videoRequest{}, err
	}

	request.ReferenceImages = referenceImages
	request.ReferenceVideos = referenceVideos
	request.ReferenceAudio = referenceAudio

	return request, nil
}

func (b *videoRequestBuilder) warn(feature string, details string) {
	b.warnings = append(b.warnings, model.UnsupportedWarning(feature, details))
}

// add routes an input reference by its media type.
func (r *references) add(reference model.VideoFile) error {
	mediaType := videoFileMediaType(reference)

	switch {
	case strings.HasPrefix(mediaType, "video/"):
		return appendAsset(&r.videos, reference, kindVideo)
	case strings.HasPrefix(mediaType, "audio/"):
		return appendAsset(&r.audio, reference, kindAudio)
	case strings.HasPrefix(mediaType, "image/"):
		return appendAsset(&r.images, reference, kindImage)
	default:
		return fmt.Errorf("%w: inputReferences: HeyGen references require an image, video, or audio mediaType", model.ErrInvalidArgument)
	}
}

func appendAsset(assets *[]VideoAsset, file model.VideoFile, kind assetKind) error {
	asset, err := toAsset(file, kind)
	if err != nil {
		return err
	}

	*assets = append(*assets, asset)

	return nil
}

// toAsset converts an input file. A known media type must be of the kind.
func toAsset(file model.VideoFile, kind assetKind) (VideoAsset, error) {
	mediaType := videoFileMediaType(file)
	if mediaType != "" && !hasKind(mediaType, kind) {
		return VideoAsset{}, mediaTypeError(kind)
	}

	switch file := file.(type) {
	case model.VideoURL:
		return VideoAsset{
			Type: AssetURL,
			URL:  file.URL,
		}, nil

	case model.VideoBytes:
		return VideoAsset{
			Type:      AssetBase64,
			MediaType: file.MediaType,
			Data:      base64.StdEncoding.EncodeToString(file.Data),
		}, nil

	default:
		return VideoAsset{}, fmt.Errorf("%w: video file %T", model.ErrUnsupported, file)
	}
}

func encodeAssets(assets []VideoAsset, kind assetKind) ([]assetBody, error) {
	if len(assets) == 0 {
		return nil, nil
	}

	bodies := make([]assetBody, 0, len(assets))

	for _, asset := range assets {
		body, err := encodeAsset(asset, kind)
		if err != nil {
			return nil, err
		}

		bodies = append(bodies, body)
	}

	return bodies, nil
}

// encodeAsset converts an asset to its wire form. URLs must be absolute
// HTTPS URLs and inline data must have a media type of the kind.
func encodeAsset(asset VideoAsset, kind assetKind) (assetBody, error) {
	switch asset.Type {
	case AssetURL:
		if !isHTTPSURL(asset.URL) {
			return assetBody{}, fmt.Errorf("%w: %s: HeyGen input URLs must be absolute HTTPS URLs", model.ErrInvalidArgument, kind)
		}

		return assetBody{
			Type: AssetURL,
			URL:  asset.URL,
		}, nil

	case AssetID:
		return assetBody{
			Type:    AssetID,
			AssetID: asset.AssetID,
		}, nil

	case AssetBase64:
		if !hasKind(asset.MediaType, kind) {
			return assetBody{}, mediaTypeError(kind)
		}

		return assetBody{
			Type:      AssetBase64,
			MediaType: asset.MediaType,
			Data:      asset.Data,
		}, nil

	default:
		return assetBody{}, fmt.Errorf("%w: heygen asset type %q", model.ErrInvalidArgument, asset.Type)
	}
}

func hasKind(mediaType string, kind assetKind) bool {
	return strings.HasPrefix(mediaType, string(kind)+"/")
}

func mediaTypeError(kind assetKind) error {
	return fmt.Errorf("%w: %s: expected a %s mediaType", model.ErrInvalidArgument, kind, kind)
}

func videoFileMediaType(file model.VideoFile) string {
	switch file := file.(type) {
	case model.VideoURL:
		return file.MediaType
	case model.VideoBytes:
		return file.MediaType
	default:
		return ""
	}
}

func isNotFirstFrame(frame model.VideoFrameImage) bool {
	return frame.FrameType != model.VideoFirstFrame
}

// sameVideoFile reports whether two files have the same source.
func sameVideoFile(first model.VideoFile, second model.VideoFile) bool {
	switch first := first.(type) {
	case model.VideoURL:
		other, ok := second.(model.VideoURL)

		return ok && first.URL == other.URL

	case model.VideoBytes:
		other, ok := second.(model.VideoBytes)

		return ok && bytes.Equal(first.Data, other.Data)

	default:
		return false
	}
}
