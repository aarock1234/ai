package alibaba

import (
	"encoding/base64"
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/pkg/model"
)

// videoMode is the generation mode named by a wan2.6 or wan2.7 model ID.
type videoMode int

const (
	modeTextToVideo videoMode = iota
	modeImageToVideo
	modeReferenceToVideo
)

// videoProtocol is the request protocol of a model generation.
type videoProtocol int

const (
	// protocolLegacy (wan2.6 and earlier) uses parameters.size,
	// input.img_url and input.reference_urls.
	protocolLegacy videoProtocol = iota

	// protocolWan27 uses resolution tiers and a ratio instead of a size,
	// input.media instead of reference URLs, and always generates audio.
	protocolWan27

	// protocolWan3 is like protocolWan27 with a last frame, an audio toggle,
	// a 480P tier and one model ID for every mode.
	protocolWan3
)

// versionPattern reads the version of IDs such as "wan2.6-t2v" and the older
// "wanx2.1-t2v-turbo".
var versionPattern = regexp.MustCompile(`^wanx?(\d+)(?:\.(\d+))?(?:[.-]|$)`)

// videoURLPattern matches URLs of MP4 and MOV videos.
var videoURLPattern = regexp.MustCompile(`(?i)\.(mp4|mov)([?#]|$)`)

// resolutionTiers maps "WIDTHxHEIGHT" resolutions to Alibaba resolution
// tiers.
var resolutionTiers = map[string]string{
	"1280x720":  "720P",
	"720x1280":  "720P",
	"960x960":   "720P",
	"1088x832":  "720P",
	"832x1088":  "720P",
	"1920x1080": "1080P",
	"1080x1920": "1080P",
	"1440x1440": "1080P",
	"1632x1248": "1080P",
	"1248x1632": "1080P",
	"832x480":   "480P",
	"480x832":   "480P",
	"624x624":   "480P",
}

// derivableRatios are the aspect ratios derived from resolutions.
var derivableRatios = map[string]bool{
	"16:9": true,
	"9:16": true,
	"1:1":  true,
	"4:3":  true,
	"3:4":  true,
}

type videoRequest struct {
	Model      string          `json:"model"`
	Input      videoInput      `json:"input"`
	Parameters videoParameters `json:"parameters"`
}

type videoInput struct {
	Prompt         string           `json:"prompt,omitempty"`
	NegativePrompt *string          `json:"negative_prompt,omitempty"`
	AudioURL       *string          `json:"audio_url,omitempty"`
	ImgURL         string           `json:"img_url,omitempty"`
	Media          []videoMediaItem `json:"media,omitempty"`
	ReferenceURLs  []string         `json:"reference_urls,omitempty"`
}

type videoMediaItem struct {
	Type           VideoMediaType `json:"type"`
	URL            string         `json:"url"`
	ReferenceVoice *string        `json:"reference_voice,omitempty"`
}

type videoParameters struct {
	Duration     *float64 `json:"duration,omitempty"`
	Seed         *int     `json:"seed,omitempty"`
	Resolution   string   `json:"resolution,omitempty"`
	Size         string   `json:"size,omitempty"`
	Ratio        string   `json:"ratio,omitempty"`
	PromptExtend *bool    `json:"prompt_extend,omitempty"`
	ShotType     ShotType `json:"shot_type,omitempty"`
	Watermark    *bool    `json:"watermark,omitempty"`
	Audio        *bool    `json:"audio,omitempty"`
}

// videoRequestBuilder builds one request and collects its warnings.
type videoRequestBuilder struct {
	options      model.VideoOptions
	videoOptions VideoOptions
	mode         videoMode
	protocol     videoProtocol
	warnings     []model.Warning
}

// buildRequest converts the call options into a task request for the
// model's protocol and mode.
func (m *VideoModel) buildRequest(options model.VideoOptions) (videoRequest, []model.Warning, error) {
	videoOptions, err := readVideoOptions(options.ProviderOptions)
	if err != nil {
		return videoRequest{}, nil, err
	}

	builder := &videoRequestBuilder{
		options:      options,
		videoOptions: videoOptions,
		mode:         detectMode(m.modelID),
		protocol:     detectProtocol(m.modelID),
	}

	input, err := builder.input()
	if err != nil {
		return videoRequest{}, nil, err
	}

	return videoRequest{
		Model:      m.modelID,
		Input:      input,
		Parameters: builder.parameters(),
	}, builder.warnings, nil
}

// detectMode reads the mode of IDs that name it. wan3 IDs serve every mode,
// which the request media decide.
func detectMode(modelID string) videoMode {
	switch {
	case strings.Contains(modelID, "-i2v"):
		return modeImageToVideo
	case strings.Contains(modelID, "-r2v"):
		return modeReferenceToVideo
	default:
		return modeTextToVideo
	}
}

// detectProtocol selects the protocol by model version. Unrecognized IDs use
// the latest protocol.
func detectProtocol(modelID string) videoProtocol {
	match := versionPattern.FindStringSubmatch(modelID)
	if match == nil {
		return protocolWan3
	}

	major, _ := strconv.Atoi(match[1])

	minor := 0
	if match[2] != "" {
		minor, _ = strconv.Atoi(match[2])
	}

	switch {
	case major < 2 || (major == 2 && minor < 7):
		return protocolLegacy
	case major == 2:
		return protocolWan27
	default:
		return protocolWan3
	}
}

func (b *videoRequestBuilder) isTiered() bool {
	return b.protocol == protocolWan27 || b.protocol == protocolWan3
}

// supportsRatio reports an explicit aspect ratio: wan2.7 text- and
// reference-to-video models, whose image-to-video follows the input image,
// and every wan3 generation.
func (b *videoRequestBuilder) supportsRatio() bool {
	return b.protocol == protocolWan3 || (b.protocol == protocolWan27 && b.mode != modeImageToVideo)
}

func (b *videoRequestBuilder) warn(feature string, details string) {
	b.warnings = append(b.warnings, model.UnsupportedWarning(feature, details))
}

// input builds the prompt and media of the request.
func (b *videoRequestBuilder) input() (videoInput, error) {
	input := videoInput{
		Prompt:         b.options.Prompt,
		NegativePrompt: b.videoOptions.NegativePrompt,
		AudioURL:       b.videoOptions.AudioURL,
	}

	startImage := b.startImage()

	if b.protocol != protocolWan3 && b.mode == modeImageToVideo && startImage != nil {
		image, err := fileImageString(startImage)
		if err != nil {
			return videoInput{}, err
		}

		input.ImgURL = image
	}

	if err := b.addMedia(&input, startImage); err != nil {
		return videoInput{}, err
	}

	if b.lastFrame() != nil && b.protocol != protocolWan3 {
		b.warn("frameImages", "This model does not support last_frame. The last frame image was ignored.")
	}

	if len(b.options.InputReferences) > 0 && b.mode != modeReferenceToVideo && b.protocol != protocolWan3 {
		b.warn(
			"inputReferences",
			"Alibaba only supports inputReferences (reference-to-video) on reference-to-video models. The reference images were ignored.",
		)
	}

	return input, nil
}

// addMedia sets the media of wan3 and wan2.7 reference-to-video requests and
// the reference URLs of wan2.6 reference-to-video requests.
func (b *videoRequestBuilder) addMedia(input *videoInput, startImage model.VideoFile) error {
	switch {
	case b.protocol == protocolWan3:
		media, err := b.media(startImage, b.lastFrame())
		if err != nil {
			return err
		}

		input.Media = media

	case b.mode == modeReferenceToVideo && b.protocol == protocolWan27:
		media, err := b.media(b.firstFrame(), nil)
		if err != nil {
			return err
		}

		input.Media = media

	case b.mode == modeReferenceToVideo:
		input.ReferenceURLs = b.referenceURLs()
	}

	return nil
}

// media returns the explicit media option, or the media derived from the
// input references and frames. It returns nil when there is none.
func (b *videoRequestBuilder) media(first model.VideoFile, last model.VideoFile) ([]videoMediaItem, error) {
	if len(b.videoOptions.Media) > 0 {
		media := make([]videoMediaItem, 0, len(b.videoOptions.Media))
		for _, item := range b.videoOptions.Media {
			// The wire item has the same fields under snake case names.
			media = append(media, videoMediaItem(item))
		}

		return media, nil
	}

	var media []videoMediaItem

	for _, reference := range b.options.InputReferences {
		item, ok, err := b.referenceMedia(reference)
		if err != nil {
			return nil, err
		}

		if ok {
			media = append(media, item)
		}
	}

	media, err := appendFrame(media, VideoMediaFirstFrame, first)
	if err != nil {
		return nil, err
	}

	return appendFrame(media, VideoMediaLastFrame, last)
}

// appendFrame adds a frame image to the media, when there is one.
func appendFrame(media []videoMediaItem, mediaType VideoMediaType, image model.VideoFile) ([]videoMediaItem, error) {
	if image == nil {
		return media, nil
	}

	address, err := dataurl.VideoFileURL(image)
	if err != nil {
		return nil, err
	}

	return append(media, videoMediaItem{
		Type: mediaType,
		URL:  address,
	}), nil
}

// referenceMedia converts an input reference. Inline references must be
// images; others produce a warning and report false.
func (b *videoRequestBuilder) referenceMedia(reference model.VideoFile) (videoMediaItem, bool, error) {
	switch reference := reference.(type) {
	case model.VideoURL:
		mediaType := VideoMediaReferenceImage
		if videoURLPattern.MatchString(reference.URL) {
			mediaType = VideoMediaReferenceVideo
		}

		return videoMediaItem{
			Type: mediaType,
			URL:  reference.URL,
		}, true, nil

	case model.VideoBytes:
		if !strings.HasPrefix(reference.MediaType, "image/") {
			b.warn(
				"inputReferences",
				"Alibaba reference-to-video requires URL references for videos. Non-URL video reference was skipped.",
			)

			return videoMediaItem{}, false, nil
		}

		return videoMediaItem{
			Type: VideoMediaReferenceImage,
			URL:  dataurl.Encode(reference.MediaType, reference.Data),
		}, true, nil

	default:
		return videoMediaItem{}, false, fmt.Errorf("%w: video file %T", model.ErrUnsupported, reference)
	}
}

// referenceURLs returns the reference URLs of wan2.6 reference-to-video
// requests: the URL input references, or the reference URLs option. Frame
// images exclude references.
func (b *videoRequestBuilder) referenceURLs() []string {
	if len(b.options.FrameImages) > 0 {
		return nil
	}

	if len(b.options.InputReferences) == 0 {
		return b.videoOptions.ReferenceURLs
	}

	var urls []string
	for _, reference := range b.options.InputReferences {
		if address, ok := reference.(model.VideoURL); ok {
			urls = append(urls, address.URL)

			continue
		}

		b.warn(
			"inputReferences",
			"Alibaba reference-to-video requires URL references. Non-URL reference was skipped.",
		)
	}

	return urls
}

// parameters builds the generation parameters and warns about the settings
// the model ignores.
func (b *videoRequestBuilder) parameters() videoParameters {
	parameters := videoParameters{
		Duration: b.options.Duration,
		Seed:     b.options.Seed,
	}

	b.setResolution(&parameters)

	if b.supportsRatio() {
		parameters.Ratio = b.ratio()
	}

	parameters.PromptExtend = b.videoOptions.PromptExtend

	if b.videoOptions.ShotType != "" {
		if b.isTiered() {
			b.warn("shotType", fmt.Sprintf(
				"%s models do not support the shotType option. Describe the shot structure in the prompt instead.",
				b.protocolName(),
			))
		} else {
			parameters.ShotType = b.videoOptions.ShotType
		}
	}

	parameters.Watermark = b.videoOptions.Watermark

	audio := b.options.GenerateAudio
	if audio == nil {
		audio = b.videoOptions.Audio
	}

	if audio != nil {
		if b.protocol == protocolWan27 {
			b.warn("generateAudio", "wan2.7 models always generate audio. The audio option was ignored.")
		} else {
			parameters.Audio = audio
		}
	}

	b.unsupportedSettings()

	return parameters
}

// setResolution sends image-to-video, wan2.7 and wan3 resolutions as tiers
// and other resolutions as a "WIDTH*HEIGHT" size.
func (b *videoRequestBuilder) setResolution(parameters *videoParameters) {
	resolution := b.options.Resolution
	if resolution == "" {
		return
	}

	if b.mode != modeImageToVideo && !b.isTiered() {
		parameters.Size = strings.Replace(resolution, "x", "*", 1)

		return
	}

	tier, ok := resolutionTiers[resolution]
	if !ok {
		tier = resolution
	}

	supportedTiers := []string{
		"720P",
		"1080P",
	}
	if b.protocol == protocolWan3 {
		supportedTiers = []string{
			"480P",
			"720P",
			"1080P",
		}
	}

	if b.isTiered() && !slices.Contains(supportedTiers, tier) {
		b.warn("resolution", fmt.Sprintf(
			"%s models only support the %s resolution tiers. The resolution %q was ignored.",
			b.protocolName(),
			strings.Join(supportedTiers, ", "),
			resolution,
		))

		return
	}

	parameters.Resolution = tier
}

// ratio prefers the ratio option, then the aspect ratio, then the ratio of
// the resolution.
func (b *videoRequestBuilder) ratio() string {
	if b.videoOptions.Ratio != "" {
		return string(b.videoOptions.Ratio)
	}

	if b.options.AspectRatio != "" {
		return b.options.AspectRatio
	}

	if b.options.Resolution == "" {
		return ""
	}

	return ratioOfResolution(b.options.Resolution)
}

// unsupportedSettings warns about the standard settings the model ignores.
func (b *videoRequestBuilder) unsupportedSettings() {
	if b.options.AspectRatio != "" && !b.supportsRatio() {
		b.warn(
			"aspectRatio",
			"Alibaba video models use explicit size/resolution dimensions. Use the resolution option or providerOptions.alibaba for size control.",
		)
	}

	if b.options.FPS != 0 {
		b.warn("fps", "Alibaba video models do not support custom FPS.")
	}

	if b.options.N > 1 {
		b.warn("n", "Alibaba video models only support generating 1 video per call.")
	}
}

func (b *videoRequestBuilder) protocolName() string {
	if b.protocol == protocolWan3 {
		return "wan3"
	}

	return "wan2.7"
}

// startImage is the first frame image, or the image option.
func (b *videoRequestBuilder) startImage() model.VideoFile {
	if first := b.firstFrame(); first != nil {
		return first
	}

	return b.options.Image
}

func (b *videoRequestBuilder) firstFrame() model.VideoFile {
	return b.frame(model.VideoFirstFrame)
}

func (b *videoRequestBuilder) lastFrame() model.VideoFile {
	return b.frame(model.VideoLastFrame)
}

func (b *videoRequestBuilder) frame(frameType model.VideoFrameType) model.VideoFile {
	for _, frame := range b.options.FrameImages {
		if frame.FrameType == frameType {
			return frame.Image
		}
	}

	return nil
}

// ratioOfResolution reduces a "WIDTHxHEIGHT" resolution to a supported
// aspect ratio, or returns an empty string.
func ratioOfResolution(resolution string) string {
	widthText, heightText, found := strings.Cut(resolution, "x")
	if !found {
		return ""
	}

	width, widthErr := strconv.Atoi(widthText)
	height, heightErr := strconv.Atoi(heightText)

	if widthErr != nil || heightErr != nil || width <= 0 || height <= 0 {
		return ""
	}

	divisor := greatestCommonDivisor(width, height)
	ratio := fmt.Sprintf("%d:%d", width/divisor, height/divisor)

	if !derivableRatios[ratio] {
		return ""
	}

	return ratio
}

func greatestCommonDivisor(a int, b int) int {
	for b != 0 {
		a, b = b, a%b
	}

	return a
}

// fileImageString returns the URL of a URL image or the bare base64 of
// inline bytes.
func fileImageString(file model.VideoFile) (string, error) {
	switch file := file.(type) {
	case model.VideoURL:
		return file.URL, nil
	case model.VideoBytes:
		return base64.StdEncoding.EncodeToString(file.Data), nil
	default:
		return "", fmt.Errorf("%w: video file %T", model.ErrUnsupported, file)
	}
}
