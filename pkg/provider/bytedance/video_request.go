package bytedance

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// ServiceTier is the processing tier of a generation.
type ServiceTier string

// Service tiers.
const (
	ServiceTierDefault ServiceTier = "default"
	ServiceTierFlex    ServiceTier = "flex"
)

// Content item types and roles.
const (
	contentText  = "text"
	contentImage = "image_url"
	contentVideo = "video_url"
	contentAudio = "audio_url"

	roleFirstFrame     = "first_frame"
	roleLastFrame      = "last_frame"
	roleReferenceImage = "reference_image"
	roleReferenceVideo = "reference_video"
	roleReferenceAudio = "reference_audio"
)

// videoOptionKeys are the VideoOptions members, which are mapped to request
// fields rather than passed through.
var videoOptionKeys = map[string]bool{
	"watermark":       true,
	"generateAudio":   true,
	"cameraFixed":     true,
	"returnLastFrame": true,
	"serviceTier":     true,
	"draft":           true,
	"lastFrameImage":  true,
	"referenceImages": true,
	"referenceVideos": true,
	"referenceAudio":  true,
	"pollIntervalMs":  true,
	"pollTimeoutMs":   true,
}

// videoResolutions maps "WIDTHxHEIGHT" resolutions onto Seedance resolution
// tiers.
var videoResolutions = map[string]string{
	"864x496":   "480p",
	"496x864":   "480p",
	"752x560":   "480p",
	"560x752":   "480p",
	"640x640":   "480p",
	"992x432":   "480p",
	"432x992":   "480p",
	"864x480":   "480p",
	"480x864":   "480p",
	"736x544":   "480p",
	"544x736":   "480p",
	"960x416":   "480p",
	"416x960":   "480p",
	"832x480":   "480p",
	"480x832":   "480p",
	"624x624":   "480p",
	"1280x720":  "720p",
	"720x1280":  "720p",
	"1112x834":  "720p",
	"834x1112":  "720p",
	"960x960":   "720p",
	"1470x630":  "720p",
	"630x1470":  "720p",
	"1248x704":  "720p",
	"704x1248":  "720p",
	"1120x832":  "720p",
	"832x1120":  "720p",
	"1504x640":  "720p",
	"640x1504":  "720p",
	"1920x1080": "1080p",
	"1080x1920": "1080p",
	"1664x1248": "1080p",
	"1248x1664": "1080p",
	"1440x1440": "1080p",
	"2206x946":  "1080p",
	"946x2206":  "1080p",
	"1920x1088": "1080p",
	"1088x1920": "1080p",
	"2176x928":  "1080p",
	"928x2176":  "1080p",
}

// VideoOptions are the provider options of video models, read from the
// "bytedance" namespace. Other members of the namespace are sent in the
// request body unchanged.
type VideoOptions struct {
	// Watermark adds a watermark to the video.
	Watermark *bool `json:"watermark,omitempty"`

	// GenerateAudio asks for audio. The standard GenerateAudio option takes
	// precedence.
	GenerateAudio *bool `json:"generateAudio,omitempty"`

	// CameraFixed keeps the camera still.
	CameraFixed *bool `json:"cameraFixed,omitempty"`

	// ReturnLastFrame reports the last frame of the video in the metadata.
	ReturnLastFrame *bool `json:"returnLastFrame,omitempty"`

	ServiceTier *ServiceTier `json:"serviceTier,omitempty"`

	// Draft generates a quick, low-cost preview.
	Draft *bool `json:"draft,omitempty"`

	// LastFrameImage is the URL of the last frame. A last-frame image takes
	// precedence.
	LastFrameImage *string `json:"lastFrameImage,omitempty"`

	// ReferenceImages and ReferenceVideos are reference URLs, used when the
	// call has no input references.
	ReferenceImages []string `json:"referenceImages,omitempty"`
	ReferenceVideos []string `json:"referenceVideos,omitempty"`

	// ReferenceAudio are reference audio URLs.
	ReferenceAudio []string `json:"referenceAudio,omitempty"`

	// PollIntervalMs and PollTimeoutMs are deprecated and ignored, with a
	// warning; ai.VideoPollOptions controls polling.
	PollIntervalMs *float64 `json:"pollIntervalMs,omitempty"`
	PollTimeoutMs  *float64 `json:"pollTimeoutMs,omitempty"`
}

type videoRequest struct {
	Model           string         `json:"model"`
	Content         []videoContent `json:"content"`
	Ratio           string         `json:"ratio,omitempty"`
	Duration        *float64       `json:"duration,omitempty"`
	Seed            *int           `json:"seed,omitempty"`
	Resolution      string         `json:"resolution,omitempty"`
	GenerateAudio   *bool          `json:"generate_audio,omitempty"`
	Watermark       *bool          `json:"watermark,omitempty"`
	CameraFixed     *bool          `json:"camera_fixed,omitempty"`
	ReturnLastFrame *bool          `json:"return_last_frame,omitempty"`
	ServiceTier     *ServiceTier   `json:"service_tier,omitempty"`
	Draft           *bool          `json:"draft,omitempty"`
}

// videoContent is one prompt item: text, or an image, video or audio URL
// with its role.
type videoContent struct {
	Type     string      `json:"type"`
	Text     string      `json:"text,omitempty"`
	ImageURL *contentURL `json:"image_url,omitempty"`
	VideoURL *contentURL `json:"video_url,omitempty"`
	AudioURL *contentURL `json:"audio_url,omitempty"`
	Role     string      `json:"role,omitempty"`
}

type contentURL struct {
	URL string `json:"url"`
}

// videoRequestBuilder builds one request and collects its warnings.
type videoRequestBuilder struct {
	modelID      string
	options      model.VideoStartOptions
	videoOptions VideoOptions
	warnings     []model.Warning
}

// buildVideoRequest converts the call options into a task request.
func (m *VideoModel) buildVideoRequest(options model.VideoStartOptions) (jsonobject.Object, []model.Warning, error) {
	videoOptions, passthrough, err := readVideoOptions(options.ProviderOptions)
	if err != nil {
		return nil, nil, err
	}

	builder := &videoRequestBuilder{
		modelID:      m.modelID,
		options:      options,
		videoOptions: videoOptions,
	}

	builder.addSettingWarnings()

	content, err := builder.content()
	if err != nil {
		return nil, nil, err
	}

	body, err := requestObject(builder.request(content), passthrough, options.WebhookURL)
	if err != nil {
		return nil, nil, err
	}

	return body, builder.warnings, nil
}

// addSettingWarnings warns about the deprecated poll options and the
// standard settings Seedance ignores.
func (b *videoRequestBuilder) addSettingWarnings() {
	deprecated := []struct {
		setting string
		value   *float64
	}{
		{
			setting: "pollIntervalMs",
			value:   b.videoOptions.PollIntervalMs,
		},
		{
			setting: "pollTimeoutMs",
			value:   b.videoOptions.PollTimeoutMs,
		},
	}

	for _, option := range deprecated {
		if option.value == nil {
			continue
		}

		b.warnings = append(b.warnings, model.Warning{
			Type:    model.WarningDeprecated,
			Setting: option.setting,
			Message: fmt.Sprintf(
				"`%s` is ignored. Polling is orchestrated by the AI SDK: pass `poll: { intervalMs, timeoutMs }` to `generateVideo` instead.",
				option.setting,
			),
		})
	}

	if b.options.FPS != 0 {
		b.warn("fps", "ByteDance video models do not support custom FPS. Frame rate is fixed at 24 fps.")
	}

	if b.options.N > 1 {
		b.warn(
			"n",
			"ByteDance video models do not support generating multiple videos per call. Only 1 video will be generated.",
		)
	}
}

// content builds the prompt items: the text, the start image, the last
// frame, the references and the reference audio. The start image is a first
// frame when a last frame is given, and a reference image when references
// are given.
func (b *videoRequestBuilder) content() ([]videoContent, error) {
	var content []videoContent

	if b.options.Prompt != "" {
		content = append(content, videoContent{
			Type: contentText,
			Text: b.options.Prompt,
		})
	}

	lastFrame, err := b.lastFrameURL()
	if err != nil {
		return nil, err
	}

	references, err := b.referenceContent()
	if err != nil {
		return nil, err
	}

	if startImage := b.startImage(); startImage != nil {
		address, err := dataurl.VideoFileURL(startImage)
		if err != nil {
			return nil, err
		}

		role := ""
		switch {
		case lastFrame != "":
			role = roleFirstFrame
		case len(references) > 0:
			role = roleReferenceImage
		}

		content = append(content, imageContent(address, role))
	}

	if lastFrame != "" {
		content = append(content, imageContent(lastFrame, roleLastFrame))
	}

	content = append(content, references...)

	for _, audioURL := range b.videoOptions.ReferenceAudio {
		content = append(content, videoContent{
			Type: contentAudio,
			AudioURL: &contentURL{
				URL: audioURL,
			},
			Role: roleReferenceAudio,
		})
	}

	return content, nil
}

// referenceContent converts the input references, or the reference URLs of
// the options when there are none. Frame images exclude references. URL
// references without a media type are sent as images with a warning.
func (b *videoRequestBuilder) referenceContent() ([]videoContent, error) {
	if len(b.options.FrameImages) > 0 {
		return nil, nil
	}

	if len(b.options.InputReferences) == 0 {
		return b.optionReferenceContent(), nil
	}

	content := make([]videoContent, 0, len(b.options.InputReferences))

	for _, reference := range b.options.InputReferences {
		if address, ok := reference.(model.VideoURL); ok && address.MediaType == "" {
			b.warn(
				"inputReferences",
				`ByteDance requires an explicit mediaType to route URL references as video or image. Pass { data: url, mediaType: "video/mp4" } for video references. The reference was treated as an image.`,
			)
		}

		address, err := dataurl.VideoFileURL(reference)
		if err != nil {
			return nil, err
		}

		if isVideoFile(reference) {
			content = append(content, videoContentOf(address))
		} else {
			content = append(content, imageContent(address, roleReferenceImage))
		}
	}

	return content, nil
}

func (b *videoRequestBuilder) optionReferenceContent() []videoContent {
	var content []videoContent

	for _, imageURL := range b.videoOptions.ReferenceImages {
		content = append(content, imageContent(imageURL, roleReferenceImage))
	}

	for _, videoURL := range b.videoOptions.ReferenceVideos {
		content = append(content, videoContentOf(videoURL))
	}

	return content
}

// request maps the standard settings and the options onto the request. A
// zero duration or seed is omitted.
func (b *videoRequestBuilder) request(content []videoContent) videoRequest {
	request := videoRequest{
		Model:           b.modelID,
		Content:         content,
		Ratio:           b.options.AspectRatio,
		GenerateAudio:   b.options.GenerateAudio,
		Watermark:       b.videoOptions.Watermark,
		CameraFixed:     b.videoOptions.CameraFixed,
		ReturnLastFrame: b.videoOptions.ReturnLastFrame,
		ServiceTier:     b.videoOptions.ServiceTier,
		Draft:           b.videoOptions.Draft,
	}

	if request.Content == nil {
		request.Content = []videoContent{}
	}

	if request.GenerateAudio == nil {
		request.GenerateAudio = b.videoOptions.GenerateAudio
	}

	if b.options.Duration != nil && *b.options.Duration != 0 {
		request.Duration = b.options.Duration
	}

	if b.options.Seed != nil && *b.options.Seed != 0 {
		request.Seed = b.options.Seed
	}

	if resolution := b.options.Resolution; resolution != "" {
		request.Resolution = resolution
		if tier, ok := videoResolutions[resolution]; ok {
			request.Resolution = tier
		}
	}

	return request
}

// startImage is the first frame image, or the image option.
func (b *videoRequestBuilder) startImage() model.VideoFile {
	if first := frameImage(b.options.FrameImages, model.VideoFirstFrame); first != nil {
		return first
	}

	return b.options.Image
}

// lastFrameURL is the last frame image, or the lastFrameImage option.
func (b *videoRequestBuilder) lastFrameURL() (string, error) {
	if last := frameImage(b.options.FrameImages, model.VideoLastFrame); last != nil {
		return dataurl.VideoFileURL(last)
	}

	if b.videoOptions.LastFrameImage == nil {
		return "", nil
	}

	return *b.videoOptions.LastFrameImage, nil
}

func (b *videoRequestBuilder) warn(feature string, details string) {
	b.warnings = append(b.warnings, model.UnsupportedWarning(feature, details))
}

// readVideoOptions reads the typed options and the passthrough members of
// the "bytedance" namespace.
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
	if o.ServiceTier != nil && *o.ServiceTier != ServiceTierDefault && *o.ServiceTier != ServiceTierFlex {
		return fmt.Errorf("%w: bytedance serviceTier %q", model.ErrInvalidArgument, *o.ServiceTier)
	}

	if o.PollIntervalMs != nil && *o.PollIntervalMs <= 0 {
		return fmt.Errorf("%w: bytedance pollIntervalMs must be positive", model.ErrInvalidArgument)
	}

	if o.PollTimeoutMs != nil && *o.PollTimeoutMs <= 0 {
		return fmt.Errorf("%w: bytedance pollTimeoutMs must be positive", model.ErrInvalidArgument)
	}

	return nil
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

func imageContent(address string, role string) videoContent {
	return videoContent{
		Type: contentImage,
		ImageURL: &contentURL{
			URL: address,
		},
		Role: role,
	}
}

func videoContentOf(address string) videoContent {
	return videoContent{
		Type: contentVideo,
		VideoURL: &contentURL{
			URL: address,
		},
		Role: roleReferenceVideo,
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
