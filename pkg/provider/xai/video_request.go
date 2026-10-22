package xai

import (
	"encoding/json"
	"fmt"
	"slices"

	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/internal/optional"
	"github.com/aarock1234/ai/pkg/model"
)

// Video endpoints of each mode.
const (
	videoGenerationsPath = "/videos/generations"
	videoEditsPath       = "/videos/edits"
	videoExtensionsPath  = "/videos/extensions"
)

// Models with their own video rules.
const (
	modelImagineVideo   = "grok-imagine-video"
	modelImagineVideo15 = "grok-imagine-video-1.5"
)

// maxReferenceAudios is the audio reference limit of reference-to-video.
const maxReferenceAudios = 3

// videoResolutions maps "WIDTHxHEIGHT" resolutions onto xAI resolutions.
var videoResolutions = map[string]VideoResolution{
	"1920x1080": VideoResolution1080p,
	"1280x720":  VideoResolution720p,
	"854x480":   VideoResolution480p,
	"640x480":   VideoResolution480p,
}

// videoRequest is the body of a video generation, edit or extension.
type videoRequest struct {
	Model           string               `json:"model"`
	Prompt          string               `json:"prompt,omitempty"`
	Duration        *float64             `json:"duration,omitempty"`
	AspectRatio     string               `json:"aspect_ratio,omitempty"`
	Resolution      VideoResolution      `json:"resolution,omitempty"`
	GenerateAudio   *bool                `json:"generate_audio,omitempty"`
	StorageOptions  *videoStorageOptions `json:"storage_options,omitempty"`
	Keyframes       []videoKeyframe      `json:"keyframes,omitempty"`
	Video           *videoLink           `json:"video,omitempty"`
	Image           *videoLink           `json:"image,omitempty"`
	LastFrame       *videoLink           `json:"last_frame,omitempty"`
	ReferenceImages []videoLink          `json:"reference_images,omitempty"`
	ReferenceAudios []referenceAudio     `json:"reference_audios,omitempty"`
	User            *string              `json:"user,omitempty"`
}

type videoLink struct {
	URL string `json:"url,omitempty"`
}

// referenceAudio is an audio URL or a preset voice.
type referenceAudio struct {
	URL     string `json:"url,omitempty"`
	VoiceID string `json:"voice_id,omitempty"`
}

type videoKeyframe struct {
	Image      videoLink `json:"image"`
	TimestampS float64   `json:"timestamp_s"`
}

type videoStorageOptions struct {
	Filename     string          `json:"filename"`
	ExpiresAfter *int            `json:"expires_after,omitempty"`
	PublicURL    *publicURLValue `json:"public_url,omitempty"`
}

// publicURLValue is a boolean or {"expires_after": seconds}.
type publicURLValue struct {
	enabled      *bool
	expiresAfter *int
}

type publicURLWireObject struct {
	ExpiresAfter *int `json:"expires_after,omitempty"`
}

// MarshalJSON encodes a boolean, or an object when enabled is nil.
func (v publicURLValue) MarshalJSON() ([]byte, error) {
	if v.enabled != nil {
		return json.Marshal(*v.enabled)
	}

	return json.Marshal(publicURLWireObject{
		ExpiresAfter: v.expiresAfter,
	})
}

// videoPlan is a request body with the endpoint of its mode.
type videoPlan struct {
	path string
	body jsonobject.Object
}

// videoRequestBuilder converts the call options into a request body and
// collects the warnings of ignored or adjusted inputs.
type videoRequestBuilder struct {
	modelID      string
	options      model.VideoOptions
	videoOptions VideoOptions
	warnings     []model.Warning

	mode        VideoMode
	isEdit      bool
	isExtension bool
	isReference bool
}

// buildVideoPlan returns the request of a generation, edit or extension.
func buildVideoPlan(modelID string, options model.VideoOptions) (videoPlan, []model.Warning, error) {
	videoOptions, extra, err := readVideoOptions(options.ProviderOptions)
	if err != nil {
		return videoPlan{}, nil, err
	}

	builder := &videoRequestBuilder{
		modelID:      modelID,
		options:      options,
		videoOptions: videoOptions,
	}

	builder.mode = builder.resolveMode()
	builder.isEdit = builder.mode == VideoModeEditVideo
	builder.isExtension = builder.mode == VideoModeExtendVideo
	builder.isReference = builder.mode == VideoModeReferenceToVideo

	body, err := encodeVideoBody(builder.build(), extra)
	if err != nil {
		return videoPlan{}, nil, err
	}

	return videoPlan{
		path: builder.path(),
		body: body,
	}, builder.warnings, nil
}

// encodeVideoBody encodes the typed body followed by the passthrough
// options, which replace members of the same name.
func encodeVideoBody(request videoRequest, extra jsonobject.Object) (jsonobject.Object, error) {
	encoded, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("encode video request: %w", err)
	}

	body, err := jsonobject.Parse(encoded)
	if err != nil {
		return nil, fmt.Errorf("encode video request: %w", err)
	}

	return body.Merge(extra), nil
}

func (b *videoRequestBuilder) warn(feature string, details string) {
	b.warnings = append(b.warnings, model.UnsupportedWarning(feature, details))
}

// resolveMode returns the explicit mode or the one the inputs imply. Video
// references alone do not select reference-to-video.
func (b *videoRequestBuilder) resolveMode() VideoMode {
	switch {
	case b.videoOptions.Mode != "":
		return b.videoOptions.Mode

	case b.videoOptions.VideoURL != nil:
		return VideoModeEditVideo

	case b.hasImageReference() || b.hasAudioReference() || len(b.videoOptions.ReferenceImageURLs) > 0:
		return VideoModeReferenceToVideo

	default:
		return ""
	}
}

func (b *videoRequestBuilder) path() string {
	switch {
	case b.isEdit:
		return videoEditsPath

	case b.isExtension:
		return videoExtensionsPath

	default:
		return videoGenerationsPath
	}
}

// operationName names the mode in warnings of edit and extension requests.
func (b *videoRequestBuilder) operationName() string {
	if b.isEdit {
		return "video editing"
	}

	return "video extension"
}

func (b *videoRequestBuilder) build() videoRequest {
	b.warnUnsupportedSettings()

	request := videoRequest{
		Model:  b.modelID,
		Prompt: b.options.Prompt,
	}

	if !b.isEdit {
		request.Duration = b.options.Duration
	}

	if !b.isEdit && !b.isExtension {
		request.AspectRatio = b.options.AspectRatio
		request.Resolution = b.resolution()
	}

	if b.options.GenerateAudio != nil {
		if b.isEdit || b.isExtension {
			b.warn("generateAudio", "xAI "+b.operationName()+" does not support generateAudio.")
		} else {
			request.GenerateAudio = b.options.GenerateAudio
		}
	}

	request.StorageOptions = storageOptionsOf(b.videoOptions.StorageOptions)
	request.Keyframes = b.keyframes()

	if b.isEdit || b.isExtension {
		request.Video = &videoLink{
			URL: optional.Or(b.videoOptions.VideoURL, ""),
		}
	}

	request.Image = b.startImage()
	request.LastFrame = b.lastFrame()

	if b.isReference {
		b.addReferences(&request)
	}

	if request.Resolution == VideoResolution1080p && b.modelID == modelImagineVideo {
		b.warn("resolution", `xAI model "grok-imagine-video" does not support 1080p. Use "grok-imagine-video-1.5" for 1080p, or a lower resolution. The request was sent with 1080p.`)
	}

	b.warnUnusedReferences()

	if !b.isExtension {
		request.User = b.videoOptions.User
	}

	return request
}

// warnUnsupportedSettings reports the settings that the models or the
// selected mode do not support.
func (b *videoRequestBuilder) warnUnsupportedSettings() {
	if b.options.FPS != 0 {
		b.warn("fps", "xAI video models do not support custom FPS.")
	}

	if b.options.Seed != nil {
		b.warn("seed", "xAI video models do not support seed.")
	}

	if b.options.N > 1 {
		b.warn("n", "xAI video models do not support generating multiple videos per call. Only 1 video will be generated.")
	}

	hasResolution := b.videoOptions.Resolution != "" || b.options.Resolution != ""

	if b.isEdit {
		if b.options.Duration != nil {
			b.warn("duration", "xAI video editing does not support custom duration.")
		}

		if b.options.AspectRatio != "" {
			b.warn("aspectRatio", "xAI video editing does not support custom aspect ratio.")
		}

		if hasResolution {
			b.warn("resolution", "xAI video editing does not support custom resolution.")
		}
	}

	if b.isExtension {
		if b.options.AspectRatio != "" {
			b.warn("aspectRatio", "xAI video extension does not support custom aspect ratio.")
		}

		if hasResolution {
			b.warn("resolution", "xAI video extension does not support custom resolution.")
		}
	}
}

// resolution returns the provider resolution, or the mapped call
// resolution.
func (b *videoRequestBuilder) resolution() VideoResolution {
	if b.videoOptions.Resolution != "" {
		return b.videoOptions.Resolution
	}

	requested := b.options.Resolution
	if requested == "" {
		return ""
	}

	if mapped, ok := videoResolutions[requested]; ok {
		return mapped
	}

	b.warn("resolution", fmt.Sprintf(
		`Unrecognized resolution %q. Use providerOptions.xai.resolution with "480p", "720p", or "1080p" instead.`,
		requested,
	))

	return ""
}

func storageOptionsOf(options *VideoStorageOptions) *videoStorageOptions {
	if options == nil {
		return nil
	}

	storage := &videoStorageOptions{
		Filename:     options.Filename,
		ExpiresAfter: options.ExpiresAfter,
	}

	if options.PublicURL != nil {
		storage.PublicURL = &publicURLValue{
			enabled:      options.PublicURL.Enabled,
			expiresAfter: options.PublicURL.ExpiresAfter,
		}
	}

	return storage
}

// keyframes returns the keyframes, which only grok-imagine-video-1.5
// generation supports.
func (b *videoRequestBuilder) keyframes() []videoKeyframe {
	keyframes := b.videoOptions.Keyframes
	if len(keyframes) == 0 {
		return nil
	}

	if b.modelID != modelImagineVideo15 {
		b.warn("keyframes", `xAI only supports keyframes with "grok-imagine-video-1.5".`)

		return nil
	}

	if b.isEdit || b.isExtension {
		b.warn("keyframes", "xAI "+b.operationName()+" does not support keyframes.")

		return nil
	}

	converted := make([]videoKeyframe, 0, len(keyframes))
	for _, keyframe := range keyframes {
		converted = append(converted, videoKeyframe{
			Image: videoLink{
				URL: keyframe.ImageURL,
			},
			TimestampS: keyframe.TimestampSeconds,
		})
	}

	return converted
}

// startImage returns the first frame or image of image-to-video
// generation. Videos are ignored.
func (b *videoRequestBuilder) startImage() *videoLink {
	firstFrame := videoFrameImage(b.options.FrameImages, model.VideoFirstFrame)

	start := firstFrame
	if start == nil {
		start = b.options.Image
	}

	if start == nil {
		return nil
	}

	if isVideoFile(start) {
		feature := "image"
		if firstFrame != nil {
			feature = "frameImages"
		}

		b.warn(feature, `xAI does not accept a video as a start/frame image. The video was ignored. Use providerOptions.xai.mode "extend-video" to continue from a video instead.`)

		return nil
	}

	return &videoLink{
		URL: videoFileURL(start),
	}
}

// lastFrame returns the pinned last frame, which only
// grok-imagine-video-1.5 generation supports.
func (b *videoRequestBuilder) lastFrame() *videoLink {
	last := videoFrameImage(b.options.FrameImages, model.VideoLastFrame)
	if last == nil {
		return nil
	}

	if b.modelID != modelImagineVideo15 {
		b.warn("frameImages", `xAI only supports last_frame with "grok-imagine-video-1.5". The last frame was ignored.`)

		return nil
	}

	if b.isEdit || b.isExtension || isVideoFile(last) {
		b.warn("frameImages", "xAI only accepts an image last_frame for video generation. The last frame was ignored.")

		return nil
	}

	return &videoLink{
		URL: videoFileURL(last),
	}
}

// addReferences adds the reference images and audios of reference-to-video
// generation, which is limited to 720p.
func (b *videoRequestBuilder) addReferences(request *videoRequest) {
	images := b.referenceImages()
	audios := b.referenceAudios()

	switch {
	case images != nil:
		request.ReferenceImages = images

	case audios == nil:
		b.warn("referenceImages", "xAI reference-to-video requires at least one image reference. The video will be generated without reference images.")
	}

	for _, voiceID := range b.videoOptions.ReferenceVoiceIDs {
		audios = append(audios, referenceAudio{
			VoiceID: voiceID,
		})
	}

	if len(audios) > maxReferenceAudios {
		b.warn("inputReferences", "xAI reference-to-video supports at most 3 audio references. Only the first 3 were used.")
		audios = audios[:maxReferenceAudios]
	}

	if len(audios) > 0 {
		request.ReferenceAudios = audios
	}

	if request.Resolution == VideoResolution1080p {
		b.warn("resolution", "xAI reference-to-video is limited to 720p. The request was downgraded from 1080p to 720p.")
		request.Resolution = VideoResolution720p
	}
}

// referenceImages returns the image input references, or else the
// referenceImageUrls option. Video references are ignored; audio
// references are added separately.
func (b *videoRequestBuilder) referenceImages() []videoLink {
	if len(b.options.InputReferences) == 0 {
		var images []videoLink
		for _, imageURL := range b.videoOptions.ReferenceImageURLs {
			images = append(images, videoLink{
				URL: imageURL,
			})
		}

		return images
	}

	var images []videoLink

	for _, reference := range b.options.InputReferences {
		if isAudioReference(reference) {
			continue
		}

		if !isImageReference(reference) {
			b.warn("inputReferences", `xAI reference-to-video does not accept video references. The video reference was ignored. Use providerOptions.xai.mode "extend-video" to continue from a video.`)

			continue
		}

		images = append(images, videoLink{
			URL: videoFileURL(reference),
		})
	}

	return images
}

func (b *videoRequestBuilder) referenceAudios() []referenceAudio {
	var audios []referenceAudio

	for _, reference := range b.options.InputReferences {
		if isAudioReference(reference) {
			audios = append(audios, referenceAudio{
				URL: videoFileURL(reference),
			})
		}
	}

	return audios
}

// warnUnusedReferences reports references and voices that the resolved
// mode does not use.
func (b *videoRequestBuilder) warnUnusedReferences() {
	if b.isReference {
		return
	}

	if len(b.options.InputReferences) > 0 {
		details := "xAI reference-to-video requires at least one image or audio reference. The references were ignored."
		if b.hasImageReference() || b.hasAudioReference() {
			details = "xAI only supports inputReferences for reference-to-video generation. The references were ignored."
		}

		b.warn("inputReferences", details)
	}

	if len(b.videoOptions.ReferenceVoiceIDs) > 0 {
		b.warn("referenceVoiceIds", "xAI only supports reference voices for reference-to-video generation. The reference voices were ignored.")
	}
}

func (b *videoRequestBuilder) hasImageReference() bool {
	return slices.ContainsFunc(b.options.InputReferences, isImageReference)
}

func (b *videoRequestBuilder) hasAudioReference() bool {
	return slices.ContainsFunc(b.options.InputReferences, isAudioReference)
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

func isVideoFile(file model.VideoFile) bool {
	return mediatype.TopLevel(videoFileMediaType(file)) == "video"
}

// isImageReference reports image references. References without a media
// type, which only URLs can have, are images.
func isImageReference(file model.VideoFile) bool {
	mediaType := videoFileMediaType(file)

	return mediaType == "" || mediatype.TopLevel(mediaType) == "image"
}

func isAudioReference(file model.VideoFile) bool {
	return mediatype.TopLevel(videoFileMediaType(file)) == "audio"
}

// videoFileURL passes URLs through and encodes inline data as a data URL.
func videoFileURL(file model.VideoFile) string {
	switch typed := file.(type) {
	case model.VideoURL:
		return typed.URL

	case model.VideoBytes:
		return dataurl.Encode(typed.MediaType, typed.Data)

	default:
		return ""
	}
}
