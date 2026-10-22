package openai

import (
	"context"
	"encoding/base64"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/download"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/pkg/model"
)

// imageMaxImagesPerCall lists the per-call image limits of known models.
var imageMaxImagesPerCall = map[string]int{
	"dall-e-3":                          1,
	"dall-e-2":                          10,
	"gpt-image-1":                       10,
	"gpt-image-1-mini":                  10,
	"gpt-image-1.5":                     10,
	"gpt-image-2":                       10,
	"gpt-image-2.5-flare":               10,
	"gpt-image-2.5-flare-2026-09-08":    10,
	"gpt-image-2.5-sunburst":            10,
	"gpt-image-2.5-sunburst-2026-09-08": 10,
	"chatgpt-image-latest":              10,
}

// imageDefaultResponseFormatPrefixes are the model families that always
// return base64 images and reject the response_format parameter.
var imageDefaultResponseFormatPrefixes = []string{
	"chatgpt-image-",
	"gpt-image-",
}

// ImageOptions are the image settings shared by generation and editing,
// read from the "openai" provider options namespace.
type ImageOptions struct {
	// Quality of the generated images. "xhigh" and "max" need a GPT Image
	// 2.5 model.
	Quality ImageQuality `json:"quality,omitempty"`

	// Background behavior. "transparent" needs the png or webp output
	// format.
	Background ImageBackground `json:"background,omitempty"`

	// OutputFormat is the format of the returned images.
	OutputFormat ImageOutputFormat `json:"outputFormat,omitempty"`

	// OutputCompression is the compression level, from 0 to 100, of jpeg
	// and webp images.
	OutputCompression *int `json:"outputCompression,omitempty"`

	// User identifies the end user for abuse monitoring.
	User string `json:"user,omitempty"`
}

// ImageGenerationOptions are the settings of an image generation call.
type ImageGenerationOptions struct {
	ImageOptions

	// Style of DALL-E 3 images: "vivid" is hyper-real and dramatic,
	// "natural" is more subdued.
	Style ImageStyle `json:"style,omitempty"`

	// Moderation sets the content moderation level.
	Moderation ImageModeration `json:"moderation,omitempty"`
}

// ImageEditOptions are the settings of an image edit call, which is made
// when input files are given.
type ImageEditOptions struct {
	ImageOptions

	// InputFidelity sets how closely the output follows the input images.
	InputFidelity ImageInputFidelity `json:"inputFidelity,omitempty"`
}

// ImageQuality is the quality of generated images.
type ImageQuality string

// Image qualities.
const (
	ImageQualityStandard ImageQuality = "standard"
	ImageQualityHD       ImageQuality = "hd"
	ImageQualityLow      ImageQuality = "low"
	ImageQualityMedium   ImageQuality = "medium"
	ImageQualityHigh     ImageQuality = "high"
	ImageQualityXHigh    ImageQuality = "xhigh"
	ImageQualityMax      ImageQuality = "max"
	ImageQualityAuto     ImageQuality = "auto"
)

// ImageBackground is the background behavior of generated images.
type ImageBackground string

// Image backgrounds.
const (
	ImageBackgroundTransparent ImageBackground = "transparent"
	ImageBackgroundOpaque      ImageBackground = "opaque"
	ImageBackgroundAuto        ImageBackground = "auto"
)

// ImageOutputFormat is the file format of generated images.
type ImageOutputFormat string

// Image output formats.
const (
	ImageOutputPNG  ImageOutputFormat = "png"
	ImageOutputJPEG ImageOutputFormat = "jpeg"
	ImageOutputWebP ImageOutputFormat = "webp"
)

// ImageStyle is the style of DALL-E 3 images.
type ImageStyle string

// Image styles.
const (
	ImageStyleVivid   ImageStyle = "vivid"
	ImageStyleNatural ImageStyle = "natural"
)

// ImageModeration is the content moderation level.
type ImageModeration string

// Image moderation levels.
const (
	ImageModerationAuto ImageModeration = "auto"
	ImageModerationLow  ImageModeration = "low"
)

// ImageInputFidelity is how closely edits follow the input images.
type ImageInputFidelity string

// Image input fidelities.
const (
	ImageInputFidelityHigh ImageInputFidelity = "high"
	ImageInputFidelityLow  ImageInputFidelity = "low"
)

// ImageMetadata is the "openai" provider metadata of an image result.
type ImageMetadata struct {
	// Images describes each generated image, in result order.
	Images []ImageDetails `json:"images"`
}

// ImageDetails describes one generated image. Input token details are
// distributed evenly across images, with the remainder on the last image.
type ImageDetails struct {
	RevisedPrompt string `json:"revisedPrompt,omitempty"`
	Created       *int64 `json:"created,omitempty"`
	Size          string `json:"size,omitempty"`
	Quality       string `json:"quality,omitempty"`
	Background    string `json:"background,omitempty"`
	OutputFormat  string `json:"outputFormat,omitempty"`
	ImageTokens   *int   `json:"imageTokens,omitempty"`
	TextTokens    *int   `json:"textTokens,omitempty"`
}

// ImageModel is an OpenAI image generation and editing model.
type ImageModel struct {
	modelID string
	config  config
}

func newImageModel(modelID string, c config) *ImageModel {
	return &ImageModel{
		modelID: modelID,
		config:  c,
	}
}

// Provider returns the provider identifier, such as "openai.image".
func (m *ImageModel) Provider() string {
	return m.config.modelProvider("image")
}

// ModelID returns the model identifier.
func (m *ImageModel) ModelID() string {
	return m.modelID
}

// MaxImagesPerCall returns the model's per-call limit: 10 for GPT Image
// models and DALL-E 2, and 1 for DALL-E 3 and unknown models.
func (m *ImageModel) MaxImagesPerCall() int {
	if limit, ok := imageMaxImagesPerCall[m.modelID]; ok {
		return limit
	}

	if strings.HasPrefix(m.modelID, "gpt-image-") {
		return 10
	}

	return 1
}

// imageGenerationRequest is the body of an image generation request.
type imageGenerationRequest struct {
	Model             string            `json:"model"`
	Prompt            string            `json:"prompt"`
	N                 int               `json:"n,omitempty"`
	Size              string            `json:"size,omitempty"`
	Quality           ImageQuality      `json:"quality,omitempty"`
	Style             ImageStyle        `json:"style,omitempty"`
	Background        ImageBackground   `json:"background,omitempty"`
	Moderation        ImageModeration   `json:"moderation,omitempty"`
	OutputFormat      ImageOutputFormat `json:"output_format,omitempty"`
	OutputCompression *int              `json:"output_compression,omitempty"`
	User              string            `json:"user,omitempty"`
	ResponseFormat    string            `json:"response_format,omitempty"`
}

// imageResponse is an image generation or edit response.
type imageResponse struct {
	Created      *int64      `json:"created"`
	Data         []imageData `json:"data"`
	Background   string      `json:"background"`
	OutputFormat string      `json:"output_format"`
	Size         string      `json:"size"`
	Quality      string      `json:"quality"`
	Usage        *imageUsage `json:"usage"`
}

// imageData is one generated image.
type imageData struct {
	B64JSON       *string `json:"b64_json"`
	RevisedPrompt string  `json:"revised_prompt"`
}

// imageUsage is the token usage of an image call.
type imageUsage struct {
	InputTokens        *int                     `json:"input_tokens"`
	OutputTokens       *int                     `json:"output_tokens"`
	TotalTokens        *int                     `json:"total_tokens"`
	InputTokensDetails *imageInputTokensDetails `json:"input_tokens_details"`
}

// imageInputTokensDetails breaks down image input tokens.
type imageInputTokensDetails struct {
	ImageTokens *int `json:"image_tokens"`
	TextTokens  *int `json:"text_tokens"`
}

// Generate creates images from the prompt, or edits the input files when
// any are given.
func (m *ImageModel) Generate(ctx context.Context, options model.ImageOptions) (*model.ImageResult, error) {
	warnings := imageWarnings(options)
	timestamp := time.Now()

	var response *apicall.Response
	var err error

	if len(options.Files) > 0 {
		response, err = m.edit(ctx, options)
	} else {
		response, err = m.generate(ctx, options)
	}

	if err != nil {
		return nil, err
	}

	var decoded imageResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	images, err := decoded.images()
	if err != nil {
		return nil, err
	}

	return &model.ImageResult{
		Images: images,
		Usage:  decoded.Usage.result(),
		ProviderMetadata: model.ProviderMetadata{
			Namespace: decoded.metadata(),
		},
		Warnings: warnings,
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: timestamp,
				ModelID:   m.modelID,
			},
			Headers: response.Headers,
		},
	}, nil
}

func imageWarnings(options model.ImageOptions) []model.Warning {
	var warnings []model.Warning

	if options.AspectRatio != "" {
		warnings = append(warnings, model.UnsupportedWarning(
			"aspectRatio",
			"This model does not support aspect ratio. Use `size` instead.",
		))
	}

	if options.Seed != nil {
		warnings = append(warnings, model.UnsupportedWarning("seed", ""))
	}

	return warnings
}

func (m *ImageModel) generate(ctx context.Context, options model.ImageOptions) (*apicall.Response, error) {
	openaiOptions, _, err := model.ProviderValue[ImageGenerationOptions](options.ProviderOptions, Namespace)
	if err != nil {
		return nil, err
	}

	if err := openaiOptions.validate(); err != nil {
		return nil, err
	}

	request, err := m.config.request(ctx, "/images/generations", options.Headers)
	if err != nil {
		return nil, err
	}

	body := imageGenerationRequest{
		Model:             m.modelID,
		Prompt:            options.Prompt,
		N:                 options.N,
		Size:              options.Size,
		Quality:           openaiOptions.Quality,
		Style:             openaiOptions.Style,
		Background:        openaiOptions.Background,
		Moderation:        openaiOptions.Moderation,
		OutputFormat:      openaiOptions.OutputFormat,
		OutputCompression: openaiOptions.OutputCompression,
		User:              openaiOptions.User,
	}

	if !imageHasDefaultResponseFormat(m.modelID) {
		body.ResponseFormat = "b64_json"
	}

	return apicall.PostJSON(ctx, request, body)
}

func (m *ImageModel) edit(ctx context.Context, options model.ImageOptions) (*apicall.Response, error) {
	openaiOptions, _, err := model.ProviderValue[ImageEditOptions](options.ProviderOptions, Namespace)
	if err != nil {
		return nil, err
	}

	if err := openaiOptions.validate(); err != nil {
		return nil, err
	}

	fields, err := m.editFields(ctx, options, openaiOptions)
	if err != nil {
		return nil, err
	}

	request, err := m.config.request(ctx, "/images/edits", options.Headers)
	if err != nil {
		return nil, err
	}

	return apicall.PostForm(ctx, request, fields)
}

// editFields builds the multipart form of an edit. Several input images are
// sent as "image[]" fields.
func (m *ImageModel) editFields(
	ctx context.Context,
	options model.ImageOptions,
	openaiOptions ImageEditOptions,
) ([]apicall.FormField, error) {
	form := imageForm{
		fields: []apicall.FormField{
			{
				Name:  "model",
				Value: m.modelID,
			},
		},
	}

	form.add("prompt", options.Prompt)

	imageField := "image"
	if len(options.Files) > 1 {
		imageField = "image[]"
	}

	for _, file := range options.Files {
		if err := form.addFile(ctx, imageField, file); err != nil {
			return nil, err
		}
	}

	if options.Mask != nil {
		if err := form.addFile(ctx, "mask", options.Mask); err != nil {
			return nil, err
		}
	}

	if options.N > 0 {
		form.add("n", strconv.Itoa(options.N))
	}

	form.add("size", options.Size)
	form.add("quality", string(openaiOptions.Quality))
	form.add("background", string(openaiOptions.Background))
	form.add("output_format", string(openaiOptions.OutputFormat))

	if openaiOptions.OutputCompression != nil {
		form.add("output_compression", strconv.Itoa(*openaiOptions.OutputCompression))
	}

	form.add("input_fidelity", string(openaiOptions.InputFidelity))
	form.add("user", openaiOptions.User)

	return form.fields, nil
}

// imageForm collects multipart fields in order.
type imageForm struct {
	fields []apicall.FormField
}

// add appends a value field unless value is empty.
func (f *imageForm) add(name, value string) {
	if value == "" {
		return
	}

	f.fields = append(f.fields, apicall.FormField{
		Name:  name,
		Value: value,
	})
}

// addFile appends an image, downloading it first when it is a URL.
func (f *imageForm) addFile(ctx context.Context, name string, file model.ImageFile) error {
	data, mediaType, err := imageFileContent(ctx, file)
	if err != nil {
		return err
	}

	f.fields = append(f.fields, apicall.FormField{
		Name:      name,
		Filename:  "blob",
		MediaType: mediaType,
		Data:      data,
	})

	return nil
}

// imageFileContent returns the bytes and media type of an input image.
func imageFileContent(ctx context.Context, file model.ImageFile) ([]byte, string, error) {
	switch file := file.(type) {
	case model.ImageBytes:
		mediaType, _ := mediatype.Resolve(file.MediaType, file.Data)
		if mediaType == "" {
			mediaType = file.MediaType
		}

		return file.Data, mediaType, nil

	case model.ImageURL:
		return imageDownload(ctx, file.URL)

	default:
		return nil, "", fmt.Errorf("%w: unsupported image file %T", model.ErrInvalidArgument, file)
	}
}

// imageDownload fetches an input image from an untrusted URL without the
// provider's credentials.
func imageDownload(ctx context.Context, location string) ([]byte, string, error) {
	downloaded, err := download.Fetch(ctx, download.Request{
		URL: location,
	})
	if err != nil {
		return nil, "", err
	}

	return downloaded.Data, downloaded.MediaType, nil
}

func imageHasDefaultResponseFormat(modelID string) bool {
	for _, prefix := range imageDefaultResponseFormatPrefixes {
		if strings.HasPrefix(modelID, prefix) {
			return true
		}
	}

	return false
}

// images decodes the base64 image data.
func (r imageResponse) images() ([][]byte, error) {
	if r.Data == nil {
		return nil, fmt.Errorf("%w: response has no data", model.ErrInvalidResponse)
	}

	images := make([][]byte, 0, len(r.Data))
	for index, item := range r.Data {
		if item.B64JSON == nil {
			return nil, fmt.Errorf("%w: image %d has no b64_json data", model.ErrInvalidResponse, index)
		}

		image, err := base64.StdEncoding.DecodeString(*item.B64JSON)
		if err != nil {
			return nil, fmt.Errorf("%w: decode image %d: %w", model.ErrInvalidResponse, index, err)
		}

		images = append(images, image)
	}

	return images, nil
}

func (r imageResponse) metadata() ImageMetadata {
	details := make([]ImageDetails, 0, len(r.Data))

	for index, item := range r.Data {
		imageDetails := ImageDetails{
			RevisedPrompt: item.RevisedPrompt,
			Created:       r.Created,
			Size:          r.Size,
			Quality:       r.Quality,
			Background:    r.Background,
			OutputFormat:  r.OutputFormat,
		}

		if r.Usage != nil && r.Usage.InputTokensDetails != nil {
			tokens := r.Usage.InputTokensDetails
			imageDetails.ImageTokens = imageTokenShare(tokens.ImageTokens, index, len(r.Data))
			imageDetails.TextTokens = imageTokenShare(tokens.TextTokens, index, len(r.Data))
		}

		details = append(details, imageDetails)
	}

	return ImageMetadata{
		Images: details,
	}
}

// imageTokenShare splits total evenly across count images and gives the
// remainder to the last one, so the shares sum to the total.
func imageTokenShare(total *int, index, count int) *int {
	if total == nil {
		return nil
	}

	share := *total / count
	if index == count-1 {
		return new(*total - share*(count-1))
	}

	return &share
}

func (u *imageUsage) result() *model.ImageUsage {
	if u == nil {
		return nil
	}

	return &model.ImageUsage{
		InputTokens:  u.InputTokens,
		OutputTokens: u.OutputTokens,
		TotalTokens:  u.TotalTokens,
	}
}

func (o ImageOptions) validate() error {
	switch o.Quality {
	case "",
		ImageQualityStandard,
		ImageQualityHD,
		ImageQualityLow,
		ImageQualityMedium,
		ImageQualityHigh,
		ImageQualityXHigh,
		ImageQualityMax,
		ImageQualityAuto:
	default:
		return imageOptionError("unknown quality %q", o.Quality)
	}

	switch o.Background {
	case "", ImageBackgroundTransparent, ImageBackgroundOpaque, ImageBackgroundAuto:
	default:
		return imageOptionError("unknown background %q", o.Background)
	}

	switch o.OutputFormat {
	case "", ImageOutputPNG, ImageOutputJPEG, ImageOutputWebP:
	default:
		return imageOptionError("unknown outputFormat %q", o.OutputFormat)
	}

	if o.OutputCompression == nil {
		return nil
	}

	compression := *o.OutputCompression
	if compression < 0 || compression > 100 {
		return imageOptionError("outputCompression must be between 0 and 100, got %d", compression)
	}

	return nil
}

func (o ImageGenerationOptions) validate() error {
	if err := o.ImageOptions.validate(); err != nil {
		return err
	}

	switch o.Style {
	case "", ImageStyleVivid, ImageStyleNatural:
	default:
		return imageOptionError("unknown style %q", o.Style)
	}

	switch o.Moderation {
	case "", ImageModerationAuto, ImageModerationLow:
	default:
		return imageOptionError("unknown moderation %q", o.Moderation)
	}

	return nil
}

func (o ImageEditOptions) validate() error {
	if err := o.ImageOptions.validate(); err != nil {
		return err
	}

	switch o.InputFidelity {
	case "", ImageInputFidelityHigh, ImageInputFidelityLow:
	default:
		return imageOptionError("unknown inputFidelity %q", o.InputFidelity)
	}

	return nil
}

func imageOptionError(format string, args ...any) error {
	return fmt.Errorf("%w: openai image options: %s", model.ErrInvalidArgument, fmt.Sprintf(format, args...))
}
