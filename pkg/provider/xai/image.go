package xai

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/internal/download"
	"github.com/aarock1234/ai/pkg/model"
)

// Image endpoints.
const (
	imageGenerationPath = "/images/generations"
	imageEditPath       = "/images/edits"
)

// maxImagesPerCall is the per-call image limit.
const maxImagesPerCall = 3

// errModeratedImage reports an image blocked by content moderation.
var errModeratedImage = errors.New("image generation was blocked due to a content policy violation")

// ImageOptions are the provider options of image models, read from the
// "xai" namespace. The JSON names follow the wire format.
type ImageOptions struct {
	// AspectRatio is used when the call does not set one.
	AspectRatio *string `json:"aspect_ratio,omitempty"`

	OutputFormat *string `json:"output_format,omitempty"`
	SyncMode     *bool   `json:"sync_mode,omitempty"`

	// Resolution is "1k", "1.5k" or "2k".
	Resolution *string `json:"resolution,omitempty"`

	// Quality is "low", "medium", "high" or "auto".
	Quality *string `json:"quality,omitempty"`

	User *string `json:"user,omitempty"`
}

// ImageMetadata is the result provider metadata of image models, stored
// under "xai".
type ImageMetadata struct {
	// Images has one entry per generated image.
	Images []GeneratedImage `json:"images"`

	CostInUSDTicks *float64 `json:"costInUsdTicks,omitempty"`
}

// GeneratedImage describes one generated image.
type GeneratedImage struct {
	RevisedPrompt string `json:"revisedPrompt,omitempty"`
}

// ImageModel is an xAI image generation and editing model. Calls with input
// files edit them; masks are not supported.
type ImageModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "xai.image" by default.
func (m *ImageModel) Provider() string {
	return m.provider.name + ".image"
}

// ModelID returns the model identifier.
func (m *ImageModel) ModelID() string {
	return m.modelID
}

// MaxImagesPerCall returns 3.
func (m *ImageModel) MaxImagesPerCall() int {
	return maxImagesPerCall
}

type imageRequest struct {
	Model          string           `json:"model"`
	Prompt         string           `json:"prompt"`
	N              int              `json:"n,omitempty"`
	ResponseFormat string           `json:"response_format"`
	AspectRatio    *string          `json:"aspect_ratio,omitempty"`
	OutputFormat   *string          `json:"output_format,omitempty"`
	SyncMode       *bool            `json:"sync_mode,omitempty"`
	Resolution     *string          `json:"resolution,omitempty"`
	Quality        *string          `json:"quality,omitempty"`
	User           *string          `json:"user,omitempty"`
	Image          *imageReference  `json:"image,omitempty"`
	Images         []imageReference `json:"images,omitempty"`
}

type imageReference struct {
	URL  string `json:"url"`
	Type string `json:"type"`
}

type imageResponse struct {
	Data  []imageData `json:"data"`
	Usage *imageUsage `json:"usage"`
}

type imageData struct {
	URL               *string `json:"url"`
	B64JSON           *string `json:"b64_json"`
	RevisedPrompt     *string `json:"revised_prompt"`
	RespectModeration *bool   `json:"respect_moderation"`
}

type imageUsage struct {
	CostInUSDTicks *float64 `json:"cost_in_usd_ticks"`
}

// Generate generates images, or edits the input files when there are any.
func (m *ImageModel) Generate(ctx context.Context, options model.ImageOptions) (*model.ImageResult, error) {
	timestamp := time.Now()

	imageOptions, err := readImageOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	path := imageGenerationPath
	if len(options.Files) > 0 {
		path = imageEditPath
	}

	request, err := m.provider.apiRequest(path, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, m.requestBody(options, imageOptions))
	if err != nil {
		return nil, err
	}

	var decoded imageResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	images, err := m.images(ctx, decoded.Data, request)
	if err != nil {
		return nil, err
	}

	return &model.ImageResult{
		Images:           images,
		ProviderMetadata: imageMetadata(decoded),
		Warnings:         unsupportedImageSettings(options),
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: timestamp,
				ModelID:   m.modelID,
			},
			Headers: response.Headers,
		},
	}, nil
}

func unsupportedImageSettings(options model.ImageOptions) []model.Warning {
	var warnings []model.Warning

	if options.Size != "" {
		warnings = append(warnings, model.UnsupportedWarning(
			"size",
			"This model does not support the `size` option. Use `aspectRatio` instead.",
		))
	}

	if options.Seed != nil {
		warnings = append(warnings, model.UnsupportedWarning("seed", ""))
	}

	if options.Mask != nil {
		warnings = append(warnings, model.UnsupportedWarning("mask", ""))
	}

	return warnings
}

// requestBody builds the request. The call's aspect ratio takes precedence
// over the provider option.
func (m *ImageModel) requestBody(options model.ImageOptions, imageOptions ImageOptions) imageRequest {
	body := imageRequest{
		Model:          m.modelID,
		Prompt:         options.Prompt,
		N:              options.N,
		ResponseFormat: "b64_json",
		AspectRatio:    imageOptions.AspectRatio,
		OutputFormat:   imageOptions.OutputFormat,
		SyncMode:       imageOptions.SyncMode,
		Resolution:     imageOptions.Resolution,
		Quality:        imageOptions.Quality,
		User:           imageOptions.User,
	}

	if options.AspectRatio != "" {
		body.AspectRatio = new(options.AspectRatio)
	}

	references := make([]imageReference, 0, len(options.Files))
	for _, file := range options.Files {
		references = append(references, imageReference{
			URL:  imageFileURL(file),
			Type: "image_url",
		})
	}

	switch len(references) {
	case 0:
	case 1:
		body.Image = &references[0]
	default:
		body.Images = references
	}

	return body
}

// imageFileURL returns a URL file unchanged and inline bytes as a data URL.
func imageFileURL(file model.ImageFile) string {
	switch file := file.(type) {
	case model.ImageURL:
		return file.URL
	case model.ImageBytes:
		return dataurl.Encode(file.MediaType, file.Data)
	default:
		return ""
	}
}

// images decodes inline images, or downloads every image when any arrived
// by URL. Moderated images fail the call.
func (m *ImageModel) images(ctx context.Context, data []imageData, request apicall.Request) ([][]byte, error) {
	allInline := true

	for _, item := range data {
		if item.RespectModeration != nil && !*item.RespectModeration {
			return nil, fmt.Errorf("%w: %w", model.ErrNoContentGenerated, errModeratedImage)
		}

		if item.B64JSON == nil {
			allInline = false
		}
	}

	images := make([][]byte, 0, len(data))
	for _, item := range data {
		image, err := m.image(ctx, item, allInline, request)
		if err != nil {
			return nil, err
		}

		images = append(images, image)
	}

	return images, nil
}

func (m *ImageModel) image(ctx context.Context, item imageData, inline bool, request apicall.Request) ([]byte, error) {
	if inline {
		image, err := base64.StdEncoding.DecodeString(*item.B64JSON)
		if err != nil {
			return nil, fmt.Errorf("%w: decode image: %w", model.ErrInvalidResponse, err)
		}

		return image, nil
	}

	if item.URL == nil {
		return nil, fmt.Errorf("%w: expected b64_json or url in the image response", model.ErrInvalidResponse)
	}

	downloaded, err := download.Fetch(ctx, download.Request{
		URL:           *item.URL,
		TrustedOrigin: m.provider.baseURL,
		Client:        request.Client,
	})
	if err != nil {
		return nil, err
	}

	return downloaded.Data, nil
}

func imageMetadata(response imageResponse) model.ProviderMetadata {
	metadata := ImageMetadata{
		Images: make([]GeneratedImage, 0, len(response.Data)),
	}

	for _, item := range response.Data {
		image := GeneratedImage{}
		if item.RevisedPrompt != nil {
			image.RevisedPrompt = *item.RevisedPrompt
		}

		metadata.Images = append(metadata.Images, image)
	}

	if response.Usage != nil {
		metadata.CostInUSDTicks = response.Usage.CostInUSDTicks
	}

	return model.ProviderMetadata{
		Namespace: metadata,
	}
}

// readImageOptions decodes and validates the image options.
func readImageOptions(options model.ProviderOptions) (ImageOptions, error) {
	imageOptions, _, err := model.ProviderValue[ImageOptions](options, Namespace)
	if err != nil {
		return ImageOptions{}, err
	}

	if imageOptions.Resolution != nil && !slices.Contains([]string{"1k", "1.5k", "2k"}, *imageOptions.Resolution) {
		return ImageOptions{}, fmt.Errorf("%w: xai provider options: unsupported resolution %q", model.ErrInvalidArgument, *imageOptions.Resolution)
	}

	if imageOptions.Quality != nil && !slices.Contains([]string{"low", "medium", "high", "auto"}, *imageOptions.Quality) {
		return ImageOptions{}, fmt.Errorf("%w: xai provider options: unsupported quality %q", model.ErrInvalidArgument, *imageOptions.Quality)
	}

	return imageOptions, nil
}

var _ model.ImageModel = (*ImageModel)(nil)
