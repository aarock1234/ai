package openaicompatible

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"maps"
	"net/http"
	"strconv"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/download"
	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// Image endpoints.
const (
	imageGenerationPath = "/images/generations"
	imageEditPath       = "/images/edits"
)

// maxImagesPerCall is the per-call image limit.
const maxImagesPerCall = 10

// ImageConfig configures an ImageModel.
type ImageConfig struct {
	// Provider is the model's provider identifier, such as "name.image".
	Provider string

	// BuildURL returns the full URL of an API path such as
	// "/chat/completions", or an error when the settings needed to build it
	// are missing.
	BuildURL URLBuilder

	// Headers returns the provider headers of one request.
	Headers func() (http.Header, error)

	// HTTPClient sends requests. Nil uses http.DefaultClient.
	HTTPClient *http.Client

	// ParseError reads failed response bodies. Nil uses ParseError.
	ParseError ErrorParser
}

// ImageModel is an OpenAI-compatible image model. Calls with input files edit
// images; other calls generate them.
type ImageModel struct {
	modelID string
	config  ImageConfig
}

// NewImageModel returns an image model that uses config.
func NewImageModel(modelID string, config ImageConfig) *ImageModel {
	return &ImageModel{
		modelID: modelID,
		config:  config,
	}
}

// Provider returns the configured provider identifier.
func (m *ImageModel) Provider() string {
	return m.config.Provider
}

// ModelID returns the model identifier.
func (m *ImageModel) ModelID() string {
	return m.modelID
}

// MaxImagesPerCall returns the per-call image limit.
func (m *ImageModel) MaxImagesPerCall() int {
	return maxImagesPerCall
}

type imageRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt,omitempty"`
	N      int    `json:"n,omitempty"`
	Size   string `json:"size,omitempty"`
}

type imageResponse struct {
	Data  []imageData  `json:"data"`
	Usage *imageTokens `json:"usage"`
}

type imageData struct {
	B64JSON *string `json:"b64_json"`
	URL     *string `json:"url"`
}

type imageTokens struct {
	InputTokens  *int `json:"input_tokens"`
	OutputTokens *int `json:"output_tokens"`
	TotalTokens  *int `json:"total_tokens"`
}

// Generate generates images, or edits the input files when there are any.
func (m *ImageModel) Generate(ctx context.Context, options model.ImageOptions) (*model.ImageResult, error) {
	timestamp := time.Now()
	warnings := unsupportedImageSettings(options)

	name := optionsName(m.config.Provider)
	warnings = append(warnings, deprecatedNameWarnings(name, options.ProviderOptions)...)

	args, err := mergedMembers(options.ProviderOptions, name, toCamelCase(name))
	if err != nil {
		return nil, err
	}

	editing := len(options.Files) > 0

	path := imageGenerationPath
	if editing {
		path = imageEditPath
	}

	request, err := newAPIRequest(apiRequestSettings{
		path:        path,
		buildURL:    m.config.BuildURL,
		headers:     m.config.Headers,
		callHeaders: options.Headers,
		client:      m.config.HTTPClient,
		parseError:  m.config.ParseError,
	})
	if err != nil {
		return nil, err
	}

	var response *apicall.Response
	if editing {
		response, err = m.edit(ctx, request, options, args)
	} else {
		response, err = m.generate(ctx, request, options, args)
	}

	if err != nil {
		return nil, err
	}

	var decoded imageResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	images, err := m.images(ctx, request, decoded.Data)
	if err != nil {
		return nil, err
	}

	return &model.ImageResult{
		Images:   images,
		Usage:    imageUsage(decoded.Usage),
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

func unsupportedImageSettings(options model.ImageOptions) []model.Warning {
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

// generate sends a JSON generation request. Provider options replace the
// standard fields.
func (m *ImageModel) generate(
	ctx context.Context,
	request apicall.Request,
	options model.ImageOptions,
	args map[string]json.RawMessage,
) (*apicall.Response, error) {
	body, err := jsonobject.Members(imageRequest{
		Model:  m.modelID,
		Prompt: options.Prompt,
		N:      options.N,
		Size:   options.Size,
	})
	if err != nil {
		return nil, fmt.Errorf("encode request body: %w", err)
	}

	maps.Copy(body, args)

	return apicall.PostJSON(ctx, request, body)
}

// edit sends a multipart edit request with the input images and mask.
func (m *ImageModel) edit(
	ctx context.Context,
	request apicall.Request,
	options model.ImageOptions,
	args map[string]json.RawMessage,
) (*apicall.Response, error) {
	images := make([]formFile, 0, len(options.Files))
	for _, file := range options.Files {
		image, err := m.loadImageFile(ctx, request, file)
		if err != nil {
			return nil, err
		}

		images = append(images, image)
	}

	var form imageForm
	form.setValue("model", m.modelID)

	if options.Prompt != "" {
		form.setValue("prompt", options.Prompt)
	}

	form.setFiles("image", images)

	if options.Mask != nil {
		mask, err := m.loadImageFile(ctx, request, options.Mask)
		if err != nil {
			return nil, err
		}

		form.setFiles("mask", []formFile{
			mask,
		})
	}

	if options.N > 0 {
		form.setValue("n", strconv.Itoa(options.N))
	}

	if options.Size != "" {
		form.setValue("size", options.Size)
	}

	if err := form.setJSONMembers(args); err != nil {
		return nil, err
	}

	return apicall.PostForm(ctx, request, form.fields())
}

// loadImageFile returns inline image bytes, downloading URL images with the
// provider HTTP client.
func (m *ImageModel) loadImageFile(ctx context.Context, request apicall.Request, file model.ImageFile) (formFile, error) {
	switch file := file.(type) {
	case model.ImageBytes:
		return formFile{
			MediaType: file.MediaType,
			Data:      file.Data,
		}, nil

	case model.ImageURL:
		downloaded, err := download.Fetch(ctx, download.Request{
			URL:    file.URL,
			Client: request.Client,
		})
		if err != nil {
			return formFile{}, err
		}

		return formFile{
			MediaType: downloaded.MediaType,
			Data:      downloaded.Data,
		}, nil

	default:
		return formFile{}, fmt.Errorf("%w: image file %T", model.ErrUnsupported, file)
	}
}

// images decodes base64 images and downloads URL images. Provider headers are
// sent only to the provider's own origin.
func (m *ImageModel) images(ctx context.Context, request apicall.Request, data []imageData) ([][]byte, error) {
	images := make([][]byte, 0, len(data))

	for _, item := range data {
		image, err := m.image(ctx, request, item)
		if err != nil {
			return nil, err
		}

		images = append(images, image)
	}

	return images, nil
}

func (m *ImageModel) image(ctx context.Context, request apicall.Request, item imageData) ([]byte, error) {
	if item.B64JSON != nil {
		image, err := base64.StdEncoding.DecodeString(*item.B64JSON)
		if err != nil {
			return nil, fmt.Errorf("%w: decode base64 image: %w", model.ErrInvalidResponse, err)
		}

		return image, nil
	}

	if item.URL == nil {
		return nil, fmt.Errorf("%w: expected b64_json or url in the image response", model.ErrInvalidResponse)
	}

	downloaded, err := download.Fetch(ctx, download.Request{
		URL:           *item.URL,
		Headers:       request.Headers,
		TrustedOrigin: request.URL,
		Client:        request.Client,
	})
	if err != nil {
		return nil, err
	}

	return downloaded.Data, nil
}

// imageUsage converts reported usage. Many providers omit it.
func imageUsage(usage *imageTokens) *model.ImageUsage {
	if usage == nil {
		return nil
	}

	return &model.ImageUsage{
		InputTokens:  usage.InputTokens,
		OutputTokens: usage.OutputTokens,
		TotalTokens:  usage.TotalTokens,
	}
}

var _ model.ImageModel = (*ImageModel)(nil)
