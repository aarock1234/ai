package gateway

import (
	"context"
	"encoding/base64"
	"fmt"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// imageModelPath is the image model endpoint.
const imageModelPath = "/image-model"

// ImageModel is an AI Gateway image model. It is safe for concurrent use.
type ImageModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "gateway" by default.
func (m *ImageModel) Provider() string {
	return m.provider.name
}

// ModelID returns the model identifier.
func (m *ImageModel) ModelID() string {
	return m.modelID
}

// MaxImagesPerCall returns 0: the gateway applies the underlying model's
// limit itself.
func (m *ImageModel) MaxImagesPerCall() int {
	return 0
}

type imageRequest struct {
	Prompt          string                `json:"prompt,omitempty"`
	N               int                   `json:"n"`
	Size            string                `json:"size,omitempty"`
	AspectRatio     string                `json:"aspectRatio,omitempty"`
	Seed            *int                  `json:"seed,omitempty"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
	Files           []imageFile           `json:"files,omitempty"`
	Mask            *imageFile            `json:"mask,omitempty"`
}

// imageFile is an input image on the wire: base64 data or a URL.
type imageFile struct {
	Type            string                `json:"type"`
	MediaType       string                `json:"mediaType,omitempty"`
	Data            string                `json:"data,omitempty"`
	URL             string                `json:"url,omitempty"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

type imageResponse struct {
	Images           []string      `json:"images"`
	Warnings         []wireWarning `json:"warnings"`
	ProviderMetadata wireMetadata  `json:"providerMetadata"`
	Usage            *imageUsage   `json:"usage"`
}

type imageUsage struct {
	InputTokens  *int `json:"inputTokens"`
	OutputTokens *int `json:"outputTokens"`
	TotalTokens  *int `json:"totalTokens"`
}

// Generate generates or edits images through the gateway.
func (m *ImageModel) Generate(ctx context.Context, options model.ImageOptions) (*model.ImageResult, error) {
	body, err := imageBody(options)
	if err != nil {
		return nil, err
	}

	request, method, err := m.provider.modelRequest(imageModelPath, options.Headers, modelConfigHeaders("image", m.modelID))
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, asGatewayError(err, method)
	}

	var decoded imageResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	images, err := decodeImages(decoded.Images)
	if err != nil {
		return nil, err
	}

	result := &model.ImageResult{
		Images:           images,
		ProviderMetadata: decoded.ProviderMetadata.providerMetadata(),
		Warnings:         decodeWarnings(decoded.Warnings),
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: time.Now(),
				ModelID:   m.modelID,
			},
			Headers: response.Headers,
		},
	}

	if decoded.Usage != nil {
		result.Usage = &model.ImageUsage{
			InputTokens:  decoded.Usage.InputTokens,
			OutputTokens: decoded.Usage.OutputTokens,
			TotalTokens:  decoded.Usage.TotalTokens,
		}
	}

	return result, nil
}

// imageBody builds the request. A zero seed is omitted, as upstream does.
func imageBody(options model.ImageOptions) (imageRequest, error) {
	body := imageRequest{
		Prompt:          options.Prompt,
		N:               options.N,
		Size:            options.Size,
		AspectRatio:     options.AspectRatio,
		ProviderOptions: options.ProviderOptions,
	}

	if options.Seed != nil && *options.Seed != 0 {
		body.Seed = options.Seed
	}

	for _, file := range options.Files {
		encoded, err := encodeImageFile(file)
		if err != nil {
			return imageRequest{}, err
		}

		body.Files = append(body.Files, encoded)
	}

	if options.Mask != nil {
		mask, err := encodeImageFile(options.Mask)
		if err != nil {
			return imageRequest{}, err
		}

		body.Mask = &mask
	}

	return body, nil
}

func encodeImageFile(file model.ImageFile) (imageFile, error) {
	switch file := file.(type) {
	case model.ImageBytes:
		return imageFile{
			Type:            "file",
			MediaType:       file.MediaType,
			Data:            base64.StdEncoding.EncodeToString(file.Data),
			ProviderOptions: file.ProviderOptions,
		}, nil

	case model.ImageURL:
		return imageFile{
			Type:            "url",
			URL:             file.URL,
			ProviderOptions: file.ProviderOptions,
		}, nil

	default:
		return imageFile{}, fmt.Errorf("%w: image file %T", model.ErrUnsupported, file)
	}
}

func decodeImages(encoded []string) ([][]byte, error) {
	images := make([][]byte, 0, len(encoded))

	for _, image := range encoded {
		decoded, err := base64.StdEncoding.DecodeString(image)
		if err != nil {
			return nil, fmt.Errorf("%w: decode base64 image: %w", model.ErrInvalidResponse, err)
		}

		images = append(images, decoded)
	}

	return images, nil
}

var _ model.ImageModel = (*ImageModel)(nil)
