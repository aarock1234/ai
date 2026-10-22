package togetherai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"maps"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// imagePath is the image generation endpoint.
const imagePath = "/images/generations"

// imageResponseFormat asks for inline base64 images.
const imageResponseFormat = "base64"

// nonDiffusionImageModels are image models that reject diffusion settings
// and seeds.
var nonDiffusionImageModels = map[string]bool{
	"google/gemini-3-pro-image": true,
}

// ImageModel is a Together AI image generation model. FLUX Kontext models
// edit the first input file; mask-based editing is not supported.
type ImageModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "togetherai.image" by default.
func (m *ImageModel) Provider() string {
	return m.provider.name + ".image"
}

// ModelID returns the model identifier.
func (m *ImageModel) ModelID() string {
	return m.modelID
}

// MaxImagesPerCall returns 1.
func (m *ImageModel) MaxImagesPerCall() int {
	return 1
}

type imageRequest struct {
	Model          string `json:"model"`
	Prompt         string `json:"prompt"`
	Seed           *int   `json:"seed,omitempty"`
	N              int    `json:"n,omitempty"`
	Width          *int   `json:"width,omitempty"`
	Height         *int   `json:"height,omitempty"`
	ImageURL       string `json:"image_url,omitempty"`
	ResponseFormat string `json:"response_format"`
}

type imageResponse struct {
	Data []imageData `json:"data"`
}

type imageData struct {
	B64JSON string `json:"b64_json"`
}

// Generate generates one image, using the first input file as the reference
// image of editing models.
func (m *ImageModel) Generate(ctx context.Context, options model.ImageOptions) (*model.ImageResult, error) {
	if options.Mask != nil {
		return nil, fmt.Errorf(
			"%w: Together AI does not support mask-based image editing. "+
				"Use FLUX Kontext models (e.g., black-forest-labs/FLUX.1-kontext-pro) "+
				"with a reference image and descriptive prompt instead",
			model.ErrUnsupported,
		)
	}

	timestamp := time.Now()

	body, warnings, err := m.requestBody(options)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(imagePath, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, err
	}

	var decoded imageResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	images, err := decodeImages(decoded.Data)
	if err != nil {
		return nil, err
	}

	return &model.ImageResult{
		Images:   images,
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

// requestBody builds the request with the provider options layered over the
// standard settings.
func (m *ImageModel) requestBody(options model.ImageOptions) (map[string]json.RawMessage, []model.Warning, error) {
	var warnings []model.Warning

	if options.AspectRatio != "" {
		warnings = append(warnings, model.UnsupportedWarning(
			"aspectRatio",
			"This model does not support the `aspectRatio` option. Use `size` instead.",
		))
	}

	providerMembers, err := imageOptionMembers(options.ProviderOptions)
	if err != nil {
		return nil, nil, err
	}

	nonDiffusion := nonDiffusionImageModels[m.modelID]

	request := imageRequest{
		Model:          m.modelID,
		Prompt:         options.Prompt,
		ResponseFormat: imageResponseFormat,
	}

	if nonDiffusion {
		for _, key := range diffusionOptionKeys {
			delete(providerMembers, key)
		}

		if options.Seed != nil {
			warnings = append(warnings, model.UnsupportedWarning(
				"seed",
				fmt.Sprintf("The %s model does not support the `seed` option.", m.modelID),
			))
		}
	} else {
		request.Seed = options.Seed
	}

	if options.N > 1 {
		request.N = options.N
	}

	if options.Size != "" {
		width, height, err := parseSize(options.Size)
		if err != nil {
			return nil, nil, err
		}

		request.Width = &width
		request.Height = &height
	}

	if len(options.Files) > 0 {
		request.ImageURL = imageFileURL(options.Files[0])

		if len(options.Files) > 1 {
			warnings = append(warnings, model.Warning{
				Type:    model.WarningOther,
				Message: "Together AI only supports a single input image. Additional images are ignored.",
			})
		}
	}

	body, err := jsonobject.Members(request)
	if err != nil {
		return nil, nil, fmt.Errorf("encode request body: %w", err)
	}

	maps.Copy(body, providerMembers)

	return body, warnings, nil
}

// parseSize splits a "widthxheight" size.
func parseSize(size string) (int, int, error) {
	widthText, heightText, _ := strings.Cut(size, "x")

	width, err := strconv.Atoi(widthText)
	if err != nil {
		return 0, 0, fmt.Errorf("%w: size %q must have the form widthxheight", model.ErrInvalidArgument, size)
	}

	height, err := strconv.Atoi(heightText)
	if err != nil {
		return 0, 0, fmt.Errorf("%w: size %q must have the form widthxheight", model.ErrInvalidArgument, size)
	}

	return width, height, nil
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

func decodeImages(data []imageData) ([][]byte, error) {
	images := make([][]byte, 0, len(data))
	for _, item := range data {
		image, err := base64.StdEncoding.DecodeString(item.B64JSON)
		if err != nil {
			return nil, fmt.Errorf("%w: decode image: %w", model.ErrInvalidResponse, err)
		}

		images = append(images, image)
	}

	return images, nil
}

var _ model.ImageModel = (*ImageModel)(nil)
