package azure

import (
	"context"
	"fmt"
	"slices"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

var _ model.ImageModel = (*ImageModel)(nil)

// maiImageModels are the lowercase model IDs served by the MAI image API.
var maiImageModels = []string{
	"mai-image-2.5",
	"mai-image-2.5-flash",
	"mai-image-2.5-pro",
	"mai-image-2.6",
	"mai-image-2.6-flash",
}

// ImageAPI selects the API of an image call.
type ImageAPI string

// Image APIs.
const (
	ImageAPIOpenAI ImageAPI = "openai"
	ImageAPIMAI    ImageAPI = "mai"
)

// ImageOptions are the Azure options of an image call, read from
// ProviderOptions["azure"]. OpenAI image options stay in the "openai"
// namespace.
type ImageOptions struct {
	// API selects the API. The default is the MAI image API for MAI-Image
	// models and the OpenAI image API otherwise.
	API ImageAPI `json:"api,omitempty"`

	// AutoAspectRatio lets the model pick the output aspect ratio from the
	// prompt and any reference images. MAI-Image-2.6 models only.
	AutoAspectRatio *bool `json:"autoAspectRatio,omitempty"`

	// WebGrounding grounds generation in Bing web search results.
	// MAI-Image-2.6 models only.
	WebGrounding *bool `json:"webGrounding,omitempty"`
}

func (o ImageOptions) validate() error {
	switch o.API {
	case "", ImageAPIOpenAI, ImageAPIMAI:
		return nil
	default:
		return fmt.Errorf("%w: providerOptions.azure.api must be %q or %q", model.ErrInvalidArgument, ImageAPIOpenAI, ImageAPIMAI)
	}
}

// maiOptionNames returns the names of the MAI-only options that are set.
func (o ImageOptions) maiOptionNames() []string {
	var names []string

	if o.AutoAspectRatio != nil {
		names = append(names, "autoAspectRatio")
	}

	if o.WebGrounding != nil {
		names = append(names, "webGrounding")
	}

	return names
}

// ImageModel is an Azure image deployment. It selects the MAI or OpenAI
// image API for each call.
type ImageModel struct {
	modelID  string
	provider string
	openai   *openai.ImageModel
	mai      *maiImageModel
}

// Provider returns "azure.image".
func (m *ImageModel) Provider() string {
	return m.provider
}

// ModelID returns the deployment ID.
func (m *ImageModel) ModelID() string {
	return m.modelID
}

// MaxImagesPerCall returns the limit of the default API: 1 for the MAI
// image API, which returns one image per request.
func (m *ImageModel) MaxImagesPerCall() int {
	if isMAIImageModel(m.modelID) {
		return maiMaxImagesPerCall
	}

	return m.openai.MaxImagesPerCall()
}

// Generate generates or edits images with the selected API. MAI options sent
// to the OpenAI image API are reported as unsupported.
func (m *ImageModel) Generate(ctx context.Context, options model.ImageOptions) (*model.ImageResult, error) {
	azureOptions, _, err := model.ProviderValue[ImageOptions](options.ProviderOptions, Namespace)
	if err != nil {
		return nil, err
	}

	if err := azureOptions.validate(); err != nil {
		return nil, err
	}

	if m.selectedAPI(azureOptions.API) == ImageAPIMAI {
		return m.mai.generate(ctx, options, azureOptions)
	}

	result, err := m.openai.Generate(ctx, options)
	if err != nil {
		return nil, err
	}

	result.Warnings = append(result.Warnings, unsupportedOptionWarnings(azureOptions.maiOptionNames(), maiImageOnlyDetails)...)

	return result, nil
}

// selectedAPI returns the requested API or the model's default.
func (m *ImageModel) selectedAPI(requested ImageAPI) ImageAPI {
	if requested != "" {
		return requested
	}

	if isMAIImageModel(m.modelID) {
		return ImageAPIMAI
	}

	return ImageAPIOpenAI
}

func isMAIImageModel(modelID string) bool {
	return slices.Contains(maiImageModels, strings.ToLower(modelID))
}
