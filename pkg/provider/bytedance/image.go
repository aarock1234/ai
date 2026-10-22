package bytedance

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"maps"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// imagePath is the image generation endpoint.
const imagePath = "/images/generations"

// OutputFormat is the file format of generated images.
type OutputFormat string

// Output formats.
const (
	OutputPNG  OutputFormat = "png"
	OutputJPEG OutputFormat = "jpeg"
)

// SequentialImageGeneration controls batches of related images.
type SequentialImageGeneration string

// Sequential image generation modes.
const (
	SequentialAuto     SequentialImageGeneration = "auto"
	SequentialDisabled SequentialImageGeneration = "disabled"
)

// OptimizePromptMode selects prompt optimization.
type OptimizePromptMode string

// Prompt optimization modes.
const (
	OptimizePromptStandard OptimizePromptMode = "standard"
	OptimizePromptFast     OptimizePromptMode = "fast"
)

// ImageOptions are the provider options of image models, read from the
// "bytedance" namespace. Members other than these fields are sent in the
// request body unchanged.
type ImageOptions struct {
	// Watermark adds an "AI generated" watermark to the bottom-right corner.
	Watermark *bool `json:"watermark,omitempty"`

	// OutputFormat is supported by seedream-5-0 and dola-seedream-5-0-pro;
	// older models always return JPEG.
	OutputFormat *OutputFormat `json:"outputFormat,omitempty"`

	// Size is a resolution level such as "2K". It overrides the size call
	// option.
	Size *string `json:"size,omitempty"`

	// SequentialImageGeneration set to "auto" generates a batch of related
	// images. The default is "disabled".
	SequentialImageGeneration *SequentialImageGeneration `json:"sequentialImageGeneration,omitempty"`

	// MaxImages limits a sequential batch.
	MaxImages *int `json:"maxImages,omitempty"`

	// OptimizePromptMode selects prompt optimization. Only seedream-4-0
	// supports "fast".
	OptimizePromptMode *OptimizePromptMode `json:"optimizePromptMode,omitempty"`
}

// imageOptionKeys are the ImageOptions members, which are mapped to request
// fields rather than passed through.
var imageOptionKeys = []string{
	"watermark",
	"outputFormat",
	"size",
	"sequentialImageGeneration",
	"maxImages",
	"optimizePromptMode",
}

// ImageModel is a ByteDance Seedream image model. Calls with input files
// edit or combine those images.
type ImageModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "bytedance.image" by default.
func (m *ImageModel) Provider() string {
	return m.provider.name + ".image"
}

// ModelID returns the model identifier.
func (m *ImageModel) ModelID() string {
	return m.modelID
}

// MaxImagesPerCall returns 1: the API has no output count. Use
// SequentialImageGeneration for batches of related images.
func (m *ImageModel) MaxImagesPerCall() int {
	return 1
}

type imageRequest struct {
	Model                            string                     `json:"model"`
	Prompt                           string                     `json:"prompt,omitempty"`
	Image                            json.RawMessage            `json:"image,omitempty"`
	Size                             string                     `json:"size,omitempty"`
	Watermark                        *bool                      `json:"watermark,omitempty"`
	OutputFormat                     *OutputFormat              `json:"output_format,omitempty"`
	SequentialImageGeneration        *SequentialImageGeneration `json:"sequential_image_generation,omitempty"`
	SequentialImageGenerationOptions *sequentialOptions         `json:"sequential_image_generation_options,omitempty"`
	OptimizePromptOptions            *optimizePromptOptions     `json:"optimize_prompt_options,omitempty"`
}

type sequentialOptions struct {
	MaxImages int `json:"max_images"`
}

type optimizePromptOptions struct {
	Mode OptimizePromptMode `json:"mode"`
}

type imageResponse struct {
	Data  []imageData  `json:"data"`
	Usage *imageTokens `json:"usage"`
}

type imageData struct {
	B64JSON string `json:"b64_json"`
}

type imageTokens struct {
	OutputTokens *int `json:"output_tokens"`
	TotalTokens  *int `json:"total_tokens"`
}

// Generate generates images, using the input files as references.
func (m *ImageModel) Generate(ctx context.Context, options model.ImageOptions) (*model.ImageResult, error) {
	timestamp := time.Now()
	warnings := unsupportedImageSettings(options)

	imageOptions, passthrough, err := readImageOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	body, err := m.requestBody(options, imageOptions, passthrough)
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
			`ByteDance does not support aspectRatio. Use `+"`size`"+` (e.g. "2048x2048" or a resolution level like "2K" via providerOptions) instead.`,
		))
	}

	if options.Seed != nil {
		warnings = append(warnings, model.UnsupportedWarning("seed", ""))
	}

	if options.Mask != nil {
		warnings = append(warnings, model.UnsupportedWarning(
			"mask",
			"ByteDance Seedream does not support a separate mask. Provide edit instructions in the prompt, optionally with markings on the input image.",
		))
	}

	return warnings
}

// readImageOptions reads and validates the call's ByteDance options and
// returns the members to pass through to the request body.
func readImageOptions(options model.ProviderOptions) (ImageOptions, map[string]json.RawMessage, error) {
	imageOptions, _, err := model.ProviderValue[ImageOptions](options, Namespace)
	if err != nil {
		return ImageOptions{}, nil, err
	}

	if err := imageOptions.validate(); err != nil {
		return ImageOptions{}, nil, err
	}

	passthrough, _, err := model.ProviderValue[map[string]json.RawMessage](options, Namespace)
	if err != nil {
		return ImageOptions{}, nil, err
	}

	for _, key := range imageOptionKeys {
		delete(passthrough, key)
	}

	return imageOptions, passthrough, nil
}

func (o ImageOptions) validate() error {
	if o.OutputFormat != nil && *o.OutputFormat != OutputPNG && *o.OutputFormat != OutputJPEG {
		return fmt.Errorf("%w: bytedance outputFormat %q", model.ErrInvalidArgument, *o.OutputFormat)
	}

	if o.SequentialImageGeneration != nil &&
		*o.SequentialImageGeneration != SequentialAuto &&
		*o.SequentialImageGeneration != SequentialDisabled {
		return fmt.Errorf("%w: bytedance sequentialImageGeneration %q", model.ErrInvalidArgument, *o.SequentialImageGeneration)
	}

	if o.MaxImages != nil && *o.MaxImages <= 0 {
		return fmt.Errorf("%w: bytedance maxImages must be positive", model.ErrInvalidArgument)
	}

	if o.OptimizePromptMode != nil &&
		*o.OptimizePromptMode != OptimizePromptStandard &&
		*o.OptimizePromptMode != OptimizePromptFast {
		return fmt.Errorf("%w: bytedance optimizePromptMode %q", model.ErrInvalidArgument, *o.OptimizePromptMode)
	}

	return nil
}

// requestBody maps the call and provider options onto the request. Base64
// output is always requested, after the passthrough options.
func (m *ImageModel) requestBody(
	options model.ImageOptions,
	imageOptions ImageOptions,
	passthrough map[string]json.RawMessage,
) (map[string]json.RawMessage, error) {
	image, err := inputImages(options.Files)
	if err != nil {
		return nil, err
	}

	body := imageRequest{
		Model:                     m.modelID,
		Prompt:                    options.Prompt,
		Image:                     image,
		Size:                      options.Size,
		Watermark:                 imageOptions.Watermark,
		OutputFormat:              imageOptions.OutputFormat,
		SequentialImageGeneration: imageOptions.SequentialImageGeneration,
	}

	if imageOptions.Size != nil {
		body.Size = *imageOptions.Size
	}

	if imageOptions.MaxImages != nil {
		body.SequentialImageGenerationOptions = &sequentialOptions{
			MaxImages: *imageOptions.MaxImages,
		}
	}

	if imageOptions.OptimizePromptMode != nil {
		body.OptimizePromptOptions = &optimizePromptOptions{
			Mode: *imageOptions.OptimizePromptMode,
		}
	}

	members, err := jsonobject.Members(body)
	if err != nil {
		return nil, fmt.Errorf("encode request body: %w", err)
	}

	maps.Copy(members, passthrough)
	members["response_format"] = json.RawMessage(`"b64_json"`)

	return members, nil
}

// inputImages encodes one input file as a string and several as an array.
func inputImages(files []model.ImageFile) (json.RawMessage, error) {
	if len(files) == 0 {
		return nil, nil
	}

	uris := make([]string, 0, len(files))
	for _, file := range files {
		uri, err := dataurl.ImageFileURL(file)
		if err != nil {
			return nil, err
		}

		uris = append(uris, uri)
	}

	var encoded []byte
	var err error
	if len(uris) == 1 {
		encoded, err = json.Marshal(uris[0])
	} else {
		encoded, err = json.Marshal(uris)
	}

	if err != nil {
		return nil, fmt.Errorf("encode input images: %w", err)
	}

	return encoded, nil
}

func decodeImages(data []imageData) ([][]byte, error) {
	images := make([][]byte, 0, len(data))

	for _, item := range data {
		image, err := base64.StdEncoding.DecodeString(item.B64JSON)
		if err != nil {
			return nil, fmt.Errorf("%w: decode base64 image: %w", model.ErrInvalidResponse, err)
		}

		images = append(images, image)
	}

	return images, nil
}

// imageUsage converts reported usage. ModelArk does not report input tokens.
func imageUsage(usage *imageTokens) *model.ImageUsage {
	if usage == nil {
		return nil
	}

	return &model.ImageUsage{
		OutputTokens: usage.OutputTokens,
		TotalTokens:  usage.TotalTokens,
	}
}

var _ model.ImageModel = (*ImageModel)(nil)
