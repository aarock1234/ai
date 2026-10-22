package googlevertex

import (
	"context"
	"fmt"
	"net/url"
	"strings"
	"time"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/google"
)

var _ model.ImageModel = (*ImageModel)(nil)

// ImageOptions are the options of an image call, read from
// ProviderOptions["googleVertex"] or ["vertex"]. They are the language model
// options; ResponseModalities is ignored because the output is always
// images.
type ImageOptions = google.ChatOptions

// ImageMetadata is the metadata of an image response, stored under
// ProviderMetadata["googleVertex"] and ["vertex"].
type ImageMetadata struct {
	// Images has one empty entry per generated image.
	Images []struct{} `json:"images"`
}

// ImageModel generates and edits images with Gemini image models, such as
// gemini-2.5-flash-image, through the generateContent API. It is safe for
// concurrent use.
type ImageModel struct {
	modelID  string
	provider string
	language *google.ChatModel
}

// Provider returns "google.vertex.image".
func (m *ImageModel) Provider() string {
	return m.provider
}

// ModelID returns the model ID.
func (m *ImageModel) ModelID() string {
	return m.modelID
}

// MaxImagesPerCall returns 1.
func (m *ImageModel) MaxImagesPerCall() int {
	return 1
}

// Generate generates images from the prompt and optional input images.
// Gemini image models do not support masks or explicit sizes.
func (m *ImageModel) Generate(ctx context.Context, options model.ImageOptions) (*model.ImageResult, error) {
	if !strings.HasPrefix(m.modelID, "gemini-") {
		return nil, fmt.Errorf("%w: Google image models other than Gemini are no longer supported; use a model ID that starts with gemini-", model.ErrUnsupported)
	}

	if options.Mask != nil {
		return nil, fmt.Errorf("%w: Gemini image models do not support mask-based image editing", model.ErrUnsupported)
	}

	var warnings []model.Warning
	if options.Size != "" {
		warnings = append(warnings, model.UnsupportedWarning("size", "This model does not support the `size` option. Use `aspectRatio` instead."))
	}

	callOptions, err := languageModelCall(options)
	if err != nil {
		return nil, err
	}

	result, err := m.language.Generate(ctx, callOptions)
	if err != nil {
		return nil, err
	}

	images := generatedImages(result.Content)

	return &model.ImageResult{
		Images: images,
		Usage:  imageUsage(result.Usage),
		ProviderMetadata: wrapMetadata(ImageMetadata{
			Images: make([]struct{}, len(images)),
		}),
		Warnings: warnings,
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: time.Now(),
				ModelID:   m.modelID,
			},
			Headers: result.Response.Headers,
		},
	}, nil
}

// languageModelCall builds the generateContent call with image output.
func languageModelCall(options model.ImageOptions) (model.CallOptions, error) {
	imageOptions, err := readImageOptions(options.ProviderOptions)
	if err != nil {
		return model.CallOptions{}, err
	}

	imageOptions.ResponseModalities = []google.ResponseModality{
		google.ResponseModalityImage,
	}

	if options.AspectRatio != "" {
		imageConfig := google.ImageConfig{}
		if imageOptions.ImageConfig != nil {
			imageConfig = *imageOptions.ImageConfig
		}

		imageConfig.AspectRatio = google.AspectRatio(options.AspectRatio)
		imageOptions.ImageConfig = &imageConfig
	}

	prompt, err := imagePrompt(options)
	if err != nil {
		return model.CallOptions{}, err
	}

	return model.CallOptions{
		Prompt: prompt,
		Seed:   options.Seed,
		ProviderOptions: model.ProviderOptions{
			Namespace:       imageOptions,
			LegacyNamespace: imageOptions,
		},
		Headers: options.Headers,
	}, nil
}

// readImageOptions reads the Vertex namespaces only; image calls do not
// fall back to the "google" namespace.
func readImageOptions(options model.ProviderOptions) (ImageOptions, error) {
	for _, namespace := range []string{Namespace, LegacyNamespace} {
		value, ok, err := model.ProviderValue[ImageOptions](options, namespace)
		if err != nil || ok {
			return value, err
		}
	}

	return ImageOptions{}, nil
}

func imagePrompt(options model.ImageOptions) ([]model.Message, error) {
	var parts []model.UserPart

	if options.Prompt != "" {
		parts = append(parts, model.TextPart{
			Text: options.Prompt,
		})
	}

	for _, file := range options.Files {
		filePart, err := imageFilePart(file)
		if err != nil {
			return nil, err
		}

		parts = append(parts, filePart)
	}

	return []model.Message{
		model.UserMessage{
			Content: parts,
		},
	}, nil
}

func imageFilePart(file model.ImageFile) (model.FilePart, error) {
	switch file := file.(type) {
	case model.ImageBytes:
		return model.FilePart{
			MediaType: file.MediaType,
			Data: model.FileBytes{
				Data: file.Data,
			},
		}, nil

	case model.ImageURL:
		parsed, err := url.Parse(file.URL)
		if err != nil {
			return model.FilePart{}, fmt.Errorf("%w: input image URL: %w", model.ErrInvalidArgument, err)
		}

		return model.FilePart{
			MediaType: "image/*",
			Data: model.FileURL{
				URL: parsed,
			},
		}, nil

	default:
		return model.FilePart{}, fmt.Errorf("%w: unsupported input image %T", model.ErrInvalidArgument, file)
	}
}

func generatedImages(contents []model.Content) [][]byte {
	var images [][]byte

	for _, item := range contents {
		file, ok := item.(model.File)
		if !ok || !strings.HasPrefix(file.MediaType, "image/") {
			continue
		}

		data, ok := file.Data.(model.FileBytes)
		if ok {
			images = append(images, data.Data)
		}
	}

	return images
}

// imageUsage reports the language model's token totals; missing counts add
// zero to the total.
func imageUsage(usage model.Usage) *model.ImageUsage {
	total := 0
	if usage.InputTokens.Total != nil {
		total += *usage.InputTokens.Total
	}

	if usage.OutputTokens.Total != nil {
		total += *usage.OutputTokens.Total
	}

	return &model.ImageUsage{
		InputTokens:  usage.InputTokens.Total,
		OutputTokens: usage.OutputTokens.Total,
		TotalTokens:  &total,
	}
}
