package google

import (
	"context"
	"fmt"
	"net/url"
	"strings"
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

var _ model.ImageModel = (*ImageModel)(nil)

// ImageOptions are the Google provider options of an image call, read
// from ProviderOptions["google"]. They accept the language model options;
// response modalities are always images.
type ImageOptions struct {
	ChatOptions

	// GoogleSearch enables Google Search grounding for image generation with
	// these tool arguments. Use a zero value for the defaults.
	GoogleSearch *SearchArgs `json:"googleSearch,omitempty"`
}

// ImageMetadata is the Google metadata of an image response, stored under
// ProviderMetadata["google"].
type ImageMetadata struct {
	ChatMetadata

	FinishReason *string `json:"finishReason"`

	// Images has one empty entry per generated image.
	Images []struct{} `json:"images"`
}

// ImageModel generates and edits images with Gemini image models, such as
// gemini-2.5-flash-image, through the generateContent API. It is safe for
// concurrent use.
type ImageModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider name, "google.generative-ai" by default.
func (m *ImageModel) Provider() string {
	return m.provider.name
}

// ModelID returns the model ID.
func (m *ImageModel) ModelID() string {
	return m.modelID
}

// MaxImagesPerCall returns 1. Set ai.GenerateImageOptions.MaxImagesPerCall to
// request more images per call.
func (m *ImageModel) MaxImagesPerCall() int {
	return 1
}

// Generate generates images from the prompt and optional input images. Gemini
// image models do not support masks or explicit sizes.
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

	callOptions, err := m.languageModelCall(options)
	if err != nil {
		return nil, err
	}

	languageModel := m.provider.Chat(m.modelID)

	result, err := languageModel.Generate(ctx, callOptions)
	if err != nil {
		return nil, err
	}

	images := generatedImages(result.Content)

	return &model.ImageResult{
		Images:           images,
		Usage:            imageUsage(result.Usage),
		ProviderMetadata: imageMetadata(result, len(images)),
		Warnings:         warnings,
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
func (m *ImageModel) languageModelCall(options model.ImageOptions) (model.CallOptions, error) {
	imageOptions, _, err := model.ProviderValue[ImageOptions](options.ProviderOptions, Namespace)
	if err != nil {
		return model.CallOptions{}, err
	}

	prompt, err := imagePrompt(options)
	if err != nil {
		return model.CallOptions{}, err
	}

	googleOptions := imageOptions.ChatOptions
	googleOptions.ResponseModalities = []ResponseModality{
		ResponseModalityImage,
	}

	if options.AspectRatio != "" {
		imageConfig := ImageConfig{}
		if googleOptions.ImageConfig != nil {
			imageConfig = *googleOptions.ImageConfig
		}

		imageConfig.AspectRatio = AspectRatio(options.AspectRatio)
		googleOptions.ImageConfig = &imageConfig
	}

	callOptions := model.CallOptions{
		Prompt: prompt,
		Seed:   options.Seed,
		ProviderOptions: model.ProviderOptions{
			Namespace: googleOptions,
		},
		Headers: options.Headers,
	}

	if imageOptions.GoogleSearch != nil {
		callOptions.Tools = []model.Tool{
			Search(*imageOptions.GoogleSearch),
		}
	}

	return callOptions, nil
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

func imageUsage(usage model.Usage) *model.ImageUsage {
	total := valueOrZero(usage.InputTokens.Total) + valueOrZero(usage.OutputTokens.Total)

	return &model.ImageUsage{
		InputTokens:  usage.InputTokens.Total,
		OutputTokens: usage.OutputTokens.Total,
		TotalTokens:  new(total),
	}
}

func imageMetadata(result *model.GenerateResult, imageCount int) model.ProviderMetadata {
	languageMetadata, _ := result.ProviderMetadata[Namespace].(ChatMetadata)

	var finishReason *string
	if result.FinishReason.Raw != "" {
		finishReason = new(result.FinishReason.Raw)
	}

	return model.ProviderMetadata{
		Namespace: ImageMetadata{
			ChatMetadata: languageMetadata,
			FinishReason: finishReason,
			Images:       make([]struct{}, imageCount),
		},
	}
}
