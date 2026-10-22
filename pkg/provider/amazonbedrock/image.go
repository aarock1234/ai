package amazonbedrock

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/optional"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/amazonbedrock/internal/bedrock"
)

var _ model.ImageModel = (*ImageModel)(nil)

const (
	// novaCanvasModelID accepts several images per call and image inputs.
	novaCanvasModelID = "amazon.nova-canvas-v1:0"

	novaCanvasMaxImagesPerCall = 5

	// moderatedStatus is the status of a request that moderation blocked.
	moderatedStatus = "Request Moderated"
)

// ImageModel is an Amazon Bedrock image model, such as Nova Canvas.
type ImageModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "amazon-bedrock" by default.
func (m *ImageModel) Provider() string {
	return m.provider.name
}

// ModelID returns the model ID.
func (m *ImageModel) ModelID() string {
	return m.modelID
}

// MaxImagesPerCall returns 5 for Nova Canvas and 1 for other models.
func (m *ImageModel) MaxImagesPerCall() int {
	if m.modelID == novaCanvasModelID {
		return novaCanvasMaxImagesPerCall
	}

	return 1
}

type imageRequest struct {
	TaskType                ImageTaskType            `json:"taskType"`
	TextToImageParams       *textToImageParams       `json:"textToImageParams,omitempty"`
	InPaintingParams        *imageEditParams         `json:"inPaintingParams,omitempty"`
	OutPaintingParams       *imageEditParams         `json:"outPaintingParams,omitempty"`
	BackgroundRemovalParams *backgroundRemovalParams `json:"backgroundRemovalParams,omitempty"`
	ImageVariationParams    *imageVariationParams    `json:"imageVariationParams,omitempty"`
	ImageGenerationConfig   *imageGenerationConfig   `json:"imageGenerationConfig,omitempty"`
}

type textToImageParams struct {
	Text         string      `json:"text"`
	NegativeText string      `json:"negativeText,omitempty"`
	Style        *ImageStyle `json:"style,omitempty"`
}

// imageEditParams are the inpainting and outpainting parameters. Images
// are base64.
type imageEditParams struct {
	Image           string           `json:"image"`
	Text            string           `json:"text,omitempty"`
	NegativeText    string           `json:"negativeText,omitempty"`
	OutPaintingMode *OutPaintingMode `json:"outPaintingMode,omitempty"`
	MaskImage       string           `json:"maskImage,omitempty"`
	MaskPrompt      *string          `json:"maskPrompt,omitempty"`
}

type backgroundRemovalParams struct {
	Image string `json:"image"`
}

type imageVariationParams struct {
	Images             []string `json:"images"`
	Text               string   `json:"text,omitempty"`
	NegativeText       string   `json:"negativeText,omitempty"`
	SimilarityStrength *float64 `json:"similarityStrength,omitempty"`
}

type imageGenerationConfig struct {
	Width          int           `json:"width,omitempty"`
	Height         int           `json:"height,omitempty"`
	Seed           int           `json:"seed,omitempty"`
	NumberOfImages int           `json:"numberOfImages,omitempty"`
	Quality        *ImageQuality `json:"quality,omitempty"`
	CfgScale       float64       `json:"cfgScale,omitempty"`
}

// imageResponse holds the generated images or the moderation outcome.
type imageResponse struct {
	Images  []string                   `json:"images"`
	Status  string                     `json:"status"`
	Details map[string]json.RawMessage `json:"details"`
}

// Generate generates images, or edits the input images when the call has
// files.
func (m *ImageModel) Generate(ctx context.Context, options model.ImageOptions) (*model.ImageResult, error) {
	imageOptions, err := readImageOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	body, err := imageRequestOf(options, imageOptions)
	if err != nil {
		return nil, err
	}

	var warnings []model.Warning
	if options.AspectRatio != "" {
		warnings = append(warnings, model.UnsupportedWarning(
			"aspectRatio",
			"This model does not support aspect ratio. Use `size` instead.",
		))
	}

	request, err := m.provider.apiRequest(ctx, bedrock.Runtime, bedrock.ModelPath(m.modelID, "invoke"), options.Headers)
	if err != nil {
		return nil, err
	}

	timestamp := time.Now()

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, err
	}

	var decoded imageResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	images, err := decoded.decodedImages()
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
			Body:    response.Body,
		},
	}, nil
}

// imageRequestOf builds a text-to-image request, or an editing request
// when the call has input images.
func imageRequestOf(options model.ImageOptions, imageOptions ImageOptions) (imageRequest, error) {
	config := generationConfigOf(options, imageOptions)
	negativeText := optional.Or(imageOptions.NegativeText, "")

	if len(options.Files) == 0 {
		return imageRequest{
			TaskType: ImageTaskTextImage,
			TextToImageParams: &textToImageParams{
				Text:         options.Prompt,
				NegativeText: negativeText,
				Style:        imageOptions.Style,
			},
			ImageGenerationConfig: config,
		}, nil
	}

	source, err := imageBase64(options.Files[0])
	if err != nil {
		return imageRequest{}, err
	}

	hasMask := options.Mask != nil
	hasMaskPrompt := imageOptions.MaskPrompt != nil

	taskType := ImageTaskImageVariation
	if hasMask || hasMaskPrompt {
		taskType = ImageTaskInpainting
	}

	taskType = optional.Or(imageOptions.TaskType, taskType)

	switch taskType {
	case ImageTaskInpainting, ImageTaskOutpainting:
		params := &imageEditParams{
			Image:        source,
			Text:         options.Prompt,
			NegativeText: negativeText,
		}

		if taskType == ImageTaskOutpainting {
			params.OutPaintingMode = imageOptions.OutPaintingMode
		}

		if err := setMask(params, options.Mask, imageOptions.MaskPrompt); err != nil {
			return imageRequest{}, err
		}

		request := imageRequest{
			TaskType:              taskType,
			ImageGenerationConfig: config,
		}

		if taskType == ImageTaskInpainting {
			request.InPaintingParams = params
		} else {
			request.OutPaintingParams = params
		}

		return request, nil

	case ImageTaskBackgroundRemoval:
		return imageRequest{
			TaskType: ImageTaskBackgroundRemoval,
			BackgroundRemovalParams: &backgroundRemovalParams{
				Image: source,
			},
		}, nil

	case ImageTaskImageVariation:
		images := make([]string, 0, len(options.Files))
		for _, file := range options.Files {
			encoded, err := imageBase64(file)
			if err != nil {
				return imageRequest{}, err
			}

			images = append(images, encoded)
		}

		return imageRequest{
			TaskType: ImageTaskImageVariation,
			ImageVariationParams: &imageVariationParams{
				Images:             images,
				Text:               options.Prompt,
				NegativeText:       negativeText,
				SimilarityStrength: imageOptions.SimilarityStrength,
			},
			ImageGenerationConfig: config,
		}, nil

	default:
		return imageRequest{}, fmt.Errorf("%w: unsupported task type: %s", model.ErrUnsupported, taskType)
	}
}

// setMask sets the mask image, or the mask prompt when there is no mask.
func setMask(params *imageEditParams, mask model.ImageFile, maskPrompt *string) error {
	if mask == nil {
		params.MaskPrompt = maskPrompt

		return nil
	}

	encoded, err := imageBase64(mask)
	if err != nil {
		return err
	}

	params.MaskImage = encoded

	return nil
}

// generationConfigOf returns the image generation settings that are set.
func generationConfigOf(options model.ImageOptions, imageOptions ImageOptions) *imageGenerationConfig {
	config := &imageGenerationConfig{
		NumberOfImages: options.N,
		Seed:           optional.Or(options.Seed, 0),
		Quality:        imageOptions.Quality,
		CfgScale:       optional.Or(imageOptions.CfgScale, 0),
	}

	if width, height, found := strings.Cut(options.Size, "x"); found {
		config.Width, _ = strconv.Atoi(width)
		config.Height, _ = strconv.Atoi(height)
	}

	return config
}

// imageBase64 returns the base64 data of an input image. Bedrock does not
// download images.
func imageBase64(file model.ImageFile) (string, error) {
	switch file := file.(type) {
	case model.ImageBytes:
		return base64.StdEncoding.EncodeToString(file.Data), nil

	case model.ImageURL:
		return "", fmt.Errorf(
			"%w: URL-based images are not supported for Amazon Bedrock image editing; provide the image data directly",
			model.ErrUnsupported,
		)

	default:
		return "", fmt.Errorf("%w: image file %T", model.ErrUnsupported, file)
	}
}

// decodedImages returns the images, or an error for moderated requests and
// responses without images.
func (r imageResponse) decodedImages() ([][]byte, error) {
	if r.Status == moderatedStatus {
		reasons := []string{
			"Unknown",
		}

		var listed []string
		if err := json.Unmarshal(r.Details["Moderation Reasons"], &listed); err == nil && listed != nil {
			reasons = listed
		}

		return nil, fmt.Errorf("%w: Amazon Bedrock request was moderated: %s", model.ErrNoContentGenerated, strings.Join(reasons, ", "))
	}

	if len(r.Images) == 0 {
		message := "Amazon Bedrock returned no images"
		if r.Status != "" {
			message += ". Status: " + r.Status
		}

		return nil, fmt.Errorf("%w: %s", model.ErrNoContentGenerated, message)
	}

	images := make([][]byte, 0, len(r.Images))
	for _, encoded := range r.Images {
		image, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil {
			return nil, fmt.Errorf("%w: image is not base64: %w", model.ErrInvalidResponse, err)
		}

		images = append(images, image)
	}

	return images, nil
}
