package ai

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"math/big"
	"net/http"
	"regexp"
	"strings"

	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/internal/merge"
	"github.com/aarock1234/ai/pkg/model"
)

// gatewayProvider is the provider namespace whose cost metadata is summed
// across calls instead of collected per image.
const gatewayProvider = "gateway"

// gatewayCostKeys are the gateway metadata keys that hold decimal costs.
var gatewayCostKeys = []string{
	"cost",
	"gatewayCost",
	"inferenceCost",
	"inputInferenceCost",
	"marketCost",
	"outputInferenceCost",
	"surchargeCost",
}

// errNoImageInResponse marks an image call that returned no images, which is
// retried like a transient failure.
var errNoImageInResponse = errors.New("no image generated")

// GenerateImageOptions configures GenerateImage.
type GenerateImageOptions struct {
	Model model.ImageModel

	// Prompt describes the image to generate or the edit to make.
	Prompt string

	// Images are input images for editing or variation: inline bytes,
	// http(s) URLs that the provider downloads, or data URLs.
	Images []model.FileData

	// Mask optionally marks the region of the first image to edit.
	Mask model.FileData

	// N is the number of images to generate. Zero means one.
	N int

	// MaxImagesPerCall overrides the model's per-call limit. Zero uses the
	// model's limit, and one image per call when the model has none.
	MaxImagesPerCall int

	// Size has the form "{width}x{height}".
	Size string

	// AspectRatio has the form "{width}:{height}".
	AspectRatio string

	Seed *int

	// MaxRetries is the number of retries per model call, including retries
	// after responses without images. Nil selects 2; zero disables retries.
	MaxRetries *int

	Headers         http.Header
	ProviderOptions model.ProviderOptions

	// LogWarnings receives the warnings of every model call together. Nil
	// discards them; they are always returned on the result.
	LogWarnings LogWarningsFunc
}

// GenerateImageResult is the outcome of GenerateImage.
type GenerateImageResult struct {
	// Images are the generated images of every call, in call order.
	Images []GeneratedFile

	// Calls are the underlying model calls, including attempts that returned
	// no images.
	Calls []GenerateImageCall

	Warnings []model.Warning

	// Responses has one entry per model call.
	Responses []model.ResponseInfo

	// ProviderMetadata collects the per-image metadata of every call.
	ProviderMetadata model.ProviderMetadata

	// Usage sums the usage reported by every call.
	Usage model.ImageUsage
}

// GenerateImageCall is the outcome of one image model call.
type GenerateImageCall struct {
	Images           []GeneratedFile
	ProviderMetadata model.ProviderMetadata
	Response         model.ResponseInfo
	Warnings         []model.Warning

	// Usage is nil when the provider did not report it.
	Usage *model.ImageUsage
}

// GenerateImage generates images. Requests for more images than the model
// allows per call are split into concurrent calls.
func GenerateImage(ctx context.Context, options GenerateImageOptions) (*GenerateImageResult, error) {
	if options.Model == nil {
		return nil, &InvalidArgumentError{
			Parameter: "model",
			Message:   "model is required",
		}
	}

	if options.N < 0 {
		return nil, &InvalidArgumentError{
			Parameter: "n",
			Value:     options.N,
			Message:   "n must be >= 0",
		}
	}

	if options.MaxImagesPerCall < 0 {
		return nil, &InvalidArgumentError{
			Parameter: "maxImagesPerCall",
			Value:     options.MaxImagesPerCall,
			Message:   "maxImagesPerCall must be >= 0",
		}
	}

	retry, err := prepareRetries(options.MaxRetries)
	if err != nil {
		return nil, err
	}

	retry.isRetryable = func(err error) bool {
		return errors.Is(err, errNoImageInResponse)
	}

	files, err := imageModelFiles(options.Images)
	if err != nil {
		return nil, err
	}

	mask, err := imageModelMask(options.Mask)
	if err != nil {
		return nil, err
	}

	request := model.ImageOptions{
		Prompt:          options.Prompt,
		Size:            options.Size,
		AspectRatio:     options.AspectRatio,
		Seed:            options.Seed,
		Files:           files,
		Mask:            mask,
		Headers:         options.Headers,
		ProviderOptions: options.ProviderOptions,
	}

	counts := callCounts(max(options.N, 1), maxImagesPerCall(options))

	callGroups, err := mapConcurrently(ctx, counts, 0, func(ctx context.Context, count int) ([]*model.ImageResult, error) {
		callRequest := request
		callRequest.N = count

		return generateImagesWithRetry(ctx, options.Model, retry, callRequest)
	})
	if err != nil {
		return nil, err
	}

	result := collectImageResults(callGroups)

	options.LogWarnings.log(ctx, WarningsEvent{
		Warnings: result.Warnings,
		Provider: options.Model.Provider(),
		ModelID:  options.Model.ModelID(),
	})

	if len(result.Images) == 0 {
		return nil, &NoImageGeneratedError{
			Calls:     result.Calls,
			Responses: result.Responses,
		}
	}

	return result, nil
}

func maxImagesPerCall(options GenerateImageOptions) int {
	if options.MaxImagesPerCall > 0 {
		return options.MaxImagesPerCall
	}

	if limit := options.Model.MaxImagesPerCall(); limit > 0 {
		return limit
	}

	return 1
}

// callCounts splits n generated items, such as images or videos, into calls
// of at most perCall items. It does not overflow for very large limits.
func callCounts(n, perCall int) []int {
	calls := n / perCall
	if n%perCall != 0 {
		calls++
	}

	counts := make([]int, 0, calls)
	for remaining := n; remaining > 0; remaining -= perCall {
		counts = append(counts, min(remaining, perCall))
	}

	return counts
}

// generateImagesWithRetry calls the model until it returns images, keeping
// every response. When the retries end on a response without images, the
// empty responses are returned without an error.
func generateImagesWithRetry(
	ctx context.Context,
	imageModel model.ImageModel,
	retry retryPolicy,
	request model.ImageOptions,
) ([]*model.ImageResult, error) {
	var responses []*model.ImageResult

	_, err := retryCall(ctx, retry, func(ctx context.Context) (*model.ImageResult, error) {
		response, err := imageModel.Generate(ctx, request)
		if err != nil {
			return nil, err
		}

		if response == nil {
			return nil, fmt.Errorf("%w: image model returned no result", model.ErrInvalidResponse)
		}

		responses = append(responses, response)

		if len(response.Images) == 0 {
			return nil, errNoImageInResponse
		}

		return response, nil
	})
	if err != nil && !isNoImageFailure(err) {
		return nil, err
	}

	return responses, nil
}

// isNoImageFailure reports whether the final attempt failed only because it
// returned no images.
func isNoImageFailure(err error) bool {
	if retryErr, ok := errors.AsType[*RetryError](err); ok {
		return errors.Is(retryErr.LastError(), errNoImageInResponse)
	}

	return errors.Is(err, errNoImageInResponse)
}

func collectImageResults(callGroups [][]*model.ImageResult) *GenerateImageResult {
	result := &GenerateImageResult{
		ProviderMetadata: model.ProviderMetadata{},
	}

	for _, group := range callGroups {
		for _, response := range group {
			images := generatedImages(response)

			result.Images = append(result.Images, images...)
			result.Calls = append(result.Calls, GenerateImageCall{
				Images:           images,
				ProviderMetadata: response.ProviderMetadata,
				Response:         response.Response,
				Warnings:         response.Warnings,
				Usage:            response.Usage,
			})
			result.Warnings = append(result.Warnings, response.Warnings...)
			result.Responses = append(result.Responses, response.Response)
			result.Usage = addImageUsage(result.Usage, response.Usage)

			mergeImageProviderMetadata(result.ProviderMetadata, response.ProviderMetadata)
		}
	}

	return result
}

func generatedImages(response *model.ImageResult) []GeneratedFile {
	images := make([]GeneratedFile, len(response.Images))

	for index, data := range response.Images {
		mediaType, ok := mediatype.Detect(data, "image")
		if !ok {
			mediaType = "image/png"
		}

		images[index] = GeneratedFile{
			Data:             data,
			MediaType:        mediaType,
			ProviderMetadata: imageProviderMetadata(response.ProviderMetadata, index),
		}
	}

	return images
}

// imageMetadataEnvelope is the per-provider shape of image call metadata:
// one metadata object per generated image.
type imageMetadataEnvelope struct {
	Images []map[string]any `json:"images"`
}

// imageProviderMetadata extracts the metadata of one image from the
// "images" array of each provider's call metadata.
func imageProviderMetadata(metadata model.ProviderMetadata, index int) model.ProviderMetadata {
	var perImage model.ProviderMetadata

	for provider := range metadata {
		envelope, ok, err := model.ProviderValue[imageMetadataEnvelope](metadata, provider)
		if err != nil || !ok || index >= len(envelope.Images) || envelope.Images[index] == nil {
			continue
		}

		if perImage == nil {
			perImage = model.ProviderMetadata{}
		}

		perImage[provider] = envelope.Images[index]
	}

	return perImage
}

// mergeImageProviderMetadata adds one call's metadata to the combined
// metadata: per-image entries are concatenated, and gateway metadata is
// merged with its costs summed.
func mergeImageProviderMetadata(combined, metadata model.ProviderMetadata) {
	for provider := range metadata {
		if provider == gatewayProvider {
			combined[provider] = mergeGatewayMetadata(combined[provider], metadata[provider])

			continue
		}

		images := []map[string]any{}
		if existing, ok := combined[provider].(imageMetadataEnvelope); ok {
			images = existing.Images
		}

		envelope, _, err := model.ProviderValue[imageMetadataEnvelope](metadata, provider)
		if err == nil {
			images = append(images, envelope.Images...)
		}

		combined[provider] = imageMetadataEnvelope{
			Images: images,
		}
	}
}

func mergeGatewayMetadata(existing, next any) any {
	nextObject, ok := merge.Object(next)
	if !ok {
		return next
	}

	existingObject, hasExisting := merge.Object(existing)

	merged := make(map[string]any, len(existingObject)+len(nextObject))
	maps.Copy(merged, existingObject)
	maps.Copy(merged, nextObject)

	if hasExisting {
		for _, key := range gatewayCostKeys {
			if total, ok := addDecimalStrings(existingObject[key], nextObject[key]); ok {
				merged[key] = total
			}
		}
	}

	if images, ok := merged["images"].([]any); ok && len(images) == 0 {
		delete(merged, "images")
	}

	return merged
}

var decimalPattern = regexp.MustCompile(`^\d+(?:\.\d+)?$`)

// addDecimalStrings adds two non-negative decimal strings exactly. It reports
// false when either value is not such a string.
func addDecimalStrings(first, second any) (string, bool) {
	firstText, firstOK := first.(string)
	secondText, secondOK := second.(string)

	if !firstOK || !secondOK || !decimalPattern.MatchString(firstText) || !decimalPattern.MatchString(secondText) {
		return "", false
	}

	firstValue, _ := new(big.Rat).SetString(firstText)
	secondValue, _ := new(big.Rat).SetString(secondText)
	sum := new(big.Rat).Add(firstValue, secondValue)

	precision := max(fractionDigits(firstText), fractionDigits(secondText))
	text := sum.FloatString(precision)

	if strings.Contains(text, ".") {
		text = strings.TrimRight(strings.TrimRight(text, "0"), ".")
	}

	return text, true
}

func fractionDigits(decimal string) int {
	_, fraction, _ := strings.Cut(decimal, ".")

	return len(fraction)
}

func addImageUsage(total model.ImageUsage, usage *model.ImageUsage) model.ImageUsage {
	if usage == nil {
		return total
	}

	return model.ImageUsage{
		InputTokens:  addTokenCounts(total.InputTokens, usage.InputTokens),
		OutputTokens: addTokenCounts(total.OutputTokens, usage.OutputTokens),
		TotalTokens:  addTokenCounts(total.TotalTokens, usage.TotalTokens),
	}
}

// addTokenCounts adds optional counts; the sum is unknown only when both are.
func addTokenCounts(first, second *int) *int {
	if first == nil && second == nil {
		return nil
	}

	sum := 0
	if first != nil {
		sum += *first
	}

	if second != nil {
		sum += *second
	}

	return &sum
}

// imageModelFiles converts input images into image model files.
func imageModelFiles(images []model.FileData) ([]model.ImageFile, error) {
	if len(images) == 0 {
		return nil, nil
	}

	files := make([]model.ImageFile, len(images))
	for index, image := range images {
		file, err := imageModelFile(image)
		if err != nil {
			return nil, err
		}

		files[index] = file
	}

	return files, nil
}

func imageModelMask(mask model.FileData) (model.ImageFile, error) {
	if mask == nil {
		return nil, nil
	}

	return imageModelFile(mask)
}

// imageModelFile passes http(s) URLs to the provider and sends other images
// inline, with a media type from the data URL or the image signature.
func imageModelFile(image model.FileData) (model.ImageFile, error) {
	data, mediaType, err := normalizeFileData(image, "")
	if err != nil {
		return nil, err
	}

	switch typed := data.(type) {
	case model.FileURL:
		return model.ImageURL{
			URL: typed.URL.String(),
		}, nil

	case model.FileBytes:
		if mediaType == "" {
			mediaType = detectImageMediaType(typed.Data)
		}

		return model.ImageBytes{
			MediaType: mediaType,
			Data:      typed.Data,
		}, nil

	default:
		return nil, &InvalidArgumentError{
			Parameter: "images",
			Message:   "images must be inline bytes or URLs",
		}
	}
}

func detectImageMediaType(data []byte) string {
	if detected, ok := mediatype.Detect(data, "image"); ok {
		return detected
	}

	return "image/png"
}
