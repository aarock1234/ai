package luma

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/url"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/download"
	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/internal/poll"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// generationsPath is the generations endpoint. New image generations are
	// posted to generationsPath + "/image"; status checks append the ID.
	generationsPath = "/dream-machine/v1/generations"

	defaultPollInterval    = 500 * time.Millisecond
	defaultMaxPollAttempts = 120

	// maxReferenceImages is the image and character reference limit.
	maxReferenceImages = 4
)

// defaultWeights are the reference weights used when ImageReference.Weight
// is nil.
var defaultWeights = map[ReferenceType]float64{
	ReferenceImage:       0.85,
	ReferenceStyle:       0.8,
	ReferenceCharacter:   1,
	ReferenceModifyImage: 1,
}

var (
	// ErrImageGenerationFailed reports a generation that failed. The error
	// names the reported reason.
	ErrImageGenerationFailed = errors.New("luma image generation failed")

	// ErrImageGenerationTimeout reports a generation that did not finish
	// within the polling attempts.
	ErrImageGenerationTimeout = errors.New("luma image generation timed out")
)

// ImageModel is a Luma image model. Input files must be URLs; Luma does not
// support masks.
type ImageModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "luma.image" by default.
func (m *ImageModel) Provider() string {
	return m.provider.name + ".image"
}

// ModelID returns the model identifier.
func (m *ImageModel) ModelID() string {
	return m.modelID
}

// MaxImagesPerCall returns 1: each generation produces one image.
func (m *ImageModel) MaxImagesPerCall() int {
	return 1
}

type generationRequest struct {
	Prompt      string                       `json:"prompt,omitempty"`
	AspectRatio string                       `json:"aspect_ratio,omitempty"`
	Model       string                       `json:"model"`
	Image       []weightedImage              `json:"image,omitempty"`
	Style       []weightedImage              `json:"style,omitempty"`
	Character   map[string]characterIdentity `json:"character,omitempty"`
	ModifyImage *weightedImage               `json:"modify_image,omitempty"`
}

type weightedImage struct {
	URL    string  `json:"url"`
	Weight float64 `json:"weight"`
}

type characterIdentity struct {
	Images []string `json:"images"`
}

// generationState is the state of a queued generation.
type generationState string

const (
	stateQueued    generationState = "queued"
	stateDreaming  generationState = "dreaming"
	stateCompleted generationState = "completed"
	stateFailed    generationState = "failed"
)

type generationResponse struct {
	ID            string            `json:"id"`
	State         generationState   `json:"state"`
	FailureReason *string           `json:"failure_reason"`
	Assets        *generationAssets `json:"assets"`
}

type generationAssets struct {
	Image string `json:"image"`
}

// pollSettings controls how a generation is awaited.
type pollSettings struct {
	interval    time.Duration
	maxAttempts int
}

// Generate queues a generation, waits for it and downloads the image.
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

	request, err := m.provider.apiRequest(m.provider.baseURL+generationsPath+"/image", options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, err
	}

	var generation generationResponse
	if err := apicall.DecodeJSON(response.Body, &generation); err != nil {
		return nil, err
	}

	imageURL, err := m.pollForImageURL(ctx, generation.ID, options, pollSettingsOf(imageOptions))
	if err != nil {
		return nil, err
	}

	image, err := download.Fetch(ctx, download.Request{
		URL:           imageURL,
		TrustedOrigin: m.provider.baseURL,
		Client:        m.provider.httpClient,
	})
	if err != nil {
		return nil, err
	}

	return &model.ImageResult{
		Images: [][]byte{
			image.Data,
		},
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

	if options.Seed != nil {
		warnings = append(warnings, model.UnsupportedWarning(
			"seed",
			"This model does not support the `seed` option.",
		))
	}

	if options.Size != "" {
		warnings = append(warnings, model.UnsupportedWarning(
			"size",
			"This model does not support the `size` option. Use `aspectRatio` instead.",
		))
	}

	return warnings
}

// requestBody combines the standard fields, the image references and the
// passthrough provider options, which take precedence.
func (m *ImageModel) requestBody(
	options model.ImageOptions,
	imageOptions ImageOptions,
	passthrough map[string]json.RawMessage,
) (map[string]json.RawMessage, error) {
	body := generationRequest{
		Prompt:      options.Prompt,
		AspectRatio: options.AspectRatio,
		Model:       m.modelID,
	}

	if err := addReferences(&body, options, imageOptions); err != nil {
		return nil, err
	}

	members, err := jsonobject.Members(body)
	if err != nil {
		return nil, fmt.Errorf("encode request body: %w", err)
	}

	maps.Copy(members, passthrough)

	return members, nil
}

// addReferences adds the input files to the request as the configured kind
// of reference.
func addReferences(body *generationRequest, options model.ImageOptions, imageOptions ImageOptions) error {
	if options.Mask != nil {
		return fmt.Errorf(
			"%w: Luma AI does not support mask-based image editing. Use the prompt to describe the changes you want to make, along with input image URLs",
			model.ErrUnsupported,
		)
	}

	if len(options.Files) == 0 {
		return nil
	}

	urls, err := referenceURLs(options.Files)
	if err != nil {
		return err
	}

	referenceType := ReferenceImage
	if imageOptions.ReferenceType != nil {
		referenceType = *imageOptions.ReferenceType
	}

	switch referenceType {
	case ReferenceStyle:
		body.Style = weightedImages(urls, imageOptions.Images, referenceType)

	case ReferenceCharacter:
		character, err := characterIdentities(urls, imageOptions.Images)
		if err != nil {
			return err
		}

		body.Character = character

	case ReferenceModifyImage:
		if len(urls) > 1 {
			return fmt.Errorf(
				"%w: Luma AI modify_image only supports a single input image, got %d",
				model.ErrInvalidArgument,
				len(urls),
			)
		}

		body.ModifyImage = &weightedImages(urls, imageOptions.Images, referenceType)[0]

	default:
		if len(urls) > maxReferenceImages {
			return fmt.Errorf(
				"%w: Luma AI image supports up to %d reference images, got %d",
				model.ErrInvalidArgument,
				maxReferenceImages,
				len(urls),
			)
		}

		body.Image = weightedImages(urls, imageOptions.Images, referenceType)
	}

	return nil
}

// referenceURLs returns the URLs of the input files. Luma downloads the
// images itself, so inline image data is not supported.
func referenceURLs(files []model.ImageFile) ([]string, error) {
	urls := make([]string, 0, len(files))

	for _, file := range files {
		image, ok := file.(model.ImageURL)
		if !ok {
			return nil, fmt.Errorf(
				"%w: Luma AI only supports URL-based images. Provide publicly accessible image URLs; inline image data is not supported",
				model.ErrUnsupported,
			)
		}

		urls = append(urls, image.URL)
	}

	return urls, nil
}

func weightedImages(urls []string, references []ImageReference, referenceType ReferenceType) []weightedImage {
	images := make([]weightedImage, 0, len(urls))

	for index, url := range urls {
		weight := defaultWeights[referenceType]
		if index < len(references) && references[index].Weight != nil {
			weight = *references[index].Weight
		}

		images = append(images, weightedImage{
			URL:    url,
			Weight: weight,
		})
	}

	return images
}

// characterIdentities groups the character images by identity. Each
// identity accepts up to four images.
func characterIdentities(urls []string, references []ImageReference) (map[string]characterIdentity, error) {
	identities := make(map[string]characterIdentity)

	for index, url := range urls {
		identityID := "identity0"
		if index < len(references) && references[index].ID != nil {
			identityID = *references[index].ID
		}

		identity := identities[identityID]
		identity.Images = append(identity.Images, url)
		identities[identityID] = identity
	}

	for identityID, identity := range identities {
		if len(identity.Images) > maxReferenceImages {
			return nil, fmt.Errorf(
				"%w: Luma AI character supports up to %d images per identity, identity %q has %d",
				model.ErrInvalidArgument,
				maxReferenceImages,
				identityID,
				len(identity.Images),
			)
		}
	}

	return identities, nil
}

func pollSettingsOf(options ImageOptions) pollSettings {
	settings := pollSettings{
		interval:    defaultPollInterval,
		maxAttempts: defaultMaxPollAttempts,
	}

	if options.PollIntervalMillis != nil {
		settings.interval = time.Duration(*options.PollIntervalMillis) * time.Millisecond
	}

	if options.MaxPollAttempts != nil {
		settings.maxAttempts = *options.MaxPollAttempts
	}

	return settings
}

// pollForImageURL checks the generation until it completes or fails and
// returns the URL of the generated image.
func (m *ImageModel) pollForImageURL(
	ctx context.Context,
	generationID string,
	options model.ImageOptions,
	settings pollSettings,
) (string, error) {
	request, err := m.provider.apiRequest(m.provider.baseURL+generationsPath+"/"+url.PathEscape(generationID), options.Headers)
	if err != nil {
		return "", err
	}

	for range settings.maxAttempts {
		response, err := apicall.Get(ctx, request)
		if err != nil {
			return "", err
		}

		var status generationResponse
		if err := apicall.DecodeJSON(response.Body, &status); err != nil {
			return "", err
		}

		imageURL, done, err := completedImageURL(status)
		if done || err != nil {
			return imageURL, err
		}

		if err := poll.Wait(ctx, settings.interval); err != nil {
			return "", err
		}
	}

	return "", fmt.Errorf("%w after %d attempts", ErrImageGenerationTimeout, settings.maxAttempts)
}

// completedImageURL returns the image URL of a completed generation. It
// reports false while the generation is still queued or running.
func completedImageURL(status generationResponse) (string, bool, error) {
	switch status.State {
	case stateQueued, stateDreaming:
		return "", false, nil

	case stateCompleted:
		if status.Assets == nil || status.Assets.Image == "" {
			return "", true, fmt.Errorf("%w: image generation completed but no image was found", model.ErrInvalidResponse)
		}

		return status.Assets.Image, true, nil

	case stateFailed:
		reason := "unknown reason"
		if status.FailureReason != nil {
			reason = *status.FailureReason
		}

		return "", true, fmt.Errorf("%w: %s", ErrImageGenerationFailed, reason)

	default:
		return "", true, fmt.Errorf("%w: unknown generation state %q", model.ErrInvalidResponse, status.State)
	}
}

var _ model.ImageModel = (*ImageModel)(nil)
