package blackforestlabs

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/download"
	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/internal/origin"
	"github.com/aarock1234/ai/pkg/model"
)

// maxInputImages is the input image limit of every model.
const maxInputImages = 10

// legacyModelPattern matches FLUX.1 and FLUX.2 model IDs.
var legacyModelPattern = regexp.MustCompile(`^flux-[12](?:[.-]|$)`)

// flux3AspectRatios are the aspect ratios of FLUX 3 and later models.
var flux3AspectRatios = []string{
	"21:9",
	"2:1",
	"16:9",
	"3:2",
	"7:5",
	"4:3",
	"5:4",
	"1:1",
	"4:5",
	"3:4",
	"5:7",
	"2:3",
	"9:16",
	"1:2",
	"9:21",
}

// latestSupportedOptions and flux3SupportedOptions are the options that FLUX
// 3 and later models accept; others are reported as unsupported.
var (
	latestSupportedOptions = []string{
		"height",
		"outputFormat",
		"pollIntervalMillis",
		"pollTimeoutMillis",
		"resolution",
		"safetyTolerance",
		"steps",
		"width",
	}

	flux3SupportedOptions = []string{
		"grounding",
		"pollIntervalMillis",
		"pollTimeoutMillis",
		"resolution",
		"safetyTolerance",
		"version",
	}
)

// modelAPI is the request format a model accepts.
type modelAPI int

const (
	// apiLegacy is the format of FLUX.1, FLUX.2 and Kontext models.
	apiLegacy modelAPI = iota

	// apiFlux3 is the format of flux-3-image.
	apiFlux3

	// apiLatest is the format of later models, including unknown IDs.
	apiLatest
)

// ImageMetadata is the Black Forest Labs metadata of an image response,
// stored under ProviderMetadata["blackForestLabs"].
type ImageMetadata struct {
	// Images has one entry for the generated image.
	Images []ImageInfo `json:"images"`
}

// ImageInfo describes one generated image. Nil fields were not reported.
type ImageInfo struct {
	Seed             *float64 `json:"seed,omitempty"`
	StartTime        *float64 `json:"start_time,omitempty"`
	EndTime          *float64 `json:"end_time,omitempty"`
	Duration         *float64 `json:"duration,omitempty"`
	Cost             *float64 `json:"cost,omitempty"`
	InputMegapixels  *float64 `json:"inputMegapixels,omitempty"`
	OutputMegapixels *float64 `json:"outputMegapixels,omitempty"`
}

// ImageModel is a Black Forest Labs image model.
type ImageModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "black-forest-labs.image" by
// default.
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

// api returns the request format of the model. Unknown IDs use the latest
// API.
func (m *ImageModel) api() modelAPI {
	switch {
	case m.modelID == "flux-pro",
		m.modelID == "flux-dev",
		strings.HasPrefix(m.modelID, "flux-pro-"),
		strings.HasPrefix(m.modelID, "flux-kontext-"),
		legacyModelPattern.MatchString(m.modelID):
		return apiLegacy

	case m.modelID == "flux-3-image":
		return apiFlux3

	default:
		return apiLatest
	}
}

// imageSize is a parsed "WIDTHxHEIGHT" size.
type imageSize struct {
	width  int
	height int
}

// submitResponse acknowledges a queued generation.
type submitResponse struct {
	ID               string   `json:"id"`
	PollingURL       string   `json:"polling_url"`
	Cost             *float64 `json:"cost"`
	InputMegapixels  *float64 `json:"input_mp"`
	OutputMegapixels *float64 `json:"output_mp"`
}

// Generate submits a generation, polls until it is ready and downloads the
// image.
func (m *ImageModel) Generate(ctx context.Context, options model.ImageOptions) (*model.ImageResult, error) {
	timestamp := time.Now()

	body, imageOptions, warnings, err := m.requestBody(options)
	if err != nil {
		return nil, err
	}

	headers, err := m.provider.authHeaders(options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, m.provider.apiRequest(m.provider.baseURL+"/"+m.modelID, headers), body)
	if err != nil {
		return nil, classifyError(err)
	}

	var submitted submitResponse
	if err := apicall.DecodeJSON(response.Body, &submitted); err != nil {
		return nil, err
	}

	result, err := m.pollForResult(ctx, submitted, headers, m.pollSettings(imageOptions))
	if err != nil {
		return nil, err
	}

	image, err := m.downloadImage(ctx, result.Sample, headers)
	if err != nil {
		return nil, err
	}

	return &model.ImageResult{
		Images: [][]byte{
			image,
		},
		Warnings: warnings,
		ProviderMetadata: model.ProviderMetadata{
			Namespace: ImageMetadata{
				Images: []ImageInfo{
					{
						Seed:             result.Seed,
						StartTime:        result.StartTime,
						EndTime:          result.EndTime,
						Duration:         result.Duration,
						Cost:             submitted.Cost,
						InputMegapixels:  submitted.InputMegapixels,
						OutputMegapixels: submitted.OutputMegapixels,
					},
				},
			},
		},
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: timestamp,
				ModelID:   m.modelID,
			},
			Headers: response.Headers,
		},
	}, nil
}

// requestBody builds the encoded request of the model's API and its
// warnings.
func (m *ImageModel) requestBody(options model.ImageOptions) (json.RawMessage, ImageOptions, []model.Warning, error) {
	api := m.api()

	size, err := parseSize(options.Size)
	if err != nil && api == apiLatest {
		return nil, ImageOptions{}, nil, err
	}

	imageOptions, err := readImageOptions(options.ProviderOptions, api, size)
	if err != nil {
		return nil, ImageOptions{}, nil, err
	}

	warnings := sizeWarnings(options, api)

	inputImages, err := encodeInputImages(options.Files)
	if err != nil {
		return nil, ImageOptions{}, nil, err
	}

	aspectRatio := options.AspectRatio
	if aspectRatio == "" && size != nil {
		aspectRatio = reducedAspectRatio(*size)
	}

	var body any

	if api == apiLegacy {
		body, err = m.legacyBody(options, imageOptions, inputImages, aspectRatio, size)
		if err != nil {
			return nil, ImageOptions{}, nil, err
		}
	} else {
		if options.Mask != nil {
			return nil, ImageOptions{}, nil, fmt.Errorf("%w: FLUX 3 image masks", model.ErrUnsupported)
		}

		var bodyWarnings []model.Warning
		body, bodyWarnings = currentBody(options, imageOptions, inputImages, aspectRatio, api)
		warnings = append(warnings, bodyWarnings...)
	}

	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, ImageOptions{}, nil, fmt.Errorf("encode request body: %w", err)
	}

	return encoded, imageOptions, warnings, nil
}

// sizeWarnings explains how a size is applied to models without explicit
// dimensions.
func sizeWarnings(options model.ImageOptions, api modelAPI) []model.Warning {
	if options.Size == "" || api == apiLatest {
		return nil
	}

	var details string

	switch {
	case options.AspectRatio == "" && api == apiFlux3:
		details = "Deriving aspect_ratio from size. FLUX 3 uses the resolution provider option to select output resolution."
	case options.AspectRatio == "":
		details = "Deriving aspect_ratio from size. Use the width and height provider options to specify dimensions for models that support them."
	case api == apiFlux3:
		details = "FLUX 3 ignores size when aspectRatio is provided. Use the resolution provider option to select output resolution."
	default:
		details = "Black Forest Labs ignores size when aspectRatio is provided. Use the width and height provider options to specify dimensions for models that support them"
	}

	return []model.Warning{
		model.UnsupportedWarning("size", details),
	}
}

// currentRequest is the request of FLUX 3 and later models.
type currentRequest struct {
	Prompt          string        `json:"prompt,omitempty"`
	AspectRatio     string        `json:"aspect_ratio,omitempty"`
	Images          []string      `json:"images,omitempty"`
	Resolution      *Resolution   `json:"resolution,omitempty"`
	SafetyTolerance *int          `json:"safety_tolerance,omitempty"`
	Width           *int          `json:"width,omitempty"`
	Height          *int          `json:"height,omitempty"`
	Seed            *int          `json:"seed,omitempty"`
	Steps           *int          `json:"steps,omitempty"`
	OutputFormat    *OutputFormat `json:"output_format,omitempty"`
	Grounding       *bool         `json:"grounding,omitempty"`
	Version         *string       `json:"version,omitempty"`
}

// currentBody builds the request of FLUX 3 and later models.
func currentBody(
	options model.ImageOptions,
	imageOptions ImageOptions,
	inputImages []string,
	aspectRatio string,
	api modelAPI,
) (currentRequest, []model.Warning) {
	var warnings []model.Warning

	hasDimensions := api == apiLatest && imageOptions.Width != nil

	requestedAspectRatio := aspectRatio
	switch {
	case hasDimensions:
		requestedAspectRatio = ""
	case aspectRatio == "7:3":
		requestedAspectRatio = "21:9"
	case aspectRatio == "3:7":
		requestedAspectRatio = "9:21"
	}

	if requestedAspectRatio != "" && !slices.Contains(flux3AspectRatios, requestedAspectRatio) {
		warnings = append(warnings, model.UnsupportedWarning(
			"aspectRatio",
			fmt.Sprintf("FLUX 3 does not support aspect ratio %s. Using the endpoint's default auto aspect ratio.", aspectRatio),
		))

		requestedAspectRatio = ""
	}

	if options.Seed != nil && api != apiLatest {
		warnings = append(warnings, model.UnsupportedWarning("seed", ""))
	}

	warnings = append(warnings, unsupportedOptionWarnings(imageOptions, api)...)

	body := currentRequest{
		Prompt:          options.Prompt,
		AspectRatio:     requestedAspectRatio,
		Images:          inputImages,
		SafetyTolerance: imageOptions.SafetyTolerance,
	}

	if !hasDimensions {
		body.Resolution = imageOptions.Resolution
	}

	if api == apiLatest {
		body.Width = imageOptions.Width
		body.Height = imageOptions.Height
		body.Seed = options.Seed
		body.Steps = imageOptions.Steps
		body.OutputFormat = imageOptions.OutputFormat
	} else {
		body.Grounding = imageOptions.Grounding
		body.Version = imageOptions.Version
	}

	return body, warnings
}

func unsupportedOptionWarnings(imageOptions ImageOptions, api modelAPI) []model.Warning {
	supported := flux3SupportedOptions
	if api == apiLatest {
		supported = latestSupportedOptions
	}

	var warnings []model.Warning
	for _, option := range imageOptions.presence() {
		if option.set && !slices.Contains(supported, option.key) {
			warnings = append(warnings, model.UnsupportedWarning(Namespace+"."+option.key, ""))
		}
	}

	return warnings
}

// legacyRequest is the request of FLUX.1, FLUX.2 and Kontext models. Input
// images are added as "input_image", "input_image_2" and so on, or as
// "image" for flux-pro-1.0-fill.
type legacyRequest struct {
	Prompt              string        `json:"prompt,omitempty"`
	Seed                *int          `json:"seed,omitempty"`
	AspectRatio         string        `json:"aspect_ratio,omitempty"`
	Width               *int          `json:"width,omitempty"`
	Height              *int          `json:"height,omitempty"`
	Steps               *int          `json:"steps,omitempty"`
	Guidance            *float64      `json:"guidance,omitempty"`
	ImagePromptStrength *float64      `json:"image_prompt_strength,omitempty"`
	ImagePrompt         *string       `json:"image_prompt,omitempty"`
	Mask                string        `json:"mask,omitempty"`
	OutputFormat        *OutputFormat `json:"output_format,omitempty"`
	PromptUpsampling    *bool         `json:"prompt_upsampling,omitempty"`
	Raw                 *bool         `json:"raw,omitempty"`
	SafetyTolerance     *int          `json:"safety_tolerance,omitempty"`
	WebhookSecret       *string       `json:"webhook_secret,omitempty"`
	WebhookURL          *string       `json:"webhook_url,omitempty"`
}

// legacyBody builds the request of legacy models.
func (m *ImageModel) legacyBody(
	options model.ImageOptions,
	imageOptions ImageOptions,
	inputImages []string,
	aspectRatio string,
	size *imageSize,
) (map[string]json.RawMessage, error) {
	body := legacyRequest{
		Prompt:              options.Prompt,
		Seed:                options.Seed,
		AspectRatio:         aspectRatio,
		Width:               imageOptions.Width,
		Height:              imageOptions.Height,
		Steps:               imageOptions.Steps,
		Guidance:            imageOptions.Guidance,
		ImagePromptStrength: imageOptions.ImagePromptStrength,
		ImagePrompt:         imageOptions.ImagePrompt,
		OutputFormat:        imageOptions.OutputFormat,
		PromptUpsampling:    imageOptions.PromptUpsampling,
		Raw:                 imageOptions.Raw,
		SafetyTolerance:     imageOptions.SafetyTolerance,
		WebhookSecret:       imageOptions.WebhookSecret,
		WebhookURL:          imageOptions.WebhookURL,
	}

	if size != nil && body.Width == nil {
		body.Width = new(size.width)
	}

	if size != nil && body.Height == nil {
		body.Height = new(size.height)
	}

	if options.Mask != nil {
		mask, err := encodeInputImage(options.Mask)
		if err != nil {
			return nil, err
		}

		body.Mask = mask
	}

	members, err := jsonobject.Members(body)
	if err != nil {
		return nil, fmt.Errorf("encode request body: %w", err)
	}

	fieldName := "input_image"
	if m.modelID == "flux-pro-1.0-fill" {
		fieldName = "image"
	}

	for index, image := range inputImages {
		key := fieldName
		if index > 0 {
			key = fieldName + "_" + strconv.Itoa(index+1)
		}

		encoded, err := json.Marshal(image)
		if err != nil {
			return nil, fmt.Errorf("encode input image: %w", err)
		}

		members[key] = encoded
	}

	return members, nil
}

// encodeInputImages returns the input files as URLs or raw base64.
func encodeInputImages(files []model.ImageFile) ([]string, error) {
	if len(files) > maxInputImages {
		return nil, fmt.Errorf("%w: Black Forest Labs supports up to %d input images", model.ErrInvalidArgument, maxInputImages)
	}

	images := make([]string, 0, len(files))
	for _, file := range files {
		image, err := encodeInputImage(file)
		if err != nil {
			return nil, err
		}

		images = append(images, image)
	}

	return images, nil
}

func encodeInputImage(file model.ImageFile) (string, error) {
	switch file := file.(type) {
	case model.ImageURL:
		return file.URL, nil

	case model.ImageBytes:
		return base64.StdEncoding.EncodeToString(file.Data), nil

	default:
		return "", fmt.Errorf("%w: image file %T", model.ErrUnsupported, file)
	}
}

// parseSize reads a "WIDTHxHEIGHT" size. It returns nil without error when
// size is empty.
func parseSize(size string) (*imageSize, error) {
	if size == "" {
		return nil, nil
	}

	widthText, heightText, _ := strings.Cut(size, "x")

	width, widthErr := strconv.Atoi(widthText)
	height, heightErr := strconv.Atoi(heightText)
	if widthErr != nil || heightErr != nil {
		return nil, fmt.Errorf("%w: size %q must have the form WIDTHxHEIGHT", model.ErrInvalidArgument, size)
	}

	return &imageSize{
		width:  width,
		height: height,
	}, nil
}

// reducedAspectRatio returns the size's aspect ratio in lowest terms, or an
// empty string when a dimension is not positive.
func reducedAspectRatio(size imageSize) string {
	if size.width <= 0 || size.height <= 0 {
		return ""
	}

	divisor := gcd(size.width, size.height)

	return fmt.Sprintf("%d:%d", size.width/divisor, size.height/divisor)
}

func gcd(a int, b int) int {
	for b != 0 {
		a, b = b, a%b
	}

	return a
}

// downloadImage downloads the generated image. Signed Black Forest Labs
// URLs need no credentials; a custom base URL, such as a proxy, receives
// the provider headers for images on its own origin.
func (m *ImageModel) downloadImage(ctx context.Context, imageURL string, headers http.Header) ([]byte, error) {
	request := download.Request{
		URL:           imageURL,
		TrustedOrigin: m.provider.baseURL,
		Client:        m.provider.httpClient,
	}

	if origin.Same(imageURL, m.provider.baseURL) && !isBFLHost(hostname(m.provider.baseURL)) {
		request.Headers = headers
		request.CredentialedOrigin = m.provider.baseURL
	} else if userAgent := headers.Get("User-Agent"); userAgent != "" {
		request.Headers = http.Header{
			"User-Agent": {
				userAgent,
			},
		}
	}

	image, err := download.Fetch(ctx, request)
	if err != nil {
		return nil, err
	}

	return image.Data, nil
}

var _ model.ImageModel = (*ImageModel)(nil)
