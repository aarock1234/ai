package fireworks

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/internal/download"
	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/internal/poll"
	"github.com/aarock1234/ai/pkg/model"
)

// Polling of asynchronous image models.
const (
	pollInterval = 500 * time.Millisecond
	pollTimeout  = 2 * time.Minute
)

// Poll statuses of asynchronous image generation.
const (
	pollStatusReady  = "Ready"
	pollStatusError  = "Error"
	pollStatusFailed = "Failed"
)

var (
	// ErrImageGenerationFailed reports an asynchronous image generation that
	// failed or returned no image.
	ErrImageGenerationFailed = errors.New("fireworks image generation failed")

	// ErrImageGenerationTimeout reports an asynchronous image generation that
	// did not finish within the polling deadline. It is also the
	// cancellation cause of the expired deadline.
	ErrImageGenerationTimeout = errors.New("fireworks image generation timed out")
)

// endpointFormat selects the endpoint layout of an image model.
type endpointFormat int

const (
	// endpointWorkflows returns the image synchronously from a workflow.
	endpointWorkflows endpointFormat = iota

	// endpointWorkflowsAsync returns a request ID to poll for the image.
	endpointWorkflowsAsync

	// endpointImageGeneration is the legacy image generation endpoint.
	endpointImageGeneration
)

// imageBackend describes how an image model is served.
type imageBackend struct {
	format endpointFormat

	// supportsSize reports models that take a size instead of an aspect
	// ratio.
	supportsSize bool
}

// imageBackends are the known image models. Unknown models use synchronous
// workflows.
var imageBackends = map[string]imageBackend{
	"accounts/fireworks/models/flux-1-dev-fp8": {
		format: endpointWorkflows,
	},
	"accounts/fireworks/models/flux-1-schnell-fp8": {
		format: endpointWorkflows,
	},
	"accounts/fireworks/models/flux-kontext-pro": {
		format: endpointWorkflowsAsync,
	},
	"accounts/fireworks/models/flux-kontext-max": {
		format: endpointWorkflowsAsync,
	},
	"accounts/fireworks/models/playground-v2-5-1024px-aesthetic": {
		format:       endpointImageGeneration,
		supportsSize: true,
	},
	"accounts/fireworks/models/japanese-stable-diffusion-xl": {
		format:       endpointImageGeneration,
		supportsSize: true,
	},
	"accounts/fireworks/models/playground-v2-1024px-aesthetic": {
		format:       endpointImageGeneration,
		supportsSize: true,
	},
	"accounts/fireworks/models/stable-diffusion-xl-1024-v1-0": {
		format:       endpointImageGeneration,
		supportsSize: true,
	},
	"accounts/fireworks/models/SSD-1B": {
		format:       endpointImageGeneration,
		supportsSize: true,
	},
}

// ImageModel is a Fireworks image model. Kontext models generate and edit
// images asynchronously; other models return images synchronously.
type ImageModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "fireworks.image" by default.
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

// imageRequest holds the standard members of a generation request. The
// provider options are added after them.
type imageRequest struct {
	Prompt      string `json:"prompt,omitempty"`
	AspectRatio string `json:"aspect_ratio,omitempty"`
	Seed        *int   `json:"seed,omitempty"`
	Samples     int    `json:"samples,omitempty"`
	InputImage  string `json:"input_image,omitempty"`
	Width       string `json:"width,omitempty"`
	Height      string `json:"height,omitempty"`
}

type submitResponse struct {
	RequestID string `json:"request_id"`
}

type pollRequest struct {
	ID string `json:"id"`
}

type pollResponse struct {
	ID     string      `json:"id"`
	Status string      `json:"status"`
	Result *pollResult `json:"result"`
}

type pollResult struct {
	Sample *string `json:"sample"`
}

// Generate generates one image. Kontext models edit the first input file.
func (m *ImageModel) Generate(ctx context.Context, options model.ImageOptions) (*model.ImageResult, error) {
	timestamp := time.Now()
	backend := imageBackends[m.modelID]
	warnings := imageWarnings(options, backend)

	imageOptions, err := readImageOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	body, err := m.requestBody(options, imageOptions)
	if err != nil {
		return nil, err
	}

	request, err := m.apiRequest(m.endpointURL(backend), options.Headers)
	if err != nil {
		return nil, err
	}

	var image []byte
	var headers http.Header

	if backend.format == endpointWorkflowsAsync {
		image, headers, err = m.generateAsync(ctx, request, body)
	} else {
		image, headers, err = m.generateSync(ctx, request, body)
	}

	if err != nil {
		return nil, err
	}

	return &model.ImageResult{
		Images: [][]byte{
			image,
		},
		Warnings: warnings,
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: timestamp,
				ModelID:   m.modelID,
			},
			Headers: headers,
		},
	}, nil
}

// imageWarnings reports the settings the model ignores. Models that take a
// size do not take an aspect ratio.
func imageWarnings(options model.ImageOptions, backend imageBackend) []model.Warning {
	var warnings []model.Warning

	if !backend.supportsSize && options.Size != "" {
		warnings = append(warnings, model.UnsupportedWarning(
			"size",
			"This model does not support the `size` option. Use `aspectRatio` instead.",
		))
	}

	if backend.supportsSize && options.AspectRatio != "" {
		warnings = append(warnings, model.UnsupportedWarning(
			"aspectRatio",
			"This model does not support the `aspectRatio` option.",
		))
	}

	if len(options.Files) > 1 {
		warnings = append(warnings, model.Warning{
			Type:    model.WarningOther,
			Message: "Fireworks only supports a single input image. Additional images are ignored.",
		})
	}

	if options.Mask != nil {
		warnings = append(warnings, model.UnsupportedWarning(
			"mask",
			"Fireworks Kontext models do not support explicit masks. Use the prompt to describe the areas to edit.",
		))
	}

	return warnings
}

// requestBody returns the standard members followed by the provider options.
func (m *ImageModel) requestBody(options model.ImageOptions, imageOptions jsonobject.Object) (jsonobject.Object, error) {
	width, height, _ := strings.Cut(options.Size, "x")

	request := imageRequest{
		Prompt:      options.Prompt,
		AspectRatio: options.AspectRatio,
		Seed:        options.Seed,
		Samples:     options.N,
		Width:       width,
		Height:      height,
	}

	if len(options.Files) > 0 {
		inputImage, err := dataurl.ImageFileURL(options.Files[0])
		if err != nil {
			return nil, err
		}

		request.InputImage = inputImage
	}

	encoded, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("encode request body: %w", err)
	}

	head, err := jsonobject.Parse(encoded)
	if err != nil {
		return nil, fmt.Errorf("encode request body: %w", err)
	}

	return head.Merge(imageOptions), nil
}

// generateSync posts the request and returns the image bytes of the
// response.
func (m *ImageModel) generateSync(ctx context.Context, request apicall.Request, body jsonobject.Object) ([]byte, http.Header, error) {
	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, nil, err
	}

	return response.Body, response.Headers, nil
}

// generateAsync submits the request, polls until the image is ready and
// downloads it. Provider credentials are sent only to the provider's own
// origin, never to the CDN that usually serves the image.
func (m *ImageModel) generateAsync(ctx context.Context, request apicall.Request, body jsonobject.Object) ([]byte, http.Header, error) {
	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, nil, err
	}

	var submitted submitResponse
	if err := apicall.DecodeJSON(response.Body, &submitted); err != nil {
		return nil, nil, err
	}

	imageURL, headers, err := m.pollForImageURL(ctx, request, submitted.RequestID)
	if err != nil {
		return nil, nil, err
	}

	downloaded, err := download.Fetch(ctx, download.Request{
		URL:                imageURL,
		Headers:            request.Headers,
		CredentialedOrigin: m.provider.baseURL,
		TrustedOrigin:      m.provider.baseURL,
		Client:             m.provider.httpClient,
	})
	if err != nil {
		return nil, nil, err
	}

	return downloaded.Data, headers, nil
}

// pollForImageURL polls the result endpoint until the image is ready, the
// generation fails or the polling deadline expires. It returns the image URL
// and the headers of the last poll response.
func (m *ImageModel) pollForImageURL(ctx context.Context, request apicall.Request, requestID string) (string, http.Header, error) {
	pollCtx, cancel := context.WithTimeoutCause(ctx, pollTimeout, ErrImageGenerationTimeout)
	defer cancel()

	request.URL = m.provider.baseURL + "/workflows/" + m.modelID + "/get_result"

	for {
		response, err := apicall.PostJSON(pollCtx, request, pollRequest{
			ID: requestID,
		})
		if err != nil {
			return "", nil, pollError(pollCtx, err)
		}

		var polled pollResponse
		if err := apicall.DecodeJSON(response.Body, &polled); err != nil {
			return "", nil, err
		}

		switch polled.Status {
		case pollStatusReady:
			if polled.Result == nil || polled.Result.Sample == nil {
				return "", nil, fmt.Errorf("%w: poll response is ready but missing result.sample", ErrImageGenerationFailed)
			}

			return *polled.Result.Sample, response.Headers, nil

		case pollStatusError, pollStatusFailed:
			return "", nil, fmt.Errorf("%w: status %s", ErrImageGenerationFailed, polled.Status)
		}

		if err := poll.Wait(pollCtx, pollInterval); err != nil {
			return "", nil, pollError(pollCtx, err)
		}
	}
}

// pollError reports an expired polling deadline as a timeout and returns
// other errors unchanged.
func pollError(pollCtx context.Context, err error) error {
	if !errors.Is(context.Cause(pollCtx), ErrImageGenerationTimeout) {
		return err
	}

	return fmt.Errorf("%w after %s: %w", ErrImageGenerationTimeout, pollTimeout, context.DeadlineExceeded)
}

// endpointURL returns the generation endpoint of the model.
func (m *ImageModel) endpointURL(backend imageBackend) string {
	switch backend.format {
	case endpointImageGeneration:
		return m.provider.baseURL + "/image_generation/" + m.modelID
	case endpointWorkflowsAsync:
		return m.provider.baseURL + "/workflows/" + m.modelID
	default:
		return m.provider.baseURL + "/workflows/" + m.modelID + "/text_to_image"
	}
}

// apiRequest describes a call whose failures report the HTTP status text.
func (m *ImageModel) apiRequest(url string, callHeaders http.Header) (apicall.Request, error) {
	headers, err := m.provider.requestHeaders()
	if err != nil {
		return apicall.Request{}, err
	}

	return apicall.Request{
		Client:  m.provider.httpClient,
		URL:     url,
		Headers: apicall.MergeHeaders(headers, callHeaders),
	}, nil
}

var _ model.ImageModel = (*ImageModel)(nil)
