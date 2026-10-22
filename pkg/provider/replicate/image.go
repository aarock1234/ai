package replicate

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/internal/download"
	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/internal/poll"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	defaultPollInterval    = 500 * time.Millisecond
	defaultMaxPollAttempts = 240

	// maxFlux2InputImages is the input image limit of FLUX.2 models.
	maxFlux2InputImages = 8
)

// flux2ModelPattern matches FLUX.2 models, which accept several input images.
var flux2ModelPattern = regexp.MustCompile(`^black-forest-labs/flux-2-`)

var (
	// ErrImageGenerationFailed reports a prediction that failed or was
	// canceled. The error names the status and the reported reason.
	ErrImageGenerationFailed = errors.New("replicate image generation failed")

	// ErrImageGenerationTimeout reports a prediction that did not finish
	// within the polling attempts.
	ErrImageGenerationTimeout = errors.New("replicate image generation timed out")
)

// ImageModel is a Replicate image model.
type ImageModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "replicate" by default.
func (m *ImageModel) Provider() string {
	return m.provider.name
}

// ModelID returns the model identifier.
func (m *ImageModel) ModelID() string {
	return m.modelID
}

// MaxImagesPerCall returns 8 for FLUX.2 models and 1 otherwise.
func (m *ImageModel) MaxImagesPerCall() int {
	if m.isFlux2() {
		return maxFlux2InputImages
	}

	return 1
}

func (m *ImageModel) isFlux2() bool {
	return flux2ModelPattern.MatchString(m.modelID)
}

type predictionRequest struct {
	Input   map[string]json.RawMessage `json:"input"`
	Version string                     `json:"version,omitempty"`
}

// predictionInput holds the standard input fields.
type predictionInput struct {
	Prompt      string `json:"prompt,omitempty"`
	AspectRatio string `json:"aspect_ratio,omitempty"`
	Size        string `json:"size,omitempty"`
	Seed        *int   `json:"seed,omitempty"`
	NumOutputs  int    `json:"num_outputs,omitempty"`
}

// predictionStatus is the state of a prediction.
type predictionStatus string

const (
	statusStarting   predictionStatus = "starting"
	statusProcessing predictionStatus = "processing"
	statusSucceeded  predictionStatus = "succeeded"
	statusFailed     predictionStatus = "failed"
	statusCanceled   predictionStatus = "canceled"
)

type prediction struct {
	Status predictionStatus `json:"status"`
	Output predictionOutput `json:"output"`
	Error  *string          `json:"error"`
	URLs   predictionURLs   `json:"urls"`
}

type predictionURLs struct {
	Get string `json:"get"`
}

// predictionOutput is the list of output URLs. Replicate returns a single
// URL as a string; null leaves the list nil.
type predictionOutput []string

// UnmarshalJSON reads a URL string or an array of URLs.
func (o *predictionOutput) UnmarshalJSON(data []byte) error {
	var single string
	if err := json.Unmarshal(data, &single); err == nil {
		*o = predictionOutput{
			single,
		}

		return nil
	}

	var urls []string
	if err := json.Unmarshal(data, &urls); err != nil {
		return fmt.Errorf("output must be a string or an array of strings: %w", err)
	}

	*o = urls

	return nil
}

// pollSettings controls how a prediction is awaited.
type pollSettings struct {
	interval    time.Duration
	maxAttempts int
}

// Generate creates a prediction, waits for it and downloads its images.
func (m *ImageModel) Generate(ctx context.Context, options model.ImageOptions) (*model.ImageResult, error) {
	timestamp := time.Now()

	imageOptions, inputOptions, err := readImageOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	input, warnings, err := m.predictionInput(options, inputOptions)
	if err != nil {
		return nil, err
	}

	headers, err := m.provider.authHeaders(options.Headers)
	if err != nil {
		return nil, err
	}

	modelID, version, _ := strings.Cut(m.modelID, ":")

	endpoint := m.provider.baseURL + "/models/" + modelID + "/predictions"
	if version != "" {
		endpoint = m.provider.baseURL + "/predictions"
	}

	createHeaders := apicall.MergeHeaders(headers, http.Header{
		"Prefer": {
			preferHeader(imageOptions.MaxWaitTimeInSeconds),
		},
	})

	response, err := apicall.PostJSON(ctx, m.provider.apiRequest(endpoint, createHeaders), predictionRequest{
		Input:   input,
		Version: version,
	})
	if err != nil {
		return nil, err
	}

	var created prediction
	if err := apicall.DecodeJSON(response.Body, &created); err != nil {
		return nil, err
	}

	completed, err := m.awaitPrediction(ctx, created, headers, pollSettingsOf(imageOptions))
	if err != nil {
		return nil, err
	}

	if completed.Output == nil {
		return nil, fmt.Errorf("%w: replicate image generation completed without output", model.ErrInvalidResponse)
	}

	images, err := m.downloadImages(ctx, completed.Output)
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

// preferHeader asks Replicate to hold the request open until the prediction
// finishes or the wait time passes.
func preferHeader(maxWaitTimeInSeconds *float64) string {
	if maxWaitTimeInSeconds == nil {
		return "wait"
	}

	return "wait=" + strconv.FormatFloat(*maxWaitTimeInSeconds, 'f', -1, 64)
}

// predictionInput combines the standard fields, input images and the
// provider options, which take precedence.
func (m *ImageModel) predictionInput(
	options model.ImageOptions,
	inputOptions map[string]json.RawMessage,
) (map[string]json.RawMessage, []model.Warning, error) {
	input, err := jsonobject.Members(predictionInput{
		Prompt:      options.Prompt,
		AspectRatio: options.AspectRatio,
		Size:        options.Size,
		Seed:        options.Seed,
		NumOutputs:  options.N,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("encode prediction input: %w", err)
	}

	images, warnings, err := m.inputImages(options.Files)
	if err != nil {
		return nil, nil, err
	}

	maps.Copy(input, images)

	mask, maskWarnings, err := m.maskInput(options.Mask)
	if err != nil {
		return nil, nil, err
	}

	warnings = append(warnings, maskWarnings...)
	maps.Copy(input, mask)
	maps.Copy(input, inputOptions)

	return input, warnings, nil
}

// inputImages maps the input files onto "input_image", "input_image_2" and
// so on for FLUX.2 models, and onto "image" for other models.
func (m *ImageModel) inputImages(files []model.ImageFile) (map[string]json.RawMessage, []model.Warning, error) {
	if len(files) == 0 {
		return nil, nil, nil
	}

	limit := 1
	if m.isFlux2() {
		limit = maxFlux2InputImages
	}

	images := make(map[string]json.RawMessage)
	for index, file := range files[:min(len(files), limit)] {
		uri, err := dataURIValue(file)
		if err != nil {
			return nil, nil, err
		}

		images[inputImageKey(m.isFlux2(), index)] = uri
	}

	if len(files) <= limit {
		return images, nil, nil
	}

	message := "This Replicate model only supports a single input image. Additional images are ignored."
	if m.isFlux2() {
		message = fmt.Sprintf("Flux-2 models support up to %d input images. Additional images are ignored.", maxFlux2InputImages)
	}

	return images, []model.Warning{
		{
			Type:    model.WarningOther,
			Message: message,
		},
	}, nil
}

func inputImageKey(flux2 bool, index int) string {
	switch {
	case !flux2:
		return "image"
	case index == 0:
		return "input_image"
	default:
		return fmt.Sprintf("input_image_%d", index+1)
	}
}

// maskInput adds the mask, which FLUX.2 models do not support.
func (m *ImageModel) maskInput(mask model.ImageFile) (map[string]json.RawMessage, []model.Warning, error) {
	if mask == nil {
		return nil, nil, nil
	}

	if m.isFlux2() {
		return nil, []model.Warning{
			{
				Type:    model.WarningOther,
				Message: "Flux-2 models do not support mask input. The mask will be ignored.",
			},
		}, nil
	}

	uri, err := dataURIValue(mask)
	if err != nil {
		return nil, nil, err
	}

	return map[string]json.RawMessage{
		"mask": uri,
	}, nil, nil
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

// awaitPrediction fetches the prediction until it completes. A failed or
// canceled prediction is an error wrapping ErrImageGenerationFailed.
func (m *ImageModel) awaitPrediction(
	ctx context.Context,
	current prediction,
	headers http.Header,
	settings pollSettings,
) (prediction, error) {
	for attempt := range settings.maxAttempts {
		done, err := isComplete(current)
		if done || err != nil {
			return current, err
		}

		current, err = m.fetchPrediction(ctx, current.URLs.Get, headers)
		if err != nil {
			return prediction{}, err
		}

		if attempt < settings.maxAttempts-1 {
			if err := poll.Wait(ctx, settings.interval); err != nil {
				return prediction{}, err
			}
		}
	}

	done, err := isComplete(current)
	if done || err != nil {
		return current, err
	}

	return prediction{}, fmt.Errorf("%w after %d polling attempts", ErrImageGenerationTimeout, settings.maxAttempts)
}

// isComplete reports whether the prediction has output or succeeded.
func isComplete(current prediction) (bool, error) {
	switch current.Status {
	case statusFailed, statusCanceled:
		reason := "Unknown error"
		if current.Error != nil {
			reason = *current.Error
		}

		return true, fmt.Errorf("%w: %s: %s", ErrImageGenerationFailed, current.Status, reason)

	case statusSucceeded:
		return true, nil

	case statusStarting, statusProcessing:
		return current.Output != nil, nil

	default:
		return true, fmt.Errorf("%w: unknown prediction status %q", model.ErrInvalidResponse, current.Status)
	}
}

// fetchPrediction reads the prediction at location, which comes from a
// response. Credentials are sent only to the API origin; other URLs are
// validated and receive only non-credential headers.
func (m *ImageModel) fetchPrediction(ctx context.Context, location string, headers http.Header) (prediction, error) {
	_, body, err := m.provider.fetchStatus(ctx, location, headers)
	if err != nil {
		return prediction{}, err
	}

	var current prediction
	if err := apicall.DecodeJSON(body, &current); err != nil {
		return prediction{}, err
	}

	return current, nil
}

// downloadImages downloads the output images without credentials.
func (m *ImageModel) downloadImages(ctx context.Context, urls []string) ([][]byte, error) {
	images := make([][]byte, 0, len(urls))

	for _, location := range urls {
		image, err := download.Fetch(ctx, download.Request{
			URL:           location,
			TrustedOrigin: m.provider.baseURL,
			Client:        m.provider.httpClient,
		})
		if err != nil {
			return nil, err
		}

		images = append(images, image.Data)
	}

	return images, nil
}

// dataURIValue returns an input image as its URL or as a base64 data URL,
// encoded as a JSON string.
func dataURIValue(file model.ImageFile) (json.RawMessage, error) {
	uri, err := dataurl.ImageFileURL(file)
	if err != nil {
		return nil, err
	}

	encoded, err := json.Marshal(uri)
	if err != nil {
		return nil, fmt.Errorf("encode input image: %w", err)
	}

	return encoded, nil
}

var _ model.ImageModel = (*ImageModel)(nil)
