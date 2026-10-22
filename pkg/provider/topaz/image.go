package topaz

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/download"
	"github.com/aarock1234/ai/internal/poll"
	"github.com/aarock1234/ai/pkg/model"
)

// Image API paths. Status, cancel and download paths end with the process
// ID.
const (
	enhancePath  = "/image/v1/enhance-gen/async"
	statusPath   = "/image/v1/status/"
	cancelPath   = "/image/v1/cancel/"
	downloadPath = "/image/v1/download/"
)

const (
	defaultPollInterval = 2 * time.Second
	defaultPollTimeout  = 10 * time.Minute

	// cancelTimeout bounds the best-effort cancellation of an abandoned job.
	cancelTimeout = 10 * time.Second
)

// imageAPIModelIDs maps readable model IDs onto Topaz API model names.
var imageAPIModelIDs = map[string]string{
	"wonder-3.5": "Wonder 3.5",
}

var (
	// ErrImageEnhancementFailed reports a job that Topaz failed or
	// cancelled. The error names the status and the process ID.
	ErrImageEnhancementFailed = errors.New("topaz image enhancement failed")

	// ErrImageEnhancementTimeout reports a job that did not finish within
	// the poll timeout. Such a job is cancelled.
	ErrImageEnhancementTimeout = errors.New("topaz image enhancement timed out")
)

// Process statuses. Unknown statuses keep polling.
const (
	processCompleted = "Completed"
	processFailed    = "Failed"
	processCancelled = "Cancelled"
)

// ImageMetadata is the Topaz metadata of an image response, stored under
// ProviderMetadata["topaz"].
type ImageMetadata struct {
	// Images has one entry for the enhanced image.
	Images []ImageInfo `json:"images"`
}

// ImageInfo describes an enhanced image. Nil fields were not reported.
type ImageInfo struct {
	ProcessID string `json:"processId"`

	// Credits are the credits billed for the completed job.
	Credits *float64 `json:"credits,omitempty"`

	Width  *float64 `json:"width,omitempty"`
	Height *float64 `json:"height,omitempty"`
	Format *string  `json:"format,omitempty"`
}

// ImageModel is a Topaz image enhancement model. It enhances the first input
// file; a text prompt is not used.
type ImageModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "topaz.image" by default.
func (m *ImageModel) Provider() string {
	return m.provider.name + ".image"
}

// ModelID returns the model identifier.
func (m *ImageModel) ModelID() string {
	return m.modelID
}

// MaxImagesPerCall returns 1: each call enhances one image.
func (m *ImageModel) MaxImagesPerCall() int {
	return 1
}

// apiModelID returns the Topaz API name of the model.
func (m *ImageModel) apiModelID() string {
	if name, ok := imageAPIModelIDs[m.modelID]; ok {
		return name
	}

	return m.modelID
}

type submitResponse struct {
	ProcessID *string `json:"process_id"`
}

type statusResponse struct {
	Status       *string  `json:"status"`
	Credits      *float64 `json:"credits"`
	OutputWidth  *float64 `json:"output_width"`
	OutputHeight *float64 `json:"output_height"`
	OutputFormat *string  `json:"output_format"`
}

type downloadResponse struct {
	DownloadURL *string `json:"download_url"`
}

// pollSettings controls how a job is awaited.
type pollSettings struct {
	interval time.Duration
	timeout  time.Duration
}

// Generate submits an enhancement job, waits for it and downloads the
// enhanced image.
func (m *ImageModel) Generate(ctx context.Context, options model.ImageOptions) (*model.ImageResult, error) {
	timestamp := time.Now()

	imageOptions, err := readImageOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	headers, err := m.provider.authHeaders(options.Headers)
	if err != nil {
		return nil, err
	}

	warnings := unsupportedImageSettings(options)

	fields, formWarnings, err := m.enhanceForm(options, imageOptions)
	if err != nil {
		return nil, err
	}

	warnings = append(warnings, formWarnings...)

	processID, err := m.submitEnhancement(ctx, fields, headers)
	if err != nil {
		return nil, err
	}

	status, err := m.waitForCompletion(ctx, processID, headers, pollSettingsOf(imageOptions))
	if err != nil {
		if ctx.Err() != nil {
			m.cancelQuietly(ctx, processID, headers)
		}

		return nil, err
	}

	image, responseHeaders, err := m.downloadImage(ctx, processID, headers)
	if err != nil {
		return nil, err
	}

	return &model.ImageResult{
		Images: [][]byte{
			image,
		},
		Warnings:         warnings,
		ProviderMetadata: imageMetadata(processID, status),
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: timestamp,
				ModelID:   m.modelID,
			},
			Headers: responseHeaders,
		},
	}, nil
}

// submitEnhancement submits the enhance form and returns the process ID of
// the queued job.
func (m *ImageModel) submitEnhancement(ctx context.Context, fields []apicall.FormField, headers http.Header) (string, error) {
	submitted, err := apicall.PostForm(ctx, m.provider.apiRequest(enhancePath, headers), fields)
	if err != nil {
		return "", err
	}

	var submit submitResponse
	if err := apicall.DecodeJSON(submitted.Body, &submit); err != nil {
		return "", err
	}

	if submit.ProcessID == nil || *submit.ProcessID == "" {
		return "", fmt.Errorf("%w: Topaz did not return a process_id for the enhance request", model.ErrInvalidResponse)
	}

	return *submit.ProcessID, nil
}

// imageMetadata describes the enhanced image of a completed job.
func imageMetadata(processID string, status statusResponse) model.ProviderMetadata {
	return model.ProviderMetadata{
		Namespace: ImageMetadata{
			Images: []ImageInfo{
				{
					ProcessID: processID,
					Credits:   status.Credits,
					Width:     status.OutputWidth,
					Height:    status.OutputHeight,
					Format:    status.OutputFormat,
				},
			},
		},
	}
}

func unsupportedImageSettings(options model.ImageOptions) []model.Warning {
	var warnings []model.Warning

	if options.Prompt != "" {
		warnings = append(warnings, model.UnsupportedWarning(
			"prompt",
			"Topaz image models enhance an existing image and do not take a text prompt. The prompt was ignored.",
		))
	}

	if options.AspectRatio != "" {
		warnings = append(warnings, model.UnsupportedWarning(
			"aspectRatio",
			"Topaz image models do not support aspectRatio. Use `size`, or the `outputWidth` / `outputHeight` provider options, to set output dimensions.",
		))
	}

	if options.Seed != nil {
		warnings = append(warnings, model.UnsupportedWarning(
			"seed",
			"Topaz image models do not support seed.",
		))
	}

	if options.Mask != nil {
		warnings = append(warnings, model.UnsupportedWarning(
			"mask",
			"Topaz image models do not support masks. The mask was ignored.",
		))
	}

	if options.N > 1 {
		warnings = append(warnings, model.UnsupportedWarning(
			"n",
			"Topaz image models enhance one image per call. Only 1 image will be returned.",
		))
	}

	return warnings
}

// enhanceForm builds the multipart enhance request from the first input
// file and the options.
func (m *ImageModel) enhanceForm(options model.ImageOptions, imageOptions ImageOptions) ([]apicall.FormField, []model.Warning, error) {
	if len(options.Files) == 0 {
		return nil, nil, fmt.Errorf(
			"%w: files: Topaz image models enhance an existing image. Pass the input image as an input file",
			model.ErrInvalidArgument,
		)
	}

	var warnings []model.Warning
	if len(options.Files) > 1 {
		warnings = append(warnings, model.UnsupportedWarning(
			"files",
			"Topaz image models enhance one image per call. Only the first file was used.",
		))
	}

	source, err := sourceField(options.Files[0])
	if err != nil {
		return nil, nil, err
	}

	var form enhanceForm
	form.add("model", m.apiModelID())
	form.fields = append(form.fields, source)

	width, height := parseSize(options.Size)
	form.addInt("output_width", firstInt(imageOptions.OutputWidth, width))
	form.addInt("output_height", firstInt(imageOptions.OutputHeight, height))
	form.addString("output_format", (*string)(imageOptions.OutputFormat))
	form.addBool("crop_to_fill", imageOptions.CropToFill)
	form.addString("webhook_url", imageOptions.WebhookURL)

	form.addString("enhancementStrength", (*string)(imageOptions.EnhancementStrength))
	form.addBool("grain", imageOptions.Grain)
	form.addFloat("grainDensity", imageOptions.GrainDensity)
	form.addString("grainModel", (*string)(imageOptions.GrainModel))
	form.addFloat("grainSize", imageOptions.GrainSize)
	form.addFloat("grainStrength", imageOptions.GrainStrength)
	form.addInt("inputWidth", imageOptions.InputWidth)
	form.addInt("inputHeight", imageOptions.InputHeight)

	return form.fields, warnings, nil
}

// sourceField sends a URL for Topaz to fetch, or the image bytes.
func sourceField(file model.ImageFile) (apicall.FormField, error) {
	switch file := file.(type) {
	case model.ImageURL:
		return apicall.FormField{
			Name:  "source_url",
			Value: file.URL,
		}, nil

	case model.ImageBytes:
		return apicall.FormField{
			Name:      "image",
			Filename:  "image." + fileExtension(file.MediaType),
			MediaType: file.MediaType,
			Data:      file.Data,
		}, nil

	default:
		return apicall.FormField{}, fmt.Errorf("%w: image file %T", model.ErrUnsupported, file)
	}
}

// fileExtension returns the media subtype, or "png" when there is none.
func fileExtension(mediaType string) string {
	_, subtype, _ := strings.Cut(strings.ToLower(mediaType), "/")
	if subtype == "" {
		return "png"
	}

	return subtype
}

// parseSize reads a "WIDTHxHEIGHT" size. Missing or invalid dimensions are
// nil.
func parseSize(size string) (*int, *int) {
	if size == "" {
		return nil, nil
	}

	widthText, heightText, _ := strings.Cut(size, "x")

	var width *int
	if value, err := strconv.Atoi(widthText); err == nil {
		width = new(value)
	}

	var height *int
	if value, err := strconv.Atoi(heightText); err == nil {
		height = new(value)
	}

	return width, height
}

func firstInt(preferred *int, fallback *int) *int {
	if preferred != nil {
		return preferred
	}

	return fallback
}

// enhanceForm collects the multipart fields of an enhance request.
type enhanceForm struct {
	fields []apicall.FormField
}

func (f *enhanceForm) add(name string, value string) {
	f.fields = append(f.fields, apicall.FormField{
		Name:  name,
		Value: value,
	})
}

func (f *enhanceForm) addString(name string, value *string) {
	if value != nil {
		f.add(name, *value)
	}
}

func (f *enhanceForm) addInt(name string, value *int) {
	if value != nil {
		f.add(name, strconv.Itoa(*value))
	}
}

func (f *enhanceForm) addFloat(name string, value *float64) {
	if value != nil {
		f.add(name, strconv.FormatFloat(*value, 'f', -1, 64))
	}
}

func (f *enhanceForm) addBool(name string, value *bool) {
	if value != nil {
		f.add(name, strconv.FormatBool(*value))
	}
}

func pollSettingsOf(options ImageOptions) pollSettings {
	settings := pollSettings{
		interval: defaultPollInterval,
		timeout:  defaultPollTimeout,
	}

	if options.PollIntervalMillis != nil {
		settings.interval = time.Duration(*options.PollIntervalMillis) * time.Millisecond
	}

	if options.PollTimeoutMillis != nil {
		settings.timeout = time.Duration(*options.PollTimeoutMillis) * time.Millisecond
	}

	return settings
}

// waitForCompletion polls the job until it completes. A job that does not
// finish before the timeout is cancelled, since Topaz bills on completion.
func (m *ImageModel) waitForCompletion(
	ctx context.Context,
	processID string,
	headers http.Header,
	settings pollSettings,
) (statusResponse, error) {
	deadline := time.Now().Add(settings.timeout)
	request := m.provider.apiRequest(statusPath+url.PathEscape(processID), headers)

	for {
		response, err := apicall.Get(ctx, request)
		if err != nil {
			return statusResponse{}, err
		}

		var status statusResponse
		if err := apicall.DecodeJSON(response.Body, &status); err != nil {
			return statusResponse{}, err
		}

		current := ""
		if status.Status != nil {
			current = *status.Status
		}

		switch current {
		case processCompleted:
			return status, nil

		case processFailed, processCancelled:
			return statusResponse{}, fmt.Errorf("%w: %s for process %s", ErrImageEnhancementFailed, strings.ToLower(current), processID)
		}

		if time.Now().Add(settings.interval).After(deadline) {
			m.cancelQuietly(ctx, processID, headers)

			return statusResponse{}, timeoutError(processID, current, settings.timeout)
		}

		if err := poll.Wait(ctx, settings.interval); err != nil {
			return statusResponse{}, err
		}
	}
}

func timeoutError(processID string, status string, timeout time.Duration) error {
	if status == "" {
		status = "unknown"
	}

	return fmt.Errorf(
		"%w after %dms (process %s, last status %s); increase the pollTimeoutMillis provider option if the job needs longer",
		ErrImageEnhancementTimeout,
		timeout.Milliseconds(),
		processID,
		status,
	)
}

// cancelQuietly asks Topaz to cancel the job. It is detached from the
// caller's cancellation, which may be why polling stopped, and its failure
// is ignored in favor of the original error.
func (m *ImageModel) cancelQuietly(ctx context.Context, processID string, headers http.Header) {
	cancelCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), cancelTimeout)
	defer cancel()

	request, err := http.NewRequestWithContext(cancelCtx, http.MethodDelete, m.provider.baseURL+cancelPath+url.PathEscape(processID), nil)
	if err != nil {
		return
	}

	request.Header = headers.Clone()

	response, err := m.provider.client().Do(request)
	if err != nil {
		return
	}

	_ = response.Body.Close()
}

// downloadImage resolves the download URL of the job and downloads the
// image. The URL normally points at object storage, so the API key is sent
// only to the API origin.
func (m *ImageModel) downloadImage(ctx context.Context, processID string, headers http.Header) ([]byte, http.Header, error) {
	response, err := apicall.Get(ctx, m.provider.apiRequest(downloadPath+url.PathEscape(processID), headers))
	if err != nil {
		return nil, nil, err
	}

	var decoded downloadResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, nil, err
	}

	if decoded.DownloadURL == nil {
		return nil, nil, fmt.Errorf("%w: Topaz did not return a download URL for process %s", model.ErrInvalidResponse, processID)
	}

	image, err := download.Fetch(ctx, download.Request{
		URL:                *decoded.DownloadURL,
		Headers:            headers,
		CredentialedOrigin: m.provider.baseURL,
		TrustedOrigin:      m.provider.baseURL,
		Client:             m.provider.httpClient,
	})
	if err != nil {
		return nil, nil, err
	}

	return image.Data, response.Headers, nil
}

var _ model.ImageModel = (*ImageModel)(nil)
