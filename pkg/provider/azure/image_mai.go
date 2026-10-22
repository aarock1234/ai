package azure

import (
	"context"
	"encoding/base64"
	"fmt"
	"math"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/download"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// maiMaxImagesPerCall is the number of images a MAI request returns.
	maiMaxImagesPerCall = 1

	// maiMinSide is the smallest width or height MAI accepts, in pixels.
	maiMinSide = 768

	// maiDefaultPixels is the pixel budget of MAI's default 1024x1024 output.
	maiDefaultPixels = 1024 * 1024
)

// ImageMetadata is the Azure metadata of a MAI image response, stored under
// ProviderMetadata["azure"].
type ImageMetadata struct {
	// Images has one entry per generated image.
	Images []ImageResultMetadata `json:"images"`
}

// ImageResultMetadata describes one generated MAI image. Nil fields were not
// reported.
type ImageResultMetadata struct {
	Created     *int64  `json:"created,omitempty"`
	Size        *string `json:"size,omitempty"`
	TextTokens  *int    `json:"textTokens,omitempty"`
	ImageTokens *int    `json:"imageTokens,omitempty"`
}

// maiImageModel generates and edits images with the MAI image API
// (/mai/v1/images/*) on Azure AI Foundry. The API returns one PNG per request
// and takes a width and height instead of a size.
type maiImageModel struct {
	modelID  string
	settings *settings
}

// maiImageRequest is the body of POST /images/generations.
type maiImageRequest struct {
	Model           string `json:"model"`
	Prompt          string `json:"prompt"`
	Width           *int   `json:"width,omitempty"`
	Height          *int   `json:"height,omitempty"`
	AutoAspectRatio *bool  `json:"auto_aspect_ratio,omitempty"`
	WebGrounding    *bool  `json:"web_grounding,omitempty"`
}

// maiImageResponse holds the response fields the model reads.
type maiImageResponse struct {
	Created *int64          `json:"created"`
	Size    *string         `json:"size"`
	Usage   *maiImageUsage  `json:"usage"`
	Data    []maiImageDatum `json:"data"`
}

type maiImageUsage struct {
	InputTextTokens  *int `json:"num_input_text_tokens"`
	InputImageTokens *int `json:"num_input_image_tokens"`
	OutputTokens     *int `json:"num_output_tokens"`
}

type maiImageDatum struct {
	B64JSON string `json:"b64_json"`
}

// imageDimensions is an output size in pixels.
type imageDimensions struct {
	width  int
	height int
}

// maiExchange is one successful MAI response.
type maiExchange struct {
	headers  http.Header
	response maiImageResponse
}

// generate sends one request per requested image, because the API ignores
// the image count.
func (m *maiImageModel) generate(ctx context.Context, options model.ImageOptions, azureOptions ImageOptions) (*model.ImageResult, error) {
	timestamp := time.Now()

	dimensions, warnings := maiDimensions(options)
	warnings = append(unsupportedMAISettings(options), warnings...)

	body := maiImageRequest{
		Model:           m.modelID,
		Prompt:          options.Prompt,
		AutoAspectRatio: azureOptions.AutoAspectRatio,
		WebGrounding:    azureOptions.WebGrounding,
	}

	if dimensions != nil {
		body.Width = &dimensions.width
		body.Height = &dimensions.height
	}

	images, err := maiInputImages(ctx, options.Files)
	if err != nil {
		return nil, err
	}

	exchanges, err := sendAll(ctx, max(1, options.N), func(ctx context.Context) (maiExchange, error) {
		return m.send(ctx, body, images, options.Headers)
	})
	if err != nil {
		return nil, err
	}

	return maiImageResult(exchanges, warnings, timestamp, m.modelID)
}

// unsupportedMAISettings reports the settings MAI ignores.
func unsupportedMAISettings(options model.ImageOptions) []model.Warning {
	var warnings []model.Warning

	if options.Seed != nil {
		warnings = append(warnings, model.UnsupportedWarning("seed", ""))
	}

	if options.Mask != nil {
		warnings = append(warnings, model.UnsupportedWarning("mask", "MAI image edits do not support masks."))
	}

	return warnings
}

// maiDimensions returns the requested size, or the dimensions of the
// requested aspect ratio. A size takes precedence over an aspect ratio.
func maiDimensions(options model.ImageOptions) (*imageDimensions, []model.Warning) {
	size, ok := parseImageSize(options.Size)
	if !ok {
		return nil, []model.Warning{
			model.UnsupportedWarning("size", "Invalid size: "+options.Size+"."),
		}
	}

	if size != nil && options.AspectRatio != "" {
		return size, []model.Warning{
			model.UnsupportedWarning("aspectRatio", "aspectRatio is ignored when size is set."),
		}
	}

	if options.AspectRatio == "" {
		return size, nil
	}

	dimensions, ok := dimensionsForAspectRatio(options.AspectRatio)
	if !ok {
		return nil, []model.Warning{
			model.UnsupportedWarning("aspectRatio", "Invalid aspect ratio: "+options.AspectRatio+"."),
		}
	}

	return &dimensions, nil
}

// parseImageSize parses a "{width}x{height}" size. It returns nil for an
// empty size and reports false for a malformed one.
func parseImageSize(size string) (*imageDimensions, bool) {
	if size == "" {
		return nil, true
	}

	widthText, heightText, found := strings.Cut(size, "x")
	width, widthErr := strconv.Atoi(widthText)
	height, heightErr := strconv.Atoi(heightText)

	if !found || widthErr != nil || heightErr != nil {
		return nil, false
	}

	return &imageDimensions{
		width:  width,
		height: height,
	}, true
}

// dimensionsForAspectRatio keeps the default 1024x1024 pixel budget, raises
// the short side to the 768 pixel minimum and rounds down to multiples of
// 16.
func dimensionsForAspectRatio(aspectRatio string) (imageDimensions, bool) {
	widthText, heightText, _ := strings.Cut(aspectRatio, ":")
	widthRatio, widthErr := strconv.ParseFloat(widthText, 64)
	heightRatio, heightErr := strconv.ParseFloat(heightText, 64)

	if widthErr != nil || heightErr != nil || !(widthRatio > 0 && heightRatio > 0) {
		return imageDimensions{}, false
	}

	ratio := widthRatio / heightRatio
	width := math.Sqrt(maiDefaultPixels * ratio)
	height := width / ratio

	if min(width, height) < maiMinSide {
		if width < height {
			width, height = maiMinSide, maiMinSide/ratio
		} else {
			width, height = maiMinSide*ratio, maiMinSide
		}
	}

	return imageDimensions{
		width:  max(maiMinSide, floorToMultipleOf16(width)),
		height: max(maiMinSide, floorToMultipleOf16(height)),
	}, true
}

func floorToMultipleOf16(value float64) int {
	return int(math.Floor(value/16+1e-9)) * 16
}

// maiInputImage is an input image ready for upload.
type maiInputImage struct {
	mediaType string
	data      []byte
}

// maiInputImages downloads URL images and collects inline images.
func maiInputImages(ctx context.Context, files []model.ImageFile) ([]maiInputImage, error) {
	images := make([]maiInputImage, 0, len(files))

	for _, file := range files {
		switch file := file.(type) {
		case model.ImageBytes:
			images = append(images, maiInputImage{
				mediaType: file.MediaType,
				data:      file.Data,
			})

		case model.ImageURL:
			downloaded, err := download.Fetch(ctx, download.Request{
				URL: file.URL,
			})
			if err != nil {
				return nil, err
			}

			images = append(images, maiInputImage{
				mediaType: downloaded.MediaType,
				data:      downloaded.Data,
			})

		default:
			return nil, fmt.Errorf("%w: unsupported input image %T", model.ErrInvalidArgument, file)
		}
	}

	return images, nil
}

// sendAll runs count requests concurrently and returns their responses in
// order. The first failure cancels the other requests and is returned.
func sendAll(ctx context.Context, count int, send func(context.Context) (maiExchange, error)) ([]maiExchange, error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	exchanges := make([]maiExchange, count)

	var (
		group    sync.WaitGroup
		failure  sync.Once
		firstErr error
	)

	for index := range count {
		group.Go(func() {
			exchange, err := send(ctx)
			if err != nil {
				failure.Do(func() {
					firstErr = err
					cancel()
				})

				return
			}

			exchanges[index] = exchange
		})
	}

	group.Wait()

	if firstErr != nil {
		return nil, firstErr
	}

	return exchanges, nil
}

// send performs one generation, or one edit when there are input images.
func (m *maiImageModel) send(ctx context.Context, body maiImageRequest, images []maiInputImage, callHeaders http.Header) (maiExchange, error) {
	path := "/images/generations"
	if len(images) > 0 {
		path = "/images/edits"
	}

	requestURL, err := m.settings.maiURL(path)
	if err != nil {
		return maiExchange{}, err
	}

	request, err := m.settings.serviceRequest(apiMAI, requestURL, callHeaders)
	if err != nil {
		return maiExchange{}, err
	}

	var response *apicall.Response
	if len(images) > 0 {
		response, err = apicall.PostForm(ctx, request, maiEditForm(body, images))
	} else {
		response, err = apicall.PostJSON(ctx, request, body)
	}

	if err != nil {
		return maiExchange{}, err
	}

	var decoded maiImageResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return maiExchange{}, err
	}

	return maiExchange{
		headers:  response.Headers,
		response: decoded,
	}, nil
}

// maiEditForm returns the multipart fields of an edit request.
func maiEditForm(body maiImageRequest, images []maiInputImage) []apicall.FormField {
	fields := []apicall.FormField{
		formValue("model", body.Model),
		formValue("prompt", body.Prompt),
	}

	if body.Width != nil {
		fields = append(fields, formValue("width", strconv.Itoa(*body.Width)))
	}

	if body.Height != nil {
		fields = append(fields, formValue("height", strconv.Itoa(*body.Height)))
	}

	if body.AutoAspectRatio != nil {
		fields = append(fields, formValue("auto_aspect_ratio", strconv.FormatBool(*body.AutoAspectRatio)))
	}

	if body.WebGrounding != nil {
		fields = append(fields, formValue("web_grounding", strconv.FormatBool(*body.WebGrounding)))
	}

	for index, image := range images {
		extension := "png"
		if image.mediaType == "image/jpeg" {
			extension = "jpg"
		}

		fields = append(fields, apicall.FormField{
			Name:      "image",
			Filename:  fmt.Sprintf("image-%d.%s", index+1, extension),
			MediaType: image.mediaType,
			Data:      image.data,
		})
	}

	return fields
}

func formValue(name string, value string) apicall.FormField {
	return apicall.FormField{
		Name:  name,
		Value: value,
	}
}

// maiImageResult combines the responses of one call.
func maiImageResult(exchanges []maiExchange, warnings []model.Warning, timestamp time.Time, modelID string) (*model.ImageResult, error) {
	var images [][]byte
	var usage *model.ImageUsage
	metadata := []ImageResultMetadata{}

	for _, exchange := range exchanges {
		response := exchange.response

		for _, datum := range response.Data {
			image, err := base64.StdEncoding.DecodeString(datum.B64JSON)
			if err != nil {
				return nil, fmt.Errorf("%w: MAI image data: %w", model.ErrInvalidResponse, err)
			}

			images = append(images, image)
			metadata = append(metadata, imageResultMetadata(response))
		}

		if response.Usage != nil {
			usage = addImageUsage(usage, *response.Usage)
		}
	}

	result := &model.ImageResult{
		Images: images,
		Usage:  usage,
		ProviderMetadata: model.ProviderMetadata{
			Namespace: ImageMetadata{
				Images: metadata,
			},
		},
		Warnings: warnings,
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: timestamp,
				ModelID:   modelID,
			},
		},
	}

	if len(exchanges) > 0 {
		result.Response.Headers = exchanges[0].headers
	}

	return result, nil
}

func imageResultMetadata(response maiImageResponse) ImageResultMetadata {
	metadata := ImageResultMetadata{
		Created: response.Created,
		Size:    response.Size,
	}

	if response.Usage != nil {
		metadata.TextTokens = response.Usage.InputTextTokens
		metadata.ImageTokens = response.Usage.InputImageTokens
	}

	return metadata
}

// addImageUsage adds one response's usage to the running total. Counts stay
// nil until a response reports them.
func addImageUsage(total *model.ImageUsage, usage maiImageUsage) *model.ImageUsage {
	if total == nil {
		total = &model.ImageUsage{}
	}

	inputTokens := addTokens(usage.InputTextTokens, usage.InputImageTokens)

	return &model.ImageUsage{
		InputTokens:  addTokens(total.InputTokens, inputTokens),
		OutputTokens: addTokens(total.OutputTokens, usage.OutputTokens),
		TotalTokens:  addTokens(total.TotalTokens, addTokens(inputTokens, usage.OutputTokens)),
	}
}

// addTokens returns the sum, or nil when neither count is known.
func addTokens(a *int, b *int) *int {
	if a == nil && b == nil {
		return nil
	}

	sum := 0
	if a != nil {
		sum += *a
	}

	if b != nil {
		sum += *b
	}

	return &sum
}
