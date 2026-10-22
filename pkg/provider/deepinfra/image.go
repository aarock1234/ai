package deepinfra

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// imageEditPath is the OpenAI-compatible image edit endpoint.
const imageEditPath = "/images/edits"

// dataURLPrefix matches the data URL prefix of a generated image.
var dataURLPrefix = regexp.MustCompile(`^data:image/\w+;base64,`)

// ImageModel is a DeepInfra image model. Calls with input files edit images
// through the OpenAI-compatible endpoint; other calls generate images through
// the native inference API.
type ImageModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "deepinfra.image" by default.
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

// generationRequest holds the standard members of a generation request. The
// provider options are added after them.
type generationRequest struct {
	Prompt      string `json:"prompt,omitempty"`
	NumImages   int    `json:"num_images,omitempty"`
	AspectRatio string `json:"aspect_ratio,omitempty"`
	Width       string `json:"width,omitempty"`
	Height      string `json:"height,omitempty"`
	Seed        *int   `json:"seed,omitempty"`
}

type generationResponse struct {
	Images []string `json:"images"`
}

type editResponse struct {
	Data []editImage `json:"data"`
}

type editImage struct {
	B64JSON string `json:"b64_json"`
}

// Generate generates images, or edits the input files when there are any.
// Some models accept a size and others an aspect ratio; both are sent and
// the API validates them.
func (m *ImageModel) Generate(ctx context.Context, options model.ImageOptions) (*model.ImageResult, error) {
	timestamp := time.Now()

	imageOptions, err := readImageOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	var images [][]byte
	var headers http.Header

	if len(options.Files) > 0 {
		images, headers, err = m.edit(ctx, options, imageOptions)
	} else {
		images, headers, err = m.generate(ctx, options, imageOptions)
	}

	if err != nil {
		return nil, err
	}

	return &model.ImageResult{
		Images: images,
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: timestamp,
				ModelID:   m.modelID,
			},
			Headers: headers,
		},
	}, nil
}

// generate posts a JSON request to the model's inference endpoint.
func (m *ImageModel) generate(
	ctx context.Context,
	options model.ImageOptions,
	imageOptions jsonobject.Object,
) ([][]byte, http.Header, error) {
	width, height, _ := strings.Cut(options.Size, "x")

	encoded, err := json.Marshal(generationRequest{
		Prompt:      options.Prompt,
		NumImages:   options.N,
		AspectRatio: options.AspectRatio,
		Width:       width,
		Height:      height,
		Seed:        options.Seed,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("encode request body: %w", err)
	}

	head, err := jsonobject.Parse(encoded)
	if err != nil {
		return nil, nil, fmt.Errorf("encode request body: %w", err)
	}

	request, err := m.apiRequest(m.inferenceURL(), options.Headers, parseGenerationError)
	if err != nil {
		return nil, nil, err
	}

	response, err := apicall.PostJSON(ctx, request, head.Merge(imageOptions))
	if err != nil {
		return nil, nil, err
	}

	var decoded generationResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, nil, err
	}

	images := make([][]byte, 0, len(decoded.Images))
	for _, image := range decoded.Images {
		decodedImage, err := decodeImage(dataURLPrefix.ReplaceAllString(image, ""))
		if err != nil {
			return nil, nil, err
		}

		images = append(images, decodedImage)
	}

	return images, response.Headers, nil
}

// edit posts a multipart request to the OpenAI-compatible edit endpoint.
func (m *ImageModel) edit(
	ctx context.Context,
	options model.ImageOptions,
	imageOptions jsonobject.Object,
) ([][]byte, http.Header, error) {
	request, err := m.apiRequest(m.editURL(), options.Headers, parseEditError)
	if err != nil {
		return nil, nil, err
	}

	fields, err := editFormFields(ctx, request.Client, m.modelID, options, imageOptions)
	if err != nil {
		return nil, nil, err
	}

	response, err := apicall.PostForm(ctx, request, fields)
	if err != nil {
		return nil, nil, err
	}

	var decoded editResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, nil, err
	}

	images := make([][]byte, 0, len(decoded.Data))
	for _, item := range decoded.Data {
		image, err := decodeImage(item.B64JSON)
		if err != nil {
			return nil, nil, err
		}

		images = append(images, image)
	}

	return images, response.Headers, nil
}

// inferenceURL returns the native inference endpoint of the model.
func (m *ImageModel) inferenceURL() string {
	return m.provider.baseURL + inferencePrefix + "/" + m.modelID
}

// editURL returns the OpenAI-compatible edit endpoint, derived from the
// inference base URL.
func (m *ImageModel) editURL() string {
	inferenceBase := m.provider.baseURL + inferencePrefix

	return strings.Replace(inferenceBase, inferencePrefix, openAIPrefix, 1) + imageEditPath
}

func (m *ImageModel) apiRequest(url string, callHeaders http.Header, parse apicall.ErrorParser) (apicall.Request, error) {
	headers, err := m.provider.requestHeaders()
	if err != nil {
		return apicall.Request{}, err
	}

	return apicall.Request{
		Client:     m.provider.httpClient,
		URL:        url,
		Headers:    apicall.MergeHeaders(headers, callHeaders),
		ParseError: parse,
	}, nil
}

func decodeImage(encoded string) ([]byte, error) {
	image, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, fmt.Errorf("%w: decode base64 image: %w", model.ErrInvalidResponse, err)
	}

	return image, nil
}

// generationError is the error body of the inference API.
type generationError struct {
	Detail struct {
		Error string `json:"error"`
	} `json:"detail"`
}

// parseGenerationError reads the message of an inference API error.
func parseGenerationError(body []byte) string {
	var data generationError
	if err := json.Unmarshal(body, &data); err != nil {
		return ""
	}

	return data.Detail.Error
}

// editError is the error body of the OpenAI-compatible edit endpoint.
type editError struct {
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// parseEditError reads the message of an edit endpoint error. A JSON body
// without a message reports an unknown error.
func parseEditError(body []byte) string {
	var data editError
	if err := json.Unmarshal(body, &data); err != nil {
		return ""
	}

	if data.Error == nil {
		return "unknown error"
	}

	return data.Error.Message
}

var _ model.ImageModel = (*ImageModel)(nil)
