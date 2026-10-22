package fal

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/internal/download"
	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// aspectRatioSizes maps aspect ratios onto fal.ai image sizes.
var aspectRatioSizes = map[string]imageSize{
	"1:1": {
		Preset: "square_hd",
	},
	"16:9": {
		Preset: "landscape_16_9",
	},
	"9:16": {
		Preset: "portrait_16_9",
	},
	"4:3": {
		Preset: "landscape_4_3",
	},
	"3:4": {
		Preset: "portrait_4_3",
	},
	"16:10": {
		Width:  1280,
		Height: 800,
	},
	"10:16": {
		Width:  800,
		Height: 1280,
	},
	"21:9": {
		Width:  2560,
		Height: 1080,
	},
	"9:21": {
		Width:  1080,
		Height: 2560,
	},
}

// imageResponseOmittedKeys are response members that are not copied into
// the provider metadata: the images themselves, the echoed prompt and the
// NSFW flags, which move into each image's metadata.
var imageResponseOmittedKeys = []string{
	"images",
	"image",
	"prompt",
	"has_nsfw_concepts",
	"nsfw_content_detected",
}

// ImageMetadata is the fal.ai metadata of an image response, stored under
// ProviderMetadata["fal"]. Response members other than the images, such as
// timings and seed, are kept in Response.
type ImageMetadata struct {
	Images []ImageInfo

	// Response holds the other members of the model's response.
	Response map[string]json.RawMessage
}

// MarshalJSON writes the images and the response members as one object.
func (m ImageMetadata) MarshalJSON() ([]byte, error) {
	members := make(map[string]json.RawMessage, len(m.Response)+1)
	maps.Copy(members, m.Response)

	images, err := json.Marshal(m.Images)
	if err != nil {
		return nil, err
	}

	members["images"] = images

	return json.Marshal(members)
}

// UnmarshalJSON reads the images and keeps the other members in Response.
func (m *ImageMetadata) UnmarshalJSON(data []byte) error {
	var members map[string]json.RawMessage
	if err := json.Unmarshal(data, &members); err != nil {
		return err
	}

	var images []ImageInfo
	if encoded, ok := members["images"]; ok {
		if err := json.Unmarshal(encoded, &images); err != nil {
			return fmt.Errorf("decode images: %w", err)
		}
	}

	delete(members, "images")

	*m = ImageMetadata{
		Images:   images,
		Response: members,
	}

	return nil
}

// ImageInfo describes one generated image. Nil fields were not reported.
type ImageInfo struct {
	Width       *float64 `json:"width,omitempty"`
	Height      *float64 `json:"height,omitempty"`
	ContentType *string  `json:"contentType,omitempty"`
	FileName    *string  `json:"fileName,omitempty"`
	FileData    *string  `json:"fileData,omitempty"`
	FileSize    *float64 `json:"fileSize,omitempty"`

	// NSFW reports whether the image was flagged.
	NSFW *bool `json:"nsfw,omitempty"`
}

// ImageModel is a fal.ai image model. Calls with input files edit those
// images.
type ImageModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "fal.image" by default.
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

// imageSize is a preset name or explicit dimensions.
type imageSize struct {
	Preset string
	Width  float64
	Height float64
}

// MarshalJSON writes the preset name or a dimensions object.
func (s imageSize) MarshalJSON() ([]byte, error) {
	if s.Preset != "" {
		return json.Marshal(s.Preset)
	}

	return json.Marshal(imageDimensions{
		Width:  s.Width,
		Height: s.Height,
	})
}

type imageDimensions struct {
	Width  float64 `json:"width"`
	Height float64 `json:"height"`
}

type imageRequest struct {
	Prompt    string     `json:"prompt,omitempty"`
	Seed      *int       `json:"seed,omitempty"`
	ImageSize *imageSize `json:"image_size,omitempty"`
	NumImages int        `json:"num_images,omitempty"`
	ImageURL  string     `json:"image_url,omitempty"`
	ImageURLs []string   `json:"image_urls,omitempty"`
	MaskURL   string     `json:"mask_url,omitempty"`
}

// imageResponse holds the images of a response. Most models return an
// "images" array; some return a single "image".
type imageResponse struct {
	Images              []falImage `json:"images"`
	Image               *falImage  `json:"image"`
	HasNSFWConcepts     []bool     `json:"has_nsfw_concepts"`
	NSFWContentDetected []bool     `json:"nsfw_content_detected"`
}

type falImage struct {
	URL         string   `json:"url"`
	Width       *float64 `json:"width"`
	Height      *float64 `json:"height"`
	ContentType *string  `json:"content_type"`
	FileName    *string  `json:"file_name"`
	FileData    *string  `json:"file_data"`
	FileSize    *float64 `json:"file_size"`
}

// Generate generates or edits an image and downloads it.
func (m *ImageModel) Generate(ctx context.Context, options model.ImageOptions) (*model.ImageResult, error) {
	timestamp := time.Now()

	requestOptions, err := readImageOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	body, warnings, err := imageBody(options, requestOptions)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(m.provider.baseURL+"/"+m.modelID, options.Headers, parseImageError)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, err
	}

	decoded, rest, err := decodeImageResponse(response.Body)
	if err != nil {
		return nil, err
	}

	images, err := m.downloadImages(ctx, decoded.Images)
	if err != nil {
		return nil, err
	}

	return &model.ImageResult{
		Images:   images,
		Warnings: warnings,
		ProviderMetadata: model.ProviderMetadata{
			Namespace: ImageMetadata{
				Images:   imageInfos(decoded),
				Response: rest,
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

// imageBody builds the request. Mapped provider options replace the
// standard fields.
func imageBody(options model.ImageOptions, requestOptions imageRequestOptions) (map[string]json.RawMessage, []model.Warning, error) {
	var warnings []model.Warning

	body := imageRequest{
		Prompt:    options.Prompt,
		Seed:      options.Seed,
		ImageSize: requestedImageSize(options),
		NumImages: options.N,
	}

	if len(options.Files) > 0 {
		fileWarnings, err := addInputImages(&body, options.Files, requestOptions.useMultipleImages)
		if err != nil {
			return nil, nil, err
		}

		warnings = append(warnings, fileWarnings...)
	}

	if options.Mask != nil {
		mask, err := dataurl.ImageFileURL(options.Mask)
		if err != nil {
			return nil, nil, err
		}

		body.MaskURL = mask
	}

	if len(requestOptions.deprecatedKeys) > 0 {
		warnings = append(warnings, deprecationWarning(requestOptions.deprecatedKeys))
	}

	members, err := jsonobject.Members(body)
	if err != nil {
		return nil, nil, fmt.Errorf("encode request body: %w", err)
	}

	maps.Copy(members, requestOptions.members)

	return members, warnings, nil
}

// requestedImageSize converts the size or aspect ratio. Unknown aspect
// ratios leave the model default.
func requestedImageSize(options model.ImageOptions) *imageSize {
	if options.Size != "" {
		widthText, heightText, _ := strings.Cut(options.Size, "x")

		width, _ := strconv.ParseFloat(widthText, 64)
		height, _ := strconv.ParseFloat(heightText, 64)

		return &imageSize{
			Width:  width,
			Height: height,
		}
	}

	if size, ok := aspectRatioSizes[options.AspectRatio]; ok {
		return &size
	}

	return nil
}

// addInputImages sends the input files as "image_urls" or, by default, the
// first file as "image_url".
func addInputImages(body *imageRequest, files []model.ImageFile, useMultipleImages bool) ([]model.Warning, error) {
	if !useMultipleImages {
		uri, err := dataurl.ImageFileURL(files[0])
		if err != nil {
			return nil, err
		}

		body.ImageURL = uri

		if len(files) == 1 {
			return nil, nil
		}

		return []model.Warning{
			{
				Type: model.WarningOther,
				Message: "Multiple input images provided but useMultipleImages is not enabled. " +
					"Only the first image will be used. Set providerOptions.fal.useMultipleImages " +
					"to true for models that support multiple images (e.g., fal-ai/flux-2/edit).",
			},
		}, nil
	}

	for _, file := range files {
		uri, err := dataurl.ImageFileURL(file)
		if err != nil {
			return nil, err
		}

		body.ImageURLs = append(body.ImageURLs, uri)
	}

	return nil, nil
}

// decodeImageResponse decodes the images and returns the response members
// that become provider metadata.
func decodeImageResponse(body []byte) (imageResponse, map[string]json.RawMessage, error) {
	var decoded imageResponse
	if err := apicall.DecodeJSON(body, &decoded); err != nil {
		return imageResponse{}, nil, err
	}

	if decoded.Images == nil {
		if decoded.Image == nil {
			return imageResponse{}, nil, fmt.Errorf("%w: fal response has no images", model.ErrInvalidResponse)
		}

		decoded.Images = []falImage{
			*decoded.Image,
		}
	}

	var rest map[string]json.RawMessage
	if err := apicall.DecodeJSON(body, &rest); err != nil {
		return imageResponse{}, nil, err
	}

	for _, key := range imageResponseOmittedKeys {
		delete(rest, key)
	}

	return decoded, rest, nil
}

func imageInfos(decoded imageResponse) []ImageInfo {
	infos := make([]ImageInfo, 0, len(decoded.Images))

	for index, image := range decoded.Images {
		infos = append(infos, ImageInfo{
			Width:       image.Width,
			Height:      image.Height,
			ContentType: image.ContentType,
			FileName:    image.FileName,
			FileData:    image.FileData,
			FileSize:    image.FileSize,
			NSFW:        nsfwFlag(decoded, index),
		})
	}

	return infos
}

// nsfwFlag returns the image's NSFW flag from either reported list.
func nsfwFlag(decoded imageResponse, index int) *bool {
	if index < len(decoded.HasNSFWConcepts) {
		return new(decoded.HasNSFWConcepts[index])
	}

	if index < len(decoded.NSFWContentDetected) {
		return new(decoded.NSFWContentDetected[index])
	}

	return nil
}

// downloadImages downloads the generated images without credentials.
func (m *ImageModel) downloadImages(ctx context.Context, images []falImage) ([][]byte, error) {
	downloaded := make([][]byte, 0, len(images))

	for _, image := range images {
		file, err := download.Fetch(ctx, download.Request{
			URL:           image.URL,
			TrustedOrigin: m.provider.baseURL,
			Client:        m.provider.httpClient,
		})
		if err != nil {
			return nil, err
		}

		downloaded = append(downloaded, file.Data)
	}

	return downloaded, nil
}

var _ model.ImageModel = (*ImageModel)(nil)
