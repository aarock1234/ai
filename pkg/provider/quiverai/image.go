package quiverai

import (
	"context"
	"fmt"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// maxImagesPerCall is the per-call SVG limit.
const maxImagesPerCall = 16

// svgMediaType is the media type of every generated document.
const svgMediaType = "image/svg+xml"

// ImageMetadata is the QuiverAI metadata of an image response, stored under
// ProviderMetadata["quiverai"].
type ImageMetadata struct {
	// Credits are the credits charged, when reported.
	Credits *int `json:"credits,omitempty"`

	Images []ImageInfo `json:"images"`
}

// ImageInfo describes one generated SVG.
type ImageInfo struct {
	Index    int    `json:"index"`
	MimeType string `json:"mimeType"`

	// LoopPeriodMs and OpeningAnimationMs describe animated SVGs.
	LoopPeriodMs       *int `json:"loopPeriodMs,omitempty"`
	OpeningAnimationMs *int `json:"openingAnimationMs,omitempty"`
}

// ImageModel is a QuiverAI SVG model. Images are SVG documents; set the
// operation option to vectorize, animate or edit input images.
type ImageModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "quiverai.image" by default.
func (m *ImageModel) Provider() string {
	return m.provider.name + ".image"
}

// ModelID returns the model identifier.
func (m *ImageModel) ModelID() string {
	return m.modelID
}

// MaxImagesPerCall returns 16.
func (m *ImageModel) MaxImagesPerCall() int {
	return maxImagesPerCall
}

type svgResponse struct {
	ID      string        `json:"id"`
	Created int64         `json:"created"`
	Data    []svgDocument `json:"data"`
	Usage   *svgUsage     `json:"usage"`
	Credits *int          `json:"credits"`
}

type svgDocument struct {
	SVG                string `json:"svg"`
	MimeType           string `json:"mime_type"`
	LoopPeriodMs       *int   `json:"loop_period_ms"`
	OpeningAnimationMs *int   `json:"opening_animation_ms"`
}

type svgUsage struct {
	TotalTokens  int `json:"total_tokens"`
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// Generate runs the configured operation and returns the SVG documents.
func (m *ImageModel) Generate(ctx context.Context, options model.ImageOptions) (*model.ImageResult, error) {
	timestamp := time.Now()

	imageOptions, err := readImageOptions(options.ProviderOptions)
	if err != nil {
		return nil, err
	}

	operation := OperationGenerate
	if imageOptions.Operation != nil {
		operation = *imageOptions.Operation
	}

	warnings := unsupportedImageSettings(options)

	body, err := requestBody(m.modelID, options, imageOptions, operation)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(operationPath(operation), options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, body)
	if err != nil {
		return nil, err
	}

	var decoded svgResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	if err := decoded.validate(); err != nil {
		return nil, err
	}

	if decoded.Created != 0 {
		timestamp = time.Unix(decoded.Created, 0)
	}

	return &model.ImageResult{
		Images:   svgImages(decoded.Data),
		Usage:    imageUsage(decoded.Usage),
		Warnings: warnings,
		ProviderMetadata: model.ProviderMetadata{
			Namespace: imageMetadata(decoded),
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

func unsupportedImageSettings(options model.ImageOptions) []model.Warning {
	var warnings []model.Warning

	if options.Size != "" {
		warnings = append(warnings, model.UnsupportedWarning(
			"size",
			"QuiverAI SVG generation does not support the `size` option. The setting was ignored.",
		))
	}

	if options.AspectRatio != "" {
		warnings = append(warnings, model.UnsupportedWarning(
			"aspectRatio",
			"QuiverAI SVG generation does not support the `aspectRatio` option. The setting was ignored.",
		))
	}

	if options.Seed != nil {
		warnings = append(warnings, model.UnsupportedWarning(
			"seed",
			"QuiverAI SVG generation does not support the `seed` option. The setting was ignored.",
		))
	}

	if options.Mask != nil {
		warnings = append(warnings, model.UnsupportedWarning(
			"mask",
			"QuiverAI SVG generation does not support masks. The mask was ignored.",
		))
	}

	return warnings
}

// validate checks the documented response constraints.
func (r svgResponse) validate() error {
	if r.ID == "" || r.Created < 0 || len(r.Data) == 0 {
		return fmt.Errorf("%w: QuiverAI response must have an id, a creation time and at least one SVG", model.ErrInvalidResponse)
	}

	for index, document := range r.Data {
		if document.SVG == "" || document.MimeType != svgMediaType {
			return fmt.Errorf("%w: QuiverAI SVG %d must be non-empty with media type %s", model.ErrInvalidResponse, index, svgMediaType)
		}
	}

	return nil
}

func svgImages(documents []svgDocument) [][]byte {
	images := make([][]byte, 0, len(documents))
	for _, document := range documents {
		images = append(images, []byte(document.SVG))
	}

	return images
}

func imageUsage(usage *svgUsage) *model.ImageUsage {
	if usage == nil {
		return nil
	}

	return &model.ImageUsage{
		InputTokens:  new(usage.InputTokens),
		OutputTokens: new(usage.OutputTokens),
		TotalTokens:  new(usage.TotalTokens),
	}
}

func imageMetadata(response svgResponse) ImageMetadata {
	infos := make([]ImageInfo, 0, len(response.Data))
	for index, document := range response.Data {
		infos = append(infos, ImageInfo{
			Index:              index,
			MimeType:           document.MimeType,
			LoopPeriodMs:       document.LoopPeriodMs,
			OpeningAnimationMs: document.OpeningAnimationMs,
		})
	}

	return ImageMetadata{
		Credits: response.Credits,
		Images:  infos,
	}
}

var _ model.ImageModel = (*ImageModel)(nil)
