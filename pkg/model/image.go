package model

import (
	"context"
	"net/http"
)

// ImageModel generates or edits images.
type ImageModel interface {
	Provider() string
	ModelID() string

	// MaxImagesPerCall returns the provider's per-call limit, or zero when
	// there is none.
	MaxImagesPerCall() int

	Generate(ctx context.Context, options ImageOptions) (*ImageResult, error)
}

// ImageOptions are the inputs of one image call. Empty values leave the
// provider default.
type ImageOptions struct {
	Prompt string

	// N is the number of images to generate.
	N int

	// Size has the form "1024x1024".
	Size string

	// AspectRatio has the form "16:9".
	AspectRatio string

	Seed *int

	// Files are input images for editing or variation.
	Files []ImageFile

	// Mask optionally marks the region to edit.
	Mask ImageFile

	Headers         http.Header
	ProviderOptions ProviderOptions
}

// ImageFile is an input image: ImageBytes or ImageURL.
type ImageFile interface {
	isImageFile()
}

// ImageBytes is an inline input image.
type ImageBytes struct {
	MediaType       string
	Data            []byte
	ProviderOptions ProviderOptions
}

// ImageURL is an input image the provider downloads.
type ImageURL struct {
	URL             string
	ProviderOptions ProviderOptions
}

func (ImageBytes) isImageFile() {}
func (ImageURL) isImageFile()   {}

// ImageResult is the outcome of one image call.
type ImageResult struct {
	Images [][]byte

	// Usage is nil when the provider does not report it.
	Usage *ImageUsage

	ProviderMetadata ProviderMetadata
	Warnings         []Warning
	Response         ResponseInfo
}

// ImageUsage reports image token consumption. Nil counts were not reported.
type ImageUsage struct {
	InputTokens  *int
	OutputTokens *int
	TotalTokens  *int
}
