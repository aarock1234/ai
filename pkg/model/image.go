package model

import (
	"context"
	"net/http"
)

// ImageModel is the interface for image generation providers.
type ImageModel interface {
	// Provider returns the provider name (e.g. "openai").
	Provider() string

	// ModelID returns the model identifier.
	ModelID() string

	// DoGenerateImage performs an image generation request.
	DoGenerateImage(ctx context.Context, req *ImageRequest) (*ImageResponse, error)
}

// ImageInput is an input image for image generation or editing.
type ImageInput struct {
	URL      string
	Data     []byte
	MIMEType string
	Filename string
}

// ImageRequest is the normalized request shape for image generation.
type ImageRequest struct {
	Prompt          string
	Images          []ImageInput
	Size            string
	Quality         string
	Background      string
	OutputFormat    string
	OutputCompress  *int
	Count           int
	Headers         http.Header
	ProviderOptions ProviderOptions
}

// GeneratedImage is a single generated image.
type GeneratedImage struct {
	URL           string
	Data          []byte
	MIMEType      string
	RevisedPrompt string
}

// ImageResponse is the normalized response from an image request.
type ImageResponse struct {
	Images   []GeneratedImage
	Usage    Usage
	Metadata ProviderMetadata
}
