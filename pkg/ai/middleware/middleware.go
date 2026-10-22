// Package middleware wraps models to change their inputs and outputs, a port
// of the Vercel AI SDK middleware. A middleware is a struct of optional
// functions: TransformParams rewrites the call options, and the Wrap
// functions surround the model call. WrapLanguageModel, WrapEmbeddingModel,
// WrapImageModel and WrapProvider apply middleware; the remaining functions
// build ready-made middleware.
package middleware

import (
	"context"
	"regexp"

	"github.com/aarock1234/ai/pkg/model"
)

// CallType names the language model operation being transformed.
type CallType string

// Call types.
const (
	CallGenerate CallType = "generate"
	CallStream   CallType = "stream"
)

// Language is middleware for language models. Every field is optional.
type Language struct {
	// OverrideProvider replaces the provider name of the wrapped model.
	OverrideProvider func(base model.LanguageModel) string

	// OverrideModelID replaces the model ID of the wrapped model.
	OverrideModelID func(base model.LanguageModel) string

	// OverrideSupportedURLs replaces the URLs the wrapped model downloads
	// itself.
	OverrideSupportedURLs func(base model.LanguageModel) map[string][]*regexp.Regexp

	// TransformParams rewrites the call options before the call.
	TransformParams func(ctx context.Context, transform LanguageTransform) (model.CallOptions, error)

	// WrapGenerate surrounds a non-streaming call.
	WrapGenerate func(ctx context.Context, call LanguageCall) (*model.GenerateResult, error)

	// WrapStream surrounds a streaming call.
	WrapStream func(ctx context.Context, call LanguageCall) (*model.StreamResult, error)
}

// LanguageTransform is the input of Language.TransformParams.
type LanguageTransform struct {
	Type   CallType
	Params model.CallOptions
	Model  model.LanguageModel
}

// LanguageCall is the input of Language.WrapGenerate and Language.WrapStream.
// DoGenerate and DoStream call the wrapped model with Params, so a wrapper
// may serve a streaming call with a non-streaming one and vice versa.
type LanguageCall struct {
	DoGenerate func(ctx context.Context) (*model.GenerateResult, error)
	DoStream   func(ctx context.Context) (*model.StreamResult, error)

	// Params are the call options after TransformParams.
	Params model.CallOptions

	Model model.LanguageModel
}

// Embedding is middleware for embedding models. Every field is optional.
type Embedding struct {
	OverrideProvider              func(base model.EmbeddingModel) string
	OverrideModelID               func(base model.EmbeddingModel) string
	OverrideMaxEmbeddingsPerCall  func(base model.EmbeddingModel) int
	OverrideSupportsParallelCalls func(base model.EmbeddingModel) bool

	// TransformParams rewrites the call options before the call.
	TransformParams func(ctx context.Context, transform EmbeddingTransform) (model.EmbedOptions, error)

	// WrapEmbed surrounds the call.
	WrapEmbed func(ctx context.Context, call EmbeddingCall) (*model.EmbedResult, error)
}

// EmbeddingTransform is the input of Embedding.TransformParams.
type EmbeddingTransform struct {
	Params model.EmbedOptions
	Model  model.EmbeddingModel
}

// EmbeddingCall is the input of Embedding.WrapEmbed.
type EmbeddingCall struct {
	DoEmbed func(ctx context.Context) (*model.EmbedResult, error)
	Params  model.EmbedOptions
	Model   model.EmbeddingModel
}

// Image is middleware for image models. Every field is optional.
type Image struct {
	OverrideProvider         func(base model.ImageModel) string
	OverrideModelID          func(base model.ImageModel) string
	OverrideMaxImagesPerCall func(base model.ImageModel) int

	// TransformParams rewrites the call options before the call.
	TransformParams func(ctx context.Context, transform ImageTransform) (model.ImageOptions, error)

	// WrapGenerate surrounds the call.
	WrapGenerate func(ctx context.Context, call ImageCall) (*model.ImageResult, error)
}

// ImageTransform is the input of Image.TransformParams.
type ImageTransform struct {
	Params model.ImageOptions
	Model  model.ImageModel
}

// ImageCall is the input of Image.WrapGenerate.
type ImageCall struct {
	DoGenerate func(ctx context.Context) (*model.ImageResult, error)
	Params     model.ImageOptions
	Model      model.ImageModel
}
