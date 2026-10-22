package middleware

import (
	"context"
	"regexp"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// WrapLanguageModelOptions configures WrapLanguageModel.
type WrapLanguageModelOptions struct {
	Model model.LanguageModel

	// Middleware is applied in order: the first middleware transforms the
	// input first, and the last one wraps the model directly.
	Middleware []Language

	// ModelID and ProviderID optionally replace the wrapped model's
	// identity, taking precedence over the middleware overrides.
	ModelID    string
	ProviderID string
}

// WrapLanguageModel returns options.Model with the middleware applied.
func WrapLanguageModel(options WrapLanguageModelOptions) model.LanguageModel {
	wrapped := options.Model

	for _, middleware := range slices.Backward(options.Middleware) {
		wrapped = &languageModel{
			modelIdentity: modelIdentity[model.LanguageModel]{
				base:             wrapped,
				providerID:       options.ProviderID,
				modelID:          options.ModelID,
				overrideProvider: middleware.OverrideProvider,
				overrideModelID:  middleware.OverrideModelID,
			},
			base:       wrapped,
			middleware: middleware,
		}
	}

	return wrapped
}

type languageModel struct {
	modelIdentity[model.LanguageModel]

	base       model.LanguageModel
	middleware Language
}

func (m *languageModel) SupportedURLs() map[string][]*regexp.Regexp {
	if m.middleware.OverrideSupportedURLs != nil {
		return m.middleware.OverrideSupportedURLs(m.base)
	}

	return m.base.SupportedURLs()
}

func (m *languageModel) Generate(ctx context.Context, options model.CallOptions) (*model.GenerateResult, error) {
	call, err := m.prepareCall(ctx, CallGenerate, options)
	if err != nil {
		return nil, err
	}

	if m.middleware.WrapGenerate == nil {
		return call.DoGenerate(ctx)
	}

	return m.middleware.WrapGenerate(ctx, call)
}

func (m *languageModel) Stream(ctx context.Context, options model.CallOptions) (*model.StreamResult, error) {
	call, err := m.prepareCall(ctx, CallStream, options)
	if err != nil {
		return nil, err
	}

	if m.middleware.WrapStream == nil {
		return call.DoStream(ctx)
	}

	return m.middleware.WrapStream(ctx, call)
}

func (m *languageModel) prepareCall(ctx context.Context, callType CallType, options model.CallOptions) (LanguageCall, error) {
	params := options

	if m.middleware.TransformParams != nil {
		transformed, err := m.middleware.TransformParams(ctx, LanguageTransform{
			Type:   callType,
			Params: options,
			Model:  m.base,
		})
		if err != nil {
			return LanguageCall{}, err
		}

		params = transformed
	}

	return LanguageCall{
		DoGenerate: func(ctx context.Context) (*model.GenerateResult, error) {
			return m.base.Generate(ctx, params)
		},
		DoStream: func(ctx context.Context) (*model.StreamResult, error) {
			return m.base.Stream(ctx, params)
		},
		Params: params,
		Model:  m.base,
	}, nil
}

// WrapEmbeddingModelOptions configures WrapEmbeddingModel.
type WrapEmbeddingModelOptions struct {
	Model model.EmbeddingModel

	// Middleware is applied in order: the first middleware transforms the
	// input first, and the last one wraps the model directly.
	Middleware []Embedding

	// ModelID and ProviderID optionally replace the wrapped model's
	// identity, taking precedence over the middleware overrides.
	ModelID    string
	ProviderID string
}

// WrapEmbeddingModel returns options.Model with the middleware applied.
func WrapEmbeddingModel(options WrapEmbeddingModelOptions) model.EmbeddingModel {
	wrapped := options.Model

	for _, middleware := range slices.Backward(options.Middleware) {
		wrapped = &embeddingModel{
			modelIdentity: modelIdentity[model.EmbeddingModel]{
				base:             wrapped,
				providerID:       options.ProviderID,
				modelID:          options.ModelID,
				overrideProvider: middleware.OverrideProvider,
				overrideModelID:  middleware.OverrideModelID,
			},
			base:       wrapped,
			middleware: middleware,
		}
	}

	return wrapped
}

type embeddingModel struct {
	modelIdentity[model.EmbeddingModel]

	base       model.EmbeddingModel
	middleware Embedding
}

func (m *embeddingModel) MaxEmbeddingsPerCall() int {
	if m.middleware.OverrideMaxEmbeddingsPerCall != nil {
		return m.middleware.OverrideMaxEmbeddingsPerCall(m.base)
	}

	return m.base.MaxEmbeddingsPerCall()
}

func (m *embeddingModel) SupportsParallelCalls() bool {
	if m.middleware.OverrideSupportsParallelCalls != nil {
		return m.middleware.OverrideSupportsParallelCalls(m.base)
	}

	return m.base.SupportsParallelCalls()
}

func (m *embeddingModel) Embed(ctx context.Context, options model.EmbedOptions) (*model.EmbedResult, error) {
	params := options

	if m.middleware.TransformParams != nil {
		transformed, err := m.middleware.TransformParams(ctx, EmbeddingTransform{
			Params: options,
			Model:  m.base,
		})
		if err != nil {
			return nil, err
		}

		params = transformed
	}

	doEmbed := func(ctx context.Context) (*model.EmbedResult, error) {
		return m.base.Embed(ctx, params)
	}

	if m.middleware.WrapEmbed == nil {
		return doEmbed(ctx)
	}

	return m.middleware.WrapEmbed(ctx, EmbeddingCall{
		DoEmbed: doEmbed,
		Params:  params,
		Model:   m.base,
	})
}

// WrapImageModelOptions configures WrapImageModel.
type WrapImageModelOptions struct {
	Model model.ImageModel

	// Middleware is applied in order: the first middleware transforms the
	// input first, and the last one wraps the model directly.
	Middleware []Image

	// ModelID and ProviderID optionally replace the wrapped model's
	// identity, taking precedence over the middleware overrides.
	ModelID    string
	ProviderID string
}

// WrapImageModel returns options.Model with the middleware applied.
func WrapImageModel(options WrapImageModelOptions) model.ImageModel {
	wrapped := options.Model

	for _, middleware := range slices.Backward(options.Middleware) {
		wrapped = &imageModel{
			modelIdentity: modelIdentity[model.ImageModel]{
				base:             wrapped,
				providerID:       options.ProviderID,
				modelID:          options.ModelID,
				overrideProvider: middleware.OverrideProvider,
				overrideModelID:  middleware.OverrideModelID,
			},
			base:       wrapped,
			middleware: middleware,
		}
	}

	return wrapped
}

type imageModel struct {
	modelIdentity[model.ImageModel]

	base       model.ImageModel
	middleware Image
}

func (m *imageModel) MaxImagesPerCall() int {
	if m.middleware.OverrideMaxImagesPerCall != nil {
		return m.middleware.OverrideMaxImagesPerCall(m.base)
	}

	return m.base.MaxImagesPerCall()
}

func (m *imageModel) Generate(ctx context.Context, options model.ImageOptions) (*model.ImageResult, error) {
	params := options

	if m.middleware.TransformParams != nil {
		transformed, err := m.middleware.TransformParams(ctx, ImageTransform{
			Params: options,
			Model:  m.base,
		})
		if err != nil {
			return nil, err
		}

		params = transformed
	}

	doGenerate := func(ctx context.Context) (*model.ImageResult, error) {
		return m.base.Generate(ctx, params)
	}

	if m.middleware.WrapGenerate == nil {
		return doGenerate(ctx)
	}

	return m.middleware.WrapGenerate(ctx, ImageCall{
		DoGenerate: doGenerate,
		Params:     params,
		Model:      m.base,
	})
}

// WrapProviderOptions configures WrapProvider.
type WrapProviderOptions struct {
	Provider model.Provider

	// LanguageModelMiddleware wraps every language model of the provider.
	LanguageModelMiddleware []Language

	// ImageModelMiddleware wraps every image model of the provider.
	ImageModelMiddleware []Image
}

// WrapProvider returns a provider whose language and image models have the
// middleware applied. Other model types are returned unchanged.
func WrapProvider(options WrapProviderOptions) model.Provider {
	return &provider{
		Provider:                options.Provider,
		languageModelMiddleware: slices.Clone(options.LanguageModelMiddleware),
		imageModelMiddleware:    slices.Clone(options.ImageModelMiddleware),
	}
}

// provider embeds the wrapped provider for the model types that pass through
// unchanged.
type provider struct {
	model.Provider

	languageModelMiddleware []Language
	imageModelMiddleware    []Image
}

func (p *provider) LanguageModel(modelID string) (model.LanguageModel, error) {
	base, err := p.Provider.LanguageModel(modelID)
	if err != nil {
		return nil, err
	}

	return WrapLanguageModel(WrapLanguageModelOptions{
		Model:      base,
		Middleware: p.languageModelMiddleware,
	}), nil
}

func (p *provider) ImageModel(modelID string) (model.ImageModel, error) {
	base, err := p.Provider.ImageModel(modelID)
	if err != nil {
		return nil, err
	}

	return WrapImageModel(WrapImageModelOptions{
		Model:      base,
		Middleware: p.imageModelMiddleware,
	}), nil
}
