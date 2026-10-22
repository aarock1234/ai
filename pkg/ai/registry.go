package ai

import (
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/aarock1234/ai/pkg/ai/middleware"
	"github.com/aarock1234/ai/pkg/model"
)

// defaultRegistrySeparator separates the provider ID from the model ID.
const defaultRegistrySeparator = ":"

// RegistryOptions configures NewRegistry.
type RegistryOptions struct {
	// Providers maps a provider ID to its provider.
	Providers map[string]model.Provider

	// Separator splits combined IDs such as "openai:gpt-5". The default is
	// ":".
	Separator string

	// LanguageModelMiddleware wraps every language model the registry
	// returns. The first middleware transforms the input first.
	LanguageModelMiddleware []middleware.Language

	// ImageModelMiddleware wraps every image model the registry returns.
	ImageModelMiddleware []middleware.Image
}

// Registry resolves combined "provider:model" IDs to models of the
// registered providers. It implements model.Provider and is safe for
// concurrent use.
type Registry struct {
	providers               map[string]model.Provider
	separator               string
	languageModelMiddleware []middleware.Language
	imageModelMiddleware    []middleware.Image
}

// NewRegistry returns a registry of the given providers.
func NewRegistry(options RegistryOptions) *Registry {
	separator := options.Separator
	if separator == "" {
		separator = defaultRegistrySeparator
	}

	return &Registry{
		providers:               maps.Clone(options.Providers),
		separator:               separator,
		languageModelMiddleware: slices.Clone(options.LanguageModelMiddleware),
		imageModelMiddleware:    slices.Clone(options.ImageModelMiddleware),
	}
}

// LanguageModel returns the language model with the combined ID, wrapped in
// the registry's language model middleware.
func (r *Registry) LanguageModel(id string) (model.LanguageModel, error) {
	provider, modelID, err := r.resolve(id, model.LanguageModelKind)
	if err != nil {
		return nil, err
	}

	languageModel, err := provider.LanguageModel(modelID)
	if err != nil {
		return nil, err
	}

	if len(r.languageModelMiddleware) == 0 {
		return languageModel, nil
	}

	return middleware.WrapLanguageModel(middleware.WrapLanguageModelOptions{
		Model:      languageModel,
		Middleware: r.languageModelMiddleware,
	}), nil
}

// EmbeddingModel returns the embedding model with the combined ID.
func (r *Registry) EmbeddingModel(id string) (model.EmbeddingModel, error) {
	provider, modelID, err := r.resolve(id, model.EmbeddingModelKind)
	if err != nil {
		return nil, err
	}

	return provider.EmbeddingModel(modelID)
}

// ImageModel returns the image model with the combined ID, wrapped in the
// registry's image model middleware.
func (r *Registry) ImageModel(id string) (model.ImageModel, error) {
	provider, modelID, err := r.resolve(id, model.ImageModelKind)
	if err != nil {
		return nil, err
	}

	imageModel, err := provider.ImageModel(modelID)
	if err != nil {
		return nil, err
	}

	if len(r.imageModelMiddleware) == 0 {
		return imageModel, nil
	}

	return middleware.WrapImageModel(middleware.WrapImageModelOptions{
		Model:      imageModel,
		Middleware: r.imageModelMiddleware,
	}), nil
}

// TranscriptionModel returns the transcription model with the combined ID.
func (r *Registry) TranscriptionModel(id string) (model.TranscriptionModel, error) {
	provider, modelID, err := r.resolve(id, model.TranscriptionModelKind)
	if err != nil {
		return nil, err
	}

	return provider.TranscriptionModel(modelID)
}

// SpeechModel returns the speech model with the combined ID.
func (r *Registry) SpeechModel(id string) (model.SpeechModel, error) {
	provider, modelID, err := r.resolve(id, model.SpeechModelKind)
	if err != nil {
		return nil, err
	}

	return provider.SpeechModel(modelID)
}

// RerankingModel returns the reranking model with the combined ID.
func (r *Registry) RerankingModel(id string) (model.RerankingModel, error) {
	provider, modelID, err := r.resolve(id, model.RerankingModelKind)
	if err != nil {
		return nil, err
	}

	return provider.RerankingModel(modelID)
}

// VideoModel returns the video model with the combined ID. The provider
// must implement model.VideoProvider.
func (r *Registry) VideoModel(id string) (model.VideoModel, error) {
	provider, modelID, err := r.resolve(id, model.VideoModelKind)
	if err != nil {
		return nil, err
	}

	videoProvider, ok := provider.(model.VideoProvider)
	if !ok {
		return nil, model.NoSuchModelError("registry", model.VideoModelKind, id)
	}

	return videoProvider.VideoModel(modelID)
}

// DecisionModel returns the decision model with the combined ID. The
// provider must implement model.DecisionProvider.
func (r *Registry) DecisionModel(id string) (model.DecisionModel, error) {
	provider, modelID, err := r.resolve(id, model.DecisionModelKind)
	if err != nil {
		return nil, err
	}

	decisionProvider, ok := provider.(model.DecisionProvider)
	if !ok {
		return nil, model.NoSuchModelError("registry", model.DecisionModelKind, id)
	}

	return decisionProvider.DecisionModel(modelID)
}

// RealtimeModel returns the realtime model with the combined ID. The
// provider must implement model.RealtimeProvider.
func (r *Registry) RealtimeModel(id string) (model.RealtimeModel, error) {
	provider, modelID, err := r.resolve(id, model.RealtimeModelKind)
	if err != nil {
		return nil, err
	}

	realtimeProvider, ok := provider.(model.RealtimeProvider)
	if !ok {
		return nil, model.NoSuchModelError("registry", model.RealtimeModelKind, id)
	}

	return realtimeProvider.RealtimeModel(modelID)
}

// SpeechTranslationModel returns the speech translation model with the
// combined ID. The provider must implement model.SpeechTranslationProvider.
func (r *Registry) SpeechTranslationModel(id string) (model.SpeechTranslationModel, error) {
	provider, modelID, err := r.resolve(id, model.SpeechTranslationModelKind)
	if err != nil {
		return nil, err
	}

	translationProvider, ok := provider.(model.SpeechTranslationProvider)
	if !ok {
		return nil, model.NoSuchModelError("registry", model.SpeechTranslationModelKind, id)
	}

	return translationProvider.SpeechTranslationModel(modelID)
}

// Batch returns the batch API of the provider registered as providerID.
// The provider must implement model.BatchProvider.
func (r *Registry) Batch(providerID string) (model.Batch, error) {
	provider, err := r.provider(providerID, providerID, model.LanguageModelKind)
	if err != nil {
		return nil, err
	}

	batchProvider, ok := provider.(model.BatchProvider)
	if !ok || batchProvider.Batch() == nil {
		return nil, fmt.Errorf(
			"%w: the provider %q does not support batch processing; it must implement model.BatchProvider",
			model.ErrUnsupported,
			providerID,
		)
	}

	return batchProvider.Batch(), nil
}

// Files returns the files interface of the provider registered as
// providerID. The provider must implement model.FilesProvider.
func (r *Registry) Files(providerID string) (model.Files, error) {
	provider, err := r.provider(providerID, providerID, model.LanguageModelKind)
	if err != nil {
		return nil, err
	}

	filesProvider, ok := provider.(model.FilesProvider)
	if !ok || filesProvider.Files() == nil {
		return nil, fmt.Errorf(
			"%w: the provider %q does not support file uploads; it must implement model.FilesProvider",
			model.ErrUnsupported,
			providerID,
		)
	}

	return filesProvider.Files(), nil
}

// Skills returns the skills interface of the provider registered as
// providerID. The provider must implement model.SkillsProvider.
func (r *Registry) Skills(providerID string) (model.Skills, error) {
	provider, err := r.provider(providerID, providerID, model.LanguageModelKind)
	if err != nil {
		return nil, err
	}

	skillsProvider, ok := provider.(model.SkillsProvider)
	if !ok || skillsProvider.Skills() == nil {
		return nil, fmt.Errorf(
			"%w: the provider %q does not support skills; it must implement model.SkillsProvider",
			model.ErrUnsupported,
			providerID,
		)
	}

	return skillsProvider.Skills(), nil
}

// resolve splits a combined ID at the first separator and finds the
// provider.
func (r *Registry) resolve(id string, modelType model.Kind) (model.Provider, string, error) {
	providerID, modelID, found := strings.Cut(id, r.separator)
	if !found {
		return nil, "", fmt.Errorf(
			"%w: invalid %s id for registry: %s (must be in the format \"providerId%smodelId\")",
			model.ErrNoSuchModel,
			modelType,
			id,
			r.separator,
		)
	}

	provider, err := r.provider(providerID, id, modelType)
	if err != nil {
		return nil, "", err
	}

	return provider, modelID, nil
}

// provider finds a registered provider. id and modelType describe the
// lookup in the error.
func (r *Registry) provider(providerID string, id string, modelType model.Kind) (model.Provider, error) {
	provider, ok := r.providers[providerID]
	if !ok || provider == nil {
		return nil, &NoSuchProviderError{
			ProviderID:         providerID,
			ModelID:            id,
			ModelType:          modelType,
			AvailableProviders: slices.Sorted(maps.Keys(r.providers)),
		}
	}

	return provider, nil
}

// CustomProviderOptions configures NewCustomProvider. Each map assigns
// models to IDs, which may be aliases.
type CustomProviderOptions struct {
	LanguageModels      map[string]model.LanguageModel
	EmbeddingModels     map[string]model.EmbeddingModel
	ImageModels         map[string]model.ImageModel
	TranscriptionModels map[string]model.TranscriptionModel
	SpeechModels        map[string]model.SpeechModel
	RerankingModels     map[string]model.RerankingModel
	VideoModels         map[string]model.VideoModel

	DecisionModels          map[string]model.DecisionModel
	RealtimeModels          map[string]model.RealtimeModel
	SpeechTranslationModels map[string]model.SpeechTranslationModel

	// Batch is the batch API. Without it, the fallback provider's is used
	// when it implements model.BatchProvider.
	Batch model.Batch

	// Files is the files interface. Without it, the fallback provider's is
	// used when it implements model.FilesProvider.
	Files model.Files

	// Skills is the skills interface. Without it, the fallback provider's is
	// used when it implements model.SkillsProvider.
	Skills model.Skills

	// FallbackProvider resolves IDs that the maps do not contain. Without it
	// those lookups fail with an error wrapping model.ErrNoSuchModel.
	FallbackProvider model.Provider
}

// CustomProvider is a provider assembled from individual models, with an
// optional fallback provider. It implements model.Provider.
type CustomProvider struct {
	options CustomProviderOptions
}

// NewCustomProvider returns a provider that serves the given models.
func NewCustomProvider(options CustomProviderOptions) *CustomProvider {
	return &CustomProvider{
		options: CustomProviderOptions{
			LanguageModels:      maps.Clone(options.LanguageModels),
			EmbeddingModels:     maps.Clone(options.EmbeddingModels),
			ImageModels:         maps.Clone(options.ImageModels),
			TranscriptionModels: maps.Clone(options.TranscriptionModels),
			SpeechModels:        maps.Clone(options.SpeechModels),
			RerankingModels:     maps.Clone(options.RerankingModels),
			VideoModels:         maps.Clone(options.VideoModels),

			DecisionModels:          maps.Clone(options.DecisionModels),
			RealtimeModels:          maps.Clone(options.RealtimeModels),
			SpeechTranslationModels: maps.Clone(options.SpeechTranslationModels),

			Batch:            options.Batch,
			Files:            options.Files,
			Skills:           options.Skills,
			FallbackProvider: options.FallbackProvider,
		},
	}
}

// LanguageModel returns the language model with the given ID.
func (p *CustomProvider) LanguageModel(modelID string) (model.LanguageModel, error) {
	return customModel(customModelLookup[model.LanguageModel]{
		models:    p.options.LanguageModels,
		modelID:   modelID,
		modelType: model.LanguageModelKind,
		fallback:  p.fallback(model.Provider.LanguageModel),
	})
}

// EmbeddingModel returns the embedding model with the given ID.
func (p *CustomProvider) EmbeddingModel(modelID string) (model.EmbeddingModel, error) {
	return customModel(customModelLookup[model.EmbeddingModel]{
		models:    p.options.EmbeddingModels,
		modelID:   modelID,
		modelType: model.EmbeddingModelKind,
		fallback:  p.fallback(model.Provider.EmbeddingModel),
	})
}

// ImageModel returns the image model with the given ID.
func (p *CustomProvider) ImageModel(modelID string) (model.ImageModel, error) {
	return customModel(customModelLookup[model.ImageModel]{
		models:    p.options.ImageModels,
		modelID:   modelID,
		modelType: model.ImageModelKind,
		fallback:  p.fallback(model.Provider.ImageModel),
	})
}

// TranscriptionModel returns the transcription model with the given ID.
func (p *CustomProvider) TranscriptionModel(modelID string) (model.TranscriptionModel, error) {
	return customModel(customModelLookup[model.TranscriptionModel]{
		models:    p.options.TranscriptionModels,
		modelID:   modelID,
		modelType: model.TranscriptionModelKind,
		fallback:  p.fallback(model.Provider.TranscriptionModel),
	})
}

// SpeechModel returns the speech model with the given ID.
func (p *CustomProvider) SpeechModel(modelID string) (model.SpeechModel, error) {
	return customModel(customModelLookup[model.SpeechModel]{
		models:    p.options.SpeechModels,
		modelID:   modelID,
		modelType: model.SpeechModelKind,
		fallback:  p.fallback(model.Provider.SpeechModel),
	})
}

// RerankingModel returns the reranking model with the given ID.
func (p *CustomProvider) RerankingModel(modelID string) (model.RerankingModel, error) {
	return customModel(customModelLookup[model.RerankingModel]{
		models:    p.options.RerankingModels,
		modelID:   modelID,
		modelType: model.RerankingModelKind,
		fallback:  p.fallback(model.Provider.RerankingModel),
	})
}

// VideoModel returns the video model with the given ID. The fallback
// provider is used when it implements model.VideoProvider.
func (p *CustomProvider) VideoModel(modelID string) (model.VideoModel, error) {
	lookup := customModelLookup[model.VideoModel]{
		models:    p.options.VideoModels,
		modelID:   modelID,
		modelType: model.VideoModelKind,
	}

	if videoProvider, ok := p.options.FallbackProvider.(model.VideoProvider); ok {
		lookup.fallback = videoProvider.VideoModel
	}

	return customModel(lookup)
}

// DecisionModel returns the decision model with the given ID. The fallback
// provider is used when it implements model.DecisionProvider.
func (p *CustomProvider) DecisionModel(modelID string) (model.DecisionModel, error) {
	lookup := customModelLookup[model.DecisionModel]{
		models:    p.options.DecisionModels,
		modelID:   modelID,
		modelType: model.DecisionModelKind,
	}

	if decisionProvider, ok := p.options.FallbackProvider.(model.DecisionProvider); ok {
		lookup.fallback = decisionProvider.DecisionModel
	}

	return customModel(lookup)
}

// RealtimeModel returns the realtime model with the given ID. The fallback
// provider is used when it implements model.RealtimeProvider.
func (p *CustomProvider) RealtimeModel(modelID string) (model.RealtimeModel, error) {
	lookup := customModelLookup[model.RealtimeModel]{
		models:    p.options.RealtimeModels,
		modelID:   modelID,
		modelType: model.RealtimeModelKind,
	}

	if realtimeProvider, ok := p.options.FallbackProvider.(model.RealtimeProvider); ok {
		lookup.fallback = realtimeProvider.RealtimeModel
	}

	return customModel(lookup)
}

// SpeechTranslationModel returns the speech translation model with the
// given ID. The fallback provider is used when it implements
// model.SpeechTranslationProvider.
func (p *CustomProvider) SpeechTranslationModel(modelID string) (model.SpeechTranslationModel, error) {
	lookup := customModelLookup[model.SpeechTranslationModel]{
		models:    p.options.SpeechTranslationModels,
		modelID:   modelID,
		modelType: model.SpeechTranslationModelKind,
	}

	if translationProvider, ok := p.options.FallbackProvider.(model.SpeechTranslationProvider); ok {
		lookup.fallback = translationProvider.SpeechTranslationModel
	}

	return customModel(lookup)
}

// Batch returns the configured batch API, or the fallback provider's. It
// returns nil when neither offers one.
func (p *CustomProvider) Batch() model.Batch {
	if p.options.Batch != nil {
		return p.options.Batch
	}

	if batchProvider, ok := p.options.FallbackProvider.(model.BatchProvider); ok {
		return batchProvider.Batch()
	}

	return nil
}

// Files returns the configured files interface, or the fallback provider's.
// It returns nil when neither offers one.
func (p *CustomProvider) Files() model.Files {
	if p.options.Files != nil {
		return p.options.Files
	}

	if filesProvider, ok := p.options.FallbackProvider.(model.FilesProvider); ok {
		return filesProvider.Files()
	}

	return nil
}

// Skills returns the configured skills interface, or the fallback
// provider's. It returns nil when neither offers one.
func (p *CustomProvider) Skills() model.Skills {
	if p.options.Skills != nil {
		return p.options.Skills
	}

	if skillsProvider, ok := p.options.FallbackProvider.(model.SkillsProvider); ok {
		return skillsProvider.Skills()
	}

	return nil
}

// fallback binds a provider method to the fallback provider, or returns nil
// when there is none.
func (p *CustomProvider) fallback[Model any](
	method func(provider model.Provider, modelID string) (Model, error),
) func(modelID string) (Model, error) {
	if p.options.FallbackProvider == nil {
		return nil
	}

	return func(modelID string) (Model, error) {
		return method(p.options.FallbackProvider, modelID)
	}
}

// customModelLookup finds one model of a custom provider.
type customModelLookup[Model any] struct {
	models    map[string]Model
	modelID   string
	modelType model.Kind

	// fallback is nil without a fallback provider.
	fallback func(modelID string) (Model, error)
}

func customModel[Model any](lookup customModelLookup[Model]) (Model, error) {
	if found, ok := lookup.models[lookup.modelID]; ok {
		return found, nil
	}

	if lookup.fallback != nil {
		return lookup.fallback(lookup.modelID)
	}

	var zero Model

	return zero, model.NoSuchModelError("custom", lookup.modelType, lookup.modelID)
}
