package gateway

import (
	"context"
	"fmt"
	"slices"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// configPath is the model catalog endpoint.
const configPath = "/config"

// ModelType is the kind of a catalog model.
type ModelType string

// Model types.
const (
	ModelTypeEmbedding     ModelType = "embedding"
	ModelTypeDecision      ModelType = "decision"
	ModelTypeEvaluation    ModelType = "evaluation"
	ModelTypeImage         ModelType = "image"
	ModelTypeLanguage      ModelType = "language"
	ModelTypeRealtime      ModelType = "realtime"
	ModelTypeReranking     ModelType = "reranking"
	ModelTypeSpeech        ModelType = "speech"
	ModelTypeTranscription ModelType = "transcription"
	ModelTypeVideo         ModelType = "video"
)

// knownModelTypes are the model types this package recognizes. Catalog
// entries of other types are skipped.
var knownModelTypes = []ModelType{
	ModelTypeEmbedding,
	ModelTypeDecision,
	ModelTypeEvaluation,
	ModelTypeImage,
	ModelTypeLanguage,
	ModelTypeRealtime,
	ModelTypeReranking,
	ModelTypeSpeech,
	ModelTypeTranscription,
	ModelTypeVideo,
}

// AvailableModels is the gateway's model catalog.
type AvailableModels struct {
	Models []ModelEntry
}

// ModelEntry describes one model of the catalog.
type ModelEntry struct {
	// ID is the model ID to pass to the provider's model constructors.
	ID string

	// Name is the display name.
	Name string

	// Description is empty when the catalog has none.
	Description string

	// Pricing is nil when the catalog has none.
	Pricing *Pricing

	Specification Specification

	// Type is empty for entries without a model type.
	Type ModelType
}

// Pricing is a model's price per token in USD, as decimal strings. Cache
// prices are empty for models without prompt caching.
type Pricing struct {
	Input                    string
	Output                   string
	CachedInputTokens        string
	CacheCreationInputTokens string
}

// Specification identifies the model behind a catalog entry.
type Specification struct {
	SpecificationVersion string
	Provider             string
	ModelID              string
}

type catalogResponse struct {
	Models []catalogModel `json:"models"`
}

type catalogModel struct {
	ID            string               `json:"id"`
	Name          string               `json:"name"`
	Description   *string              `json:"description"`
	Pricing       *catalogPricing      `json:"pricing"`
	Specification catalogSpecification `json:"specification"`
	ModelType     *ModelType           `json:"modelType"`
}

type catalogPricing struct {
	Input           string  `json:"input"`
	Output          string  `json:"output"`
	InputCacheRead  *string `json:"input_cache_read"`
	InputCacheWrite *string `json:"input_cache_write"`
}

type catalogSpecification struct {
	SpecificationVersion string `json:"specificationVersion"`
	Provider             string `json:"provider"`
	ModelID              string `json:"modelId"`
}

// AvailableModels returns the models the gateway offers. The catalog is
// cached for the metadata cache refresh period; when a refresh fails, the
// previous catalog is returned.
func (p *Provider) AvailableModels(ctx context.Context) (*AvailableModels, error) {
	select {
	case p.catalog.lock <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	defer func() {
		<-p.catalog.lock
	}()

	if p.catalog.models != nil && time.Since(p.catalog.fetchedAt) <= p.metadataCacheRefresh {
		return p.catalog.models, nil
	}

	models, err := p.fetchCatalog(ctx)
	if err != nil {
		if p.catalog.models != nil {
			return p.catalog.models, nil
		}

		return nil, err
	}

	p.catalog.models = models
	p.catalog.fetchedAt = time.Now()

	return models, nil
}

func (p *Provider) fetchCatalog(ctx context.Context) (*AvailableModels, error) {
	request, method, err := p.accountRequest(p.baseURL + configPath)
	if err != nil {
		return nil, err
	}

	response, err := apicall.Get(ctx, request)
	if err != nil {
		return nil, asGatewayError(err, method)
	}

	var decoded catalogResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	models := make([]ModelEntry, 0, len(decoded.Models))

	for _, entry := range decoded.Models {
		if entry.Specification.SpecificationVersion != "v4" {
			return nil, fmt.Errorf(
				"%w: model %q has specification version %q, want \"v4\"",
				model.ErrInvalidResponse,
				entry.ID,
				entry.Specification.SpecificationVersion,
			)
		}

		if entry.ModelType != nil && !slices.Contains(knownModelTypes, *entry.ModelType) {
			continue
		}

		models = append(models, entry.entry())
	}

	return &AvailableModels{
		Models: models,
	}, nil
}

func (m catalogModel) entry() ModelEntry {
	entry := ModelEntry{
		ID:          m.ID,
		Name:        m.Name,
		Description: stringValue(m.Description),
		Specification: Specification{
			SpecificationVersion: m.Specification.SpecificationVersion,
			Provider:             m.Specification.Provider,
			ModelID:              m.Specification.ModelID,
		},
	}

	if m.ModelType != nil {
		entry.Type = *m.ModelType
	}

	if m.Pricing != nil {
		entry.Pricing = &Pricing{
			Input:                    m.Pricing.Input,
			Output:                   m.Pricing.Output,
			CachedInputTokens:        stringValue(m.Pricing.InputCacheRead),
			CacheCreationInputTokens: stringValue(m.Pricing.InputCacheWrite),
		}
	}

	return entry
}
