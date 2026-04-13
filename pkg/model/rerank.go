package model

import (
	"context"
	"net/http"
)

// RerankingModel is the interface for reranking providers.
type RerankingModel interface {
	// Provider returns the provider name (e.g. "cohere").
	Provider() string

	// ModelID returns the model identifier.
	ModelID() string

	// DoRerank performs a reranking request.
	DoRerank(ctx context.Context, req *RerankRequest) (*RerankResponse, error)
}

// RerankRequest is the normalized request shape for reranking.
type RerankRequest struct {
	Query           string
	Documents       []string
	TopN            *int
	MaxTokensPerDoc *int
	Headers         http.Header
	ProviderOptions ProviderOptions
}

// RerankResult is a single reranked document.
type RerankResult struct {
	Index    int
	Score    float64
	Document string
}

// RerankResponse is the normalized response from a reranking request.
type RerankResponse struct {
	Results  []RerankResult
	Metadata ProviderMetadata
}
