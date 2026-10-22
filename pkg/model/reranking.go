package model

import (
	"context"
	"encoding/json"
	"net/http"
)

// RerankingModel orders documents by relevance to a query.
type RerankingModel interface {
	Provider() string
	ModelID() string

	Rerank(ctx context.Context, options RerankOptions) (*RerankResult, error)
}

// RerankOptions are the inputs of one reranking call. Exactly one of
// TextDocuments and ObjectDocuments is set.
type RerankOptions struct {
	Query string

	TextDocuments []string

	// ObjectDocuments are JSON objects.
	ObjectDocuments []json.RawMessage

	// TopN optionally limits the number of results.
	TopN *int

	Headers         http.Header
	ProviderOptions ProviderOptions
}

// RerankResult is the outcome of one reranking call.
type RerankResult struct {
	// Ranking is ordered from most to least relevant.
	Ranking []RankedDocument

	ProviderMetadata ProviderMetadata
	Warnings         []Warning
	Response         ResponseInfo
}

// RankedDocument is one entry of a ranking.
type RankedDocument struct {
	// Index is the position of the document in the input.
	Index int

	RelevanceScore float64
}
