package ai

import (
	"github.com/aarock1234/ai/pkg/model"
)

// Reranking document types reported by reranking model call events.
const (
	rerankDocumentsText   = "text"
	rerankDocumentsObject = "object"
)

// RerankingModelCallStartEvent describes one reranking model call attempt
// before it is sent.
type RerankingModelCallStartEvent struct {
	CallID string

	// OperationID is "ai.rerank.doRerank".
	OperationID string

	Provider  string
	ModelID   string
	Documents []any

	// DocumentsType is "text" for string documents and "object" otherwise.
	DocumentsType string

	Query string
	TopN  *int
}

// RerankingModelCallEndEvent describes one successful reranking model call.
type RerankingModelCallEndEvent struct {
	CallID        string
	OperationID   string
	Provider      string
	ModelID       string
	DocumentsType string

	// Ranking is the model's ranking, before it is applied to the documents.
	Ranking []model.RankedDocument
}

// anyDocuments returns the documents as a slice of any, for telemetry
// events, which are not generic.
func anyDocuments[Document any](documents []Document) []any {
	converted := make([]any, len(documents))
	for index, document := range documents {
		converted[index] = document
	}

	return converted
}

// anyRanking returns a ranking with its documents as any.
func anyRanking[Document any](ranking []RankedDocument[Document]) []RankedDocument[any] {
	converted := make([]RankedDocument[any], len(ranking))
	for index, ranked := range ranking {
		converted[index] = RankedDocument[any]{
			OriginalIndex: ranked.OriginalIndex,
			Score:         ranked.Score,
			Document:      ranked.Document,
		}
	}

	return converted
}

// forTelemetry returns the event for telemetry integrations.
func (e RerankStartEvent[Document]) forTelemetry() RerankStartEvent[any] {
	return RerankStartEvent[any]{
		CallID:          e.CallID,
		OperationID:     e.OperationID,
		Provider:        e.Provider,
		ModelID:         e.ModelID,
		Documents:       anyDocuments(e.Documents),
		Query:           e.Query,
		TopN:            e.TopN,
		MaxRetries:      e.MaxRetries,
		Headers:         e.Headers,
		ProviderOptions: e.ProviderOptions,
	}
}

// forTelemetry returns the event for telemetry integrations.
func (e RerankEndEvent[Document]) forTelemetry() RerankEndEvent[any] {
	return RerankEndEvent[any]{
		CallID:           e.CallID,
		OperationID:      e.OperationID,
		Provider:         e.Provider,
		ModelID:          e.ModelID,
		Documents:        anyDocuments(e.Documents),
		Query:            e.Query,
		Ranking:          anyRanking(e.Ranking),
		Warnings:         e.Warnings,
		ProviderMetadata: e.ProviderMetadata,
		Response:         e.Response,
	}
}
