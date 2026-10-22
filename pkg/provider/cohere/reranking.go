package cohere

import (
	"context"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/toolresult"
	"github.com/aarock1234/ai/pkg/model"
)

// rerankingPath is the rerank endpoint.
const rerankingPath = "/rerank"

// RerankingModel is a Cohere reranking model.
type RerankingModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "cohere.reranking" by default.
func (m *RerankingModel) Provider() string {
	return m.provider.name + ".reranking"
}

// ModelID returns the model identifier.
func (m *RerankingModel) ModelID() string {
	return m.modelID
}

type rerankingRequest struct {
	Model           string   `json:"model"`
	Query           string   `json:"query"`
	Documents       []string `json:"documents"`
	TopN            *int     `json:"top_n,omitempty"`
	MaxTokensPerDoc *int     `json:"max_tokens_per_doc,omitempty"`
	Priority        *int     `json:"priority,omitempty"`
}

type rerankingResponse struct {
	ID      *string           `json:"id"`
	Results []rerankingResult `json:"results"`
}

type rerankingResult struct {
	Index          int     `json:"index"`
	RelevanceScore float64 `json:"relevance_score"`
}

// Rerank orders the documents by relevance to the query. Object documents
// are sent as JSON text.
func (m *RerankingModel) Rerank(ctx context.Context, options model.RerankOptions) (*model.RerankResult, error) {
	rerankingOptions, _, err := model.ProviderValue[RerankingOptions](options.ProviderOptions, Namespace)
	if err != nil {
		return nil, err
	}

	documents, warnings, err := rerankingDocuments(options)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(rerankingPath, options.Headers)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, rerankingRequest{
		Model:           m.modelID,
		Query:           options.Query,
		Documents:       documents,
		TopN:            options.TopN,
		MaxTokensPerDoc: rerankingOptions.MaxTokensPerDoc,
		Priority:        rerankingOptions.Priority,
	})
	if err != nil {
		return nil, err
	}

	var decoded rerankingResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	ranking := make([]model.RankedDocument, 0, len(decoded.Results))
	for _, result := range decoded.Results {
		ranking = append(ranking, model.RankedDocument{
			Index:          result.Index,
			RelevanceScore: result.RelevanceScore,
		})
	}

	rerankResult := &model.RerankResult{
		Ranking:  ranking,
		Warnings: warnings,
		Response: model.ResponseInfo{
			Headers: response.Headers,
			Body:    response.Body,
		},
	}

	if decoded.ID != nil {
		rerankResult.Response.ID = *decoded.ID
	}

	return rerankResult, nil
}

// rerankingDocuments returns the text documents, or the object documents as
// JSON text with a compatibility warning.
func rerankingDocuments(options model.RerankOptions) ([]string, []model.Warning, error) {
	if options.ObjectDocuments == nil {
		return options.TextDocuments, nil, nil
	}

	documents := make([]string, 0, len(options.ObjectDocuments))
	for _, document := range options.ObjectDocuments {
		text, err := toolresult.CompactJSON(document)
		if err != nil {
			return nil, nil, err
		}

		documents = append(documents, text)
	}

	warnings := []model.Warning{
		model.CompatibilityWarning("object documents", "Object documents are converted to strings."),
	}

	return documents, warnings, nil
}

var _ model.RerankingModel = (*RerankingModel)(nil)
