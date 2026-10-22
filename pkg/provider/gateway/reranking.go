package gateway

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// rerankingModelPath is the reranking model endpoint.
const rerankingModelPath = "/reranking-model"

// RerankingModel is an AI Gateway reranking model. It is safe for concurrent
// use.
type RerankingModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "gateway" by default.
func (m *RerankingModel) Provider() string {
	return m.provider.name
}

// ModelID returns the model identifier.
func (m *RerankingModel) ModelID() string {
	return m.modelID
}

type rerankingRequest struct {
	Documents       rerankingDocuments    `json:"documents"`
	Query           string                `json:"query"`
	TopN            *int                  `json:"topN,omitempty"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

// rerankingDocuments are text or JSON object documents on the wire.
type rerankingDocuments struct {
	Type   string          `json:"type"`
	Values json.RawMessage `json:"values"`
}

type rerankingResponse struct {
	Ranking          []rankedDocument `json:"ranking"`
	Warnings         []wireWarning    `json:"warnings"`
	ProviderMetadata wireMetadata     `json:"providerMetadata"`
}

type rankedDocument struct {
	Index          int     `json:"index"`
	RelevanceScore float64 `json:"relevanceScore"`
}

// Rerank orders the documents by relevance to the query.
func (m *RerankingModel) Rerank(ctx context.Context, options model.RerankOptions) (*model.RerankResult, error) {
	documents, err := encodeDocuments(options)
	if err != nil {
		return nil, err
	}

	request, method, err := m.provider.modelRequest(rerankingModelPath, options.Headers, modelConfigHeaders("reranking", m.modelID))
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, rerankingRequest{
		Documents:       documents,
		Query:           options.Query,
		TopN:            options.TopN,
		ProviderOptions: options.ProviderOptions,
	})
	if err != nil {
		return nil, asGatewayError(err, method)
	}

	var decoded rerankingResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	ranking := make([]model.RankedDocument, 0, len(decoded.Ranking))
	for _, document := range decoded.Ranking {
		ranking = append(ranking, model.RankedDocument{
			Index:          document.Index,
			RelevanceScore: document.RelevanceScore,
		})
	}

	return &model.RerankResult{
		Ranking:          ranking,
		ProviderMetadata: decoded.ProviderMetadata.providerMetadata(),
		Warnings:         decodeWarnings(decoded.Warnings),
		Response: model.ResponseInfo{
			Headers: response.Headers,
			Body:    response.Body,
		},
	}, nil
}

// encodeDocuments sends text documents, or JSON object documents when
// they are set.
func encodeDocuments(options model.RerankOptions) (rerankingDocuments, error) {
	if options.ObjectDocuments != nil {
		return documentsOf("object", options.ObjectDocuments)
	}

	return documentsOf("text", options.TextDocuments)
}

func documentsOf[Document any](documentType string, values []Document) (rerankingDocuments, error) {
	if values == nil {
		values = []Document{}
	}

	encoded, err := json.Marshal(values)
	if err != nil {
		return rerankingDocuments{}, fmt.Errorf("%w: encode documents: %w", model.ErrInvalidArgument, err)
	}

	return rerankingDocuments{
		Type:   documentType,
		Values: encoded,
	}, nil
}

var _ model.RerankingModel = (*RerankingModel)(nil)
