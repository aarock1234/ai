package togetherai

import (
	"context"
	"encoding/json"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// rerankingPath is the rerank endpoint.
const rerankingPath = "/rerank"

// RerankingModel is a Together AI reranking model. Object documents are
// ranked by their RerankingOptions.RankFields, or by all their keys.
type RerankingModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "togetherai.reranking" by
// default.
func (m *RerankingModel) Provider() string {
	return m.provider.name + ".reranking"
}

// ModelID returns the model identifier.
func (m *RerankingModel) ModelID() string {
	return m.modelID
}

type rerankingRequest struct {
	Model     string             `json:"model"`
	Query     string             `json:"query"`
	Documents rerankingDocuments `json:"documents"`
	TopN      *int               `json:"top_n,omitempty"`

	RankFields      []string `json:"rank_fields,omitempty"`
	ReturnDocuments bool     `json:"return_documents"`
}

type rerankingResponse struct {
	ID      *string           `json:"id"`
	Model   *string           `json:"model"`
	Results []rerankingResult `json:"results"`
	Usage   rerankingUsage    `json:"usage"`
}

type rerankingResult struct {
	Index          int     `json:"index"`
	RelevanceScore float64 `json:"relevance_score"`
}

type rerankingUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Rerank orders the documents by relevance to the query.
func (m *RerankingModel) Rerank(ctx context.Context, options model.RerankOptions) (*model.RerankResult, error) {
	rerankingOptions, _, err := model.ProviderValue[RerankingOptions](options.ProviderOptions, Namespace)
	if err != nil {
		return nil, err
	}

	request, err := m.provider.apiRequest(rerankingPath, options.Headers)
	if err != nil {
		return nil, err
	}

	body := rerankingRequest{
		Model: m.modelID,
		Query: options.Query,
		Documents: rerankingDocuments{
			text:    options.TextDocuments,
			objects: options.ObjectDocuments,
		},
		TopN:       options.TopN,
		RankFields: rerankingOptions.RankFields,

		// Documents are not returned, to reduce the response size.
		ReturnDocuments: false,
	}

	response, err := apicall.PostJSON(ctx, request, body)
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

	result := &model.RerankResult{
		Ranking: ranking,
		Response: model.ResponseInfo{
			Headers: response.Headers,
			Body:    response.Body,
		},
	}

	if decoded.ID != nil {
		result.Response.ID = *decoded.ID
	}

	if decoded.Model != nil {
		result.Response.ModelID = *decoded.Model
	}

	return result, nil
}

// rerankingDocuments encodes the object documents when they are set and the
// text documents otherwise.
type rerankingDocuments struct {
	text    []string
	objects []json.RawMessage
}

// MarshalJSON encodes the documents as a JSON array.
func (d rerankingDocuments) MarshalJSON() ([]byte, error) {
	if d.objects != nil {
		return json.Marshal(d.objects)
	}

	if d.text == nil {
		return []byte("[]"), nil
	}

	return json.Marshal(d.text)
}

var _ model.RerankingModel = (*RerankingModel)(nil)
