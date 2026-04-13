package cohere

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/aarock1234/ai/internal/apierror"
	"github.com/aarock1234/ai/internal/provideropts"
	"github.com/aarock1234/ai/pkg/model"
)

// Compile-time check that RerankModel implements model.RerankingModel.
var _ model.RerankingModel = (*RerankModel)(nil)

// RerankModel is a Cohere reranking model.
type RerankModel struct {
	modelID      string
	clientConfig *clientConfig
}

// Provider returns "cohere".
func (m *RerankModel) Provider() string { return "cohere" }

// ModelID returns the model identifier.
func (m *RerankModel) ModelID() string { return m.modelID }

type rerankRequest struct {
	Model           string   `json:"model"`
	Query           string   `json:"query"`
	Documents       []string `json:"documents"`
	TopN            *int     `json:"top_n,omitempty"`
	MaxTokensPerDoc *int     `json:"max_tokens_per_doc,omitempty"`
}

type rerankResponse struct {
	Results []rerankResult `json:"results"`
	Meta    *rerankMeta    `json:"meta,omitempty"`
}

type rerankResult struct {
	Index          int     `json:"index"`
	RelevanceScore float64 `json:"relevance_score"`
}

type rerankMeta struct {
	BilledUnits map[string]float64 `json:"billed_units,omitempty"`
}

// DoRerank performs a Cohere reranking request.
func (m *RerankModel) DoRerank(ctx context.Context, req *model.RerankRequest) (*model.RerankResponse, error) {
	if req == nil {
		req = &model.RerankRequest{}
	}

	body, err := provideropts.JSONBody(rerankRequest{
		Model:           m.modelID,
		Query:           req.Query,
		Documents:       req.Documents,
		TopN:            req.TopN,
		MaxTokensPerDoc: req.MaxTokensPerDoc,
	}, "cohere", req.ProviderOptions,
		"model",
		"query",
		"documents",
		"top_n",
		"max_tokens_per_doc",
	)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, m.clientConfig.baseURL+"/rerank", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	if m.clientConfig.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+m.clientConfig.apiKey)
	}
	for key, values := range req.Headers {
		for _, value := range values {
			httpReq.Header.Set(key, value)
		}
	}

	resp, err := m.clientConfig.client().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("executing request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, apierror.New(resp.StatusCode, string(bodyBytes))
	}

	var result rerankResponse
	if err := json.Unmarshal(bodyBytes, &result); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}

	ranked := make([]model.RerankResult, len(result.Results))
	for i, item := range result.Results {
		document := ""
		if item.Index >= 0 && item.Index < len(req.Documents) {
			document = req.Documents[item.Index]
		}

		ranked[i] = model.RerankResult{
			Index:    item.Index,
			Score:    item.RelevanceScore,
			Document: document,
		}
	}

	response := &model.RerankResponse{
		Results: ranked,
	}
	if result.Meta != nil && len(result.Meta.BilledUnits) > 0 {
		response.Metadata.Provider = make(map[string]string, len(result.Meta.BilledUnits))
		for key, value := range result.Meta.BilledUnits {
			response.Metadata.Provider[key] = fmt.Sprintf("%g", value)
		}
	}

	return response, nil
}
