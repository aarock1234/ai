package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/aarock1234/ai/internal/apierror"
	"github.com/aarock1234/ai/pkg/model"
)

// Compile-time check that EmbeddingModel implements model.EmbeddingModel.
var _ model.EmbeddingModel = (*EmbeddingModel)(nil)

// EmbeddingModel is an OpenAI embedding model that implements
// model.EmbeddingModel.
type EmbeddingModel struct {
	modelID      string
	clientConfig *ClientConfig
}

// Provider returns "openai".
func (m *EmbeddingModel) Provider() string { return "openai" }

// ModelID returns the model identifier (e.g. "text-embedding-3-small").
func (m *EmbeddingModel) ModelID() string { return m.modelID }

// embeddingRequest is the OpenAI embeddings request body.
type embeddingRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

// embeddingResponse is the OpenAI embeddings response.
type embeddingResponse struct {
	Data  []embeddingData `json:"data"`
	Usage embeddingUsage  `json:"usage"`
}

type embeddingData struct {
	Embedding []float64 `json:"embedding"`
	Index     int       `json:"index"`
}

type embeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// DoEmbed embeds the given texts using the OpenAI embeddings API.
func (m *EmbeddingModel) DoEmbed(ctx context.Context, texts []string) (*model.EmbedResponse, error) {
	reqBody := embeddingRequest{
		Model: m.modelID,
		Input: texts,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshalling request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", m.clientConfig.BaseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	setHeaders(httpReq, m.clientConfig)

	resp, err := m.clientConfig.httpClient().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("executing request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, apierror.New(resp.StatusCode, string(respBody))
	}

	var result embeddingResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}

	// Sort by index to ensure correct ordering.
	embeddings := make([][]float64, len(result.Data))
	for _, d := range result.Data {
		embeddings[d.Index] = d.Embedding
	}

	return &model.EmbedResponse{
		Embeddings: embeddings,
		Usage: model.Usage{
			InputTokens: result.Usage.PromptTokens,
		},
	}, nil
}
