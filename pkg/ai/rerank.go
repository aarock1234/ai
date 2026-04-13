package ai

import (
	"context"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// Rerank reranks a list of documents for a query.
func Rerank(ctx context.Context, m model.RerankingModel, req model.RerankRequest) (*model.RerankResponse, error) {
	resp, err := m.DoRerank(ctx, &req)
	if err != nil {
		return nil, fmt.Errorf("rerank: %w", err)
	}

	return resp, nil
}
