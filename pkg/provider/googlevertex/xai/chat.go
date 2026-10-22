package xai

import (
	"encoding/json"

	"github.com/aarock1234/ai/internal/optional"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openaicompatible"
)

// transformRequestBody removes reasoning_effort, which Vertex AI rejects for
// Grok models.
func transformRequestBody(body map[string]json.RawMessage) (map[string]json.RawMessage, []model.Warning, error) {
	delete(body, "reasoning_effort")

	return body, nil, nil
}

// convertUsage converts Vertex AI usage, whose completion tokens exclude the
// reasoning tokens.
func convertUsage(usage *openaicompatible.ChatUsage) model.Usage {
	if usage == nil {
		return model.Usage{}
	}

	promptTokens := optional.Or(usage.PromptTokens, 0)
	completionTokens := optional.Or(usage.CompletionTokens, 0)

	cacheReadTokens := 0
	if usage.PromptTokensDetails != nil {
		cacheReadTokens = optional.Or(usage.PromptTokensDetails.CachedTokens, 0)
	}

	reasoningTokens := 0
	if usage.CompletionTokensDetails != nil {
		reasoningTokens = optional.Or(usage.CompletionTokensDetails.ReasoningTokens, 0)
	}

	return model.Usage{
		InputTokens: model.InputTokens{
			Total:     new(promptTokens),
			NoCache:   new(promptTokens - cacheReadTokens),
			CacheRead: new(cacheReadTokens),
		},
		OutputTokens: model.OutputTokens{
			Total:     new(completionTokens + reasoningTokens),
			Text:      new(completionTokens),
			Reasoning: new(reasoningTokens),
		},
		Raw: usage.Raw,
	}
}
