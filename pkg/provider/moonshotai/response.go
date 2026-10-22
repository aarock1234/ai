package moonshotai

import (
	"time"

	"github.com/aarock1234/ai/internal/optional"
	"github.com/aarock1234/ai/pkg/model"
)

// convertUsage converts Moonshot usage. Missing usage reports no counts.
func convertUsage(usage *chatUsage) model.Usage {
	if usage == nil {
		return model.Usage{}
	}

	promptTokens := optional.Or(usage.PromptTokens, 0)
	completionTokens := optional.Or(usage.CompletionTokens, 0)
	cacheReadTokens := usage.cacheReadTokens()
	reasoningTokens := usage.reasoningTokens()

	return model.Usage{
		InputTokens: model.InputTokens{
			Total:     new(promptTokens),
			NoCache:   new(promptTokens - cacheReadTokens),
			CacheRead: new(cacheReadTokens),
		},
		OutputTokens: model.OutputTokens{
			Total:     new(completionTokens),
			Text:      new(max(0, completionTokens-reasoningTokens)),
			Reasoning: new(reasoningTokens),
		},
		Raw: usage.raw,
	}
}

// cacheReadTokens prefers the top-level cached token count.
func (u *chatUsage) cacheReadTokens() int {
	if u.CachedTokens != nil {
		return *u.CachedTokens
	}

	if u.PromptTokensDetails != nil && u.PromptTokensDetails.CachedTokens != nil {
		return *u.PromptTokensDetails.CachedTokens
	}

	return 0
}

func (u *chatUsage) reasoningTokens() int {
	if u.CompletionTokensDetails != nil && u.CompletionTokensDetails.ReasoningTokens != nil {
		return *u.CompletionTokensDetails.ReasoningTokens
	}

	return 0
}

// mapFinishReason converts a chat completion finish reason.
func mapFinishReason(finishReason *string) model.FinishReason {
	raw := optional.Or(finishReason, "")

	return model.FinishReason{
		Unified: unifiedFinishReason(raw),
		Raw:     raw,
	}
}

func unifiedFinishReason(finishReason string) model.UnifiedFinishReason {
	switch finishReason {
	case "stop":
		return model.FinishStop
	case "length":
		return model.FinishLength
	case "content_filter":
		return model.FinishContentFilter
	case "function_call", "tool_calls":
		return model.FinishToolCalls
	default:
		return model.FinishOther
	}
}

// responseMetadata converts the response identification fields.
func responseMetadata(id *string, modelID *string, created *float64) model.ResponseMetadata {
	metadata := model.ResponseMetadata{
		ID:      optional.Or(id, ""),
		ModelID: optional.Or(modelID, ""),
	}

	if created != nil {
		metadata.Timestamp = time.Unix(int64(*created), 0)
	}

	return metadata
}
