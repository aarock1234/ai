package alibaba

import (
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

// convertUsage converts Alibaba usage. Cache reads and writes are part of
// the prompt tokens, so both are excluded from the uncached count.
func convertUsage(usage *Usage) model.Usage {
	if usage == nil {
		return model.Usage{}
	}

	cacheReadTokens := 0
	cacheWriteTokens := 0
	if details := usage.PromptTokensDetails; details != nil {
		cacheReadTokens = valueOrZero(details.CachedTokens)
		cacheWriteTokens = valueOrZero(details.CacheCreationInputTokens)
	}

	reasoningTokens := 0
	if usage.CompletionTokensDetails != nil {
		reasoningTokens = valueOrZero(usage.CompletionTokensDetails.ReasoningTokens)
	}

	return model.Usage{
		InputTokens: model.InputTokens{
			Total:      new(usage.PromptTokens),
			NoCache:    new(usage.PromptTokens - cacheReadTokens - cacheWriteTokens),
			CacheRead:  new(cacheReadTokens),
			CacheWrite: new(cacheWriteTokens),
		},
		OutputTokens: model.OutputTokens{
			Total:     new(usage.CompletionTokens),
			Text:      new(max(0, usage.CompletionTokens-reasoningTokens)),
			Reasoning: new(reasoningTokens),
		},
		Raw: usage.Raw,
	}
}

// mapFinishReason converts a chat completion finish reason.
func mapFinishReason(finishReason *string) model.FinishReason {
	raw := valueOrEmpty(finishReason)

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
		ID:      valueOrEmpty(id),
		ModelID: valueOrEmpty(modelID),
	}

	if created != nil {
		metadata.Timestamp = time.Unix(int64(*created), 0)
	}

	return metadata
}

func valueOrZero(value *int) int {
	if value == nil {
		return 0
	}

	return *value
}

func valueOrEmpty(value *string) string {
	if value == nil {
		return ""
	}

	return *value
}
