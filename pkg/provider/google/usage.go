package google

import (
	"encoding/json"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// convertUsage maps Gemini usage to the normalized usage. Tool-use prompt
// tokens count as input and thoughts count as reasoning output.
func convertUsage(usage *UsageMetadata) model.Usage {
	if usage == nil {
		return model.Usage{}
	}

	promptTokens := valueOrZero(usage.PromptTokenCount)
	candidatesTokens := valueOrZero(usage.CandidatesTokenCount)
	toolUsePromptTokens := valueOrZero(usage.ToolUsePromptTokenCount)
	cachedContentTokens := valueOrZero(usage.CachedContentTokenCount)
	thoughtsTokens := valueOrZero(usage.ThoughtsTokenCount)
	inputTokens := promptTokens + toolUsePromptTokens

	// Encoding a typed struct without custom marshalers cannot fail.
	raw, _ := json.Marshal(usage)

	return model.Usage{
		InputTokens: model.InputTokens{
			Total:     new(inputTokens),
			NoCache:   new(inputTokens - cachedContentTokens),
			CacheRead: new(cachedContentTokens),
		},
		OutputTokens: model.OutputTokens{
			Total:     new(candidatesTokens + thoughtsTokens),
			Text:      new(candidatesTokens),
			Reasoning: new(thoughtsTokens),
		},
		Raw: raw,
	}
}

// addUsageMetadata sums the usage of two requests, such as the requests of an
// automatically continued response. Non-numeric fields come from the later
// request when it reports them.
func addUsageMetadata(earlier, later *UsageMetadata) *UsageMetadata {
	if earlier == nil {
		return later
	}

	if later == nil {
		return earlier
	}

	sum := *earlier
	sum.PromptTokenCount = addCounts(earlier.PromptTokenCount, later.PromptTokenCount)
	sum.CandidatesTokenCount = addCounts(earlier.CandidatesTokenCount, later.CandidatesTokenCount)
	sum.ToolUsePromptTokenCount = addCounts(earlier.ToolUsePromptTokenCount, later.ToolUsePromptTokenCount)
	sum.TotalTokenCount = addCounts(earlier.TotalTokenCount, later.TotalTokenCount)
	sum.CachedContentTokenCount = addCounts(earlier.CachedContentTokenCount, later.CachedContentTokenCount)
	sum.ThoughtsTokenCount = addCounts(earlier.ThoughtsTokenCount, later.ThoughtsTokenCount)

	sum.PromptTokensDetails = addTokenDetails(earlier.PromptTokensDetails, later.PromptTokensDetails)
	sum.CacheTokensDetails = addTokenDetails(earlier.CacheTokensDetails, later.CacheTokensDetails)
	sum.CandidatesTokensDetails = addTokenDetails(earlier.CandidatesTokensDetails, later.CandidatesTokensDetails)
	sum.ToolUsePromptTokensDetails = addTokenDetails(earlier.ToolUsePromptTokensDetails, later.ToolUsePromptTokensDetails)

	if later.TrafficType != nil {
		sum.TrafficType = later.TrafficType
	}

	if later.ServiceTier != nil {
		sum.ServiceTier = later.ServiceTier
	}

	return &sum
}

func addCounts(earlier, later *int) *int {
	if earlier == nil && later == nil {
		return nil
	}

	return new(valueOrZero(earlier) + valueOrZero(later))
}

// addTokenDetails sums token counts per modality, in first-seen order.
func addTokenDetails(earlier, later []ModalityTokenCount) []ModalityTokenCount {
	if earlier == nil {
		return later
	}

	if later == nil {
		return earlier
	}

	var sums []ModalityTokenCount
	positions := make(map[string]int)

	for _, detail := range slices.Concat(earlier, later) {
		position, seen := positions[detail.Modality]
		if !seen {
			positions[detail.Modality] = len(sums)
			sums = append(sums, detail)

			continue
		}

		sums[position].TokenCount += detail.TokenCount
	}

	return sums
}

func valueOrZero(value *int) int {
	if value == nil {
		return 0
	}

	return *value
}
