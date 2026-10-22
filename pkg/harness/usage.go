package harness

import (
	"encoding/json"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// languageModelUsage converts harness usage to the ai usage shape, deriving
// the total.
func languageModelUsage(usage model.Usage) ai.LanguageModelUsage {
	return ai.LanguageModelUsage{
		InputTokens: usage.InputTokens.Total,
		InputTokenDetails: ai.InputTokenDetails{
			NoCacheTokens:    usage.InputTokens.NoCache,
			CacheReadTokens:  usage.InputTokens.CacheRead,
			CacheWriteTokens: usage.InputTokens.CacheWrite,
		},
		OutputTokens: usage.OutputTokens.Total,
		OutputTokenDetails: ai.OutputTokenDetails{
			TextTokens:      usage.OutputTokens.Text,
			ReasoningTokens: usage.OutputTokens.Reasoning,
		},
		TotalTokens: addTokenCounts(usage.InputTokens.Total, usage.OutputTokens.Total),
		Raw:         usage.Raw,
	}
}

// addUsage sums two usages. A count stays unknown only when both are
// unknown; the raw usage survives only when the other usage is empty.
func addUsage(first ai.LanguageModelUsage, second ai.LanguageModelUsage) ai.LanguageModelUsage {
	var raw json.RawMessage

	switch {
	case isEmptyUsage(first):
		raw = second.Raw
	case isEmptyUsage(second):
		raw = first.Raw
	}

	return ai.LanguageModelUsage{
		InputTokens: addTokenCounts(first.InputTokens, second.InputTokens),
		InputTokenDetails: ai.InputTokenDetails{
			NoCacheTokens:    addTokenCounts(first.InputTokenDetails.NoCacheTokens, second.InputTokenDetails.NoCacheTokens),
			CacheReadTokens:  addTokenCounts(first.InputTokenDetails.CacheReadTokens, second.InputTokenDetails.CacheReadTokens),
			CacheWriteTokens: addTokenCounts(first.InputTokenDetails.CacheWriteTokens, second.InputTokenDetails.CacheWriteTokens),
		},
		OutputTokens: addTokenCounts(first.OutputTokens, second.OutputTokens),
		OutputTokenDetails: ai.OutputTokenDetails{
			TextTokens:      addTokenCounts(first.OutputTokenDetails.TextTokens, second.OutputTokenDetails.TextTokens),
			ReasoningTokens: addTokenCounts(first.OutputTokenDetails.ReasoningTokens, second.OutputTokenDetails.ReasoningTokens),
		},
		TotalTokens: addTokenCounts(first.TotalTokens, second.TotalTokens),
		Raw:         raw,
	}
}

// isEmptyUsage reports whether no count and no raw usage is known.
func isEmptyUsage(usage ai.LanguageModelUsage) bool {
	return usage.InputTokens == nil &&
		usage.OutputTokens == nil &&
		usage.TotalTokens == nil &&
		usage.InputTokenDetails == (ai.InputTokenDetails{}) &&
		usage.OutputTokenDetails == (ai.OutputTokenDetails{}) &&
		len(usage.Raw) == 0
}

// addTokenCounts sums two optional counts; the sum is unknown only when both
// are.
func addTokenCounts(first *int, second *int) *int {
	if first == nil && second == nil {
		return nil
	}

	sum := 0
	if first != nil {
		sum += *first
	}

	if second != nil {
		sum += *second
	}

	return &sum
}
