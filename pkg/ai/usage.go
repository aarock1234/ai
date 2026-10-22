package ai

import (
	"encoding/json"

	"github.com/aarock1234/ai/pkg/model"
)

// LanguageModelUsage reports the tokens used by language model calls. Nil
// counts mean the provider did not report them.
type LanguageModelUsage struct {
	InputTokens       *int
	InputTokenDetails InputTokenDetails

	OutputTokens       *int
	OutputTokenDetails OutputTokenDetails

	// TotalTokens is the sum of input and output tokens.
	TotalTokens *int

	// Raw is the provider's native usage object. It is kept for single calls
	// and dropped when the usage of several calls is summed.
	Raw json.RawMessage
}

// InputTokenDetails breaks down the input tokens.
type InputTokenDetails struct {
	NoCacheTokens    *int
	CacheReadTokens  *int
	CacheWriteTokens *int
}

// OutputTokenDetails breaks down the output tokens.
type OutputTokenDetails struct {
	TextTokens      *int
	ReasoningTokens *int
}

// newLanguageModelUsage converts provider usage, deriving the total.
func newLanguageModelUsage(usage model.Usage) LanguageModelUsage {
	return LanguageModelUsage{
		InputTokens: usage.InputTokens.Total,
		InputTokenDetails: InputTokenDetails{
			NoCacheTokens:    usage.InputTokens.NoCache,
			CacheReadTokens:  usage.InputTokens.CacheRead,
			CacheWriteTokens: usage.InputTokens.CacheWrite,
		},
		OutputTokens: usage.OutputTokens.Total,
		OutputTokenDetails: OutputTokenDetails{
			TextTokens:      usage.OutputTokens.Text,
			ReasoningTokens: usage.OutputTokens.Reasoning,
		},
		TotalTokens: addTokenCounts(usage.InputTokens.Total, usage.OutputTokens.Total),
		Raw:         usage.Raw,
	}
}

// addLanguageModelUsage sums two usages. A count stays unknown only when it
// is unknown in both. The raw usage survives only when the other usage is
// empty.
func addLanguageModelUsage(first, second LanguageModelUsage) LanguageModelUsage {
	var raw json.RawMessage

	switch {
	case first.isEmpty():
		raw = second.Raw
	case second.isEmpty():
		raw = first.Raw
	}

	return LanguageModelUsage{
		InputTokens: addTokenCounts(first.InputTokens, second.InputTokens),
		InputTokenDetails: InputTokenDetails{
			NoCacheTokens:    addTokenCounts(first.InputTokenDetails.NoCacheTokens, second.InputTokenDetails.NoCacheTokens),
			CacheReadTokens:  addTokenCounts(first.InputTokenDetails.CacheReadTokens, second.InputTokenDetails.CacheReadTokens),
			CacheWriteTokens: addTokenCounts(first.InputTokenDetails.CacheWriteTokens, second.InputTokenDetails.CacheWriteTokens),
		},
		OutputTokens: addTokenCounts(first.OutputTokens, second.OutputTokens),
		OutputTokenDetails: OutputTokenDetails{
			TextTokens:      addTokenCounts(first.OutputTokenDetails.TextTokens, second.OutputTokenDetails.TextTokens),
			ReasoningTokens: addTokenCounts(first.OutputTokenDetails.ReasoningTokens, second.OutputTokenDetails.ReasoningTokens),
		},
		TotalTokens: addTokenCounts(first.TotalTokens, second.TotalTokens),
		Raw:         raw,
	}
}

// isEmpty reports whether no count and no raw usage is known.
func (u LanguageModelUsage) isEmpty() bool {
	return u.InputTokens == nil &&
		u.InputTokenDetails == InputTokenDetails{} &&
		u.OutputTokens == nil &&
		u.OutputTokenDetails == OutputTokenDetails{} &&
		u.TotalTokens == nil &&
		u.Raw == nil
}

// tokensPerSecond returns a token rate, or zero when the token count or the
// duration is unknown or zero.
func tokensPerSecond(tokens *int, durationSeconds float64) float64 {
	if tokens == nil || durationSeconds <= 0 {
		return 0
	}

	return float64(*tokens) / durationSeconds
}
