package cohere

import (
	"bytes"
	"encoding/json"

	"github.com/aarock1234/ai/pkg/model"
)

// Usage is Cohere's token usage.
type Usage struct {
	BilledUnits *UsageTokens `json:"billed_units"`
	Tokens      UsageTokens  `json:"tokens"`

	// CachedTokens counts input tokens read from the cache.
	CachedTokens *int `json:"cached_tokens"`

	// Raw is the usage object as the provider sent it.
	Raw json.RawMessage `json:"-"`
}

// UsageTokens counts input and output tokens.
type UsageTokens struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// UnmarshalJSON decodes the usage and keeps the raw object.
func (u *Usage) UnmarshalJSON(data []byte) error {
	type plain Usage

	var decoded plain
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}

	*u = Usage(decoded)
	u.Raw = bytes.Clone(data)

	return nil
}

// convertUsage converts Cohere usage. Cohere does not break down reasoning
// tokens.
func convertUsage(usage *Usage) model.Usage {
	if usage == nil {
		return model.Usage{}
	}

	input := usage.Tokens.InputTokens
	output := usage.Tokens.OutputTokens

	noCache := input
	if usage.CachedTokens != nil {
		noCache = input - *usage.CachedTokens
	}

	return model.Usage{
		InputTokens: model.InputTokens{
			Total:     new(input),
			NoCache:   new(noCache),
			CacheRead: usage.CachedTokens,
		},
		OutputTokens: model.OutputTokens{
			Total: new(output),
			Text:  new(output),
		},
		Raw: usage.Raw,
	}
}

// mapFinishReason converts a Cohere finish reason.
func mapFinishReason(finishReason string) model.FinishReason {
	return model.FinishReason{
		Unified: unifiedFinishReason(finishReason),
		Raw:     finishReason,
	}
}

func unifiedFinishReason(finishReason string) model.UnifiedFinishReason {
	switch finishReason {
	case "COMPLETE", "STOP_SEQUENCE":
		return model.FinishStop
	case "MAX_TOKENS":
		return model.FinishLength
	case "ERROR":
		return model.FinishError
	case "TOOL_CALL":
		return model.FinishToolCalls
	default:
		return model.FinishOther
	}
}
