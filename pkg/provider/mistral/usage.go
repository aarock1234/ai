package mistral

import (
	"bytes"
	"encoding/json"
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

// Usage is Mistral's token usage of a chat completion or conversation.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`

	// NumCachedTokens counts prompt tokens read from the cache.
	NumCachedTokens *int `json:"num_cached_tokens"`

	// PromptTokensDetails is spelled prompt_token_details by some endpoints.
	PromptTokensDetails *PromptTokensDetails `json:"prompt_tokens_details"`
	PromptTokenDetails  *PromptTokensDetails `json:"prompt_token_details"`

	// Raw is the usage object as the provider sent it.
	Raw json.RawMessage `json:"-"`
}

// PromptTokensDetails breaks down prompt tokens.
type PromptTokensDetails struct {
	CachedTokens *int `json:"cached_tokens"`
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

// cacheReadTokens returns the first reported cached token count, or zero.
func (u *Usage) cacheReadTokens() int {
	switch {
	case u.NumCachedTokens != nil:
		return *u.NumCachedTokens
	case u.PromptTokensDetails != nil && u.PromptTokensDetails.CachedTokens != nil:
		return *u.PromptTokensDetails.CachedTokens
	case u.PromptTokenDetails != nil && u.PromptTokenDetails.CachedTokens != nil:
		return *u.PromptTokenDetails.CachedTokens
	default:
		return 0
	}
}

// convertUsage converts Mistral usage. Mistral does not break down reasoning
// tokens.
func convertUsage(usage *Usage) model.Usage {
	if usage == nil {
		return model.Usage{}
	}

	cacheRead := usage.cacheReadTokens()

	converted := model.Usage{
		InputTokens: model.InputTokens{
			Total:   new(usage.PromptTokens),
			NoCache: new(usage.PromptTokens - cacheRead),
		},
		OutputTokens: model.OutputTokens{
			Total: new(usage.CompletionTokens),
			Text:  new(usage.CompletionTokens),
		},
		Raw: usage.Raw,
	}

	if cacheRead != 0 {
		converted.InputTokens.CacheRead = new(cacheRead)
	}

	return converted
}

// mapFinishReason converts a chat completion finish reason.
func mapFinishReason(finishReason string) model.FinishReason {
	return model.FinishReason{
		Unified: unifiedFinishReason(finishReason),
		Raw:     finishReason,
	}
}

func unifiedFinishReason(finishReason string) model.UnifiedFinishReason {
	switch finishReason {
	case "stop":
		return model.FinishStop
	case "length", "model_length":
		return model.FinishLength
	case "tool_calls":
		return model.FinishToolCalls
	case "error":
		return model.FinishError
	default:
		return model.FinishOther
	}
}

// responseMetadata converts the response identification fields.
func responseMetadata(id string, modelID string, created float64) model.ResponseMetadata {
	metadata := model.ResponseMetadata{
		ID:      id,
		ModelID: modelID,
	}

	if created != 0 {
		metadata.Timestamp = time.Unix(int64(created), 0)
	}

	return metadata
}
