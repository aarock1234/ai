package openrouter

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openaicompatible"
)

// ChatMetadata is the result provider metadata of chat models.
type ChatMetadata struct {
	// Provider is the upstream provider that served the request.
	Provider string `json:"provider,omitempty"`

	// Usage is the usage accounting of the request.
	Usage *UsageAccounting `json:"usage,omitempty"`
}

// UsageAccounting reports the cost of a request in OpenRouter credits.
type UsageAccounting struct {
	Cost *float64 `json:"cost,omitempty"`

	// IsBYOK reports whether the request used the caller's own provider key.
	IsBYOK *bool `json:"isByok,omitempty"`

	// UpstreamInferenceCost is the upstream provider's charge for BYOK
	// requests.
	UpstreamInferenceCost *float64 `json:"upstreamInferenceCost,omitempty"`
}

// metadataEnvelope holds the OpenRouter members of a response or chunk.
type metadataEnvelope struct {
	Provider string     `json:"provider"`
	Usage    *usageWire `json:"usage"`
}

// usageWire is OpenRouter's usage object.
type usageWire struct {
	PromptTokens            *int               `json:"prompt_tokens"`
	CompletionTokens        *int               `json:"completion_tokens"`
	Cost                    *float64           `json:"cost"`
	IsBYOK                  *bool              `json:"is_byok"`
	CostDetails             *costDetails       `json:"cost_details"`
	PromptTokensDetails     *promptDetails     `json:"prompt_tokens_details"`
	CompletionTokensDetails *completionDetails `json:"completion_tokens_details"`
}

type costDetails struct {
	UpstreamInferenceCost *float64 `json:"upstream_inference_cost"`
}

type promptDetails struct {
	CachedTokens     *int `json:"cached_tokens"`
	CacheWriteTokens *int `json:"cache_write_tokens"`
}

type completionDetails struct {
	ReasoningTokens *int `json:"reasoning_tokens"`
}

// accounting returns the cost fields, or nil when none were reported.
func (u *usageWire) accounting() *UsageAccounting {
	if u == nil || (u.Cost == nil && u.IsBYOK == nil && u.CostDetails == nil) {
		return nil
	}

	accounting := &UsageAccounting{
		Cost:   u.Cost,
		IsBYOK: u.IsBYOK,
	}

	if u.CostDetails != nil {
		accounting.UpstreamInferenceCost = u.CostDetails.UpstreamInferenceCost
	}

	return accounting
}

// metadataExtractor reads the upstream provider and usage accounting.
type metadataExtractor struct{}

// ExtractMetadata reads the metadata of a complete response.
func (metadataExtractor) ExtractMetadata(body json.RawMessage) (model.ProviderMetadata, error) {
	var envelope metadataEnvelope
	if err := json.Unmarshal(body, &envelope); err != nil {
		return nil, fmt.Errorf("%w: %w", model.ErrInvalidResponse, err)
	}

	return resultMetadata(envelope.Provider, envelope.Usage.accounting()), nil
}

// NewStreamExtractor returns an extractor for one stream.
func (metadataExtractor) NewStreamExtractor() openaicompatible.StreamMetadataExtractor {
	return &streamMetadataExtractor{}
}

// streamMetadataExtractor keeps the last reported provider and usage.
type streamMetadataExtractor struct {
	provider   string
	accounting *UsageAccounting
}

// ProcessChunk reads one chunk. Chunks that do not decode carry no metadata;
// the chat model reports them as stream errors.
func (e *streamMetadataExtractor) ProcessChunk(chunk json.RawMessage) {
	var envelope metadataEnvelope
	if err := json.Unmarshal(chunk, &envelope); err != nil {
		return
	}

	if envelope.Provider != "" {
		e.provider = envelope.Provider
	}

	if accounting := envelope.Usage.accounting(); accounting != nil {
		e.accounting = accounting
	}
}

// BuildMetadata returns the metadata after the last chunk.
func (e *streamMetadataExtractor) BuildMetadata() model.ProviderMetadata {
	return resultMetadata(e.provider, e.accounting)
}

func resultMetadata(provider string, accounting *UsageAccounting) model.ProviderMetadata {
	return model.ProviderMetadata{
		Namespace: ChatMetadata{
			Provider: provider,
			Usage:    accounting,
		},
	}
}

// convertUsage adds OpenRouter's cache write tokens to the standard usage
// conversion.
func convertUsage(usage *openaicompatible.ChatUsage) model.Usage {
	converted := openaicompatible.ConvertChatUsage(usage)
	if usage == nil {
		return converted
	}

	// The raw usage already decoded as ChatUsage; a cache write count of an
	// unexpected type is left out rather than failing the call.
	var wire usageWire
	if err := json.Unmarshal(usage.Raw, &wire); err != nil {
		return converted
	}

	if wire.PromptTokensDetails == nil || wire.PromptTokensDetails.CacheWriteTokens == nil {
		return converted
	}

	cacheWrite := *wire.PromptTokensDetails.CacheWriteTokens
	converted.InputTokens.CacheWrite = new(cacheWrite)

	if converted.InputTokens.NoCache != nil {
		converted.InputTokens.NoCache = new(*converted.InputTokens.NoCache - cacheWrite)
	}

	return converted
}
