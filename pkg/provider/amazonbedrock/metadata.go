package amazonbedrock

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// ChatMetadata is the provider metadata of a Converse call, stored under
// both "amazonBedrock" and "bedrock".
type ChatMetadata struct {
	// Trace is the guardrail trace, when the response included one.
	Trace json.RawMessage `json:"trace,omitempty"`

	PerformanceConfig *PerformanceConfig `json:"performanceConfig,omitempty"`
	ServiceTier       *ServiceTierInfo   `json:"serviceTier,omitempty"`

	// Usage reports prompt cache writes, when there were any.
	Usage *CacheUsage `json:"usage,omitempty"`

	// IsJSONResponseFromTool reports that the JSON response arrived as a
	// call of the synthetic "json" tool.
	IsJSONResponseFromTool bool `json:"isJsonResponseFromTool,omitempty"`

	// StopSequence is the stop sequence that ended generation of a Claude
	// model, if any.
	StopSequence *string `json:"stopSequence"`
}

// PerformanceConfig is the latency setting a call used.
type PerformanceConfig struct {
	Latency string `json:"latency"`
}

// ServiceTierInfo is the service tier a call used.
type ServiceTierInfo struct {
	Type string `json:"type"`
}

// CacheUsage reports prompt cache writes.
type CacheUsage struct {
	CacheWriteInputTokens *int          `json:"cacheWriteInputTokens,omitempty"`
	CacheDetails          []CacheDetail `json:"cacheDetails,omitempty"`
}

// CacheDetail is the number of tokens cached with one lifetime.
type CacheDetail struct {
	InputTokens int    `json:"inputTokens"`
	TTL         string `json:"ttl"`
}

// ReasoningMetadata is the provider metadata of a reasoning part, stored
// under both "amazonBedrock" and "bedrock". Passing it back as the part's
// provider options replays the reasoning in later turns.
type ReasoningMetadata struct {
	// Signature verifies Claude's reasoning text.
	Signature string `json:"signature,omitempty"`

	// RedactedData is Claude's encrypted reasoning.
	RedactedData string `json:"redactedData,omitempty"`

	// RedactedContent is the encrypted reasoning of other models, such as
	// OpenAI models on Bedrock.
	RedactedContent string `json:"redactedContent,omitempty"`
}

// providerMetadataOf stores value under both namespaces.
func providerMetadataOf(value any) model.ProviderMetadata {
	return model.ProviderMetadata{
		Namespace:       value,
		LegacyNamespace: value,
	}
}

// decodeUsage decodes and validates the usage object of a response.
// requireTotal reports whether totalTokens is required.
func decodeUsage(raw json.RawMessage, requireTotal bool) (*converseUsage, error) {
	var usage converseUsage
	if err := json.Unmarshal(raw, &usage); err != nil {
		return nil, fmt.Errorf("%w: usage: %w", model.ErrInvalidResponse, err)
	}

	if usage.InputTokens == nil || usage.OutputTokens == nil || (requireTotal && usage.TotalTokens == nil) {
		return nil, fmt.Errorf("%w: usage is missing token counts", model.ErrInvalidResponse)
	}

	return &usage, nil
}

// cacheUsageOf returns the cache metadata of usage, or nil when it reports
// no cache writes.
func cacheUsageOf(usage *converseUsage) *CacheUsage {
	if usage == nil || (usage.CacheWriteInputTokens == nil && usage.CacheDetails == nil) {
		return nil
	}

	return &CacheUsage{
		CacheWriteInputTokens: usage.CacheWriteInputTokens,
		CacheDetails:          usage.CacheDetails,
	}
}

// traceOf returns a trace when it is a JSON object.
func traceOf(trace json.RawMessage) json.RawMessage {
	if !jsonobject.IsObject(trace) {
		return nil
	}

	return trace
}

// convertUsage converts Bedrock usage, counting cache reads and writes as
// input tokens. Missing usage converts to unknown counts.
func convertUsage(usage *converseUsage, raw json.RawMessage) model.Usage {
	if usage == nil {
		return model.Usage{}
	}

	inputTokens := *usage.InputTokens
	outputTokens := *usage.OutputTokens
	cacheRead := valueOrZero(usage.CacheReadInputTokens)
	cacheWrite := valueOrZero(usage.CacheWriteInputTokens)

	return model.Usage{
		InputTokens: model.InputTokens{
			Total:      new(inputTokens + cacheRead + cacheWrite),
			NoCache:    new(inputTokens),
			CacheRead:  new(cacheRead),
			CacheWrite: new(cacheWrite),
		},
		OutputTokens: model.OutputTokens{
			Total: new(outputTokens),
			Text:  new(outputTokens),
		},
		Raw: bytes.Clone(raw),
	}
}

func valueOrZero[T any](value *T) T {
	if value == nil {
		var zero T

		return zero
	}

	return *value
}
