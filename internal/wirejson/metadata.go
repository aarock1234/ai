package wirejson

import (
	"encoding/json"

	"github.com/aarock1234/ai/pkg/model"
)

// Usage is encoded token usage. Unknown counts are omitted.
type Usage struct {
	InputTokens  InputTokens     `json:"inputTokens"`
	OutputTokens OutputTokens    `json:"outputTokens"`
	Raw          json.RawMessage `json:"raw,omitempty"`
}

// InputTokens are encoded input token counts.
type InputTokens struct {
	Total      *int `json:"total,omitempty"`
	NoCache    *int `json:"noCache,omitempty"`
	CacheRead  *int `json:"cacheRead,omitempty"`
	CacheWrite *int `json:"cacheWrite,omitempty"`
}

// OutputTokens are encoded output token counts.
type OutputTokens struct {
	Total     *int `json:"total,omitempty"`
	Text      *int `json:"text,omitempty"`
	Reasoning *int `json:"reasoning,omitempty"`
}

// FinishReason is an encoded finish reason.
type FinishReason struct {
	Unified model.UnifiedFinishReason `json:"unified"`
	Raw     string                    `json:"raw,omitempty"`
}

// Warning is an encoded call warning.
type Warning struct {
	Type    model.WarningType `json:"type"`
	Feature string            `json:"feature,omitempty"`
	Details string            `json:"details,omitempty"`
	Setting string            `json:"setting,omitempty"`
	Message string            `json:"message,omitempty"`
}

// ResponseMetadata is encoded response metadata. The timestamp is formatted
// like a JavaScript date.
type ResponseMetadata struct {
	ID        string `json:"id,omitempty"`
	Timestamp string `json:"timestamp,omitempty"`
	ModelID   string `json:"modelId,omitempty"`
}

// ResponseFormat is an encoded response format.
type ResponseFormat struct {
	Type        model.ResponseFormatType `json:"type"`
	Schema      json.RawMessage          `json:"schema,omitempty"`
	Name        string                   `json:"name,omitempty"`
	Description string                   `json:"description,omitempty"`
}

// EncodeUsage encodes token usage.
func EncodeUsage(usage model.Usage) Usage {
	encoded := Usage{
		InputTokens: InputTokens{
			Total:      usage.InputTokens.Total,
			NoCache:    usage.InputTokens.NoCache,
			CacheRead:  usage.InputTokens.CacheRead,
			CacheWrite: usage.InputTokens.CacheWrite,
		},
		OutputTokens: OutputTokens{
			Total:     usage.OutputTokens.Total,
			Text:      usage.OutputTokens.Text,
			Reasoning: usage.OutputTokens.Reasoning,
		},
	}

	if len(usage.Raw) > 0 {
		encoded.Raw = RawValue(usage.Raw)
	}

	return encoded
}

// EncodeFinishReason encodes a finish reason.
func EncodeFinishReason(reason model.FinishReason) FinishReason {
	return FinishReason{
		Unified: reason.Unified,
		Raw:     reason.Raw,
	}
}

// EncodeWarnings encodes call warnings. It never returns nil, so that the
// warnings encode as an array.
func EncodeWarnings(warnings []model.Warning) []Warning {
	encoded := make([]Warning, 0, len(warnings))

	for _, warning := range warnings {
		encoded = append(encoded, Warning{
			Type:    warning.Type,
			Feature: warning.Feature,
			Details: warning.Details,
			Setting: warning.Setting,
			Message: warning.Message,
		})
	}

	return encoded
}

// EncodeResponseMetadata encodes response metadata, omitting a zero
// timestamp.
func EncodeResponseMetadata(metadata model.ResponseMetadata) ResponseMetadata {
	encoded := ResponseMetadata{
		ID:      metadata.ID,
		ModelID: metadata.ModelID,
	}

	if !metadata.Timestamp.IsZero() {
		encoded.Timestamp = Timestamp(metadata.Timestamp)
	}

	return encoded
}

// EncodeResponseFormat encodes a response format.
func EncodeResponseFormat(format model.ResponseFormat) ResponseFormat {
	return ResponseFormat{
		Type:        format.Type,
		Schema:      format.Schema,
		Name:        format.Name,
		Description: format.Description,
	}
}
