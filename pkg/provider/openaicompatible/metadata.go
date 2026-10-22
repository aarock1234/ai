package openaicompatible

import (
	"encoding/json"
	"maps"

	"github.com/aarock1234/ai/pkg/model"
)

// ChatMetadata is the result provider metadata of chat models, stored under
// the provider's name.
type ChatMetadata struct {
	// AcceptedPredictionTokens counts predicted output tokens that appeared in
	// the completion.
	AcceptedPredictionTokens *int `json:"acceptedPredictionTokens,omitempty"`

	// RejectedPredictionTokens counts predicted output tokens that did not
	// appear in the completion.
	RejectedPredictionTokens *int `json:"rejectedPredictionTokens,omitempty"`
}

// ToolCallMetadata is the provider metadata of a tool call, stored under the
// provider's name. Replaying it lets Google Gemini verify its reasoning.
type ToolCallMetadata struct {
	ThoughtSignature string `json:"thoughtSignature,omitempty"`
}

// MetadataExtractor captures provider-specific metadata, such as
// non-standard fields or experimental metrics, from chat responses. The
// returned metadata is merged into the result metadata; a namespace it sets
// replaces the model's own metadata of that namespace.
type MetadataExtractor interface {
	// ExtractMetadata reads the metadata of a complete response body. It
	// returns nil when there is none.
	ExtractMetadata(body json.RawMessage) (model.ProviderMetadata, error)

	// NewStreamExtractor returns an extractor for one streaming response.
	NewStreamExtractor() StreamMetadataExtractor
}

// StreamMetadataExtractor accumulates metadata across the chunks of one
// streaming response.
type StreamMetadataExtractor interface {
	// ProcessChunk reads one parsed chunk.
	ProcessChunk(chunk json.RawMessage)

	// BuildMetadata returns the metadata after the last chunk, or nil when
	// there is none.
	BuildMetadata() model.ProviderMetadata
}

// chatMetadata combines the model's metadata with extracted metadata.
func chatMetadata(key string, usage *ChatUsage, extracted model.ProviderMetadata) model.ProviderMetadata {
	metadata := model.ProviderMetadata{
		key: predictionMetadata(usage),
	}

	maps.Copy(metadata, extracted)

	return metadata
}

// predictionMetadata reports the predicted output token counts.
func predictionMetadata(usage *ChatUsage) ChatMetadata {
	if usage == nil || usage.CompletionTokensDetails == nil {
		return ChatMetadata{}
	}

	return ChatMetadata{
		AcceptedPredictionTokens: usage.CompletionTokensDetails.AcceptedPredictionTokens,
		RejectedPredictionTokens: usage.CompletionTokensDetails.RejectedPredictionTokens,
	}
}

// toolCallMetadata returns the metadata of a tool call with a thought
// signature, or nil.
func toolCallMetadata(key string, signature string) model.ProviderMetadata {
	if signature == "" {
		return nil
	}

	return model.ProviderMetadata{
		key: ToolCallMetadata{
			ThoughtSignature: signature,
		},
	}
}
