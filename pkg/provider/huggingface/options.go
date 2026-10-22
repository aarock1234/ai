package huggingface

import (
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// ResponsesOptions are the provider options of Responses API models, read
// from the "huggingface" namespace.
type ResponsesOptions struct {
	// Metadata is attached to the response.
	Metadata map[string]string `json:"metadata,omitempty"`

	// Instructions are system instructions sent alongside the input.
	Instructions *string `json:"instructions,omitempty"`

	// StrictJSONSchema requests strict validation of JSON schema response
	// formats. Nil means false.
	StrictJSONSchema *bool `json:"strictJsonSchema,omitempty"`

	// ReasoningEffort is the reasoning effort of reasoning models.
	ReasoningEffort *string `json:"reasoningEffort,omitempty"`
}

// ItemMetadata is the provider metadata of text and reasoning content,
// stored under "huggingface".
type ItemMetadata struct {
	// ItemID identifies the response output item.
	ItemID string `json:"itemId"`
}

// ResponsesMetadata is the result provider metadata, stored under
// "huggingface".
type ResponsesMetadata struct {
	// ResponseID is nil when a stream ended before the response was
	// created.
	ResponseID *string `json:"responseId"`
}

// readResponsesOptions decodes the "huggingface" provider options.
func readResponsesOptions(options model.ProviderOptions) (ResponsesOptions, error) {
	responsesOptions, _, err := model.ProviderValue[ResponsesOptions](options, Namespace)
	if err != nil {
		return ResponsesOptions{}, fmt.Errorf("huggingface provider options: %w", err)
	}

	return responsesOptions, nil
}

// itemMetadata identifies the output item of text and reasoning content.
func itemMetadata(itemID string) model.ProviderMetadata {
	return model.ProviderMetadata{
		Namespace: ItemMetadata{
			ItemID: itemID,
		},
	}
}
