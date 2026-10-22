package quiverai

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openresponses"
)

// ResponsesOptions are the provider options of language models, read from
// the "quiverai" namespace.
type ResponsesOptions struct {
	// ReasoningEffort controls the amount of reasoning. It takes precedence
	// over the call's reasoning level.
	ReasoningEffort *ReasoningEffort `json:"reasoningEffort,omitempty"`

	// ReasoningSummary requests the model's safe reasoning summary. Only
	// openresponses.ReasoningSummaryAuto is supported.
	ReasoningSummary *openresponses.ReasoningSummary `json:"reasoningSummary,omitempty"`
}

// readResponsesOptions decodes and validates the language model options.
func readResponsesOptions(options model.ProviderOptions) (ResponsesOptions, error) {
	responsesOptions, _, err := model.ProviderValue[ResponsesOptions](options, Namespace)
	if err != nil {
		return ResponsesOptions{}, fmt.Errorf("quiverai provider options: %w", err)
	}

	if effort := responsesOptions.ReasoningEffort; effort != nil {
		switch *effort {
		case ReasoningLow, ReasoningMedium, ReasoningHigh, ReasoningXHigh:
		default:
			return ResponsesOptions{}, fmt.Errorf("%w: providerOptions: unsupported reasoning effort: %s", model.ErrInvalidArgument, *effort)
		}
	}

	if summary := responsesOptions.ReasoningSummary; summary != nil && *summary != openresponses.ReasoningSummaryAuto {
		return ResponsesOptions{}, fmt.Errorf("%w: providerOptions: unsupported reasoning summary: %s", model.ErrInvalidArgument, *summary)
	}

	return responsesOptions, nil
}

// reasoningReplay is the part of replayed reasoning metadata that QuiverAI
// accepts. Members keep the JSON the earlier response returned.
type reasoningReplay struct {
	ItemID           json.RawMessage `json:"itemId,omitempty"`
	ReasoningSummary json.RawMessage `json:"reasoningSummary,omitempty"`
}

// readReasoningReplay decodes the item ID and summary of replayed reasoning.
// A namespace of an unexpected shape has no replay data.
func readReasoningReplay(options model.ProviderOptions) reasoningReplay {
	replay, _, err := model.ProviderValue[reasoningReplay](options, Namespace)
	if err != nil {
		return reasoningReplay{}
	}

	return replay
}
