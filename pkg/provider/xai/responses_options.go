package xai

import (
	"errors"
	"fmt"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// ReasoningEffort constrains how hard a reasoning model thinks.
type ReasoningEffort string

// Reasoning efforts. None disables reasoning on grok-4.3 and newer
// reasoning models; xhigh is supported by grok-4.6.
const (
	ReasoningEffortNone   ReasoningEffort = "none"
	ReasoningEffortLow    ReasoningEffort = "low"
	ReasoningEffortMedium ReasoningEffort = "medium"
	ReasoningEffortHigh   ReasoningEffort = "high"
	ReasoningEffortXHigh  ReasoningEffort = "xhigh"
)

var reasoningEfforts = []ReasoningEffort{
	ReasoningEffortNone,
	ReasoningEffortLow,
	ReasoningEffortMedium,
	ReasoningEffortHigh,
	ReasoningEffortXHigh,
}

// IncludeValue names additional output data to include.
type IncludeValue string

// Include values.
const (
	IncludeFileSearchResults     IncludeValue = "file_search_call.results"
	IncludeWebSearchSources      IncludeValue = "web_search_call.action.sources"
	IncludeCodeInterpreterOutput IncludeValue = "code_interpreter_call.outputs"
	IncludeEncryptedReasoning    IncludeValue = "reasoning.encrypted_content"
	IncludeNoInlineCitations     IncludeValue = "no_inline_citations"
)

var includeValues = []IncludeValue{
	IncludeFileSearchResults,
	IncludeWebSearchSources,
	IncludeCodeInterpreterOutput,
	IncludeEncryptedReasoning,
	IncludeNoInlineCitations,
}

// Option limits.
const (
	maxTopLogprobs = 8
)

// ResponsesOptions are the provider options of Responses API models, read
// from the "xai" namespace.
type ResponsesOptions struct {
	// ReasoningEffort takes precedence over the call's reasoning level.
	ReasoningEffort ReasoningEffort `json:"reasoningEffort,omitempty"`

	// ReasoningSummary is accepted for compatibility and ignored by xAI,
	// which always returns a detailed reasoning summary.
	//
	// Deprecated: xAI ignores the requested summary level.
	ReasoningSummary string `json:"reasoningSummary,omitempty"`

	// Logprobs returns the log probabilities of the generated tokens.
	Logprobs *bool `json:"logprobs,omitempty"`

	// TopLogprobs is the number, from 0 to 8, of most likely tokens to
	// return at each position. Setting it enables Logprobs.
	TopLogprobs *int `json:"topLogprobs,omitempty"`

	// MinP is the min-p sampling threshold between 0 and 1.
	MinP *float64 `json:"minP,omitempty"`

	// MaxTurns limits the agentic tool-calling turns.
	MaxTurns *int `json:"maxTurns,omitempty"`

	// ParallelToolCalls allows parallel tool calls. Nil means true.
	ParallelToolCalls *bool `json:"parallelToolCalls,omitempty"`

	// PromptCacheKey routes requests with shared prompt prefixes.
	PromptCacheKey *string `json:"promptCacheKey,omitempty"`

	// SafetyIdentifier attributes policy violations to an end user.
	SafetyIdentifier *string `json:"safetyIdentifier,omitempty"`

	// ServiceTier is "default" or "priority".
	ServiceTier string `json:"serviceTier,omitempty"`

	// Store controls whether the input and response are stored for later
	// retrieval. It must be false for teams with Zero Data Retention. Nil
	// means true. When false, encrypted reasoning is included so that
	// reasoning can be replayed.
	Store *bool `json:"store,omitempty"`

	// PreviousResponseID continues from an earlier response.
	PreviousResponseID *string `json:"previousResponseId,omitempty"`

	// User identifies the end user for abuse monitoring.
	User *string `json:"user,omitempty"`

	// Include requests additional output data.
	Include []IncludeValue `json:"include,omitempty"`
}

// FilePartOptions are the provider options of image file parts, read from
// the "xai" namespace.
type FilePartOptions struct {
	// ImageDetail is "low", "high" or "auto". Empty processes the image at
	// full resolution. xAI ignores invalid values, so they are rejected.
	ImageDetail string `json:"imageDetail,omitempty"`
}

// ResponsesMetadata is the result provider metadata of Responses API
// models, stored under "xai" when any member is reported.
type ResponsesMetadata struct {
	CostInUSDTicks   *float64 `json:"costInUsdTicks,omitempty"`
	ServiceTier      string   `json:"serviceTier,omitempty"`
	PromptCacheKey   string   `json:"promptCacheKey,omitempty"`
	SafetyIdentifier string   `json:"safetyIdentifier,omitempty"`
}

// empty reports whether no member was reported.
func (m ResponsesMetadata) empty() bool {
	return m.CostInUSDTicks == nil && m.ServiceTier == "" && m.PromptCacheKey == "" && m.SafetyIdentifier == ""
}

// providerMetadata returns the metadata, or nil when it is empty.
func (m ResponsesMetadata) providerMetadata() model.ProviderMetadata {
	if m.empty() {
		return nil
	}

	return model.ProviderMetadata{
		Namespace: m,
	}
}

// ReasoningMetadata is the provider metadata of reasoning, stored under
// "xai". Replaying it restores the reasoning item.
type ReasoningMetadata struct {
	ItemID                    string `json:"itemId,omitempty"`
	ReasoningEncryptedContent string `json:"reasoningEncryptedContent,omitempty"`
}

// ItemMetadata is the provider metadata of text, stored under "xai".
type ItemMetadata struct {
	ItemID string `json:"itemId,omitempty"`
}

// readResponsesOptions decodes and validates the "xai" provider options.
func readResponsesOptions(options model.ProviderOptions) (ResponsesOptions, error) {
	responsesOptions, _, err := model.ProviderValue[ResponsesOptions](options, Namespace)
	if err != nil {
		return ResponsesOptions{}, err
	}

	if err := responsesOptions.validate(); err != nil {
		return ResponsesOptions{}, fmt.Errorf("%w: xai provider options: %w", model.ErrInvalidArgument, err)
	}

	return responsesOptions, nil
}

func (o ResponsesOptions) validate() error {
	if o.ReasoningEffort != "" && !slices.Contains(reasoningEfforts, o.ReasoningEffort) {
		return fmt.Errorf("unsupported reasoningEffort %q", o.ReasoningEffort)
	}

	if o.ReasoningSummary != "" && !slices.Contains([]string{"auto", "concise", "detailed"}, o.ReasoningSummary) {
		return fmt.Errorf("unsupported reasoningSummary %q", o.ReasoningSummary)
	}

	if o.TopLogprobs != nil && (*o.TopLogprobs < 0 || *o.TopLogprobs > maxTopLogprobs) {
		return fmt.Errorf("topLogprobs must be between 0 and %d", maxTopLogprobs)
	}

	if o.MinP != nil && (*o.MinP < 0 || *o.MinP > 1) {
		return errors.New("minP must be between 0 and 1")
	}

	if o.ServiceTier != "" && o.ServiceTier != "default" && o.ServiceTier != "priority" {
		return fmt.Errorf("unsupported serviceTier %q", o.ServiceTier)
	}

	for _, value := range o.Include {
		if !slices.Contains(includeValues, value) {
			return fmt.Errorf("unsupported include value %q", value)
		}
	}

	return nil
}

// readFilePartOptions decodes and validates the options of an image part.
func readFilePartOptions(options model.ProviderOptions) (FilePartOptions, error) {
	partOptions, _, err := model.ProviderValue[FilePartOptions](options, Namespace)
	if err != nil {
		return FilePartOptions{}, err
	}

	if partOptions.ImageDetail != "" && !slices.Contains([]string{"low", "high", "auto"}, partOptions.ImageDetail) {
		return FilePartOptions{}, fmt.Errorf("%w: xai file part options: unsupported imageDetail %q", model.ErrInvalidArgument, partOptions.ImageDetail)
	}

	return partOptions, nil
}

// replayedItem is the replay data of an earlier output part.
type replayedItem struct {
	ItemID                    *string `json:"itemId"`
	ReasoningEncryptedContent *string `json:"reasoningEncryptedContent"`
}

// readReplayedItem reads replay data. Data of an unexpected shape is
// ignored.
func readReplayedItem(options model.ProviderOptions) replayedItem {
	item, _, err := model.ProviderValue[replayedItem](options, Namespace)
	if err != nil {
		return replayedItem{}
	}

	return item
}
