package openrouter

import (
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// ChatOptions are the provider options of chat models, read from the
// "openrouter" namespace. The JSON names follow OpenRouter's request format
// because the options are sent in the request body unchanged, together with
// any other members of the namespace.
type ChatOptions struct {
	// Reasoning configures reasoning tokens. It takes precedence over
	// model.CallOptions.Reasoning.
	Reasoning *ReasoningOptions `json:"reasoning,omitempty"`

	// Usage configures usage accounting.
	Usage *UsageOptions `json:"usage,omitempty"`

	// Provider configures how OpenRouter routes the request to upstream
	// providers.
	Provider *ProviderRouting `json:"provider,omitempty"`

	// Models are fallback models, tried in order when the primary model
	// fails.
	Models []string `json:"models,omitempty"`

	// Transforms are prompt transforms, such as "middle-out".
	Transforms []string `json:"transforms,omitempty"`

	// Plugins enable OpenRouter plugins such as web search.
	Plugins []Plugin `json:"plugins,omitempty"`

	// User identifies the end user, to help OpenRouter detect abuse.
	User *string `json:"user,omitempty"`
}

// ReasoningEffort is an OpenRouter reasoning effort.
type ReasoningEffort string

// Reasoning efforts.
const (
	ReasoningEffortXHigh   ReasoningEffort = "xhigh"
	ReasoningEffortHigh    ReasoningEffort = "high"
	ReasoningEffortMedium  ReasoningEffort = "medium"
	ReasoningEffortLow     ReasoningEffort = "low"
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	ReasoningEffortNone    ReasoningEffort = "none"
)

// ReasoningOptions configure reasoning tokens. Set either Effort or
// MaxTokens.
type ReasoningOptions struct {
	// Enabled turns reasoning on with default settings, or off.
	Enabled *bool `json:"enabled,omitempty"`

	// Effort selects a relative reasoning budget.
	Effort ReasoningEffort `json:"effort,omitempty"`

	// MaxTokens is an absolute reasoning budget.
	MaxTokens *int `json:"max_tokens,omitempty"`

	// Exclude reasons without returning the reasoning in the response.
	Exclude *bool `json:"exclude,omitempty"`
}

// UsageOptions configure usage accounting.
type UsageOptions struct {
	// Include reports cost and detailed token counts in the usage.
	Include bool `json:"include"`
}

// DataCollection selects whether upstream providers may store prompts.
type DataCollection string

// Data collection policies.
const (
	DataCollectionAllow DataCollection = "allow"
	DataCollectionDeny  DataCollection = "deny"
)

// ProviderSort orders upstream providers.
type ProviderSort string

// Provider orderings.
const (
	ProviderSortPrice      ProviderSort = "price"
	ProviderSortThroughput ProviderSort = "throughput"
	ProviderSortLatency    ProviderSort = "latency"
)

// ProviderRouting configures upstream provider selection.
type ProviderRouting struct {
	// Order lists provider slugs to try first, in order.
	Order []string `json:"order,omitempty"`

	// AllowFallbacks permits providers outside Order. It defaults to true.
	AllowFallbacks *bool `json:"allow_fallbacks,omitempty"`

	// RequireParameters restricts routing to providers that support every
	// request parameter.
	RequireParameters *bool `json:"require_parameters,omitempty"`

	// DataCollection restricts routing by data retention policy.
	DataCollection DataCollection `json:"data_collection,omitempty"`

	// ZDR restricts routing to zero data retention endpoints.
	ZDR *bool `json:"zdr,omitempty"`

	// Only lists the only provider slugs allowed.
	Only []string `json:"only,omitempty"`

	// Ignore lists provider slugs to skip.
	Ignore []string `json:"ignore,omitempty"`

	// Quantizations lists the allowed model quantizations, such as "fp8".
	Quantizations []string `json:"quantizations,omitempty"`

	// Sort orders providers when Order is not set.
	Sort ProviderSort `json:"sort,omitempty"`

	// MaxPrice limits the price per million tokens, or per image or request.
	MaxPrice *MaxPrice `json:"max_price,omitempty"`
}

// MaxPrice limits the accepted prices in US dollars.
type MaxPrice struct {
	Prompt     *float64 `json:"prompt,omitempty"`
	Completion *float64 `json:"completion,omitempty"`
	Image      *float64 `json:"image,omitempty"`
	Audio      *float64 `json:"audio,omitempty"`
	Request    *float64 `json:"request,omitempty"`
}

// Plugin enables an OpenRouter plugin. The web plugin uses the search
// fields.
type Plugin struct {
	// ID names the plugin, such as "web" or "file-parser".
	ID string `json:"id"`

	// Engine selects the web search engine, such as "native" or "exa".
	Engine string `json:"engine,omitempty"`

	// MaxResults limits the number of web search results.
	MaxResults *int `json:"max_results,omitempty"`

	// SearchPrompt replaces the prompt that introduces search results.
	SearchPrompt string `json:"search_prompt,omitempty"`
}

// readChatOptions reads and validates the call's OpenRouter options.
func readChatOptions(options model.ProviderOptions) (ChatOptions, error) {
	chatOptions, _, err := model.ProviderValue[ChatOptions](options, Namespace)
	if err != nil {
		return ChatOptions{}, err
	}

	if err := chatOptions.validate(); err != nil {
		return ChatOptions{}, err
	}

	return chatOptions, nil
}

func (o ChatOptions) validate() error {
	if o.Reasoning != nil {
		if err := o.Reasoning.validate(); err != nil {
			return err
		}
	}

	if o.Provider != nil {
		if err := o.Provider.validate(); err != nil {
			return err
		}
	}

	for _, plugin := range o.Plugins {
		if plugin.ID == "" {
			return fmt.Errorf("%w: openrouter plugin id is required", model.ErrInvalidArgument)
		}
	}

	return nil
}

func (o ReasoningOptions) validate() error {
	switch o.Effort {
	case "", ReasoningEffortXHigh, ReasoningEffortHigh, ReasoningEffortMedium,
		ReasoningEffortLow, ReasoningEffortMinimal, ReasoningEffortNone:
	default:
		return fmt.Errorf("%w: openrouter reasoning effort %q", model.ErrInvalidArgument, o.Effort)
	}

	if o.MaxTokens != nil && *o.MaxTokens <= 0 {
		return fmt.Errorf("%w: openrouter reasoning max_tokens must be positive", model.ErrInvalidArgument)
	}

	return nil
}

func (r ProviderRouting) validate() error {
	switch r.DataCollection {
	case "", DataCollectionAllow, DataCollectionDeny:
	default:
		return fmt.Errorf("%w: openrouter data_collection %q", model.ErrInvalidArgument, r.DataCollection)
	}

	switch r.Sort {
	case "", ProviderSortPrice, ProviderSortThroughput, ProviderSortLatency:
	default:
		return fmt.Errorf("%w: openrouter provider sort %q", model.ErrInvalidArgument, r.Sort)
	}

	return nil
}
