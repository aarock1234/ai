package gateway

import "encoding/json"

// Caching selects the gateway's caching behavior.
type Caching string

// CachingAuto enables automatic caching when supported.
const CachingAuto Caching = "auto"

// ServiceTier is a unified service tier intent.
type ServiceTier string

// Service tiers.
const (
	ServiceTierFlex     ServiceTier = "flex"
	ServiceTierPriority ServiceTier = "priority"
)

// Sort orders providers by a metric before routing.
type Sort string

// Sort metrics.
const (
	SortCost Sort = "cost"
	SortTPS  Sort = "tps"
	SortTTFT Sort = "ttft"
)

// Capability requirements for Has. Weight formats are required with
// "quantization:<format>" and excluded with "!quantization:<format>".
const (
	CapabilityImplicitCaching  = "implicit-caching"
	CapabilityReasoning        = "reasoning"
	CapabilityStructuredOutput = "structured-output"
	CapabilityToolUse          = "tool-use"
	CapabilityVision           = "vision"
)

// ProviderOptions are the gateway routing options of every model kind, set
// under ProviderOptions["gateway"]. The gateway validates them; options it
// adds later can be passed as a map instead.
type ProviderOptions struct {
	// BYOK holds request-scoped credentials per provider, used instead of
	// stored credentials.
	BYOK map[string][]map[string]json.RawMessage `json:"byok,omitempty"`

	// Caching enables automatic caching.
	Caching Caching `json:"caching,omitempty"`

	// DisallowPromptTraining routes only to providers that do not train on
	// prompts.
	DisallowPromptTraining *bool `json:"disallowPromptTraining,omitempty"`

	// Has restricts routing to provider models with every listed capability
	// or weight format condition.
	Has []string `json:"has,omitempty"`

	// IdempotencyKey makes batch submissions replay instead of duplicating.
	IdempotencyKey string `json:"idempotencyKey,omitempty"`

	// Models are fallback models to try in order.
	Models []string `json:"models,omitempty"`

	// Only lists the provider slugs that may serve the request.
	Only []string `json:"only,omitempty"`

	// Order lists provider slugs in the order to attempt them.
	Order []string `json:"order,omitempty"`

	// ProviderTimeouts sets per-provider timeouts.
	ProviderTimeouts *ProviderTimeouts `json:"providerTimeouts,omitempty"`

	// QuotaEntityID is the entity that quota is tracked against.
	QuotaEntityID string `json:"quotaEntityId,omitempty"`

	// ServiceTier is the service tier intent.
	ServiceTier ServiceTier `json:"serviceTier,omitempty"`

	// Sort orders providers by a metric before routing.
	Sort Sort `json:"sort,omitempty"`

	// Tags label usage for reporting and filtering.
	Tags []string `json:"tags,omitempty"`

	// User identifies the end user for spend tracking.
	User string `json:"user,omitempty"`

	// ZeroDataRetention routes only to providers with zero data retention
	// agreements.
	ZeroDataRetention *bool `json:"zeroDataRetention,omitempty"`
}

// ProviderTimeouts are per-provider timeouts.
type ProviderTimeouts struct {
	// BYOK maps provider slugs to timeouts in milliseconds for requests
	// with your own credentials.
	BYOK map[string]float64 `json:"byok,omitempty"`
}
