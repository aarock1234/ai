package amazonbedrock

import (
	"encoding/json"
	"fmt"
	"slices"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// StructuredOutputMode selects how Claude models produce JSON with a schema.
type StructuredOutputMode string

// Structured output modes.
const (
	// StructuredOutputFormat uses the native output_config.format field.
	StructuredOutputFormat StructuredOutputMode = "outputFormat"

	// StructuredOutputJSONTool uses a synthetic "json" tool.
	StructuredOutputJSONTool StructuredOutputMode = "jsonTool"

	// StructuredOutputAuto uses the native format when the model supports
	// it, and the JSON tool otherwise. It is the default.
	StructuredOutputAuto StructuredOutputMode = "auto"
)

// ReasoningType selects the thinking mode of a reasoning configuration.
type ReasoningType string

// Reasoning types.
const (
	ReasoningEnabled  ReasoningType = "enabled"
	ReasoningDisabled ReasoningType = "disabled"

	// ReasoningAdaptive lets Claude size its thinking by effort.
	ReasoningAdaptive ReasoningType = "adaptive"
)

// ReasoningEffort is a maximum reasoning effort.
type ReasoningEffort string

// Reasoning efforts.
const (
	ReasoningEffortLow    ReasoningEffort = "low"
	ReasoningEffortMedium ReasoningEffort = "medium"
	ReasoningEffortHigh   ReasoningEffort = "high"
	ReasoningEffortXHigh  ReasoningEffort = "xhigh"
	ReasoningEffortMax    ReasoningEffort = "max"
)

// ReasoningDisplay selects how adaptive thinking is returned.
type ReasoningDisplay string

// Reasoning displays.
const (
	ReasoningDisplayOmitted    ReasoningDisplay = "omitted"
	ReasoningDisplaySummarized ReasoningDisplay = "summarized"
)

// ServiceTier is a Bedrock service tier.
type ServiceTier string

// Service tiers.
const (
	// ServiceTierReserved uses provisioned throughput capacity.
	ServiceTierReserved ServiceTier = "reserved"

	// ServiceTierPriority prioritizes low-latency inference when capacity
	// is available.
	ServiceTierPriority ServiceTier = "priority"

	// ServiceTierDefault is the standard on-demand tier.
	ServiceTierDefault ServiceTier = "default"

	// ServiceTierFlex is a lower-cost tier for flexible latency workloads.
	ServiceTierFlex ServiceTier = "flex"
)

// ReasoningConfig configures model reasoning.
type ReasoningConfig struct {
	Type *ReasoningType `json:"type,omitempty"`

	// BudgetTokens is the thinking budget of Claude models.
	BudgetTokens *int `json:"budgetTokens,omitempty"`

	MaxReasoningEffort *ReasoningEffort `json:"maxReasoningEffort,omitempty"`

	// Display applies to adaptive thinking.
	Display *ReasoningDisplay `json:"display,omitempty"`
}

// ChatOptions are the provider options of Converse models, read from the
// "amazonBedrock" namespace, or from "bedrock" when the call has no
// "amazonBedrock" options. Other members of the options object, such as
// guardrailConfig or performanceConfig, are sent as top-level Converse
// request fields.
type ChatOptions struct {
	// StructuredOutputMode selects how Claude models produce JSON with a
	// schema. The default is StructuredOutputAuto.
	StructuredOutputMode *StructuredOutputMode `json:"structuredOutputMode,omitempty"`

	// AdditionalModelRequestFields are inference parameters beyond the
	// Converse inferenceConfig, passed to the model as they are.
	AdditionalModelRequestFields jsonobject.Object `json:"additionalModelRequestFields,omitempty"`

	ReasoningConfig *ReasoningConfig `json:"reasoningConfig,omitempty"`

	// AnthropicBeta enables Anthropic beta features.
	AnthropicBeta []string `json:"anthropicBeta,omitempty"`

	ServiceTier *ServiceTier `json:"serviceTier,omitempty"`

	// RequestMetadata are up to 16 key-value pairs for invocation log
	// filtering and cost attribution.
	RequestMetadata map[string]string `json:"requestMetadata,omitempty"`
}

// chatOptionKeys are the option members that are not passed through as
// request fields. AnthropicBeta is passed through as well, as upstream
// does.
var chatOptionKeys = []string{
	"reasoningConfig",
	"additionalModelRequestFields",
	"serviceTier",
	"structuredOutputMode",
	"requestMetadata",
}

// anthropicChatOptions are the Anthropic options that Converse models read
// from the "anthropic" namespace.
type anthropicChatOptions struct {
	DisableParallelToolUse *bool                 `json:"disableParallelToolUse,omitempty"`
	StructuredOutputMode   *StructuredOutputMode `json:"structuredOutputMode,omitempty"`
}

// readOptions holds the decoded options of a call.
type readOptions struct {
	bedrock ChatOptions

	// passthrough are the remaining members of the Bedrock options object.
	passthrough jsonobject.Object

	anthropic anthropicChatOptions
}

// readChatOptions reads and validates the Bedrock and Anthropic options of a
// call.
func readChatOptions(options model.ProviderOptions) (readOptions, error) {
	raw, found, err := namespacedValue[json.RawMessage](options)
	if err != nil {
		return readOptions{}, err
	}

	result := readOptions{}

	if found {
		if result.bedrock, err = decodeChatOptions(raw); err != nil {
			return readOptions{}, err
		}

		object, err := jsonobject.Parse(raw)
		if err != nil {
			return readOptions{}, fmt.Errorf("%w: amazon bedrock provider options: %w", model.ErrInvalidArgument, err)
		}

		result.passthrough = withoutKeys(object, chatOptionKeys)
	}

	result.anthropic, _, err = model.ProviderValue[anthropicChatOptions](options, "anthropic")
	if err != nil {
		return readOptions{}, err
	}

	if err := validateStructuredOutputMode(result.anthropic.StructuredOutputMode); err != nil {
		return readOptions{}, err
	}

	return result, nil
}

func decodeChatOptions(raw json.RawMessage) (ChatOptions, error) {
	var options ChatOptions
	if err := json.Unmarshal(raw, &options); err != nil {
		return ChatOptions{}, fmt.Errorf("%w: amazon bedrock provider options: %w", model.ErrInvalidArgument, err)
	}

	if err := options.validate(); err != nil {
		return ChatOptions{}, err
	}

	return options, nil
}

func (o ChatOptions) validate() error {
	if err := validateStructuredOutputMode(o.StructuredOutputMode); err != nil {
		return err
	}

	if o.ServiceTier != nil {
		tiers := []ServiceTier{
			ServiceTierReserved,
			ServiceTierPriority,
			ServiceTierDefault,
			ServiceTierFlex,
		}

		if err := checkEnum("serviceTier", *o.ServiceTier, tiers); err != nil {
			return err
		}
	}

	if o.ReasoningConfig == nil {
		return nil
	}

	return o.ReasoningConfig.validate()
}

func (c *ReasoningConfig) validate() error {
	if c.Type != nil {
		types := []ReasoningType{
			ReasoningEnabled,
			ReasoningDisabled,
			ReasoningAdaptive,
		}

		if err := checkEnum("reasoningConfig.type", *c.Type, types); err != nil {
			return err
		}
	}

	if c.MaxReasoningEffort != nil {
		efforts := []ReasoningEffort{
			ReasoningEffortLow,
			ReasoningEffortMedium,
			ReasoningEffortHigh,
			ReasoningEffortXHigh,
			ReasoningEffortMax,
		}

		if err := checkEnum("reasoningConfig.maxReasoningEffort", *c.MaxReasoningEffort, efforts); err != nil {
			return err
		}
	}

	if c.Display != nil {
		displays := []ReasoningDisplay{
			ReasoningDisplayOmitted,
			ReasoningDisplaySummarized,
		}

		return checkEnum("reasoningConfig.display", *c.Display, displays)
	}

	return nil
}

func validateStructuredOutputMode(mode *StructuredOutputMode) error {
	if mode == nil {
		return nil
	}

	modes := []StructuredOutputMode{
		StructuredOutputFormat,
		StructuredOutputJSONTool,
		StructuredOutputAuto,
	}

	return checkEnum("structuredOutputMode", *mode, modes)
}

// checkEnum reports a value outside allowed as an invalid argument.
func checkEnum[T ~string](name string, value T, allowed []T) error {
	if slices.Contains(allowed, value) {
		return nil
	}

	return fmt.Errorf("%w: amazon bedrock %s %q", model.ErrInvalidArgument, name, value)
}

// namespacedValue reads a provider value from "amazonBedrock", falling back
// to the legacy "bedrock" namespace.
func namespacedValue[T any, Values ~map[string]any](values Values) (T, bool, error) {
	value, found, err := model.ProviderValue[T](values, Namespace)
	if err != nil || found {
		return value, found, err
	}

	return model.ProviderValue[T](values, LegacyNamespace)
}

func withoutKeys(object jsonobject.Object, keys []string) jsonobject.Object {
	for _, key := range keys {
		object = object.Without(key)
	}

	return object
}
