package anthropic

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// MessagesOptions are the Messages API call options, read from
// CallOptions.ProviderOptions under the "anthropic" key.
type MessagesOptions struct {
	// SendReasoning controls whether earlier reasoning is sent back to the
	// model. Nil sends it.
	SendReasoning *bool `json:"sendReasoning,omitempty"`

	// StructuredOutputMode selects how JSON output is produced. Empty means
	// StructuredOutputAuto.
	StructuredOutputMode StructuredOutputMode `json:"structuredOutputMode,omitempty"`

	// Thinking configures extended thinking.
	Thinking *ThinkingOptions `json:"thinking,omitempty"`

	// DisableParallelToolUse limits the model to at most one tool call per
	// response.
	DisableParallelToolUse *bool `json:"disableParallelToolUse,omitempty"`

	// CacheControl sets a request-level cache breakpoint.
	CacheControl *CacheControl `json:"cacheControl,omitempty"`

	// Metadata is sent with the request.
	Metadata *RequestMetadata `json:"metadata,omitempty"`

	// MCPServers are remote MCP servers the model may call.
	MCPServers []MCPServer `json:"mcpServers,omitempty"`

	// Container reuses a code execution container or loads agent skills.
	Container *ContainerOptions `json:"container,omitempty"`

	// ToolStreaming enables eager input streaming for function tools while
	// streaming. Nil enables it.
	ToolStreaming *bool `json:"toolStreaming,omitempty"`

	// Effort controls how much effort the model spends on thinking, text and
	// tool calls.
	Effort Effort `json:"effort,omitempty"`

	// TaskBudget informs the model of the total token budget of an agentic
	// task. It is advisory and does not enforce a limit.
	TaskBudget *TaskBudget `json:"taskBudget,omitempty"`

	// Speed selects fast or standard inference.
	Speed Speed `json:"speed,omitempty"`

	// ServiceTier selects Anthropic's service tier.
	ServiceTier ServiceTier `json:"serviceTier,omitempty"`

	// InferenceGeo restricts where inference runs.
	InferenceGeo InferenceGeo `json:"inferenceGeo,omitempty"`

	// Fallbacks configures server-side fallback models for turns that the
	// primary model's safety classifiers block.
	Fallbacks *Fallbacks `json:"fallbacks,omitempty"`

	// AnthropicBeta lists additional beta features to enable.
	AnthropicBeta []string `json:"anthropicBeta,omitempty"`

	// Safeguards are server-side checks run as part of the request. Their
	// verdicts are reported in MessagesMetadata.SafeguardResults.
	Safeguards []Safeguard `json:"safeguards,omitempty"`

	// Compaction requests an on-demand summary of the conversation. It
	// cannot be combined with ContextManagement.
	Compaction *Compaction `json:"compaction,omitempty"`

	// ContextManagement configures automatic context editing.
	ContextManagement *ContextManagement `json:"contextManagement,omitempty"`
}

// StructuredOutputMode selects how structured output is generated.
type StructuredOutputMode string

// Structured output modes.
const (
	// StructuredOutputFormat uses the native output_config.format parameter.
	StructuredOutputFormat StructuredOutputMode = "outputFormat"

	// StructuredOutputJSONTool forces a call to a synthetic "json" tool.
	StructuredOutputJSONTool StructuredOutputMode = "jsonTool"

	// StructuredOutputAuto uses the native format when the model supports it
	// and the JSON tool otherwise.
	StructuredOutputAuto StructuredOutputMode = "auto"
)

// ThinkingType selects the extended thinking mode.
type ThinkingType string

// Thinking types.
const (
	// ThinkingAdaptive lets the model decide how much to think. Use Effort to
	// steer it.
	ThinkingAdaptive ThinkingType = "adaptive"

	// ThinkingEnabled thinks within BudgetTokens.
	ThinkingEnabled ThinkingType = "enabled"

	// ThinkingDisabled turns thinking off.
	ThinkingDisabled ThinkingType = "disabled"

	// ThinkingBetweenTools skips upfront thinking but returns progress notes
	// between tool calls.
	ThinkingBetweenTools ThinkingType = "between_tools"
)

// ThinkingDisplay controls whether adaptive thinking content is returned.
type ThinkingDisplay string

// Thinking display modes.
const (
	ThinkingDisplayOmitted    ThinkingDisplay = "omitted"
	ThinkingDisplaySummarized ThinkingDisplay = "summarized"
	ThinkingDisplayUpdates    ThinkingDisplay = "updates"
)

// PrefixMismatchBehavior controls what happens when replayed thinking does
// not match the existing thinking prefix.
type PrefixMismatchBehavior string

// Prefix mismatch behaviors.
const (
	PrefixMismatchError     PrefixMismatchBehavior = "error"
	PrefixMismatchDropBlock PrefixMismatchBehavior = "drop_block"
)

// ThinkingOptions configures extended thinking. Type may be empty only when
// BlockBinding is set, for recovery requests that send binding controls
// alone.
type ThinkingOptions struct {
	Type ThinkingType `json:"type,omitempty"`

	// BudgetTokens is the thinking budget for ThinkingEnabled. Nil uses 1024
	// tokens with a warning.
	BudgetTokens *int `json:"budgetTokens,omitempty"`

	// Display applies to ThinkingAdaptive.
	Display ThinkingDisplay `json:"display,omitempty"`

	// BlockBinding applies to ThinkingAdaptive and to type-less recovery
	// requests.
	BlockBinding *BlockBinding `json:"blockBinding,omitempty"`
}

// BlockBinding controls how thinking blocks bind to an existing thinking
// prefix.
type BlockBinding struct {
	PrefixMismatchBehavior PrefixMismatchBehavior `json:"prefixMismatchBehavior"`
}

// Effort is a model effort level.
type Effort string

// Effort levels.
const (
	EffortLow    Effort = "low"
	EffortMedium Effort = "medium"
	EffortHigh   Effort = "high"
	EffortXHigh  Effort = "xhigh"
	EffortMax    Effort = "max"
)

// Speed selects the inference speed.
type Speed string

// Inference speeds.
const (
	SpeedFast     Speed = "fast"
	SpeedStandard Speed = "standard"
)

// ServiceTier selects Anthropic's service tier.
type ServiceTier string

// Service tiers.
const (
	ServiceTierAuto         ServiceTier = "auto"
	ServiceTierStandardOnly ServiceTier = "standard_only"
)

// InferenceGeo selects where inference runs.
type InferenceGeo string

// Inference geographies.
const (
	InferenceGeoUS     InferenceGeo = "us"
	InferenceGeoGlobal InferenceGeo = "global"
)

// CacheControlType is the kind of cache breakpoint.
type CacheControlType string

// CacheEphemeral is the only cache breakpoint type.
const CacheEphemeral CacheControlType = "ephemeral"

// CacheTTL is the lifetime of a cache entry.
type CacheTTL string

// Cache lifetimes.
const (
	CacheTTL5Minutes CacheTTL = "5m"
	CacheTTL1Hour    CacheTTL = "1h"
)

// CacheControl marks a prompt caching breakpoint. Anthropic accepts at most
// four breakpoints per request.
type CacheControl struct {
	Type CacheControlType `json:"type"`
	TTL  CacheTTL         `json:"ttl,omitempty"`
}

// CacheControlOptions carries a cache breakpoint for a message, a prompt
// part, a tool result output or a function tool. Set it under the
// "anthropic" provider options key.
type CacheControlOptions struct {
	CacheControl *CacheControl `json:"cacheControl,omitempty"`
}

// RequestMetadata is metadata sent with the request.
type RequestMetadata struct {
	// UserID is an opaque identifier for the end user. It must not contain
	// personal information.
	UserID *string `json:"userId,omitempty"`
}

// MCPServer is a remote MCP server the model may call.
type MCPServer struct {
	// Type must be "url".
	Type               string                `json:"type"`
	Name               string                `json:"name"`
	URL                string                `json:"url"`
	AuthorizationToken *string               `json:"authorizationToken,omitempty"`
	ToolConfiguration  *MCPToolConfiguration `json:"toolConfiguration,omitempty"`
}

// MCPToolConfiguration restricts the tools of an MCP server.
type MCPToolConfiguration struct {
	Enabled      *bool    `json:"enabled,omitempty"`
	AllowedTools []string `json:"allowedTools,omitempty"`
}

// ContainerOptions selects a code execution container and the agent skills
// loaded into it.
type ContainerOptions struct {
	ID     string  `json:"id,omitempty"`
	Skills []Skill `json:"skills,omitempty"`
}

// SkillType is the origin of an agent skill.
type SkillType string

// Skill types.
const (
	SkillAnthropic SkillType = "anthropic"
	SkillCustom    SkillType = "custom"
)

// Skill is an agent skill loaded into the container. Anthropic skills use
// SkillID; custom skills use ProviderReference, which maps provider names to
// skill IDs and must contain "anthropic".
type Skill struct {
	Type              SkillType         `json:"type"`
	SkillID           string            `json:"skillId,omitempty"`
	ProviderReference map[string]string `json:"providerReference,omitempty"`
	Version           string            `json:"version,omitempty"`
}

// TaskBudget is the advisory token budget of an agentic task.
type TaskBudget struct {
	// Type must be "tokens".
	Type string `json:"type"`

	// Total is at least 20000.
	Total int `json:"total"`

	// Remaining is optional and not negative.
	Remaining *int `json:"remaining,omitempty"`
}

// Fallbacks selects server-side fallback models: Anthropic's recommended
// fallback when Default is set, or an explicit Chain.
type Fallbacks struct {
	Default bool
	Chain   []FallbackModel
}

// FallbackModel is one entry of an explicit fallback chain. It is sent to the
// API as given, with snake_case keys.
type FallbackModel struct {
	Model        string          `json:"model"`
	MaxTokens    *int            `json:"max_tokens,omitempty"`
	Thinking     json.RawMessage `json:"thinking,omitempty"`
	OutputConfig json.RawMessage `json:"output_config,omitempty"`
	Speed        Speed           `json:"speed,omitempty"`
}

// MarshalJSON encodes "default" or the fallback chain.
func (f Fallbacks) MarshalJSON() ([]byte, error) {
	if f.Default {
		return json.Marshal("default")
	}

	if f.Chain == nil {
		return []byte("[]"), nil
	}

	return json.Marshal(f.Chain)
}

// UnmarshalJSON decodes "default" or a fallback chain.
func (f *Fallbacks) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)

	if len(trimmed) > 0 && trimmed[0] == '"' {
		var value string
		if err := json.Unmarshal(trimmed, &value); err != nil {
			return err
		}

		if value != "default" {
			return fmt.Errorf("fallbacks must be \"default\" or a list, got %q", value)
		}

		*f = Fallbacks{
			Default: true,
		}

		return nil
	}

	var chain []FallbackModel
	if err := json.Unmarshal(trimmed, &chain); err != nil {
		return err
	}

	*f = Fallbacks{
		Chain: chain,
	}

	return nil
}

func (f Fallbacks) isSet() bool {
	return f.Default || len(f.Chain) > 0
}

// Safeguard is a server-side check. Type must be "dangerous_tool_use".
type Safeguard struct {
	Type              string          `json:"type"`
	ClassifierContext json.RawMessage `json:"classifierContext,omitempty"`
}

// Compaction requests an on-demand conversation summary. Type must be
// "summarize".
type Compaction struct {
	Type         string  `json:"type"`
	Instructions *string `json:"instructions,omitempty"`
}

// ContextManagement configures automatic context editing.
type ContextManagement struct {
	Edits []ContextEdit `json:"edits"`
}

// ContextEditType is a context editing strategy.
type ContextEditType string

// Context editing strategies.
const (
	ContextEditClearToolUses ContextEditType = "clear_tool_uses_20250919"
	ContextEditClearThinking ContextEditType = "clear_thinking_20251015"
	ContextEditCompact       ContextEditType = "compact_20260112"
)

// ContextEdit is one context editing strategy. Each field applies only to
// the strategies named in its comment.
type ContextEdit struct {
	Type ContextEditType `json:"type"`

	// Trigger starts the edit: input_tokens or tool_uses for
	// ContextEditClearToolUses, input_tokens for ContextEditCompact.
	Trigger *ContextAmount `json:"trigger,omitempty"`

	// Keep preserves recent content: tool_uses for ContextEditClearToolUses,
	// "all" or thinking_turns for ContextEditClearThinking.
	Keep *ContextKeep `json:"keep,omitempty"`

	// ClearAtLeast applies to ContextEditClearToolUses.
	ClearAtLeast *ContextAmount `json:"clearAtLeast,omitempty"`

	// ClearToolInputs applies to ContextEditClearToolUses.
	ClearToolInputs *bool `json:"clearToolInputs,omitempty"`

	// ExcludeTools applies to ContextEditClearToolUses.
	ExcludeTools []string `json:"excludeTools,omitempty"`

	// PauseAfterCompaction applies to ContextEditCompact.
	PauseAfterCompaction *bool `json:"pauseAfterCompaction,omitempty"`

	// Instructions applies to ContextEditCompact.
	Instructions *string `json:"instructions,omitempty"`
}

// ContextAmount is a typed count, such as {"type": "input_tokens",
// "value": 10000}.
type ContextAmount struct {
	Type  string `json:"type"`
	Value int    `json:"value"`
}

// ContextKeep is either "all" or a typed count.
type ContextKeep struct {
	All    bool
	Amount ContextAmount
}

// MarshalJSON encodes "all" or the count.
func (k ContextKeep) MarshalJSON() ([]byte, error) {
	if k.All {
		return json.Marshal("all")
	}

	return json.Marshal(k.Amount)
}

// UnmarshalJSON decodes "all" or a count.
func (k *ContextKeep) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)

	if len(trimmed) > 0 && trimmed[0] == '"' {
		var value string
		if err := json.Unmarshal(trimmed, &value); err != nil {
			return err
		}

		if value != "all" {
			return fmt.Errorf("keep must be \"all\" or a count, got %q", value)
		}

		*k = ContextKeep{
			All: true,
		}

		return nil
	}

	var amount ContextAmount
	if err := json.Unmarshal(trimmed, &amount); err != nil {
		return err
	}

	*k = ContextKeep{
		Amount: amount,
	}

	return nil
}

// FilePartOptions configure a document or image file part, read from the
// part's provider options under the "anthropic" key.
type FilePartOptions struct {
	// ContainerUpload uploads a file reference into the code execution
	// container instead of sending it as a document or image.
	ContainerUpload *bool `json:"containerUpload,omitempty"`

	// Citations enables citations for this document.
	Citations *CitationsConfig `json:"citations,omitempty"`

	// Title overrides the document title, which defaults to the filename.
	Title *string `json:"title,omitempty"`

	// Context is passed to the model but not cited.
	Context *string `json:"context,omitempty"`

	CacheControl *CacheControl `json:"cacheControl,omitempty"`
}

// CitationsConfig enables or disables citations.
type CitationsConfig struct {
	Enabled bool `json:"enabled"`
}

// ClearAt controls when an ephemeral mid-conversation system message is
// cleared.
type ClearAt string

// ClearAtNextUserMessage clears the system message at the next user message.
const ClearAtNextUserMessage ClearAt = "next_user_message"

// SystemMessageOptions configure a system message, read from its provider
// options under the "anthropic" key. ClearAt, Effort and ToolChanges apply
// only to system messages that appear mid-conversation.
type SystemMessageOptions struct {
	ClearAt ClearAt `json:"clearAt,omitempty"`

	// Effort sets the model effort from the next user turn. An effort-only
	// system message may have empty content.
	Effort Effort `json:"effort,omitempty"`

	// ToolChanges add or remove declared tools between turns without
	// invalidating the prompt cache.
	ToolChanges []ToolChange `json:"toolChanges,omitempty"`

	CacheControl *CacheControl `json:"cacheControl,omitempty"`
}

// ToolChangeType adds or removes a tool.
type ToolChangeType string

// Tool change types.
const (
	ToolAddition ToolChangeType = "tool_addition"
	ToolRemoval  ToolChangeType = "tool_removal"
)

// ToolChange adds or removes a tool mid-conversation.
type ToolChange struct {
	Type     ToolChangeType `json:"type"`
	ToolName string         `json:"toolName"`
}

// CallerType identifies what may call a function tool, or what called it.
type CallerType string

// Caller types.
const (
	CallerDirect                CallerType = "direct"
	CallerCodeExecution20250825 CallerType = "code_execution_20250825"
	CallerCodeExecution20260120 CallerType = "code_execution_20260120"
)

// ToolOptions configure a function tool, read from its provider options
// under the "anthropic" key.
type ToolOptions struct {
	// DeferLoading loads the tool only when the tool search tool finds it.
	DeferLoading *bool `json:"deferLoading,omitempty"`

	// AllowedCallers lists the callers that may invoke the tool.
	AllowedCallers []CallerType `json:"allowedCallers,omitempty"`

	// EagerInputStreaming overrides MessagesOptions.ToolStreaming for
	// this tool.
	EagerInputStreaming *bool `json:"eagerInputStreaming,omitempty"`

	CacheControl *CacheControl `json:"cacheControl,omitempty"`
}
