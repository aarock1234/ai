package openai

import (
	"encoding/json"
	"fmt"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// responsesTopLogprobsMax is the largest number of alternatives the API
// returns per token.
const responsesTopLogprobsMax = 20

// ResponsesOptions are the Responses API settings read from the "openai"
// provider options namespace, or first from the namespace set with
// WithOptionsNamespace. Zero values leave the API default.
type ResponsesOptions struct {
	// Conversation is the ID of an OpenAI conversation to continue. It
	// cannot be combined with PreviousResponseID.
	Conversation string `json:"conversation,omitempty"`

	// Include lists extra output fields to return.
	Include []ResponsesInclude `json:"include,omitempty"`

	// IncludeWebSearchSources controls whether web search sources are
	// requested automatically when a web search tool is present. Nil means
	// true.
	IncludeWebSearchSources *bool `json:"includeWebSearchSources,omitempty"`

	// Instructions replace the system or developer message, for example when
	// continuing a conversation with PreviousResponseID.
	Instructions string `json:"instructions,omitempty"`

	// Logprobs requests token log probabilities.
	Logprobs ResponsesLogprobs `json:"logprobs,omitempty"`

	// MaxToolCalls limits the total number of built-in tool calls.
	MaxToolCalls *int `json:"maxToolCalls,omitempty"`

	// Metadata is stored with the response.
	Metadata map[string]string `json:"metadata,omitempty"`

	// ParallelToolCalls allows the model to call several tools at once. Nil
	// means true.
	ParallelToolCalls *bool `json:"parallelToolCalls,omitempty"`

	// PreviousResponseID continues from an earlier stored response.
	PreviousResponseID string `json:"previousResponseId,omitempty"`

	// PromptCacheKey ties the prompt to cached prefixes.
	PromptCacheKey string `json:"promptCacheKey,omitempty"`

	// PromptCacheOptions controls prompt caching on GPT-5.6 and later models.
	PromptCacheOptions *ResponsesPromptCacheOptions `json:"promptCacheOptions,omitempty"`

	// PromptCacheRetention selects the cache retention policy on models
	// before GPT-5.6. GPT-6 and later models ignore it with a warning.
	PromptCacheRetention ResponsesPromptCacheRetention `json:"promptCacheRetention,omitempty"`

	// ReasoningEffort overrides the call's reasoning effort. Supported
	// values vary by model.
	ReasoningEffort string `json:"reasoningEffort,omitempty"`

	// ReasoningEffortUpdate changes the reasoning effort from this response
	// onward on GPT-6 and later models without changing the request-level
	// effort, which preserves the prompt cache prefix.
	ReasoningEffortUpdate ResponsesReasoningEffort `json:"reasoningEffortUpdate,omitempty"`

	// ReasoningMode controls how much work the model does before answering.
	ReasoningMode ResponsesReasoningMode `json:"reasoningMode,omitempty"`

	// ReasoningContext selects which earlier reasoning items the model uses.
	ReasoningContext ResponsesReasoningContext `json:"reasoningContext,omitempty"`

	// ReasoningSummary requests reasoning summaries, such as "auto" or
	// "detailed". Nil defaults to "detailed" when a reasoning effort other
	// than "none" is set.
	ReasoningSummary *string `json:"reasoningSummary,omitempty"`

	// SafetyIdentifier identifies the end user for safety monitoring.
	SafetyIdentifier string `json:"safetyIdentifier,omitempty"`

	// ServiceTier selects the processing tier.
	ServiceTier ResponsesServiceTier `json:"serviceTier,omitempty"`

	// Store controls whether OpenAI stores the response. Nil means true.
	Store *bool `json:"store,omitempty"`

	// PassThroughUnsupportedFiles sends inline files of any media type as
	// input files instead of rejecting everything but images and PDFs.
	PassThroughUnsupportedFiles bool `json:"passThroughUnsupportedFiles,omitempty"`

	// StrictJSONSchema requests strict structured output validation. Nil
	// means true.
	StrictJSONSchema *bool `json:"strictJsonSchema,omitempty"`

	// TextVerbosity controls the length of responses.
	TextVerbosity ResponsesTextVerbosity `json:"textVerbosity,omitempty"`

	// Truncation controls input truncation.
	Truncation ResponsesTruncation `json:"truncation,omitempty"`

	// User identifies the end user for abuse monitoring.
	User string `json:"user,omitempty"`

	// SystemMessageMode overrides how system messages are sent. The default
	// depends on the model.
	SystemMessageMode ResponsesSystemMessageMode `json:"systemMessageMode,omitempty"`

	// ForceReasoning treats the model as a reasoning model even when its ID
	// is not recognized.
	ForceReasoning *bool `json:"forceReasoning,omitempty"`

	// ContextManagement enables server-side compaction.
	ContextManagement []ResponsesContextManagement `json:"contextManagement,omitempty"`

	// CompactionTrigger requests explicit compaction by appending a
	// compaction trigger to the input.
	CompactionTrigger bool `json:"compactionTrigger,omitempty"`

	// AllowedTools restricts the callable tools while keeping the full tool
	// list, which preserves prompt caching. It overrides the call's tool
	// choice.
	AllowedTools *ResponsesAllowedTools `json:"allowedTools,omitempty"`
}

// ResponsesLogprobs is the number of most likely alternatives returned per
// token, from 1 to 20. Zero disables log probabilities. In JSON it accepts
// true for the maximum and false for zero.
type ResponsesLogprobs int

// ResponsesLogprobsMax requests the maximum number of alternatives.
const ResponsesLogprobsMax ResponsesLogprobs = responsesTopLogprobsMax

// UnmarshalJSON accepts a boolean or a number.
func (l *ResponsesLogprobs) UnmarshalJSON(data []byte) error {
	var enabled bool
	if err := json.Unmarshal(data, &enabled); err == nil {
		*l = 0
		if enabled {
			*l = ResponsesLogprobsMax
		}

		return nil
	}

	var count int
	if err := json.Unmarshal(data, &count); err != nil {
		return fmt.Errorf("logprobs must be a boolean or a number: %w", err)
	}

	*l = ResponsesLogprobs(count)

	return nil
}

// ResponsesInclude names an extra output field.
type ResponsesInclude string

// Include values. The first four may be set in ResponsesOptions; the others
// are added automatically when the request needs them.
const (
	ResponsesIncludeReasoningEncryptedContent ResponsesInclude = "reasoning.encrypted_content"
	ResponsesIncludeFileSearchResults         ResponsesInclude = "file_search_call.results"
	ResponsesIncludeWebSearchResults          ResponsesInclude = "web_search_call.results"
	ResponsesIncludeOutputTextLogprobs        ResponsesInclude = "message.output_text.logprobs"
	ResponsesIncludeWebSearchSources          ResponsesInclude = "web_search_call.action.sources"
	ResponsesIncludeCodeInterpreterOutputs    ResponsesInclude = "code_interpreter_call.outputs"
	ResponsesIncludeComputerOutputImageURL    ResponsesInclude = "computer_call_output.output.image_url"
	ResponsesIncludeInputImageURL             ResponsesInclude = "message.input_image.image_url"
)

// ResponsesPromptCacheRetention selects how long cached prefixes live.
type ResponsesPromptCacheRetention string

// Prompt cache retention policies.
const (
	ResponsesPromptCacheInMemory ResponsesPromptCacheRetention = "in_memory"
	ResponsesPromptCache24h      ResponsesPromptCacheRetention = "24h"
)

// ResponsesPromptCacheOptions controls prompt caching on GPT-5.6 and later
// models.
type ResponsesPromptCacheOptions struct {
	// Mode controls whether OpenAI also places an implicit breakpoint.
	Mode ResponsesPromptCacheMode `json:"mode,omitempty"`

	// TTL is the minimum cache lifetime. Only "30m" is supported.
	TTL string `json:"ttl,omitempty"`
}

// ResponsesPromptCacheMode selects implicit or explicit cache breakpoints.
type ResponsesPromptCacheMode string

// Prompt cache modes.
const (
	ResponsesPromptCacheImplicit ResponsesPromptCacheMode = "implicit"
	ResponsesPromptCacheExplicit ResponsesPromptCacheMode = "explicit"
)

// ResponsesPromptCacheBreakpoint marks the end of a cacheable prompt prefix.
// Set it on message, part or tool output provider options.
type ResponsesPromptCacheBreakpoint struct {
	// Mode is "explicit".
	Mode ResponsesPromptCacheMode `json:"mode"`
}

// ResponsesReasoningEffort is a reasoning effort accepted by reasoning effort
// updates.
type ResponsesReasoningEffort string

// Reasoning effort update values.
const (
	ResponsesReasoningEffortNone   ResponsesReasoningEffort = "none"
	ResponsesReasoningEffortLow    ResponsesReasoningEffort = "low"
	ResponsesReasoningEffortMedium ResponsesReasoningEffort = "medium"
	ResponsesReasoningEffortHigh   ResponsesReasoningEffort = "high"
	ResponsesReasoningEffortXHigh  ResponsesReasoningEffort = "xhigh"
	ResponsesReasoningEffortMax    ResponsesReasoningEffort = "max"
)

// ResponsesReasoningMode controls how much work the model does before
// answering.
type ResponsesReasoningMode string

// Reasoning modes.
const (
	ResponsesReasoningStandard ResponsesReasoningMode = "standard"
	ResponsesReasoningPro      ResponsesReasoningMode = "pro"
)

// ResponsesReasoningContext selects which reasoning items the model can use.
type ResponsesReasoningContext string

// Reasoning context values.
const (
	ResponsesReasoningContextAuto        ResponsesReasoningContext = "auto"
	ResponsesReasoningContextCurrentTurn ResponsesReasoningContext = "current_turn"
	ResponsesReasoningContextAllTurns    ResponsesReasoningContext = "all_turns"
)

// ResponsesServiceTier selects a processing tier.
type ResponsesServiceTier string

// Service tiers.
const (
	ResponsesServiceTierAuto      ResponsesServiceTier = "auto"
	ResponsesServiceTierFlex      ResponsesServiceTier = "flex"
	ResponsesServiceTierPriority  ResponsesServiceTier = "priority"
	ResponsesServiceTierFast      ResponsesServiceTier = "fast"
	ResponsesServiceTierUltrafast ResponsesServiceTier = "ultrafast"
	ResponsesServiceTierDefault   ResponsesServiceTier = "default"
)

// ResponsesTextVerbosity controls the length of responses.
type ResponsesTextVerbosity string

// Text verbosity levels.
const (
	ResponsesTextVerbosityLow    ResponsesTextVerbosity = "low"
	ResponsesTextVerbosityMedium ResponsesTextVerbosity = "medium"
	ResponsesTextVerbosityHigh   ResponsesTextVerbosity = "high"
)

// ResponsesTruncation controls input truncation.
type ResponsesTruncation string

// Truncation modes.
const (
	ResponsesTruncationAuto     ResponsesTruncation = "auto"
	ResponsesTruncationDisabled ResponsesTruncation = "disabled"
)

// ResponsesSystemMessageMode controls how system messages are sent.
type ResponsesSystemMessageMode string

// System message modes.
const (
	ResponsesSystemMessageSystem    ResponsesSystemMessageMode = "system"
	ResponsesSystemMessageDeveloper ResponsesSystemMessageMode = "developer"
	ResponsesSystemMessageRemove    ResponsesSystemMessageMode = "remove"
)

// ResponsesContextManagement enables one kind of server-side context
// management.
type ResponsesContextManagement struct {
	// Type is "compaction".
	Type string `json:"type"`

	// CompactThreshold is the token count that triggers compaction.
	CompactThreshold int `json:"compactThreshold"`
}

// ResponsesAllowedTools restricts the callable tools.
type ResponsesAllowedTools struct {
	// ToolNames lists the application names of the allowed tools. It must
	// not be empty.
	ToolNames []string `json:"toolNames"`

	// Mode is "auto" (the default) or "required".
	Mode ResponsesAllowedToolsMode `json:"mode,omitempty"`
}

// ResponsesAllowedToolsMode controls whether an allowed tool must be called.
type ResponsesAllowedToolsMode string

// Allowed tools modes.
const (
	ResponsesAllowedToolsAuto     ResponsesAllowedToolsMode = "auto"
	ResponsesAllowedToolsRequired ResponsesAllowedToolsMode = "required"
)

// ResponsesSystemMessageOptions are the provider options of a system
// message.
type ResponsesSystemMessageOptions struct {
	// ReasoningEffortUpdate emits a reasoning effort update at this position
	// of the conversation instead of a message. The message content must be
	// empty, and the model must support configuration updates.
	ReasoningEffortUpdate ResponsesReasoningEffort `json:"reasoningEffortUpdate,omitempty"`

	// PromptCacheBreakpoint marks the message as the end of a cacheable
	// prefix.
	PromptCacheBreakpoint *ResponsesPromptCacheBreakpoint `json:"promptCacheBreakpoint,omitempty"`
}

// ResponsesPartOptions are the provider options of user text and file parts
// and of tool result content items.
type ResponsesPartOptions struct {
	// ImageDetail selects the image resolution: "auto", "low", "high" or
	// "original".
	ImageDetail string `json:"imageDetail,omitempty"`

	// PromptCacheBreakpoint marks the part as the end of a cacheable prefix.
	PromptCacheBreakpoint *ResponsesPromptCacheBreakpoint `json:"promptCacheBreakpoint,omitempty"`
}

// ResponsesToolResultOptions are the provider options of tool result parts.
type ResponsesToolResultOptions struct {
	// ItemID identifies the stored output item of a provider-executed tool.
	ItemID string `json:"itemId,omitempty"`

	// Caller identifies the program that called the tool.
	Caller *ResponsesToolCaller `json:"caller,omitempty"`

	// PromptCacheBreakpoint marks the result as the end of a cacheable
	// prefix.
	PromptCacheBreakpoint *ResponsesPromptCacheBreakpoint `json:"promptCacheBreakpoint,omitempty"`

	// ParallelToolCall links the result to an expanded parallel tool call.
	ParallelToolCall *ResponsesParallelToolCall `json:"parallelToolCall,omitempty"`

	// AdditionalTools are function definitions loaded right after this
	// result.
	AdditionalTools []ResponsesAdditionalTool `json:"additionalTools,omitempty"`
}

// ResponsesToolOutputOptions are the provider options of a tool result
// output value.
type ResponsesToolOutputOptions struct {
	// PromptCacheBreakpoint marks the output as the end of a cacheable
	// prefix.
	PromptCacheBreakpoint *ResponsesPromptCacheBreakpoint `json:"promptCacheBreakpoint,omitempty"`

	// ApprovalID marks a denial that an approval response already reports,
	// so the denied output is not sent again.
	ApprovalID string `json:"approvalId,omitempty"`
}

// ResponsesAdditionalTool is a function definition loaded in history after a
// tool result.
type ResponsesAdditionalTool struct {
	// Type is "function".
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
	Strict      *bool           `json:"strict,omitempty"`
}

// ResponsesFunctionToolOptions are the "openai" provider options of a
// function tool.
type ResponsesFunctionToolOptions struct {
	// AllowedCallers lists who may call the tool: "direct" and
	// "programmatic".
	AllowedCallers []string `json:"allowedCallers,omitempty"`

	// Async lets the model continue without waiting for the result. It is
	// supported by GPT-6 and later models.
	Async *bool `json:"async,omitempty"`

	// DeferLoading hides the tool until tool search loads it.
	DeferLoading *bool `json:"deferLoading,omitempty"`

	// OmitFromRequestTools leaves the tool out of the request's tool list,
	// for tools loaded in history through additional tools. The application
	// can still execute its calls.
	OmitFromRequestTools bool `json:"omitFromRequestTools,omitempty"`

	// OutputSchema is the JSON Schema of the tool's result.
	OutputSchema json.RawMessage `json:"outputSchema,omitempty"`

	// Namespace groups the tool with others under a shared name.
	Namespace *ResponsesToolNamespace `json:"namespace,omitempty"`
}

// ResponsesToolNamespace groups function tools.
type ResponsesToolNamespace struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

// readResponsesOptions decodes and validates the call options from the
// provider namespace, falling back to "openai".
func readResponsesOptions(options model.ProviderOptions, namespace string) (ResponsesOptions, error) {
	values, found, err := model.ProviderValue[ResponsesOptions](options, namespace)
	if err != nil {
		return ResponsesOptions{}, err
	}

	if !found && namespace != Namespace {
		values, _, err = model.ProviderValue[ResponsesOptions](options, Namespace)
		if err != nil {
			return ResponsesOptions{}, err
		}
	}

	if err := values.validate(); err != nil {
		return ResponsesOptions{}, err
	}

	return values, nil
}

// validate checks the enumerations and ranges the API accepts.
func (o ResponsesOptions) validate() error {
	for _, include := range o.Include {
		err := responsesCheckEnum("include", include,
			ResponsesIncludeReasoningEncryptedContent,
			ResponsesIncludeFileSearchResults,
			ResponsesIncludeWebSearchResults,
			ResponsesIncludeOutputTextLogprobs,
		)
		if err != nil {
			return err
		}
	}

	if o.Logprobs < 0 || o.Logprobs > ResponsesLogprobsMax {
		return fmt.Errorf("%w: logprobs must be between 1 and %d", model.ErrInvalidArgument, responsesTopLogprobsMax)
	}

	if err := o.validatePromptCache(); err != nil {
		return err
	}

	if err := o.validateReasoning(); err != nil {
		return err
	}

	if err := o.validateOutput(); err != nil {
		return err
	}

	return o.validateTools()
}

func (o ResponsesOptions) validatePromptCache() error {
	err := responsesCheckEnum("promptCacheRetention", o.PromptCacheRetention,
		ResponsesPromptCacheInMemory,
		ResponsesPromptCache24h,
	)
	if err != nil {
		return err
	}

	if o.PromptCacheOptions == nil {
		return nil
	}

	err = responsesCheckEnum("promptCacheOptions.mode", o.PromptCacheOptions.Mode,
		ResponsesPromptCacheImplicit,
		ResponsesPromptCacheExplicit,
	)
	if err != nil {
		return err
	}

	return responsesCheckEnum("promptCacheOptions.ttl", o.PromptCacheOptions.TTL, "30m")
}

func (o ResponsesOptions) validateReasoning() error {
	if err := responsesCheckReasoningEffort("reasoningEffortUpdate", o.ReasoningEffortUpdate); err != nil {
		return err
	}

	err := responsesCheckEnum("reasoningMode", o.ReasoningMode,
		ResponsesReasoningStandard,
		ResponsesReasoningPro,
	)
	if err != nil {
		return err
	}

	err = responsesCheckEnum("reasoningContext", o.ReasoningContext,
		ResponsesReasoningContextAuto,
		ResponsesReasoningContextCurrentTurn,
		ResponsesReasoningContextAllTurns,
	)
	if err != nil {
		return err
	}

	for _, management := range o.ContextManagement {
		if management.Type != "compaction" {
			return fmt.Errorf("%w: contextManagement.type must be \"compaction\"", model.ErrInvalidArgument)
		}
	}

	return nil
}

func (o ResponsesOptions) validateOutput() error {
	err := responsesCheckEnum("serviceTier", o.ServiceTier,
		ResponsesServiceTierAuto,
		ResponsesServiceTierFlex,
		ResponsesServiceTierPriority,
		ResponsesServiceTierFast,
		ResponsesServiceTierUltrafast,
		ResponsesServiceTierDefault,
	)
	if err != nil {
		return err
	}

	err = responsesCheckEnum("textVerbosity", o.TextVerbosity,
		ResponsesTextVerbosityLow,
		ResponsesTextVerbosityMedium,
		ResponsesTextVerbosityHigh,
	)
	if err != nil {
		return err
	}

	err = responsesCheckEnum("truncation", o.Truncation,
		ResponsesTruncationAuto,
		ResponsesTruncationDisabled,
	)
	if err != nil {
		return err
	}

	return responsesCheckEnum("systemMessageMode", o.SystemMessageMode,
		ResponsesSystemMessageSystem,
		ResponsesSystemMessageDeveloper,
		ResponsesSystemMessageRemove,
	)
}

func (o ResponsesOptions) validateTools() error {
	if o.AllowedTools == nil {
		return nil
	}

	if len(o.AllowedTools.ToolNames) == 0 {
		return fmt.Errorf("%w: allowedTools.toolNames must not be empty", model.ErrInvalidArgument)
	}

	return responsesCheckEnum("allowedTools.mode", o.AllowedTools.Mode,
		ResponsesAllowedToolsAuto,
		ResponsesAllowedToolsRequired,
	)
}

// validate checks the system message options.
func (o ResponsesSystemMessageOptions) validate() error {
	return responsesCheckReasoningEffort("reasoningEffortUpdate", o.ReasoningEffortUpdate)
}

// validate checks the additional tool definitions.
func (o ResponsesToolResultOptions) validate() error {
	for _, tool := range o.AdditionalTools {
		if tool.Type != "function" {
			return fmt.Errorf("%w: additional tools must have type \"function\"", model.ErrInvalidArgument)
		}

		if tool.Name == "" {
			return fmt.Errorf("%w: additional tools must have a name", model.ErrInvalidArgument)
		}
	}

	return nil
}

func responsesCheckReasoningEffort(setting string, effort ResponsesReasoningEffort) error {
	return responsesCheckEnum(setting, effort,
		ResponsesReasoningEffortNone,
		ResponsesReasoningEffortLow,
		ResponsesReasoningEffortMedium,
		ResponsesReasoningEffortHigh,
		ResponsesReasoningEffortXHigh,
		ResponsesReasoningEffortMax,
	)
}

// responsesCheckEnum reports an error when a set value is not one of
// allowed. The empty value means unset and is always accepted.
func responsesCheckEnum[T ~string](setting string, value T, allowed ...T) error {
	if value == "" || slices.Contains(allowed, value) {
		return nil
	}

	return fmt.Errorf("%w: unsupported %s %q", model.ErrInvalidArgument, setting, value)
}
