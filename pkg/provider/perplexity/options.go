package perplexity

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// AgentOptions are the provider options of Agent API models, read from the
// "perplexity" namespace. Keys not listed here, and unlisted keys of native
// tools, reasoning and skills, are sent in the request body unchanged, so the
// JSON names follow the wire format.
type AgentOptions struct {
	// Instructions are top-level Agent API instructions.
	Instructions *string `json:"instructions,omitempty"`

	// Tools are native Agent API tools. Function tools come from the call
	// options and are sent after them.
	Tools []NativeTool `json:"tools,omitempty"`

	// Models is a fallback model list for Agent API routing.
	Models []string `json:"models,omitempty"`

	// MaxSteps is the positive maximum number of agentic steps.
	MaxSteps *int `json:"max_steps,omitempty"`

	// MaxToolCalls is the maximum number of native tool calls.
	MaxToolCalls *int `json:"max_tool_calls,omitempty"`

	// PreviousResponseID continues a conversation from an earlier Agent API
	// response.
	PreviousResponseID *string `json:"previous_response_id,omitempty"`

	// Store controls whether the response can be retrieved later. False
	// does not disable persistence or prevent continuations.
	Store *bool `json:"store,omitempty"`

	// LanguagePreference is the preferred response language as an ISO 639-1
	// code.
	LanguagePreference *string `json:"language_preference,omitempty"`

	// Reasoning configures reasoning. Without it, the call's reasoning level
	// selects the effort.
	Reasoning *Reasoning `json:"reasoning,omitempty"`

	// Skills configures Agent API skills.
	Skills []Skill `json:"skills,omitempty"`
}

// ReasoningEffort is an Agent API reasoning effort.
type ReasoningEffort string

// Reasoning efforts.
const (
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	ReasoningEffortLow     ReasoningEffort = "low"
	ReasoningEffortMedium  ReasoningEffort = "medium"
	ReasoningEffortHigh    ReasoningEffort = "high"
	ReasoningEffortXHigh   ReasoningEffort = "xhigh"
)

// reasoningEfforts are the accepted reasoning efforts.
var reasoningEfforts = []ReasoningEffort{
	ReasoningEffortMinimal,
	ReasoningEffortLow,
	ReasoningEffortMedium,
	ReasoningEffortHigh,
	ReasoningEffortXHigh,
}

// Reasoning configures Agent API reasoning.
type Reasoning struct {
	Effort ReasoningEffort `json:"effort,omitempty"`
}

// NativeToolType names a native Agent API tool.
type NativeToolType string

// Native tool types.
const (
	NativeToolWebSearch     NativeToolType = "web_search"
	NativeToolFetchURL      NativeToolType = "fetch_url"
	NativeToolPeopleSearch  NativeToolType = "people_search"
	NativeToolFinanceSearch NativeToolType = "finance_search"
	NativeToolSandbox       NativeToolType = "sandbox"
	NativeToolMCP           NativeToolType = "mcp"
	NativeToolConnector     NativeToolType = "connector"
)

// NativeTool is a native Agent API tool. Each field applies to the tool
// types noted on it.
type NativeTool struct {
	Type NativeToolType `json:"type"`

	// Filters restrict web_search results.
	Filters *WebSearchFilters `json:"filters,omitempty"`

	// MaxResults is the positive result limit of web_search.
	MaxResults *int `json:"max_results,omitempty"`

	// MaxTokens and MaxTokensPerPage limit the web_search context.
	MaxTokens        *float64 `json:"max_tokens,omitempty"`
	MaxTokensPerPage *float64 `json:"max_tokens_per_page,omitempty"`

	// SearchContextSize is "low", "medium" or "high" for web_search.
	SearchContextSize string `json:"search_context_size,omitempty"`

	// UserLocation localizes web_search results.
	UserLocation *UserLocation `json:"user_location,omitempty"`

	// MaxURLs limits fetch_url.
	MaxURLs *float64 `json:"max_urls,omitempty"`

	// ServerLabel is required for mcp and connector tools.
	ServerLabel string `json:"server_label,omitempty"`

	// ServerURL is required for mcp tools.
	ServerURL string `json:"server_url,omitempty"`

	// AllowedTools restricts the tools of mcp and connector servers.
	AllowedTools []string `json:"allowed_tools,omitempty"`

	// Authorization, DeferLoading and Headers configure mcp servers.
	Authorization string            `json:"authorization,omitempty"`
	DeferLoading  *bool             `json:"defer_loading,omitempty"`
	Headers       map[string]string `json:"headers,omitempty"`

	// ID identifies a connector and is required for connector tools.
	ID string `json:"id,omitempty"`

	// ServerDescription describes a connector.
	ServerDescription string `json:"server_description,omitempty"`
}

// WebSearchFilters restrict web search results.
type WebSearchFilters struct {
	SearchDomainFilter []string `json:"search_domain_filter,omitempty"`

	// SearchRecencyFilter is "hour", "day", "week", "month" or "year".
	SearchRecencyFilter string `json:"search_recency_filter,omitempty"`

	SearchAfterDateFilter   string `json:"search_after_date_filter,omitempty"`
	SearchBeforeDateFilter  string `json:"search_before_date_filter,omitempty"`
	LastUpdatedAfterFilter  string `json:"last_updated_after_filter,omitempty"`
	LastUpdatedBeforeFilter string `json:"last_updated_before_filter,omitempty"`
}

// UserLocation is the approximate location of the user.
type UserLocation struct {
	Latitude  *float64 `json:"latitude,omitempty"`
	Longitude *float64 `json:"longitude,omitempty"`
	Country   string   `json:"country,omitempty"`
	City      string   `json:"city,omitempty"`
	Region    string   `json:"region,omitempty"`
}

// SkillType selects a builtin or inline skill.
type SkillType string

// Skill types.
const (
	SkillBuiltin SkillType = "builtin"
	SkillInline  SkillType = "inline"
)

// builtinSkills are the names of builtin skills.
var builtinSkills = []string{
	"office",
	"office/docx",
	"office/pdf",
	"office/pptx",
	"office/xlsx",
}

// Skill is an Agent API skill. Builtin skills are named "office",
// "office/docx", "office/pdf", "office/pptx" or "office/xlsx"; inline skills
// need a name, a description and instructions.
type Skill struct {
	Type         SkillType `json:"type"`
	Name         string    `json:"name"`
	Description  string    `json:"description,omitempty"`
	Instructions string    `json:"instructions,omitempty"`
}

// EncodingFormat is the quantized encoding of returned embeddings.
type EncodingFormat string

// Encoding formats. Perplexity does not return floating point embeddings.
const (
	// EncodingBase64Int8 returns signed int8 values, compared by cosine
	// similarity. It is the default.
	EncodingBase64Int8 EncodingFormat = "base64_int8"

	// EncodingBase64Binary returns packed bits as bytes from 0 to 255,
	// compared by Hamming distance.
	EncodingBase64Binary EncodingFormat = "base64_binary"
)

// EmbeddingOptions are the provider options of embedding models, read from
// the "perplexity" namespace.
type EmbeddingOptions struct {
	// Dimensions is the positive output size (Matryoshka truncation), from
	// 128 up to the model's full size. It takes precedence over
	// model.EmbedOptions.Dimensions.
	Dimensions *int `json:"dimensions,omitempty"`

	// EncodingFormat is the quantized encoding. Empty uses
	// EncodingBase64Int8.
	EncodingFormat EncodingFormat `json:"encodingFormat,omitempty"`
}

// agentOptionMembers decodes and validates the Agent API options and
// returns every member of the namespace, which is forwarded to the API.
func agentOptionMembers(options model.ProviderOptions) (AgentOptions, map[string]json.RawMessage, error) {
	agentOptions, _, err := model.ProviderValue[AgentOptions](options, Namespace)
	if err != nil {
		return AgentOptions{}, nil, err
	}

	if err := agentOptions.validate(); err != nil {
		return AgentOptions{}, nil, fmt.Errorf("%w: perplexity provider options: %w", model.ErrInvalidArgument, err)
	}

	members, _, err := model.ProviderValue[map[string]json.RawMessage](options, Namespace)
	if err != nil {
		return AgentOptions{}, nil, err
	}

	return agentOptions, members, nil
}

func (o AgentOptions) validate() error {
	if o.MaxSteps != nil && *o.MaxSteps <= 0 {
		return errors.New("max_steps must be positive")
	}

	if o.MaxToolCalls != nil && *o.MaxToolCalls < 0 {
		return errors.New("max_tool_calls must not be negative")
	}

	if o.Reasoning != nil && o.Reasoning.Effort != "" && !slices.Contains(reasoningEfforts, o.Reasoning.Effort) {
		return fmt.Errorf("unsupported reasoning.effort %q", o.Reasoning.Effort)
	}

	for _, tool := range o.Tools {
		if err := tool.validate(); err != nil {
			return err
		}
	}

	for _, skill := range o.Skills {
		if err := skill.validate(); err != nil {
			return err
		}
	}

	return nil
}

func (t NativeTool) validate() error {
	switch t.Type {
	case NativeToolWebSearch:
		return t.validateWebSearch()

	case NativeToolFetchURL, NativeToolPeopleSearch, NativeToolFinanceSearch, NativeToolSandbox:
		return nil

	case NativeToolMCP:
		if t.ServerLabel == "" || t.ServerURL == "" {
			return errors.New("mcp tools need server_label and server_url")
		}

		return nil

	case NativeToolConnector:
		if t.ID == "" || t.ServerLabel == "" {
			return errors.New("connector tools need id and server_label")
		}

		return nil

	default:
		return fmt.Errorf("unsupported native tool type %q", t.Type)
	}
}

func (t NativeTool) validateWebSearch() error {
	if t.MaxResults != nil && *t.MaxResults <= 0 {
		return errors.New("web_search max_results must be positive")
	}

	if t.SearchContextSize != "" && !slices.Contains(searchContextSizes, t.SearchContextSize) {
		return fmt.Errorf("unsupported web_search search_context_size %q", t.SearchContextSize)
	}

	if t.Filters == nil || t.Filters.SearchRecencyFilter == "" {
		return nil
	}

	if !slices.Contains(searchRecencyFilters, t.Filters.SearchRecencyFilter) {
		return fmt.Errorf("unsupported web_search search_recency_filter %q", t.Filters.SearchRecencyFilter)
	}

	return nil
}

// searchContextSizes are the accepted web_search context sizes.
var searchContextSizes = []string{
	"low",
	"medium",
	"high",
}

// searchRecencyFilters are the accepted web_search recency filters.
var searchRecencyFilters = []string{
	"hour",
	"day",
	"week",
	"month",
	"year",
}

func (s Skill) validate() error {
	switch s.Type {
	case SkillBuiltin:
		if !slices.Contains(builtinSkills, s.Name) {
			return fmt.Errorf("unsupported builtin skill %q", s.Name)
		}

		return nil

	case SkillInline:
		if s.Name == "" || s.Description == "" || s.Instructions == "" {
			return errors.New("inline skills need a name, a description and instructions")
		}

		return nil

	default:
		return fmt.Errorf("unsupported skill type %q", s.Type)
	}
}

// readEmbeddingOptions decodes and validates the embedding options.
func readEmbeddingOptions(options model.ProviderOptions) (EmbeddingOptions, error) {
	embeddingOptions, _, err := model.ProviderValue[EmbeddingOptions](options, Namespace)
	if err != nil {
		return EmbeddingOptions{}, err
	}

	if embeddingOptions.Dimensions != nil && *embeddingOptions.Dimensions <= 0 {
		return EmbeddingOptions{}, fmt.Errorf("%w: perplexity provider options: dimensions must be positive", model.ErrInvalidArgument)
	}

	switch embeddingOptions.EncodingFormat {
	case "", EncodingBase64Int8, EncodingBase64Binary:
		return embeddingOptions, nil
	default:
		return EmbeddingOptions{}, fmt.Errorf("%w: perplexity provider options: unsupported encodingFormat %q", model.ErrInvalidArgument, embeddingOptions.EncodingFormat)
	}
}
