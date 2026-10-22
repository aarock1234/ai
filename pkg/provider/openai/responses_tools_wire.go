package openai

import (
	"encoding/json"
)

// responsesTool is one entry of the request's tools.
type responsesTool interface {
	responsesToolType() string
}

// responsesFunctionTool is a function the application executes.
type responsesFunctionTool struct {
	Type           string          `json:"type"`
	Name           string          `json:"name"`
	Description    string          `json:"description,omitempty"`
	Parameters     json.RawMessage `json:"parameters"`
	Async          *bool           `json:"async,omitempty"`
	Strict         bool            `json:"strict"`
	DeferLoading   *bool           `json:"defer_loading,omitempty"`
	AllowedCallers []string        `json:"allowed_callers,omitempty"`
	OutputSchema   json.RawMessage `json:"output_schema,omitempty"`
}

// responsesNamespaceTool groups function tools.
type responsesNamespaceTool struct {
	Type        string                   `json:"type"`
	Name        string                   `json:"name"`
	Description string                   `json:"description"`
	Tools       []*responsesFunctionTool `json:"tools"`
}

// responsesBuiltinTool is a built-in tool without settings, such as
// apply_patch or computer.
type responsesBuiltinTool struct {
	Type string `json:"type"`
}

// responsesWebSearchTool is web_search or web_search_preview.
type responsesWebSearchTool struct {
	Type              string                  `json:"type"`
	ExternalWebAccess *bool                   `json:"external_web_access,omitempty"`
	Filters           *responsesDomainFilters `json:"filters,omitempty"`
	SearchContextSize string                  `json:"search_context_size,omitempty"`
	UserLocation      *WebSearchLocation      `json:"user_location,omitempty"`
}

// responsesDomainFilters restrict web search domains.
type responsesDomainFilters struct {
	AllowedDomains []string `json:"allowed_domains,omitempty"`
	BlockedDomains []string `json:"blocked_domains,omitempty"`
}

// responsesCodeInterpreterTool runs code in a container.
type responsesCodeInterpreterTool struct {
	Type      string                            `json:"type"`
	Container responsesCodeInterpreterContainer `json:"container"`
}

// responsesCodeInterpreterContainer is a container ID or an automatic container.
type responsesCodeInterpreterContainer struct {
	ID      string
	FileIDs []string
}

// MarshalJSON writes the container ID, or an automatic container object.
func (c responsesCodeInterpreterContainer) MarshalJSON() ([]byte, error) {
	if c.ID != "" {
		return json.Marshal(c.ID)
	}

	return json.Marshal(responsesAutoContainer{
		Type:    "auto",
		FileIDs: c.FileIDs,
	})
}

type responsesAutoContainer struct {
	Type    string   `json:"type"`
	FileIDs []string `json:"file_ids,omitempty"`
}

// responsesFileSearchTool searches vector stores.
type responsesFileSearchTool struct {
	Type           string                   `json:"type"`
	VectorStoreIDs []string                 `json:"vector_store_ids"`
	MaxNumResults  *int                     `json:"max_num_results,omitempty"`
	RankingOptions *responsesRankingOptions `json:"ranking_options,omitempty"`
	Filters        *FileSearchFilter        `json:"filters,omitempty"`
}

// responsesRankingOptions configures file search ranking.
type responsesRankingOptions struct {
	Ranker         string   `json:"ranker,omitempty"`
	ScoreThreshold *float64 `json:"score_threshold,omitempty"`
}

// responsesImageGenerationTool generates images.
type responsesImageGenerationTool struct {
	Type              string                   `json:"type"`
	Action            string                   `json:"action,omitempty"`
	Background        string                   `json:"background,omitempty"`
	InputFidelity     string                   `json:"input_fidelity,omitempty"`
	InputImageMask    *responsesInputImageMask `json:"input_image_mask,omitempty"`
	Model             string                   `json:"model,omitempty"`
	Moderation        string                   `json:"moderation,omitempty"`
	OutputCompression *int                     `json:"output_compression,omitempty"`
	OutputFormat      string                   `json:"output_format,omitempty"`
	PartialImages     *int                     `json:"partial_images,omitempty"`
	Quality           string                   `json:"quality,omitempty"`
	Size              string                   `json:"size,omitempty"`
}

// responsesInputImageMask is an inpainting mask.
type responsesInputImageMask struct {
	FileID   string `json:"file_id,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
}

// responsesMCPTool connects a remote MCP server.
type responsesMCPTool struct {
	Type              string               `json:"type"`
	ServerLabel       string               `json:"server_label"`
	AllowedTools      *responsesMCPAllowed `json:"allowed_tools,omitempty"`
	Authorization     string               `json:"authorization,omitempty"`
	ConnectorID       string               `json:"connector_id,omitempty"`
	Headers           map[string]string    `json:"headers,omitempty"`
	RequireApproval   responsesMCPApproval `json:"require_approval"`
	ServerDescription string               `json:"server_description,omitempty"`
	ServerURL         string               `json:"server_url,omitempty"`
}

// responsesMCPAllowed is a list of tool names or a filter.
type responsesMCPAllowed struct {
	ToolNames []string
	Filter    *responsesMCPFilter
}

// MarshalJSON writes the filter when set, otherwise the names.
func (a responsesMCPAllowed) MarshalJSON() ([]byte, error) {
	if a.Filter != nil {
		return json.Marshal(a.Filter)
	}

	return json.Marshal(a.ToolNames)
}

// responsesMCPFilter selects MCP tools.
type responsesMCPFilter struct {
	ReadOnly  *bool    `json:"read_only,omitempty"`
	ToolNames []string `json:"tool_names,omitempty"`
}

// responsesMCPApproval is "always", "never" or a never filter.
type responsesMCPApproval struct {
	Policy string
	Never  *responsesMCPFilter
}

type responsesMCPApprovalFilter struct {
	Never *responsesMCPFilter `json:"never,omitempty"`
}

// MarshalJSON writes the policy, or the never filter.
func (a responsesMCPApproval) MarshalJSON() ([]byte, error) {
	if a.Policy != "" {
		return json.Marshal(a.Policy)
	}

	return json.Marshal(responsesMCPApprovalFilter{
		Never: a.Never,
	})
}

// responsesCustomTool is a tool with free-text input.
type responsesCustomTool struct {
	Type        string            `json:"type"`
	Name        string            `json:"name"`
	Description string            `json:"description,omitempty"`
	Async       *bool             `json:"async,omitempty"`
	Format      *CustomToolFormat `json:"format,omitempty"`
}

// responsesShellTool runs shell commands.
type responsesShellTool struct {
	Type        string                     `json:"type"`
	Environment *responsesShellEnvironment `json:"environment,omitempty"`
}

// responsesShellEnvironment is "container_auto", "container_reference" or
// "local".
type responsesShellEnvironment struct {
	Type          string                  `json:"type"`
	FileIDs       []string                `json:"file_ids,omitempty"`
	MemoryLimit   string                  `json:"memory_limit,omitempty"`
	NetworkPolicy *responsesNetworkPolicy `json:"network_policy,omitempty"`
	Skills        []responsesShellSkill   `json:"skills,omitempty"`
	ContainerID   string                  `json:"container_id,omitempty"`
}

// responsesNetworkPolicy is "disabled" or "allowlist".
type responsesNetworkPolicy struct {
	Type           string              `json:"type"`
	AllowedDomains []string            `json:"allowed_domains,omitempty"`
	DomainSecrets  []ShellDomainSecret `json:"domain_secrets,omitempty"`
}

// responsesShellSkill is "skill_reference", "inline" or a local skill.
type responsesShellSkill struct {
	Type        string                     `json:"type,omitempty"`
	SkillID     string                     `json:"skill_id,omitempty"`
	Version     string                     `json:"version,omitempty"`
	Name        string                     `json:"name,omitempty"`
	Description string                     `json:"description,omitempty"`
	Source      *responsesShellSkillSource `json:"source,omitempty"`
	Path        string                     `json:"path,omitempty"`
}

// responsesShellSkillSource is an inline skill archive.
type responsesShellSkillSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

// responsesToolSearchTool loads deferred tools.
type responsesToolSearchTool struct {
	Type        string          `json:"type"`
	Execution   string          `json:"execution,omitempty"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

func (t *responsesFunctionTool) responsesToolType() string {
	return t.Type
}

func (t *responsesNamespaceTool) responsesToolType() string {
	return t.Type
}

func (t responsesBuiltinTool) responsesToolType() string {
	return t.Type
}

func (t responsesWebSearchTool) responsesToolType() string {
	return t.Type
}

func (t responsesCodeInterpreterTool) responsesToolType() string {
	return t.Type
}

func (t responsesFileSearchTool) responsesToolType() string {
	return t.Type
}

func (t responsesImageGenerationTool) responsesToolType() string {
	return t.Type
}

func (t responsesMCPTool) responsesToolType() string {
	return t.Type
}

func (t responsesCustomTool) responsesToolType() string {
	return t.Type
}

func (t responsesShellTool) responsesToolType() string {
	return t.Type
}

func (t responsesToolSearchTool) responsesToolType() string {
	return t.Type
}

// responsesToolChoice is "auto", "none" or "required" when Keyword is set,
// and otherwise an object naming a tool or the allowed tools.
type responsesToolChoice struct {
	Keyword string `json:"-"`

	Type  string                    `json:"type,omitempty"`
	Name  string                    `json:"name,omitempty"`
	Mode  ResponsesAllowedToolsMode `json:"mode,omitempty"`
	Tools []responsesAllowedTool    `json:"tools,omitempty"`
}

// MarshalJSON writes the keyword, or the object.
func (c responsesToolChoice) MarshalJSON() ([]byte, error) {
	if c.Keyword != "" {
		return json.Marshal(c.Keyword)
	}

	type object responsesToolChoice

	return json.Marshal(object(c))
}

// responsesAllowedTool identifies a tool in tool_choice.allowed_tools. Most
// built-in tools are identified by type alone.
type responsesAllowedTool struct {
	Type        string `json:"type"`
	Name        string `json:"name,omitempty"`
	ServerLabel string `json:"server_label,omitempty"`
}

// key identifies the tool for comparisons.
func (t responsesAllowedTool) key() string {
	switch t.Type {
	case "mcp":
		return "mcp:" + t.ServerLabel
	case "function", "custom":
		return t.Type + ":" + t.Name
	default:
		return t.Type
	}
}
