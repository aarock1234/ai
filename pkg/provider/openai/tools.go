package openai

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// Provider tool IDs.
const (
	applyPatchToolID              = "openai.apply_patch"
	codeInterpreterToolID         = "openai.code_interpreter"
	computerToolID                = "openai.computer"
	customToolID                  = "openai.custom"
	fileSearchToolID              = "openai.file_search"
	imageGenerationToolID         = "openai.image_generation"
	localShellToolID              = "openai.local_shell"
	shellToolID                   = "openai.shell"
	webSearchToolID               = "openai.web_search"
	webSearchPreviewToolID        = "openai.web_search_preview"
	mcpToolID                     = "openai.mcp"
	programmaticToolCallingToolID = "openai.programmatic_tool_calling"
	toolSearchToolID              = "openai.tool_search"
)

// ApplyPatch returns the apply patch tool, which lets the model create,
// update and delete files with structured diffs. The application applies
// each ApplyPatchInput and returns an ApplyPatchOutput.
func ApplyPatch() model.ProviderTool {
	return newOpenAITool(applyPatchToolID, "apply_patch", struct{}{})
}

// CodeInterpreterArgs configures the code interpreter.
type CodeInterpreterArgs struct {
	// Container is a container ID or the files to make available. Nil uses
	// an automatic container without files.
	Container *CodeInterpreterContainer `json:"container,omitempty"`
}

// CodeInterpreterContainer is an existing container, by ID, or an automatic
// container with uploaded files.
type CodeInterpreterContainer struct {
	ID      string
	FileIDs []string
}

// MarshalJSON writes the container ID as a string, or the file IDs as an
// object.
func (c CodeInterpreterContainer) MarshalJSON() ([]byte, error) {
	if c.ID != "" {
		return json.Marshal(c.ID)
	}

	return json.Marshal(codeInterpreterAutoContainer{
		FileIDs: c.FileIDs,
	})
}

// UnmarshalJSON reads a container ID string or a file IDs object.
func (c *CodeInterpreterContainer) UnmarshalJSON(data []byte) error {
	var id string
	if err := json.Unmarshal(data, &id); err == nil {
		*c = CodeInterpreterContainer{
			ID: id,
		}

		return nil
	}

	var auto codeInterpreterAutoContainer
	if err := json.Unmarshal(data, &auto); err != nil {
		return fmt.Errorf("container must be a string or an object: %w", err)
	}

	*c = CodeInterpreterContainer{
		FileIDs: auto.FileIDs,
	}

	return nil
}

type codeInterpreterAutoContainer struct {
	FileIDs []string `json:"fileIds,omitempty"`
}

// CodeInterpreter returns the code interpreter tool, which runs Python code
// in a sandbox. OpenAI executes it.
func CodeInterpreter(args CodeInterpreterArgs) model.ProviderTool {
	return newOpenAITool(codeInterpreterToolID, "code_interpreter", args)
}

// Computer returns the computer tool, which lets the model operate a
// browser or desktop through batched UI actions. The application executes
// each ComputerInput and returns a ComputerOutput. Run it in an isolated
// environment and treat on-screen content as untrusted.
func Computer() model.ProviderTool {
	return newOpenAITool(computerToolID, "computer", struct{}{})
}

// CustomToolArgs configures a custom tool.
type CustomToolArgs struct {
	Description string `json:"description,omitempty"`

	// Async lets the model continue without waiting for the result. It is
	// supported by GPT-6 and later models.
	Async *bool `json:"async,omitempty"`

	// Format constrains the input. Nil allows unconstrained text.
	Format *CustomToolFormat `json:"format,omitempty"`
}

// CustomToolFormat is "text" or "grammar" with a syntax and definition.
type CustomToolFormat struct {
	Type string `json:"type"`

	// Syntax is "regex" or "lark".
	Syntax     string `json:"syntax,omitempty"`
	Definition string `json:"definition,omitempty"`
}

// CustomTool returns a custom tool named name, whose input is free text
// optionally constrained by a grammar. The application executes it.
func CustomTool(name string, args CustomToolArgs) model.ProviderTool {
	return newOpenAITool(customToolID, name, args)
}

// FileSearchArgs configures file search.
type FileSearchArgs struct {
	// VectorStoreIDs lists the vector stores to search.
	VectorStoreIDs []string `json:"vectorStoreIds"`

	// MaxNumResults limits the results. The API default is 10.
	MaxNumResults *int `json:"maxNumResults,omitempty"`

	Ranking *FileSearchRanking `json:"ranking,omitempty"`

	// Filters restrict the searched files by attribute.
	Filters *FileSearchFilter `json:"filters,omitempty"`
}

// FileSearchRanking configures result ranking.
type FileSearchRanking struct {
	Ranker string `json:"ranker,omitempty"`

	// ScoreThreshold drops results below this score, between 0 and 1.
	ScoreThreshold *float64 `json:"scoreThreshold,omitempty"`
}

// FileSearchFilterType is a comparison or a compound filter operator.
type FileSearchFilterType string

// File search filter operators.
const (
	FileSearchEqual          FileSearchFilterType = "eq"
	FileSearchNotEqual       FileSearchFilterType = "ne"
	FileSearchGreater        FileSearchFilterType = "gt"
	FileSearchGreaterOrEqual FileSearchFilterType = "gte"
	FileSearchLess           FileSearchFilterType = "lt"
	FileSearchLessOrEqual    FileSearchFilterType = "lte"
	FileSearchIn             FileSearchFilterType = "in"
	FileSearchNotIn          FileSearchFilterType = "nin"
	FileSearchAnd            FileSearchFilterType = "and"
	FileSearchOr             FileSearchFilterType = "or"
)

// FileSearchFilter compares an attribute with a value, or combines filters
// with "and" or "or". Comparisons set Key and Value; compound filters set
// Filters.
type FileSearchFilter struct {
	Type    FileSearchFilterType `json:"type"`
	Key     string               `json:"key,omitempty"`
	Value   *FileSearchValue     `json:"value,omitempty"`
	Filters []FileSearchFilter   `json:"filters,omitempty"`
}

// FileSearchValue is the value a file search comparison filter compares an
// attribute with: a string, number, boolean or list of strings. Create it
// with FileSearchString, FileSearchNumber, FileSearchBool or
// FileSearchStrings.
type FileSearchValue struct {
	text    *string
	number  *float64
	boolean *bool
	texts   []string
}

// FileSearchString returns a string comparison value.
func FileSearchString(text string) *FileSearchValue {
	return &FileSearchValue{
		text: &text,
	}
}

// FileSearchNumber returns a number comparison value.
func FileSearchNumber(number float64) *FileSearchValue {
	return &FileSearchValue{
		number: &number,
	}
}

// FileSearchBool returns a boolean comparison value.
func FileSearchBool(boolean bool) *FileSearchValue {
	return &FileSearchValue{
		boolean: &boolean,
	}
}

// FileSearchStrings returns a list comparison value for the "in" and "nin"
// operators.
func FileSearchStrings(texts ...string) *FileSearchValue {
	return &FileSearchValue{
		texts: append([]string{}, texts...),
	}
}

// MarshalJSON encodes the value as a JSON string, number, boolean or array.
func (v FileSearchValue) MarshalJSON() ([]byte, error) {
	switch {
	case v.text != nil:
		return json.Marshal(*v.text)
	case v.number != nil:
		return json.Marshal(*v.number)
	case v.boolean != nil:
		return json.Marshal(*v.boolean)
	case v.texts != nil:
		return json.Marshal(v.texts)
	default:
		return []byte("null"), nil
	}
}

// UnmarshalJSON decodes a JSON string, number, boolean or array of strings.
func (v *FileSearchValue) UnmarshalJSON(data []byte) error {
	var text string
	if err := json.Unmarshal(data, &text); err == nil {
		*v = *FileSearchString(text)

		return nil
	}

	var number float64
	if err := json.Unmarshal(data, &number); err == nil {
		*v = *FileSearchNumber(number)

		return nil
	}

	var boolean bool
	if err := json.Unmarshal(data, &boolean); err == nil {
		*v = *FileSearchBool(boolean)

		return nil
	}

	var texts []string
	if err := json.Unmarshal(data, &texts); err == nil && texts != nil {
		*v = *FileSearchStrings(texts...)

		return nil
	}

	return fmt.Errorf("file search filter value must be a string, number, boolean or array of strings: %s", data)
}

// FileSearch returns the file search tool, which retrieves information from
// uploaded files. OpenAI executes it.
func FileSearch(args FileSearchArgs) model.ProviderTool {
	return newOpenAITool(fileSearchToolID, "file_search", args)
}

// ImageGenerationArgs configures image generation.
type ImageGenerationArgs struct {
	// Action is "generate", "edit" or "auto" (the default).
	Action string `json:"action,omitempty"`

	// Background is "auto", "opaque" or "transparent".
	Background string `json:"background,omitempty"`

	// InputFidelity is "low" or "high".
	InputFidelity string `json:"inputFidelity,omitempty"`

	// InputImageMask is an optional inpainting mask.
	InputImageMask *ImageGenerationMask `json:"inputImageMask,omitempty"`

	// Model is the image model. The API default is gpt-image-1.
	Model string `json:"model,omitempty"`

	// Moderation is "auto" or "low".
	Moderation string `json:"moderation,omitempty"`

	// OutputCompression is the compression level, from 0 to 100.
	OutputCompression *int `json:"outputCompression,omitempty"`

	// OutputFormat is "png", "jpeg" or "webp".
	OutputFormat string `json:"outputFormat,omitempty"`

	// PartialImages is the number of partial images to stream, from 0 to 3.
	PartialImages *int `json:"partialImages,omitempty"`

	// Quality is "auto", "low", "medium", "high", "xhigh" or "max".
	Quality string `json:"quality,omitempty"`

	// Size is "auto", "1024x1024", "1024x1536", "1536x1024" or a custom
	// WIDTHxHEIGHT size on models that support it.
	Size string `json:"size,omitempty"`
}

// ImageGenerationMask is an inpainting mask given by file ID or image URL.
type ImageGenerationMask struct {
	FileID   string `json:"fileId,omitempty"`
	ImageURL string `json:"imageUrl,omitempty"`
}

// ImageGeneration returns the image generation tool. OpenAI executes it.
func ImageGeneration(args ImageGenerationArgs) model.ProviderTool {
	return newOpenAITool(imageGenerationToolID, "image_generation", args)
}

// LocalShell returns the local shell tool, which lets the model run
// commands on a machine the application provides. The application executes
// each LocalShellInput and returns a LocalShellOutput.
func LocalShell() model.ProviderTool {
	return newOpenAITool(localShellToolID, "local_shell", struct{}{})
}

// ShellArgs configures the shell tool.
type ShellArgs struct {
	// Environment selects where commands run. Nil runs them locally.
	Environment *ShellEnvironment `json:"environment,omitempty"`
}

// ShellEnvironmentType identifies where shell commands run.
type ShellEnvironmentType string

// Shell environment types.
const (
	ShellContainerAuto      ShellEnvironmentType = "containerAuto"
	ShellContainerReference ShellEnvironmentType = "containerReference"
	ShellLocal              ShellEnvironmentType = "local"
)

// ShellEnvironment selects where shell commands run. Automatic containers
// use FileIDs, MemoryLimit, NetworkPolicy and Skills; container references
// use ContainerID; local environments use Skills with paths. OpenAI
// executes commands in containers; the application executes local ones.
type ShellEnvironment struct {
	Type          ShellEnvironmentType `json:"type,omitempty"`
	FileIDs       []string             `json:"fileIds,omitempty"`
	MemoryLimit   string               `json:"memoryLimit,omitempty"`
	NetworkPolicy *ShellNetworkPolicy  `json:"networkPolicy,omitempty"`
	Skills        []ShellSkill         `json:"skills,omitempty"`
	ContainerID   string               `json:"containerId,omitempty"`
}

// ShellNetworkPolicy is "disabled" or "allowlist" with allowed domains.
type ShellNetworkPolicy struct {
	Type           string              `json:"type"`
	AllowedDomains []string            `json:"allowedDomains,omitempty"`
	DomainSecrets  []ShellDomainSecret `json:"domainSecrets,omitempty"`
}

// ShellDomainSecret is a secret sent to an allowed domain.
type ShellDomainSecret struct {
	Domain string `json:"domain"`
	Name   string `json:"name"`
	Value  string `json:"value"`
}

// ShellSkill is a skill available to the shell. Container skills are
// "skillReference" with ProviderReference and Version, or "inline" with
// Name, Description and Source. Local skills have Name, Description and
// Path.
type ShellSkill struct {
	Type              string            `json:"type,omitempty"`
	ProviderReference map[string]string `json:"providerReference,omitempty"`
	Version           string            `json:"version,omitempty"`
	Name              string            `json:"name,omitempty"`
	Description       string            `json:"description,omitempty"`
	Source            *ShellSkillSource `json:"source,omitempty"`
	Path              string            `json:"path,omitempty"`
}

// ShellSkillSource is an inline skill archive.
type ShellSkillSource struct {
	// Type is "base64".
	Type string `json:"type"`

	// MediaType is "application/zip".
	MediaType string `json:"mediaType"`
	Data      string `json:"data"`
}

// Shell returns the shell tool. The application executes each ShellInput
// and returns a ShellOutput, unless the environment is a container, in
// which case OpenAI executes the commands. Sandbox local execution.
func Shell(args ShellArgs) model.ProviderTool {
	return newOpenAITool(shellToolID, "shell", args)
}

// WebSearchLocation approximates the user's location.
type WebSearchLocation struct {
	// Type is "approximate".
	Type string `json:"type"`

	// Country is a two-letter ISO country code.
	Country string `json:"country,omitempty"`
	City    string `json:"city,omitempty"`
	Region  string `json:"region,omitempty"`

	// Timezone is an IANA time zone.
	Timezone string `json:"timezone,omitempty"`
}

// WebSearchPreviewArgs configures the preview web search tool.
type WebSearchPreviewArgs struct {
	// SearchContextSize is "low", "medium" or "high".
	SearchContextSize string             `json:"searchContextSize,omitempty"`
	UserLocation      *WebSearchLocation `json:"userLocation,omitempty"`
}

// WebSearchPreview returns the preview web search tool. OpenAI executes it.
//
// Deprecated: use WebSearch.
func WebSearchPreview(args WebSearchPreviewArgs) model.ProviderTool {
	return newOpenAITool(webSearchPreviewToolID, "web_search_preview", args)
}

// WebSearchArgs configures web search.
type WebSearchArgs struct {
	// ExternalWebAccess fetches live content when true (the default) and
	// uses cached results when false.
	ExternalWebAccess *bool `json:"externalWebAccess,omitempty"`

	Filters *WebSearchFilters `json:"filters,omitempty"`

	// SearchContextSize is "low", "medium" or "high".
	SearchContextSize string             `json:"searchContextSize,omitempty"`
	UserLocation      *WebSearchLocation `json:"userLocation,omitempty"`
}

// WebSearchFilters restrict the searched domains. Domains omit the scheme
// and include their subdomains.
type WebSearchFilters struct {
	AllowedDomains []string `json:"allowedDomains,omitempty"`
	BlockedDomains []string `json:"blockedDomains,omitempty"`
}

// WebSearch returns the web search tool, which gives the model up-to-date
// information with cited sources. OpenAI executes it.
func WebSearch(args WebSearchArgs) model.ProviderTool {
	return newOpenAITool(webSearchToolID, "web_search", args)
}

// MCPArgs configures a remote MCP server or service connector. ServerURL or
// ConnectorID is required.
type MCPArgs struct {
	// ServerLabel identifies the server in tool calls.
	ServerLabel string `json:"serverLabel"`

	// AllowedTools restricts the server's tools.
	AllowedTools *MCPAllowedTools `json:"allowedTools,omitempty"`

	// Authorization is an OAuth access token for the server.
	Authorization string            `json:"authorization,omitempty"`
	ConnectorID   string            `json:"connectorId,omitempty"`
	Headers       map[string]string `json:"headers,omitempty"`

	// RequireApproval selects which calls need approval. Nil never requires
	// approval.
	RequireApproval *MCPApproval `json:"requireApproval,omitempty"`

	ServerDescription string `json:"serverDescription,omitempty"`
	ServerURL         string `json:"serverUrl,omitempty"`
}

// MCPAllowedTools is a list of tool names, or a filter when Filter is set.
type MCPAllowedTools struct {
	ToolNames []string
	Filter    *MCPToolFilter
}

// MCPToolFilter selects tools by name and read-only annotation.
type MCPToolFilter struct {
	ReadOnly  *bool    `json:"readOnly,omitempty"`
	ToolNames []string `json:"toolNames,omitempty"`
}

// MarshalJSON writes the filter when set, otherwise the list of names.
func (a MCPAllowedTools) MarshalJSON() ([]byte, error) {
	if a.Filter != nil {
		return json.Marshal(a.Filter)
	}

	return json.Marshal(a.ToolNames)
}

// UnmarshalJSON reads a list of names or a filter object.
func (a *MCPAllowedTools) UnmarshalJSON(data []byte) error {
	var names []string
	if err := json.Unmarshal(data, &names); err == nil {
		*a = MCPAllowedTools{
			ToolNames: names,
		}

		return nil
	}

	var filter MCPToolFilter
	if err := json.Unmarshal(data, &filter); err != nil {
		return fmt.Errorf("allowed tools must be a list or an object: %w", err)
	}

	*a = MCPAllowedTools{
		Filter: &filter,
	}

	return nil
}

// MCPApproval is the policy "always" or "never", or the tools that never
// need approval when Never is set.
type MCPApproval struct {
	Policy string
	Never  *MCPToolFilter
}

type mcpApprovalFilter struct {
	Never *MCPToolFilter `json:"never,omitempty"`
}

// MarshalJSON writes the policy string or the never filter.
func (a MCPApproval) MarshalJSON() ([]byte, error) {
	if a.Policy != "" {
		return json.Marshal(a.Policy)
	}

	return json.Marshal(mcpApprovalFilter{
		Never: a.Never,
	})
}

// UnmarshalJSON reads a policy string or a never filter object.
func (a *MCPApproval) UnmarshalJSON(data []byte) error {
	var policy string
	if err := json.Unmarshal(data, &policy); err == nil {
		*a = MCPApproval{
			Policy: policy,
		}

		return nil
	}

	var filter mcpApprovalFilter
	if err := json.Unmarshal(data, &filter); err != nil {
		return fmt.Errorf("require approval must be a string or an object: %w", err)
	}

	*a = MCPApproval{
		Never: filter.Never,
	}

	return nil
}

// MCP returns the MCP tool, which lets the model call tools on a remote MCP
// server or service connector. OpenAI executes the calls; calls that need
// approval produce tool approval requests.
func MCP(args MCPArgs) model.ProviderTool {
	return newOpenAITool(mcpToolID, "mcp", args)
}

// ProgrammaticToolCalling returns the programmatic tool calling tool, which
// lets the model write and run JavaScript that calls eligible tools. Mark
// function tools as callable from programs with
// ResponsesFunctionToolOptions.AllowedCallers.
func ProgrammaticToolCalling() model.ProviderTool {
	return newOpenAITool(programmaticToolCallingToolID, "programmatic_tool_calling", struct{}{})
}

// ToolSearchArgs configures tool search.
type ToolSearchArgs struct {
	// Execution is "server" (the default), where OpenAI searches the
	// deferred tools, or "client", where the application performs the
	// lookup and returns a ToolSearchOutput.
	Execution string `json:"execution,omitempty"`

	// Description describes a client-executed search.
	Description string `json:"description,omitempty"`

	// Parameters is the JSON Schema of client-executed search arguments.
	Parameters json.RawMessage `json:"parameters,omitempty"`
}

// ToolSearch returns the tool search tool, which loads tools marked with
// ResponsesFunctionToolOptions.DeferLoading when the model needs them.
// Parameters that are not a JSON Schema object are reported as an invalid
// argument when the tool is sent.
func ToolSearch(args ToolSearchArgs) model.ProviderTool {
	if len(args.Parameters) > 0 && !json.Valid(args.Parameters) {
		args.Parameters = jsonobject.EncodeString(string(args.Parameters))
	}

	return newOpenAITool(toolSearchToolID, "tool_search", args)
}

// newOpenAITool encodes args into a provider tool. Argument types contain
// only plain data and valid raw JSON, so encoding cannot fail.
func newOpenAITool[Args any](id, name string, args Args) model.ProviderTool {
	encoded, err := json.Marshal(args)
	if err != nil {
		panic(fmt.Sprintf("openai: encode %s tool arguments: %v", id, err))
	}

	return model.ProviderTool{
		ID:   id,
		Name: name,
		Args: encoded,
	}
}
