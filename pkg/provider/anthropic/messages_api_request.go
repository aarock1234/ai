package anthropic

import (
	"encoding/json"
)

// Content block types shared by requests and responses.
const (
	blockText                              = "text"
	blockImage                             = "image"
	blockDocument                          = "document"
	blockContainerUpload                   = "container_upload"
	blockToolResult                        = "tool_result"
	blockToolReference                     = "tool_reference"
	blockThinking                          = "thinking"
	blockRedactedThinking                  = "redacted_thinking"
	blockFallback                          = "fallback"
	blockCompaction                        = "compaction"
	blockToolUse                           = "tool_use"
	blockServerToolUse                     = "server_tool_use"
	blockMCPToolUse                        = "mcp_tool_use"
	blockMCPToolResult                     = "mcp_tool_result"
	blockWebFetchToolResult                = "web_fetch_tool_result"
	blockWebSearchToolResult               = "web_search_tool_result"
	blockCodeExecutionToolResult           = "code_execution_tool_result"
	blockBashCodeExecutionToolResult       = "bash_code_execution_tool_result"
	blockTextEditorCodeExecutionToolResult = "text_editor_code_execution_tool_result"
	blockToolSearchToolResult              = "tool_search_tool_result"
	blockAdvisorToolResult                 = "advisor_tool_result"
)

// messagesRequest is the body of a Messages API call. Field order follows the
// upstream request so that encoded bodies match.
type messagesRequest struct {
	Model             string                 `json:"model"`
	MaxTokens         int                    `json:"max_tokens"`
	Temperature       *float64               `json:"temperature,omitempty"`
	TopK              *int                   `json:"top_k,omitempty"`
	TopP              *float64               `json:"top_p,omitempty"`
	StopSequences     []string               `json:"stop_sequences,omitempty"`
	Thinking          *thinkingConfig        `json:"thinking,omitempty"`
	OutputConfig      *outputConfig          `json:"output_config,omitempty"`
	Speed             Speed                  `json:"speed,omitempty"`
	ServiceTier       ServiceTier            `json:"service_tier,omitempty"`
	InferenceGeo      InferenceGeo           `json:"inference_geo,omitempty"`
	Fallbacks         *Fallbacks             `json:"fallbacks,omitempty"`
	CacheControl      *CacheControl          `json:"cache_control,omitempty"`
	Metadata          *requestMetadata       `json:"metadata,omitempty"`
	MCPServers        []mcpServer            `json:"mcp_servers,omitempty"`
	Container         *requestContainer      `json:"container,omitempty"`
	System            []*textBlock           `json:"system,omitempty"`
	Messages          []requestMessage       `json:"messages"`
	Safeguards        []safeguard            `json:"safeguards,omitempty"`
	Compaction        *compactionRequest     `json:"compaction,omitempty"`
	ContextManagement *contextManagementWire `json:"context_management,omitempty"`
	Tools             []requestTool          `json:"tools,omitempty"`
	ToolChoice        *toolChoice            `json:"tool_choice,omitempty"`
	Stream            bool                   `json:"stream,omitempty"`
}

type thinkingConfig struct {
	Type         ThinkingType      `json:"type,omitempty"`
	BudgetTokens *int              `json:"budget_tokens,omitempty"`
	Display      ThinkingDisplay   `json:"display,omitempty"`
	BlockBinding *blockBindingWire `json:"block_binding,omitempty"`
}

type blockBindingWire struct {
	PrefixMismatchBehavior PrefixMismatchBehavior `json:"prefix_mismatch_behavior"`
}

type outputConfig struct {
	Effort     Effort        `json:"effort,omitempty"`
	TaskBudget *TaskBudget   `json:"task_budget,omitempty"`
	Format     *outputFormat `json:"format,omitempty"`
}

type outputFormat struct {
	Type   string          `json:"type"`
	Schema json.RawMessage `json:"schema"`
}

type requestMetadata struct {
	UserID string `json:"user_id"`
}

type mcpServer struct {
	Type               string                `json:"type"`
	Name               string                `json:"name"`
	URL                string                `json:"url"`
	AuthorizationToken *string               `json:"authorization_token,omitempty"`
	ToolConfiguration  *mcpToolConfiguration `json:"tool_configuration,omitempty"`
}

type mcpToolConfiguration struct {
	AllowedTools []string `json:"allowed_tools,omitempty"`
	Enabled      *bool    `json:"enabled,omitempty"`
}

// requestContainer is a container ID string, or an object when skills are
// loaded.
type requestContainer struct {
	ID     string
	Skills []containerSkill
}

type containerSkill struct {
	Type    SkillType `json:"type"`
	SkillID string    `json:"skill_id"`
	Version string    `json:"version,omitempty"`
}

type containerObject struct {
	ID     string           `json:"id,omitempty"`
	Skills []containerSkill `json:"skills"`
}

// MarshalJSON encodes the container ID alone, or the ID and skills.
func (c requestContainer) MarshalJSON() ([]byte, error) {
	if len(c.Skills) == 0 {
		return json.Marshal(c.ID)
	}

	return json.Marshal(containerObject(c))
}

type safeguard struct {
	Type              string          `json:"type"`
	ClassifierContext json.RawMessage `json:"classifier_context,omitempty"`
}

type compactionRequest struct {
	Type         string  `json:"type"`
	Instructions *string `json:"instructions,omitempty"`
}

type contextManagementWire struct {
	Edits []contextEditWire `json:"edits"`
}

type contextEditWire struct {
	Type                 ContextEditType `json:"type"`
	Trigger              *ContextAmount  `json:"trigger,omitempty"`
	Keep                 *ContextKeep    `json:"keep,omitempty"`
	ClearAtLeast         *ContextAmount  `json:"clear_at_least,omitempty"`
	ClearToolInputs      *bool           `json:"clear_tool_inputs,omitempty"`
	ExcludeTools         []string        `json:"exclude_tools,omitempty"`
	PauseAfterCompaction *bool           `json:"pause_after_compaction,omitempty"`
	Instructions         *string         `json:"instructions,omitempty"`
}

type toolChoice struct {
	Type                   string `json:"type"`
	Name                   string `json:"name,omitempty"`
	DisableParallelToolUse *bool  `json:"disable_parallel_tool_use,omitempty"`
}

// requestMessage is one entry of the messages array. System messages appear
// only mid-conversation.
type requestMessage struct {
	Role         string               `json:"role"`
	Content      []requestBlock       `json:"content"`
	ClearAt      ClearAt              `json:"clear_at,omitempty"`
	OutputConfig *messageOutputConfig `json:"output_config,omitempty"`
}

type messageOutputConfig struct {
	Effort Effort `json:"effort"`
}

// requestBlock is a content block of a request message.
type requestBlock interface {
	isRequestBlock()
}

type textBlock struct {
	Type         string        `json:"type"`
	Text         string        `json:"text"`
	Citations    []Citation    `json:"citations,omitempty"`
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

type toolChangeBlock struct {
	Type ToolChangeType `json:"type"`
	Tool toolReference  `json:"tool"`
}

type toolReference struct {
	Type     string `json:"type"`
	Name     string `json:"name,omitempty"`
	ToolName string `json:"tool_name,omitempty"`
}

type imageBlock struct {
	Type         string        `json:"type"`
	Source       contentSource `json:"source"`
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

// mediaBlock is an image or video block built by a host's media part
// converter.
type mediaBlock struct {
	Type         MediaContentType `json:"type"`
	Source       json.RawMessage  `json:"source"`
	CacheControl *CacheControl    `json:"cache_control,omitempty"`
}

type documentBlock struct {
	Type         string           `json:"type"`
	Source       contentSource    `json:"source"`
	Title        string           `json:"title,omitempty"`
	Context      string           `json:"context,omitempty"`
	Citations    *CitationsConfig `json:"citations,omitempty"`
	CacheControl *CacheControl    `json:"cache_control,omitempty"`
}

// contentSource is a base64, url, text or file source of an image or
// document.
type contentSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
	FileID    string `json:"file_id,omitempty"`
}

type containerUploadBlock struct {
	Type   string `json:"type"`
	FileID string `json:"file_id"`
}

type toolResultBlock struct {
	Type         string            `json:"type"`
	ToolUseID    string            `json:"tool_use_id"`
	ToolsetName  string            `json:"toolset_name,omitempty"`
	Content      toolResultContent `json:"content"`
	IsError      *bool             `json:"is_error,omitempty"`
	CacheControl *CacheControl     `json:"cache_control,omitempty"`
}

// toolResultContent is a text result or a list of nested blocks.
type toolResultContent struct {
	Text  string
	Parts []requestBlock
}

// MarshalJSON encodes the text, or the nested blocks when there are any.
func (c toolResultContent) MarshalJSON() ([]byte, error) {
	if c.Parts == nil {
		return json.Marshal(c.Text)
	}

	return json.Marshal(c.Parts)
}

type thinkingBlock struct {
	Type      string `json:"type"`
	Thinking  string `json:"thinking"`
	Signature string `json:"signature"`
}

type redactedThinkingBlock struct {
	Type string `json:"type"`
	Data string `json:"data"`
}

type fallbackBlock struct {
	Type string   `json:"type"`
	From ModelRef `json:"from"`
	To   ModelRef `json:"to"`
}

type compactionBlock struct {
	Type         string        `json:"type"`
	Content      string        `json:"content"`
	Signature    string        `json:"signature,omitempty"`
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

type toolUseBlock struct {
	Type         string          `json:"type"`
	ID           string          `json:"id"`
	Name         string          `json:"name"`
	ToolsetName  string          `json:"toolset_name,omitempty"`
	Input        json.RawMessage `json:"input"`
	Caller       *wireCaller     `json:"caller,omitempty"`
	CacheControl *CacheControl   `json:"cache_control,omitempty"`
}

type serverToolUseBlock struct {
	Type         string          `json:"type"`
	ID           string          `json:"id"`
	Name         string          `json:"name"`
	Input        json.RawMessage `json:"input"`
	Caller       *wireCaller     `json:"caller,omitempty"`
	CacheControl *CacheControl   `json:"cache_control,omitempty"`
}

type mcpToolUseBlock struct {
	Type         string          `json:"type"`
	ID           string          `json:"id"`
	Name         string          `json:"name"`
	Input        json.RawMessage `json:"input"`
	ServerName   string          `json:"server_name"`
	CacheControl *CacheControl   `json:"cache_control,omitempty"`
}

type mcpToolResultBlock struct {
	Type         string          `json:"type"`
	ToolUseID    string          `json:"tool_use_id"`
	IsError      bool            `json:"is_error"`
	Content      json.RawMessage `json:"content"`
	CacheControl *CacheControl   `json:"cache_control,omitempty"`
}

// serverToolResultBlock replays the result of a provider-executed tool. Type
// names the result block, such as "web_search_tool_result".
type serverToolResultBlock struct {
	Type         string                  `json:"type"`
	ToolUseID    string                  `json:"tool_use_id"`
	Content      serverToolResultContent `json:"content"`
	Caller       *wireCaller             `json:"caller,omitempty"`
	CacheControl *CacheControl           `json:"cache_control,omitempty"`
}

// serverToolResultContent is the content of a replayed provider tool result.
type serverToolResultContent interface {
	isServerToolResultContent()
}

// wireError is the error content of a provider tool result block.
type wireError struct {
	Type      string `json:"type"`
	ErrorCode string `json:"error_code"`
}

type wireWebFetchResult struct {
	Type        string              `json:"type"`
	URL         string              `json:"url"`
	RetrievedAt *string             `json:"retrieved_at"`
	Content     wireWebFetchContent `json:"content"`
}

type wireWebFetchContent struct {
	Type      string             `json:"type"`
	Title     *string            `json:"title"`
	Citations *CitationsConfig   `json:"citations,omitempty"`
	Source    wireWebFetchSource `json:"source"`
}

type wireWebFetchSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

type wireWebSearchResults []wireWebSearchResult

type wireWebSearchResult struct {
	URL              string  `json:"url"`
	Title            *string `json:"title,omitempty"`
	PageAge          *string `json:"page_age,omitempty"`
	EncryptedContent string  `json:"encrypted_content"`
	Type             string  `json:"type"`
}

type wireToolSearchResult struct {
	Type           string          `json:"type"`
	ToolReferences []toolReference `json:"tool_references"`
}

type wireAdvisorResult struct {
	Type             string  `json:"type"`
	Text             *string `json:"text,omitempty"`
	EncryptedContent *string `json:"encrypted_content,omitempty"`
	StopReason       *string `json:"stop_reason,omitempty"`
	ErrorCode        *string `json:"error_code,omitempty"`
}

// wireCaller identifies what triggered a tool call: a direct call, or code
// execution with the ID of its tool call.
type wireCaller struct {
	Type   CallerType `json:"type"`
	ToolID string     `json:"tool_id,omitempty"`
}

// requestTool is a tool definition. Name and Type are empty for definitions
// that do not have them.
type requestTool interface {
	toolName() string
	toolType() string
}

type functionTool struct {
	Name                string            `json:"name"`
	Description         string            `json:"description,omitempty"`
	InputSchema         json.RawMessage   `json:"input_schema"`
	CacheControl        *CacheControl     `json:"cache_control,omitempty"`
	EagerInputStreaming bool              `json:"eager_input_streaming,omitempty"`
	Strict              *bool             `json:"strict,omitempty"`
	DeferLoading        *bool             `json:"defer_loading,omitempty"`
	AllowedCallers      []CallerType      `json:"allowed_callers,omitempty"`
	InputExamples       []json.RawMessage `json:"input_examples,omitempty"`
}

// builtinTool is a provider tool defined by its type and name alone.
type builtinTool struct {
	Type string `json:"type"`
	Name string `json:"name"`
}

type computerTool struct {
	Name            string `json:"name"`
	Type            string `json:"type"`
	DisplayWidthPx  int    `json:"display_width_px"`
	DisplayHeightPx int    `json:"display_height_px"`
	DisplayNumber   *int   `json:"display_number,omitempty"`
	EnableZoom      *bool  `json:"enable_zoom,omitempty"`
}

type computerToolsetTool struct {
	Type    string                          `json:"type"`
	Configs map[ComputerAction]memberConfig `json:"configs,omitempty"`
}

type memberConfig struct {
	Enabled      *bool `json:"enabled,omitempty"`
	DeferLoading *bool `json:"defer_loading,omitempty"`
}

type textEditorTool struct {
	Name          string `json:"name"`
	Type          string `json:"type"`
	MaxCharacters *int   `json:"max_characters,omitempty"`
}

type webFetchTool struct {
	Type              string            `json:"type"`
	Name              string            `json:"name"`
	MaxUses           *int              `json:"max_uses,omitempty"`
	AllowedDomains    []string          `json:"allowed_domains,omitempty"`
	BlockedDomains    []string          `json:"blocked_domains,omitempty"`
	Citations         *CitationsConfig  `json:"citations,omitempty"`
	MaxContentTokens  *int              `json:"max_content_tokens,omitempty"`
	UseCache          *bool             `json:"use_cache,omitempty"`
	ResponseInclusion ResponseInclusion `json:"response_inclusion,omitempty"`
}

type webSearchTool struct {
	Type              string            `json:"type"`
	Name              string            `json:"name"`
	MaxUses           *int              `json:"max_uses,omitempty"`
	AllowedDomains    []string          `json:"allowed_domains,omitempty"`
	BlockedDomains    []string          `json:"blocked_domains,omitempty"`
	UserLocation      *UserLocation     `json:"user_location,omitempty"`
	ResponseInclusion ResponseInclusion `json:"response_inclusion,omitempty"`
}

type advisorTool struct {
	Type      string          `json:"type"`
	Name      string          `json:"name"`
	Model     string          `json:"model"`
	MaxUses   *int            `json:"max_uses,omitempty"`
	MaxTokens *int            `json:"max_tokens,omitempty"`
	Caching   *AdvisorCaching `json:"caching,omitempty"`
}

func (*textBlock) isRequestBlock()             {}
func (*toolChangeBlock) isRequestBlock()       {}
func (*toolReference) isRequestBlock()         {}
func (*imageBlock) isRequestBlock()            {}
func (*mediaBlock) isRequestBlock()            {}
func (*documentBlock) isRequestBlock()         {}
func (*containerUploadBlock) isRequestBlock()  {}
func (*toolResultBlock) isRequestBlock()       {}
func (*thinkingBlock) isRequestBlock()         {}
func (*redactedThinkingBlock) isRequestBlock() {}
func (*fallbackBlock) isRequestBlock()         {}
func (*compactionBlock) isRequestBlock()       {}
func (*toolUseBlock) isRequestBlock()          {}
func (*serverToolUseBlock) isRequestBlock()    {}
func (*mcpToolUseBlock) isRequestBlock()       {}
func (*mcpToolResultBlock) isRequestBlock()    {}
func (*serverToolResultBlock) isRequestBlock() {}

func (wireError) isServerToolResultContent()                    {}
func (wireWebFetchResult) isServerToolResultContent()           {}
func (wireWebSearchResults) isServerToolResultContent()         {}
func (wireToolSearchResult) isServerToolResultContent()         {}
func (wireAdvisorResult) isServerToolResultContent()            {}
func (CodeExecutionResult) isServerToolResultContent()          {}
func (EncryptedCodeExecutionResult) isServerToolResultContent() {}
func (BashCodeExecutionResult) isServerToolResultContent()      {}
func (CodeExecutionError) isServerToolResultContent()           {}
func (TextEditorViewResult) isServerToolResultContent()         {}
func (TextEditorCreateResult) isServerToolResultContent()       {}
func (TextEditorStrReplaceResult) isServerToolResultContent()   {}

func (t *functionTool) toolName() string {
	return t.Name
}

func (t *functionTool) toolType() string {
	return ""
}

func (t *builtinTool) toolName() string {
	return t.Name
}

func (t *builtinTool) toolType() string {
	return t.Type
}

func (t *computerTool) toolName() string {
	return t.Name
}

func (t *computerTool) toolType() string {
	return t.Type
}

func (t *computerToolsetTool) toolName() string {
	return ""
}

func (t *computerToolsetTool) toolType() string {
	return t.Type
}

func (t *textEditorTool) toolName() string {
	return t.Name
}

func (t *textEditorTool) toolType() string {
	return t.Type
}

func (t *webFetchTool) toolName() string {
	return t.Name
}

func (t *webFetchTool) toolType() string {
	return t.Type
}

func (t *webSearchTool) toolName() string {
	return t.Name
}

func (t *webSearchTool) toolType() string {
	return t.Type
}

func (t *advisorTool) toolName() string {
	return t.Name
}

func (t *advisorTool) toolType() string {
	return t.Type
}
