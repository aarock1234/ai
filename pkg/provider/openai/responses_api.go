package openai

import (
	"encoding/json"
)

// responsesRequest is the body of POST /responses.
type responsesRequest struct {
	Model           string               `json:"model"`
	Input           []responsesInputItem `json:"input"`
	Temperature     *float64             `json:"temperature,omitempty"`
	TopP            *float64             `json:"top_p,omitempty"`
	MaxOutputTokens *int                 `json:"max_output_tokens,omitempty"`
	Text            *responsesText       `json:"text,omitempty"`

	Conversation         string                        `json:"conversation,omitempty"`
	MaxToolCalls         *int                          `json:"max_tool_calls,omitempty"`
	Metadata             map[string]string             `json:"metadata,omitempty"`
	ParallelToolCalls    *bool                         `json:"parallel_tool_calls,omitempty"`
	PreviousResponseID   string                        `json:"previous_response_id,omitempty"`
	Store                *bool                         `json:"store,omitempty"`
	User                 string                        `json:"user,omitempty"`
	Instructions         string                        `json:"instructions,omitempty"`
	ServiceTier          ResponsesServiceTier          `json:"service_tier,omitempty"`
	Include              []ResponsesInclude            `json:"include,omitempty"`
	PromptCacheKey       string                        `json:"prompt_cache_key,omitempty"`
	PromptCacheOptions   *ResponsesPromptCacheOptions  `json:"prompt_cache_options,omitempty"`
	PromptCacheRetention ResponsesPromptCacheRetention `json:"prompt_cache_retention,omitempty"`
	SafetyIdentifier     string                        `json:"safety_identifier,omitempty"`
	TopLogprobs          *int                          `json:"top_logprobs,omitempty"`
	Truncation           ResponsesTruncation           `json:"truncation,omitempty"`
	ContextManagement    []responsesContextManagement  `json:"context_management,omitempty"`
	Reasoning            *responsesReasoning           `json:"reasoning,omitempty"`

	Tools      []responsesTool      `json:"tools,omitempty"`
	ToolChoice *responsesToolChoice `json:"tool_choice,omitempty"`

	Stream bool `json:"stream,omitempty"`
}

// responsesText configures text output.
type responsesText struct {
	Format    *responsesTextFormat   `json:"format,omitempty"`
	Verbosity ResponsesTextVerbosity `json:"verbosity,omitempty"`
}

// responsesTextFormat is "json_schema" with a schema or "json_object".
type responsesTextFormat struct {
	Type        string          `json:"type"`
	Strict      *bool           `json:"strict,omitempty"`
	Name        string          `json:"name,omitempty"`
	Description string          `json:"description,omitempty"`
	Schema      json.RawMessage `json:"schema,omitempty"`
}

// responsesReasoning configures reasoning models.
type responsesReasoning struct {
	Effort  string                    `json:"effort,omitempty"`
	Summary *string                   `json:"summary,omitempty"`
	Mode    ResponsesReasoningMode    `json:"mode,omitempty"`
	Context ResponsesReasoningContext `json:"context,omitempty"`
}

// responsesContextManagement enables server-side compaction.
type responsesContextManagement struct {
	Type             string `json:"type"`
	CompactThreshold int    `json:"compact_threshold"`
}

// responsesItemType is the type of an input or output item.
type responsesItemType string

// Item types.
const (
	responsesMessage              responsesItemType = "message"
	responsesFunctionCall         responsesItemType = "function_call"
	responsesFunctionCallOutput   responsesItemType = "function_call_output"
	responsesAdditionalTools      responsesItemType = "additional_tools"
	responsesProgram              responsesItemType = "program"
	responsesProgramOutput        responsesItemType = "program_output"
	responsesCustomToolCall       responsesItemType = "custom_tool_call"
	responsesCustomToolCallOutput responsesItemType = "custom_tool_call_output"
	responsesMCPCall              responsesItemType = "mcp_call"
	responsesMCPListTools         responsesItemType = "mcp_list_tools"
	responsesMCPApprovalRequest   responsesItemType = "mcp_approval_request"
	responsesMCPApprovalResponse  responsesItemType = "mcp_approval_response"
	responsesComputerCall         responsesItemType = "computer_call"
	responsesComputerCallOutput   responsesItemType = "computer_call_output"
	responsesLocalShellCall       responsesItemType = "local_shell_call"
	responsesLocalShellCallOutput responsesItemType = "local_shell_call_output"
	responsesShellCall            responsesItemType = "shell_call"
	responsesShellCallOutput      responsesItemType = "shell_call_output"
	responsesApplyPatchCall       responsesItemType = "apply_patch_call"
	responsesApplyPatchCallOutput responsesItemType = "apply_patch_call_output"
	responsesToolSearchCall       responsesItemType = "tool_search_call"
	responsesToolSearchOutput     responsesItemType = "tool_search_output"
	responsesWebSearchCall        responsesItemType = "web_search_call"
	responsesFileSearchCall       responsesItemType = "file_search_call"
	responsesCodeInterpreterCall  responsesItemType = "code_interpreter_call"
	responsesImageGenerationCall  responsesItemType = "image_generation_call"
	responsesReasoningItem        responsesItemType = "reasoning"
	responsesItemReferenceItem    responsesItemType = "item_reference"
	responsesCompaction           responsesItemType = "compaction"
	responsesConfigurationUpdate  responsesItemType = "configuration_update"
	responsesCompactionTrigger    responsesItemType = "compaction_trigger"
)

// responsesInputItem is one entry of the request input.
type responsesInputItem interface {
	responsesInputType() responsesItemType
}

// responsesSystemMessage is a system or developer message.
type responsesSystemMessage struct {
	// Type is "message" for endpoints that require explicit message items.
	Type    responsesItemType `json:"type,omitempty"`
	Role    string            `json:"role"`
	Content responsesContent  `json:"content"`
}

// responsesUserMessage is a user message.
type responsesUserMessage struct {
	Type    responsesItemType      `json:"type,omitempty"`
	Role    string                 `json:"role"`
	Content []responsesContentPart `json:"content"`
}

// responsesAssistantMessage is assistant text sent with its content.
type responsesAssistantMessage struct {
	Type    responsesItemType `json:"type,omitempty"`
	Role    string            `json:"role"`
	Content string            `json:"content"`
	Phase   string            `json:"phase,omitempty"`
}

// responsesFunctionCallInput is a function call from an earlier turn.
type responsesFunctionCallInput struct {
	Type      responsesItemType `json:"type"`
	CallID    string            `json:"call_id"`
	Name      string            `json:"name"`
	Arguments string            `json:"arguments"`
	Async     *bool             `json:"async,omitempty"`
	Namespace string            `json:"namespace,omitempty"`
	Caller    *responsesCaller  `json:"caller,omitempty"`
}

// responsesFunctionCallOutputInput is the result of a function call.
type responsesFunctionCallOutputInput struct {
	Type   responsesItemType `json:"type"`
	CallID string            `json:"call_id"`
	Output responsesContent  `json:"output"`
	Caller *responsesCaller  `json:"caller,omitempty"`
}

// responsesAdditionalToolsInput loads function definitions in history.
type responsesAdditionalToolsInput struct {
	Type  responsesItemType         `json:"type"`
	Role  string                    `json:"role"`
	Tools []ResponsesAdditionalTool `json:"tools"`
}

// responsesProgramInput is a hosted program from an earlier turn.
type responsesProgramInput struct {
	Type        responsesItemType `json:"type"`
	ID          string            `json:"id"`
	CallID      string            `json:"call_id"`
	Code        string            `json:"code"`
	Fingerprint string            `json:"fingerprint"`
}

// responsesProgramOutputInput is the result of a hosted program.
type responsesProgramOutputInput struct {
	Type   responsesItemType `json:"type"`
	ID     string            `json:"id"`
	CallID string            `json:"call_id"`
	Result string            `json:"result"`
	Status string            `json:"status"`
}

// responsesCustomToolCallInput is a custom tool call from an earlier turn.
type responsesCustomToolCallInput struct {
	Type   responsesItemType `json:"type"`
	ID     string            `json:"id,omitempty"`
	CallID string            `json:"call_id"`
	Name   string            `json:"name"`
	Input  string            `json:"input"`
	Async  *bool             `json:"async,omitempty"`
}

// responsesCustomToolCallOutputInput is the result of a custom tool call.
type responsesCustomToolCallOutputInput struct {
	Type   responsesItemType `json:"type"`
	CallID string            `json:"call_id"`
	Output responsesContent  `json:"output"`
}

// responsesMCPApprovalResponseInput answers an MCP approval request.
type responsesMCPApprovalResponseInput struct {
	Type              responsesItemType `json:"type"`
	ApprovalRequestID string            `json:"approval_request_id"`
	Approve           bool              `json:"approve"`
}

// responsesComputerCallInput is a computer call from an earlier turn.
type responsesComputerCallInput struct {
	Type                responsesItemType         `json:"type"`
	ID                  string                    `json:"id,omitempty"`
	CallID              string                    `json:"call_id"`
	Status              string                    `json:"status"`
	Actions             []responsesComputerAction `json:"actions"`
	PendingSafetyChecks []ComputerSafetyCheck     `json:"pending_safety_checks"`
}

// responsesComputerCallOutputInput is the screenshot taken after a computer
// call.
type responsesComputerCallOutputInput struct {
	Type                     responsesItemType           `json:"type"`
	CallID                   string                      `json:"call_id"`
	Output                   responsesComputerScreenshot `json:"output"`
	AcknowledgedSafetyChecks []ComputerSafetyCheck       `json:"acknowledged_safety_checks,omitempty"`
}

// responsesComputerScreenshot is a screenshot in the API's format.
type responsesComputerScreenshot struct {
	Type     string `json:"type"`
	ImageURL string `json:"image_url,omitempty"`
	FileID   string `json:"file_id,omitempty"`
	Detail   string `json:"detail,omitempty"`
}

// responsesLocalShellCallInput is a local shell call from an earlier turn.
type responsesLocalShellCallInput struct {
	Type   responsesItemType         `json:"type"`
	ID     string                    `json:"id,omitempty"`
	CallID string                    `json:"call_id"`
	Action responsesLocalShellAction `json:"action"`
}

// responsesLocalShellCallOutputInput is the output of a local shell call.
type responsesLocalShellCallOutputInput struct {
	Type   responsesItemType `json:"type"`
	CallID string            `json:"call_id"`
	Output string            `json:"output"`
}

// responsesShellCallInput is a shell call from an earlier turn.
type responsesShellCallInput struct {
	Type   responsesItemType    `json:"type"`
	ID     string               `json:"id,omitempty"`
	CallID string               `json:"call_id"`
	Status string               `json:"status"`
	Action responsesShellAction `json:"action"`
}

// responsesShellCallOutputInput is the output of a shell call.
type responsesShellCallOutputInput struct {
	Type            responsesItemType           `json:"type"`
	CallID          string                      `json:"call_id"`
	MaxOutputLength *int                        `json:"max_output_length,omitempty"`
	Output          []responsesShellOutputEntry `json:"output"`
}

// responsesApplyPatchCallInput is an apply patch call from an earlier turn.
type responsesApplyPatchCallInput struct {
	Type      responsesItemType   `json:"type"`
	ID        string              `json:"id,omitempty"`
	CallID    string              `json:"call_id"`
	Status    string              `json:"status"`
	Operation ApplyPatchOperation `json:"operation"`
}

// responsesApplyPatchCallOutputInput reports the outcome of a patch.
type responsesApplyPatchCallOutputInput struct {
	Type   responsesItemType `json:"type"`
	CallID string            `json:"call_id"`
	Status string            `json:"status"`
	Output string            `json:"output,omitempty"`
}

// responsesToolSearchCallInput is a tool search call from an earlier turn.
type responsesToolSearchCallInput struct {
	Type      responsesItemType `json:"type"`
	ID        string            `json:"id"`
	Execution string            `json:"execution"`
	CallID    *string           `json:"call_id"`
	Status    string            `json:"status"`
	Arguments json.RawMessage   `json:"arguments,omitempty"`
}

// responsesToolSearchOutputInput lists the tools a tool search loaded.
type responsesToolSearchOutputInput struct {
	Type      responsesItemType `json:"type"`
	ID        string            `json:"id,omitempty"`
	Execution string            `json:"execution"`
	CallID    *string           `json:"call_id"`
	Status    string            `json:"status"`
	Tools     []json.RawMessage `json:"tools"`
}

// responsesWebSearchCallInput is a web search call sent with its content.
type responsesWebSearchCallInput struct {
	Type   responsesItemType        `json:"type"`
	ID     string                   `json:"id"`
	Status string                   `json:"status"`
	Action responsesWebSearchAction `json:"action"`
}

// responsesReasoningInput is a reasoning item from an earlier turn.
type responsesReasoningInput struct {
	Type             responsesItemType      `json:"type"`
	ID               string                 `json:"id,omitempty"`
	EncryptedContent *string                `json:"encrypted_content,omitempty"`
	Summary          []responsesSummaryText `json:"summary"`
}

// responsesItemReference refers to a stored item by ID.
type responsesItemReference struct {
	Type responsesItemType `json:"type"`
	ID   string            `json:"id"`
}

// responsesCompactionInput is a compaction item from an earlier turn.
type responsesCompactionInput struct {
	Type             responsesItemType `json:"type"`
	ID               string            `json:"id"`
	EncryptedContent string            `json:"encrypted_content"`
}

// responsesConfigurationUpdateInput changes the reasoning effort from its
// position onward.
type responsesConfigurationUpdateInput struct {
	Type      responsesItemType               `json:"type"`
	Reasoning responsesConfigurationReasoning `json:"reasoning"`
}

// responsesConfigurationReasoning is the reasoning part of a configuration
// update.
type responsesConfigurationReasoning struct {
	Effort ResponsesReasoningEffort `json:"effort"`
}

// responsesCompactionTriggerInput requests compaction.
type responsesCompactionTriggerInput struct {
	Type responsesItemType `json:"type"`
}

func (responsesSystemMessage) responsesInputType() responsesItemType {
	return responsesMessage
}

func (responsesUserMessage) responsesInputType() responsesItemType {
	return responsesMessage
}

func (responsesAssistantMessage) responsesInputType() responsesItemType {
	return responsesMessage
}

func (i responsesFunctionCallInput) responsesInputType() responsesItemType {
	return i.Type
}

func (i responsesFunctionCallOutputInput) responsesInputType() responsesItemType {
	return i.Type
}

func (i responsesAdditionalToolsInput) responsesInputType() responsesItemType {
	return i.Type
}

func (i responsesProgramInput) responsesInputType() responsesItemType {
	return i.Type
}

func (i responsesProgramOutputInput) responsesInputType() responsesItemType {
	return i.Type
}

func (i responsesCustomToolCallInput) responsesInputType() responsesItemType {
	return i.Type
}

func (i responsesCustomToolCallOutputInput) responsesInputType() responsesItemType {
	return i.Type
}

func (i responsesMCPApprovalResponseInput) responsesInputType() responsesItemType {
	return i.Type
}

func (i responsesComputerCallInput) responsesInputType() responsesItemType {
	return i.Type
}

func (i responsesComputerCallOutputInput) responsesInputType() responsesItemType {
	return i.Type
}

func (i responsesLocalShellCallInput) responsesInputType() responsesItemType {
	return i.Type
}

func (i responsesLocalShellCallOutputInput) responsesInputType() responsesItemType {
	return i.Type
}

func (i responsesShellCallInput) responsesInputType() responsesItemType {
	return i.Type
}

func (i responsesShellCallOutputInput) responsesInputType() responsesItemType {
	return i.Type
}

func (i responsesApplyPatchCallInput) responsesInputType() responsesItemType {
	return i.Type
}

func (i responsesApplyPatchCallOutputInput) responsesInputType() responsesItemType {
	return i.Type
}

func (i responsesToolSearchCallInput) responsesInputType() responsesItemType {
	return i.Type
}

func (i responsesToolSearchOutputInput) responsesInputType() responsesItemType {
	return i.Type
}

func (i responsesWebSearchCallInput) responsesInputType() responsesItemType {
	return i.Type
}

func (i *responsesReasoningInput) responsesInputType() responsesItemType {
	return i.Type
}

func (i responsesItemReference) responsesInputType() responsesItemType {
	return i.Type
}

func (i responsesCompactionInput) responsesInputType() responsesItemType {
	return i.Type
}

func (i responsesConfigurationUpdateInput) responsesInputType() responsesItemType {
	return i.Type
}

func (i responsesCompactionTriggerInput) responsesInputType() responsesItemType {
	return i.Type
}

// responsesContent is message or tool output content: plain text, or content
// parts when Parts is not nil.
type responsesContent struct {
	Text  string
	Parts []responsesContentPart
}

// MarshalJSON writes the parts when present, otherwise the text.
func (c responsesContent) MarshalJSON() ([]byte, error) {
	if c.Parts != nil {
		return json.Marshal(c.Parts)
	}

	return json.Marshal(c.Text)
}

// responsesContentPart is an input_text, input_image or input_file part.
type responsesContentPart interface {
	responsesContentType() string
}

// responsesInputText is a text part.
type responsesInputText struct {
	Type                  string                          `json:"type"`
	Text                  string                          `json:"text"`
	PromptCacheBreakpoint *ResponsesPromptCacheBreakpoint `json:"prompt_cache_breakpoint,omitempty"`
}

// responsesInputImage is an image given by URL, data URL or file ID.
type responsesInputImage struct {
	Type                  string                          `json:"type"`
	ImageURL              string                          `json:"image_url,omitempty"`
	FileID                string                          `json:"file_id,omitempty"`
	Detail                string                          `json:"detail,omitempty"`
	PromptCacheBreakpoint *ResponsesPromptCacheBreakpoint `json:"prompt_cache_breakpoint,omitempty"`
}

// responsesInputFile is a file given by URL, inline data or file ID.
type responsesInputFile struct {
	Type                  string                          `json:"type"`
	FileURL               string                          `json:"file_url,omitempty"`
	FileID                string                          `json:"file_id,omitempty"`
	Filename              string                          `json:"filename,omitempty"`
	FileData              string                          `json:"file_data,omitempty"`
	PromptCacheBreakpoint *ResponsesPromptCacheBreakpoint `json:"prompt_cache_breakpoint,omitempty"`
}

func (p responsesInputText) responsesContentType() string {
	return p.Type
}

func (p responsesInputImage) responsesContentType() string {
	return p.Type
}

func (p responsesInputFile) responsesContentType() string {
	return p.Type
}

// responsesCaller identifies who called a tool, in the API's format.
type responsesCaller struct {
	Type     ResponsesToolCallerType `json:"type"`
	CallerID string                  `json:"caller_id,omitempty"`
}

// responsesComputerAction is a computer action in the API's format. Its
// fields match ComputerAction, so the two convert directly.
type responsesComputerAction struct {
	Type    ComputerActionType `json:"type"`
	Button  string             `json:"button,omitempty"`
	X       *float64           `json:"x,omitempty"`
	Y       *float64           `json:"y,omitempty"`
	Path    []ComputerPoint    `json:"path,omitempty"`
	Keys    []string           `json:"keys,omitempty"`
	ScrollX *float64           `json:"scroll_x,omitempty"`
	ScrollY *float64           `json:"scroll_y,omitempty"`
	Text    string             `json:"text,omitempty"`
}

// responsesLocalShellAction is a local shell action in the API's format.
// Its fields match LocalShellAction, so the two convert directly.
type responsesLocalShellAction struct {
	Type             string            `json:"type"`
	Command          []string          `json:"command"`
	TimeoutMs        *int              `json:"timeout_ms,omitempty"`
	User             string            `json:"user,omitempty"`
	WorkingDirectory string            `json:"working_directory,omitempty"`
	Env              map[string]string `json:"env,omitempty"`
}

// responsesShellAction is a shell action in the API's format. Its fields
// match ShellAction, so the two convert directly.
type responsesShellAction struct {
	Commands        []string `json:"commands"`
	TimeoutMs       *int     `json:"timeout_ms,omitempty"`
	MaxOutputLength *int     `json:"max_output_length,omitempty"`
}

// responsesShellOutputEntry is the output of one shell command.
type responsesShellOutputEntry struct {
	Stdout  string                `json:"stdout"`
	Stderr  string                `json:"stderr"`
	Outcome responsesShellOutcome `json:"outcome"`
}

// responsesShellOutcome is "timeout" or "exit" with an exit code.
type responsesShellOutcome struct {
	Type     string `json:"type"`
	ExitCode *int   `json:"exit_code,omitempty"`
}

// responsesWebSearchAction is a web search action in the API's format:
// "search", "open_page" or "find_in_page".
type responsesWebSearchAction struct {
	Type    string            `json:"type"`
	Query   *string           `json:"query,omitempty"`
	Queries []string          `json:"queries,omitempty"`
	Sources []WebSearchSource `json:"sources,omitempty"`
	URL     *string           `json:"url,omitempty"`
	Pattern *string           `json:"pattern,omitempty"`
}

// responsesSummaryText is one reasoning summary part.
type responsesSummaryText struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// responsesResponse is the body of a non-streaming response.
type responsesResponse struct {
	ID                string                      `json:"id"`
	CreatedAt         *float64                    `json:"created_at"`
	Error             *responsesResponseError     `json:"error"`
	Model             string                      `json:"model"`
	Output            responsesOutput             `json:"output"`
	ServiceTier       string                      `json:"service_tier"`
	Reasoning         *responsesReasoningState    `json:"reasoning"`
	IncompleteDetails *responsesIncompleteDetails `json:"incomplete_details"`
	Usage             *responsesUsage             `json:"usage"`
}

// responsesResponseError is an error reported in a response body.
type responsesResponseError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   string `json:"param"`
	Code    string `json:"code"`
}

// responsesReasoningState is the reasoning state reported with a response.
type responsesReasoningState struct {
	Context string `json:"context"`
}

// responsesIncompleteDetails explains why a response is incomplete.
type responsesIncompleteDetails struct {
	Reason string `json:"reason"`
}

// responsesUsage is the token usage of a response. Raw keeps the original
// object, including fields not modeled here.
type responsesUsage struct {
	InputTokens         int                           `json:"input_tokens"`
	InputTokensDetails  *responsesInputTokensDetails  `json:"input_tokens_details"`
	OutputTokens        int                           `json:"output_tokens"`
	OutputTokensDetails *responsesOutputTokensDetails `json:"output_tokens_details"`

	Raw json.RawMessage `json:"-"`
}

// responsesInputTokensDetails breaks down input tokens.
type responsesInputTokensDetails struct {
	CachedTokens     *int `json:"cached_tokens"`
	CacheWriteTokens *int `json:"cache_write_tokens"`
}

// responsesOutputTokensDetails breaks down output tokens.
type responsesOutputTokensDetails struct {
	ReasoningTokens *int `json:"reasoning_tokens"`
}

// UnmarshalJSON decodes the usage and keeps the raw object.
func (u *responsesUsage) UnmarshalJSON(data []byte) error {
	type plain responsesUsage

	var decoded plain
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}

	*u = responsesUsage(decoded)
	u.Raw = append(json.RawMessage(nil), data...)

	return nil
}
