package anthropic

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// messagesResponse is a non-streaming Messages API response.
type messagesResponse struct {
	Type                 string                 `json:"type"`
	ID                   *string                `json:"id"`
	Model                *string                `json:"model"`
	Content              []responseBlock        `json:"-"`
	RawContent           []json.RawMessage      `json:"content"`
	StopReason           *string                `json:"stop_reason"`
	StopSequence         *string                `json:"stop_sequence"`
	StopDetails          *wireStopDetails       `json:"stop_details"`
	InputTransformations []InputTransformation  `json:"input_transformations"`
	SafeguardResults     []SafeguardResult      `json:"safeguard_results"`
	Usage                json.RawMessage        `json:"usage"`
	Container            *wireContainer         `json:"container"`
	ContextManagement    *wireContextManagement `json:"context_management"`
}

// decodeMessagesResponse decodes a response body, its content blocks and
// its usage.
func decodeMessagesResponse(body []byte) (*messagesResponse, usage, error) {
	var response messagesResponse
	if err := apicall.DecodeJSON(body, &response); err != nil {
		return nil, usage{}, err
	}

	if response.Type != "message" {
		return nil, usage{}, fmt.Errorf("%w: response type must be \"message\", got %q", model.ErrInvalidResponse, response.Type)
	}

	for index, raw := range response.RawContent {
		block, err := decodeResponseBlock(raw)
		if err != nil {
			return nil, usage{}, fmt.Errorf("%w: content block %d: %w", model.ErrInvalidResponse, index, err)
		}

		response.Content = append(response.Content, block)
	}

	var decodedUsage usage
	if err := json.Unmarshal(response.Usage, &decodedUsage); err != nil {
		return nil, usage{}, fmt.Errorf("%w: usage: %w", model.ErrInvalidResponse, err)
	}

	return &response, decodedUsage, nil
}

// usage is Anthropic's usage object.
type usage struct {
	InputTokens              int                  `json:"input_tokens"`
	OutputTokens             int                  `json:"output_tokens"`
	OutputTokensDetails      *outputTokensDetails `json:"output_tokens_details"`
	CacheCreationInputTokens *int                 `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     *int                 `json:"cache_read_input_tokens"`
	Iterations               []wireUsageIteration `json:"iterations"`
}

type outputTokensDetails struct {
	ThinkingTokens *int `json:"thinking_tokens"`
}

type wireUsageIteration struct {
	Type                     UsageIterationType `json:"type"`
	Model                    *string            `json:"model"`
	InputTokens              int                `json:"input_tokens"`
	OutputTokens             int                `json:"output_tokens"`
	CacheCreationInputTokens *int               `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     *int               `json:"cache_read_input_tokens"`
}

type wireStopDetails struct {
	Type             string  `json:"type"`
	Category         *string `json:"category"`
	Explanation      *string `json:"explanation"`
	RecommendedModel *string `json:"recommended_model"`
}

type wireContainer struct {
	ExpiresAt string               `json:"expires_at"`
	ID        string               `json:"id"`
	Skills    []wireContainerSkill `json:"skills"`
}

type wireContainerSkill struct {
	Type    SkillType `json:"type"`
	SkillID string    `json:"skill_id"`
	Version string    `json:"version"`
}

type wireContextManagement struct {
	AppliedEdits []wireAppliedEdit `json:"applied_edits"`
}

type wireAppliedEdit struct {
	Type                 ContextEditType `json:"type"`
	ClearedToolUses      *int            `json:"cleared_tool_uses"`
	ClearedThinkingTurns *int            `json:"cleared_thinking_turns"`
	ClearedInputTokens   *int            `json:"cleared_input_tokens"`
}

// responseBlock is a content block of a response or of a content_block_start
// stream event.
type responseBlock interface {
	isResponseBlock()
}

type responseText struct {
	Text      string     `json:"text"`
	Citations []Citation `json:"citations"`
}

type responseThinking struct {
	Thinking  string `json:"thinking"`
	Signature string `json:"signature"`
}

type responseRedactedThinking struct {
	Data string `json:"data"`
}

type responseCompaction struct {
	Content   *string `json:"content"`
	Signature *string `json:"signature"`
}

type responseToolUse struct {
	ID          string          `json:"id"`
	Name        string          `json:"name"`
	Input       json.RawMessage `json:"input"`
	Caller      *wireCaller     `json:"caller"`
	ToolsetName *string         `json:"toolset_name"`
}

type responseServerToolUse struct {
	ID     string          `json:"id"`
	Name   string          `json:"name"`
	Input  json.RawMessage `json:"input"`
	Caller *wireCaller     `json:"caller"`
}

type responseMCPToolUse struct {
	ID         string          `json:"id"`
	Name       string          `json:"name"`
	Input      json.RawMessage `json:"input"`
	ServerName string          `json:"server_name"`
}

type responseMCPToolResult struct {
	ToolUseID string          `json:"tool_use_id"`
	IsError   bool            `json:"is_error"`
	Content   json.RawMessage `json:"content"`
}

type responseWebFetchToolResult struct {
	ToolUseID string              `json:"tool_use_id"`
	Caller    *wireCaller         `json:"caller"`
	Content   webFetchResultUnion `json:"content"`
}

// webFetchResultUnion is a web_fetch_result or a web_fetch_tool_result_error.
type webFetchResultUnion struct {
	Type        string              `json:"type"`
	URL         string              `json:"url"`
	RetrievedAt *string             `json:"retrieved_at"`
	Content     wireWebFetchContent `json:"content"`
	ErrorCode   string              `json:"error_code"`
}

type responseWebSearchToolResult struct {
	ToolUseID string               `json:"tool_use_id"`
	Caller    *wireCaller          `json:"caller"`
	Content   webSearchResultUnion `json:"content"`
}

// webSearchResultUnion is a list of results or an error object.
type webSearchResultUnion struct {
	Results []wireWebSearchResult
	Error   *wireError
}

// UnmarshalJSON decodes the result list or the error object.
func (u *webSearchResultUnion) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)

	if len(trimmed) > 0 && trimmed[0] == '[' {
		var results []wireWebSearchResult
		if err := json.Unmarshal(trimmed, &results); err != nil {
			return err
		}

		*u = webSearchResultUnion{
			Results: results,
		}

		return nil
	}

	var failure wireError
	if err := json.Unmarshal(trimmed, &failure); err != nil {
		return err
	}

	*u = webSearchResultUnion{
		Error: &failure,
	}

	return nil
}

type responseCodeExecutionToolResult struct {
	ToolUseID string                   `json:"tool_use_id"`
	Content   codeExecutionResultUnion `json:"content"`
}

// codeExecutionResultUnion is a code_execution_result, an
// encrypted_code_execution_result or a code_execution_tool_result_error.
type codeExecutionResultUnion struct {
	Type            string              `json:"type"`
	Stdout          string              `json:"stdout"`
	EncryptedStdout string              `json:"encrypted_stdout"`
	Stderr          string              `json:"stderr"`
	ReturnCode      int                 `json:"return_code"`
	Content         []CodeExecutionFile `json:"content"`
	ErrorCode       string              `json:"error_code"`
}

// responseSubtoolResult is a bash_code_execution_tool_result or a
// text_editor_code_execution_tool_result.
type responseSubtoolResult struct {
	ToolUseID string                  `json:"tool_use_id"`
	Content   serverToolResultContent `json:"-"`
	Raw       json.RawMessage         `json:"content"`
}

type responseToolSearchToolResult struct {
	ToolUseID string                `json:"tool_use_id"`
	Content   toolSearchResultUnion `json:"content"`
}

// toolSearchResultUnion is a tool_search_tool_search_result or a
// tool_search_tool_result_error.
type toolSearchResultUnion struct {
	Type           string          `json:"type"`
	ToolReferences []toolReference `json:"tool_references"`
	ErrorCode      string          `json:"error_code"`
}

type responseContainerUpload struct {
	FileID string `json:"file_id"`
}

type responseAdvisorToolResult struct {
	ToolUseID string            `json:"tool_use_id"`
	Content   wireAdvisorResult `json:"content"`
}

type responseFallback struct {
	From ModelRef `json:"from"`
	To   ModelRef `json:"to"`
}

type blockHeader struct {
	Type string `json:"type"`
}

// decodeResponseBlock decodes a content block by its type.
func decodeResponseBlock(raw json.RawMessage) (responseBlock, error) {
	var header blockHeader
	if err := json.Unmarshal(raw, &header); err != nil {
		return nil, err
	}

	switch header.Type {
	case blockText:
		return decodeBlock[responseText](raw)
	case blockThinking:
		return decodeBlock[responseThinking](raw)
	case blockRedactedThinking:
		return decodeBlock[responseRedactedThinking](raw)
	case blockCompaction:
		return decodeBlock[responseCompaction](raw)
	case blockToolUse:
		return decodeBlock[responseToolUse](raw)
	case blockServerToolUse:
		return decodeBlock[responseServerToolUse](raw)
	case blockMCPToolUse:
		return decodeBlock[responseMCPToolUse](raw)
	case blockMCPToolResult:
		return decodeBlock[responseMCPToolResult](raw)
	case blockWebFetchToolResult:
		return decodeWebFetchToolResult(raw)
	case blockWebSearchToolResult:
		return decodeBlock[responseWebSearchToolResult](raw)
	case blockCodeExecutionToolResult:
		return decodeCodeExecutionToolResult(raw)
	case blockBashCodeExecutionToolResult:
		return decodeSubtoolResult(raw, bashResultTypes)
	case blockTextEditorCodeExecutionToolResult:
		return decodeSubtoolResult(raw, textEditorResultTypes)
	case blockToolSearchToolResult:
		return decodeToolSearchToolResult(raw)
	case blockContainerUpload:
		return decodeBlock[responseContainerUpload](raw)
	case blockAdvisorToolResult:
		return decodeAdvisorToolResult(raw)
	case blockFallback:
		return decodeBlock[responseFallback](raw)
	default:
		return nil, fmt.Errorf("unsupported content block type %q", header.Type)
	}
}

func decodeBlock[Block any, Pointer interface {
	*Block
	responseBlock
}](raw json.RawMessage) (responseBlock, error) {
	var block Block
	if err := json.Unmarshal(raw, &block); err != nil {
		return nil, err
	}

	return Pointer(&block), nil
}

func decodeWebFetchToolResult(raw json.RawMessage) (responseBlock, error) {
	var block responseWebFetchToolResult
	if err := json.Unmarshal(raw, &block); err != nil {
		return nil, err
	}

	err := checkRequiredEnum("web fetch result type", block.Content.Type, resultWebFetch, resultWebFetchError)
	if err != nil {
		return nil, err
	}

	return &block, nil
}

func decodeCodeExecutionToolResult(raw json.RawMessage) (responseBlock, error) {
	var block responseCodeExecutionToolResult
	if err := json.Unmarshal(raw, &block); err != nil {
		return nil, err
	}

	err := checkRequiredEnum(
		"code execution result type",
		block.Content.Type,
		resultCodeExecution,
		resultEncryptedCodeExecution,
		resultCodeExecutionError,
	)
	if err != nil {
		return nil, err
	}

	if block.Content.Content == nil {
		block.Content.Content = []CodeExecutionFile{}
	}

	return &block, nil
}

func decodeToolSearchToolResult(raw json.RawMessage) (responseBlock, error) {
	var block responseToolSearchToolResult
	if err := json.Unmarshal(raw, &block); err != nil {
		return nil, err
	}

	err := checkRequiredEnum("tool search result type", block.Content.Type, resultToolSearch, resultToolSearchError)
	if err != nil {
		return nil, err
	}

	return &block, nil
}

func decodeAdvisorToolResult(raw json.RawMessage) (responseBlock, error) {
	var block responseAdvisorToolResult
	if err := json.Unmarshal(raw, &block); err != nil {
		return nil, err
	}

	err := checkRequiredEnum("advisor result type", block.Content.Type, resultAdvisor, resultAdvisorRedacted, resultAdvisorError)
	if err != nil {
		return nil, err
	}

	return &block, nil
}

// subtoolResultTypes lists the result variants a code execution subtool
// may return.
type subtoolResultTypes []string

var (
	bashResultTypes = subtoolResultTypes{
		resultBashCodeExecution,
		resultBashCodeExecutionError,
	}

	textEditorResultTypes = subtoolResultTypes{
		resultTextEditorError,
		resultTextEditorView,
		resultTextEditorCreate,
		resultTextEditorStrReplace,
	}
)

func decodeSubtoolResult(raw json.RawMessage, allowed subtoolResultTypes) (responseBlock, error) {
	var block responseSubtoolResult
	if err := json.Unmarshal(raw, &block); err != nil {
		return nil, err
	}

	content, err := decodeCodeExecutionOutput(block.Raw, allowed...)
	if err != nil {
		return nil, err
	}

	block.Content = content

	return &block, nil
}

// decodeCodeExecutionOutput decodes a code execution result variant whose
// type is one of allowed.
func decodeCodeExecutionOutput(raw json.RawMessage, allowed ...string) (serverToolResultContent, error) {
	var header blockHeader
	if err := json.Unmarshal(raw, &header); err != nil {
		return nil, err
	}

	if err := checkRequiredEnum("code execution result type", header.Type, allowed...); err != nil {
		return nil, err
	}

	switch header.Type {
	case resultCodeExecution:
		return decodeOutput[CodeExecutionResult](raw)
	case resultEncryptedCodeExecution:
		return decodeOutput[EncryptedCodeExecutionResult](raw)
	case resultBashCodeExecution:
		return decodeOutput[BashCodeExecutionResult](raw)
	case resultBashCodeExecutionError, resultTextEditorError:
		return decodeOutput[CodeExecutionError](raw)
	case resultTextEditorView:
		return decodeOutput[TextEditorViewResult](raw)
	case resultTextEditorCreate:
		return decodeOutput[TextEditorCreateResult](raw)
	default:
		return decodeOutput[TextEditorStrReplaceResult](raw)
	}
}

func decodeOutput[Output serverToolResultContent](raw json.RawMessage) (serverToolResultContent, error) {
	var output Output
	if err := json.Unmarshal(raw, &output); err != nil {
		return nil, err
	}

	return normalizeCodeExecutionFiles(output), nil
}

// normalizeCodeExecutionFiles replaces absent file lists with empty ones, as
// the upstream schema defaults them.
func normalizeCodeExecutionFiles(output serverToolResultContent) serverToolResultContent {
	switch result := output.(type) {
	case CodeExecutionResult:
		if result.Content == nil {
			result.Content = []CodeExecutionFile{}
		}

		return result

	case EncryptedCodeExecutionResult:
		if result.Content == nil {
			result.Content = []CodeExecutionFile{}
		}

		return result

	default:
		return output
	}
}

func (*responseText) isResponseBlock()                    {}
func (*responseThinking) isResponseBlock()                {}
func (*responseRedactedThinking) isResponseBlock()        {}
func (*responseCompaction) isResponseBlock()              {}
func (*responseToolUse) isResponseBlock()                 {}
func (*responseServerToolUse) isResponseBlock()           {}
func (*responseMCPToolUse) isResponseBlock()              {}
func (*responseMCPToolResult) isResponseBlock()           {}
func (*responseWebFetchToolResult) isResponseBlock()      {}
func (*responseWebSearchToolResult) isResponseBlock()     {}
func (*responseCodeExecutionToolResult) isResponseBlock() {}
func (*responseSubtoolResult) isResponseBlock()           {}
func (*responseToolSearchToolResult) isResponseBlock()    {}
func (*responseContainerUpload) isResponseBlock()         {}
func (*responseAdvisorToolResult) isResponseBlock()       {}
func (*responseFallback) isResponseBlock()                {}
