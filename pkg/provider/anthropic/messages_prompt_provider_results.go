package anthropic

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// providerResult is a provider-executed tool result being replayed.
type providerResult struct {
	part         model.ToolResultPart
	caller       *wireCaller
	cacheControl *CacheControl
}

// block returns a result block of blockType with content.
func (r providerResult) block(blockType string, content serverToolResultContent) requestBlock {
	return &serverToolResultBlock{
		Type:         blockType,
		ToolUseID:    r.part.ToolCallID,
		Content:      content,
		CacheControl: r.cacheControl,
	}
}

// blockWithCaller returns a result block that also carries the caller.
func (r providerResult) blockWithCaller(blockType string, content serverToolResultContent) requestBlock {
	return &serverToolResultBlock{
		Type:         blockType,
		ToolUseID:    r.part.ToolCallID,
		Content:      content,
		Caller:       r.caller,
		CacheControl: r.cacheControl,
	}
}

// assistantToolResult replays the result of a provider-executed tool from
// an earlier assistant turn. Outputs the API cannot represent are dropped
// with a warning.
func (c *promptConverter) assistantToolResult(
	part model.ToolResultPart,
	cacheControl *CacheControl,
	mcpToolUseIDs map[string]bool,
) (requestBlock, error) {
	metadata, _, err := model.ProviderValue[ToolCallMetadata](part.ProviderOptions, Namespace)
	if err != nil {
		return nil, err
	}

	result := providerResult{
		part:         part,
		caller:       wireCallerOf(metadata.Caller),
		cacheControl: cacheControl,
	}

	if mcpToolUseIDs[part.ToolCallID] {
		return c.mcpResult(result), nil
	}

	switch c.toolNames.ToProvider(part.ToolName) {
	case nameCodeExecution:
		return c.codeExecutionResult(result)
	case nameWebFetch:
		return c.webFetchResult(result)
	case nameWebSearch:
		return c.webSearchResult(result)
	case nameToolSearchRegex, nameToolSearchBM25:
		return c.toolSearchResult(result)
	case nameAdvisor:
		return c.advisorResult(result)
	default:
		c.warnings.other(fmt.Sprintf("provider executed tool result for tool %s is not supported", part.ToolName))

		return nil, nil
	}
}

func (c *promptConverter) unsupportedOutput(part model.ToolResultPart) {
	c.warnings.other(fmt.Sprintf(
		"provider executed tool result output type %s for tool %s is not supported",
		outputTypeName(part.Output),
		part.ToolName,
	))
}

func outputTypeName(output model.ToolResultOutput) string {
	switch output.(type) {
	case model.TextOutput:
		return "text"
	case model.JSONOutput:
		return "json"
	case model.ExecutionDeniedOutput:
		return "execution-denied"
	case model.ErrorTextOutput:
		return "error-text"
	case model.ErrorJSONOutput:
		return "error-json"
	case model.ContentOutput:
		return "content"
	default:
		return fmt.Sprintf("%T", output)
	}
}

func (c *promptConverter) mcpResult(result providerResult) requestBlock {
	var content json.RawMessage
	isError := false

	switch output := result.part.Output.(type) {
	case model.JSONOutput:
		content = output.Value
	case model.ErrorJSONOutput:
		content = output.Value
		isError = true
	default:
		c.unsupportedOutput(result.part)

		return nil
	}

	return &mcpToolResultBlock{
		Type:         blockMCPToolResult,
		ToolUseID:    result.part.ToolCallID,
		IsError:      isError,
		Content:      content,
		CacheControl: result.cacheControl,
	}
}

// codeExecutionResult replays a code execution result. The result's type
// distinguishes code execution 20250522, the subtools of 20250825 and the
// encrypted results of code execution provisioned for dynamic web tools.
func (c *promptConverter) codeExecutionResult(result providerResult) (requestBlock, error) {
	if failure, isFailure := errorOutputValue(result.part.Output); isFailure {
		info := parseErrorInfo(failure)

		if info.Type == resultCodeExecutionError {
			return result.block(blockCodeExecutionToolResult, wireError{
				Type:      resultCodeExecutionError,
				ErrorCode: valueOrDefault(info.ErrorCode, "unknown"),
			}), nil
		}

		return result.block(blockBashCodeExecutionToolResult, wireError{
			Type:      resultBashCodeExecutionError,
			ErrorCode: valueOrDefault(info.ErrorCode, "unknown"),
		}), nil
	}

	output, ok := result.part.Output.(model.JSONOutput)
	if !ok {
		c.unsupportedOutput(result.part)

		return nil, nil
	}

	object, err := jsonobject.Parse(output.Value)
	resultType, hasType := object.GetString("type")
	if err != nil || !hasType {
		c.warnings.other(fmt.Sprintf(
			"provider executed tool result output value is not a valid code execution result for tool %s",
			result.part.ToolName,
		))

		return nil, nil
	}

	content, err := decodeCodeExecutionOutput(output.Value, codeExecutionResultTypes...)
	if err != nil {
		return nil, invalidOutput(result.part, err)
	}

	switch resultType {
	case resultCodeExecution, resultEncryptedCodeExecution:
		return result.block(blockCodeExecutionToolResult, content), nil
	case resultBashCodeExecution, resultBashCodeExecutionError:
		return result.block(blockBashCodeExecutionToolResult, content), nil
	default:
		return result.block(blockTextEditorCodeExecutionToolResult, content), nil
	}
}

// codeExecutionResultTypes are the result variants of every code execution
// version.
var codeExecutionResultTypes = []string{
	resultCodeExecution,
	resultEncryptedCodeExecution,
	resultBashCodeExecution,
	resultBashCodeExecutionError,
	resultTextEditorError,
	resultTextEditorView,
	resultTextEditorCreate,
	resultTextEditorStrReplace,
}

func (c *promptConverter) webFetchResult(result providerResult) (requestBlock, error) {
	switch output := result.part.Output.(type) {
	case model.ErrorJSONOutput:
		return result.blockWithCaller(blockWebFetchToolResult, wireError{
			Type:      resultWebFetchError,
			ErrorCode: valueOrDefault(parseErrorInfo(output.Value).ErrorCode, "unavailable"),
		}), nil

	case model.JSONOutput:
		fetched, err := decodeOutputValue[WebFetchResult](output.Value)
		if err != nil {
			return nil, invalidOutput(result.part, err)
		}

		return result.blockWithCaller(blockWebFetchToolResult, wireWebFetchResult{
			Type:        resultWebFetch,
			URL:         fetched.URL,
			RetrievedAt: fetched.RetrievedAt,
			Content: wireWebFetchContent{
				Type:      "document",
				Title:     fetched.Content.Title,
				Citations: fetched.Content.Citations,
				Source: wireWebFetchSource{
					Type:      fetched.Content.Source.Type,
					MediaType: fetched.Content.Source.MediaType,
					Data:      fetched.Content.Source.Data,
				},
			},
		}), nil

	default:
		c.unsupportedOutput(result.part)

		return nil, nil
	}
}

func (c *promptConverter) webSearchResult(result providerResult) (requestBlock, error) {
	switch output := result.part.Output.(type) {
	case model.ErrorJSONOutput:
		return result.blockWithCaller(blockWebSearchToolResult, wireError{
			Type:      resultWebSearchError,
			ErrorCode: valueOrDefault(parseErrorInfo(output.Value).ErrorCode, "unavailable"),
		}), nil

	case model.JSONOutput:
		searched, err := decodeOutputValue[[]WebSearchResult](output.Value)
		if err != nil {
			return nil, invalidOutput(result.part, err)
		}

		results := make(wireWebSearchResults, 0, len(searched))
		for _, entry := range searched {
			results = append(results, wireWebSearchResult(entry))
		}

		return result.blockWithCaller(blockWebSearchToolResult, results), nil

	default:
		c.unsupportedOutput(result.part)

		return nil, nil
	}
}

func (c *promptConverter) toolSearchResult(result providerResult) (requestBlock, error) {
	if failure, isFailure := toolSearchFailure(result.part.Output); isFailure {
		return result.block(blockToolSearchToolResult, wireError{
			Type:      resultToolSearchError,
			ErrorCode: valueOrDefault(parseErrorInfo(failure).ErrorCode, "unavailable"),
		}), nil
	}

	output, ok := result.part.Output.(model.JSONOutput)
	if !ok {
		c.unsupportedOutput(result.part)

		return nil, nil
	}

	references, err := decodeOutputValue[[]ToolReference](output.Value)
	if err != nil {
		return nil, invalidOutput(result.part, err)
	}

	wireReferences := make([]toolReference, 0, len(references))
	for _, reference := range references {
		wireReferences = append(wireReferences, toolReference{
			Type:     blockToolReference,
			ToolName: reference.ToolName,
		})
	}

	return result.block(blockToolSearchToolResult, wireToolSearchResult{
		Type:           resultToolSearch,
		ToolReferences: wireReferences,
	}), nil
}

// toolSearchFailure returns the error value of a failed tool search: an
// error output, or a JSON output holding a tool search error.
func toolSearchFailure(output model.ToolResultOutput) (json.RawMessage, bool) {
	switch typed := output.(type) {
	case model.ErrorJSONOutput:
		return typed.Value, true

	case model.JSONOutput:
		object, err := jsonobject.Parse(typed.Value)
		if err != nil {
			return nil, false
		}

		resultType, _ := object.GetString("type")

		return typed.Value, resultType == resultToolSearchError

	default:
		return nil, false
	}
}

func (c *promptConverter) advisorResult(result providerResult) (requestBlock, error) {
	var value json.RawMessage

	switch output := result.part.Output.(type) {
	case model.JSONOutput:
		value = output.Value
	case model.ErrorJSONOutput:
		value = output.Value
	default:
		c.unsupportedOutput(result.part)

		return nil, nil
	}

	advice, err := decodeOutputValue[AdvisorResult](value)
	if err != nil {
		return nil, invalidOutput(result.part, err)
	}

	err = checkRequiredEnum("advisor result type", advice.Type, resultAdvisor, resultAdvisorRedacted, resultAdvisorError)
	if err != nil {
		return nil, invalidOutput(result.part, err)
	}

	return result.block(blockAdvisorToolResult, wireAdvisorResult(advice)), nil
}

// errorInfo is the error object of a failed provider tool result.
type errorInfo struct {
	Type      string `json:"type"`
	ErrorCode string `json:"errorCode"`
}

// errorOutputValue returns the value of an error output as JSON: an error
// JSON value, or error text, which may itself contain JSON.
func errorOutputValue(output model.ToolResultOutput) (json.RawMessage, bool) {
	switch typed := output.(type) {
	case model.ErrorJSONOutput:
		return typed.Value, true
	case model.ErrorTextOutput:
		return mustEncodeJSON(typed.Value), true
	default:
		return nil, false
	}
}

// parseErrorInfo reads an error object from a value that is either the
// object or a JSON string containing it. Unreadable values yield an empty
// error, so callers use their default error codes.
func parseErrorInfo(value json.RawMessage) errorInfo {
	var text string
	if err := json.Unmarshal(value, &text); err == nil {
		value = json.RawMessage(text)
	}

	var info errorInfo
	if err := json.Unmarshal(value, &info); err != nil {
		return errorInfo{}
	}

	return info
}

// decodeOutputValue decodes a tool output value into its typed form.
func decodeOutputValue[Output any](value json.RawMessage) (Output, error) {
	var output Output
	err := json.Unmarshal(value, &output)

	return output, err
}

func invalidOutput(part model.ToolResultPart, err error) error {
	return fmt.Errorf("%w: output of tool %s: %w", model.ErrInvalidPrompt, part.ToolName, err)
}

func valueOrDefault(value string, fallback string) string {
	if value == "" {
		return fallback
	}

	return value
}
