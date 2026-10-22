package openai

import (
	"encoding/json"

	"github.com/aarock1234/ai/pkg/model"
)

// This file converts output items into tool inputs, tool results, sources
// and metadata. Generate and Stream share these conversions.

// toolCallMetadata identifies the output item of a tool call.
func (c responsesCallContext) toolCallMetadata(itemID string) model.ProviderMetadata {
	return c.metadata(ResponsesToolCallMetadata{
		ItemID: itemID,
	})
}

// functionCallMetadata describes a function call item.
func (c responsesCallContext) functionCallMetadata(item responsesFunctionCallItem, async *bool) model.ProviderMetadata {
	return c.metadata(ResponsesToolCallMetadata{
		ItemID:    item.ID,
		Async:     async,
		Namespace: item.Namespace,
		Caller:    responsesToolCallerOf(item.Caller),
	})
}

// reasoningMetadata describes a reasoning item.
func (c responsesCallContext) reasoningMetadata(itemID string, encryptedContent *string) model.ProviderMetadata {
	return c.metadata(ResponsesReasoningMetadata{
		ItemID:                    itemID,
		ReasoningEncryptedContent: encryptedContent,
	})
}

// compaction converts a compaction item into custom content.
func (c responsesCallContext) compaction(item responsesCompactionItem) model.Custom {
	return model.Custom{
		Kind: "openai.compaction",
		ProviderMetadata: c.metadata(ResponsesCompactionMetadata{
			Type:             "compaction",
			ItemID:           item.ID,
			EncryptedContent: item.EncryptedContent,
		}),
	}
}

// webSearchCall returns the provider-executed call of a web search.
func (c responsesCallContext) webSearchCall(itemID string) model.ToolCall {
	return model.ToolCall{
		ToolCallID:       itemID,
		ToolName:         c.toolName(c.webSearchToolName),
		Input:            "{}",
		ProviderExecuted: true,
	}
}

// webSearchResult returns the result of a web search. Failed and
// incomplete searches are error results that report the status.
func (c responsesCallContext) webSearchResult(item responsesWebSearchCallItem) model.ToolResult {
	result := model.ToolResult{
		ToolCallID: item.ID,
		ToolName:   c.toolName(c.webSearchToolName),
	}

	if item.Status == "failed" || item.Status == "incomplete" {
		result.IsError = true
		result.Result = responsesEncodeJSON(WebSearchFailure{
			Status: item.Status,
		})

		return result
	}

	result.Result = responsesEncodeJSON(responsesWebSearchResultOf(item.Action))

	return result
}

// webSearchSources returns the URL sources of a search action.
func (c responsesCallContext) webSearchSources(item responsesWebSearchCallItem) []model.Source {
	if item.Action == nil || item.Action.Type != "search" {
		return nil
	}

	var sources []model.Source

	for _, source := range item.Action.Sources {
		if source.Type != "url" {
			continue
		}

		sources = append(sources, model.Source{
			SourceType: model.SourceURL,
			ID:         c.generateID(),
			URL:        source.URL,
		})
	}

	return sources
}

// fileSearchCall returns the provider-executed call of a file search.
func (c responsesCallContext) fileSearchCall(itemID string) model.ToolCall {
	return model.ToolCall{
		ToolCallID:       itemID,
		ToolName:         c.toolName("file_search"),
		Input:            "{}",
		ProviderExecuted: true,
	}
}

// fileSearchResult returns the result of a file search.
func (c responsesCallContext) fileSearchResult(item responsesFileSearchCallItem) model.ToolResult {
	result := FileSearchResult{
		Queries: item.Queries,
	}

	if item.Results != nil {
		result.Results = make([]FileSearchMatch, 0, len(item.Results))
	}

	for _, match := range item.Results {
		result.Results = append(result.Results, FileSearchMatch(match))
	}

	return model.ToolResult{
		ToolCallID: item.ID,
		ToolName:   c.toolName("file_search"),
		Result:     responsesEncodeJSON(result),
	}
}

// fileSearchSources returns a document source for each retrieved file not
// yet in retrieved, and adds it there.
func (c responsesCallContext) fileSearchSources(item responsesFileSearchCallItem, retrieved map[string]bool) []model.Source {
	var sources []model.Source

	for _, match := range item.Results {
		if retrieved[match.FileID] {
			continue
		}

		retrieved[match.FileID] = true

		sources = append(sources, model.Source{
			SourceType: model.SourceDocument,
			ID:         c.generateID(),
			MediaType:  "text/plain",
			Title:      match.Filename,
			Filename:   match.Filename,
			ProviderMetadata: c.metadata(ResponsesSourceMetadata{
				Type:   "file_search",
				FileID: match.FileID,
			}),
		})
	}

	return sources
}

// codeInterpreterCall returns the provider-executed call of the code
// interpreter.
func (c responsesCallContext) codeInterpreterCall(itemID string, code *string, containerID string) model.ToolCall {
	return model.ToolCall{
		ToolCallID: itemID,
		ToolName:   c.toolName("code_interpreter"),
		Input: string(responsesEncodeJSON(CodeInterpreterInput{
			Code:        code,
			ContainerID: containerID,
		})),
		ProviderExecuted: true,
	}
}

// codeInterpreterResult returns the outputs of the code interpreter.
func (c responsesCallContext) codeInterpreterResult(item responsesCodeInterpreterCallItem) model.ToolResult {
	return model.ToolResult{
		ToolCallID: item.ID,
		ToolName:   c.toolName("code_interpreter"),
		Result: responsesEncodeJSON(CodeInterpreterResult{
			Outputs: item.Outputs,
		}),
	}
}

// imageGenerationCall returns the provider-executed call of image
// generation.
func (c responsesCallContext) imageGenerationCall(itemID string) model.ToolCall {
	return model.ToolCall{
		ToolCallID:       itemID,
		ToolName:         c.toolName("image_generation"),
		Input:            "{}",
		ProviderExecuted: true,
	}
}

// imageGenerationResult returns a generated image. Partial images are
// preliminary results.
func (c responsesCallContext) imageGenerationResult(itemID, image string, preliminary bool) model.ToolResult {
	return model.ToolResult{
		ToolCallID: itemID,
		ToolName:   c.toolName("image_generation"),
		Result: responsesEncodeJSON(ImageGenerationResult{
			Result: image,
		}),
		Preliminary: preliminary,
	}
}

// localShellCall returns a local shell call for the application.
func (c responsesCallContext) localShellCall(item responsesLocalShellCallItem) model.ToolCall {
	return model.ToolCall{
		ToolCallID: item.CallID,
		ToolName:   c.toolName("local_shell"),
		Input: string(responsesEncodeJSON(LocalShellInput{
			Action: LocalShellAction(item.Action),
		})),
		ProviderMetadata: c.toolCallMetadata(item.ID),
	}
}

// shellCall returns a shell call, executed by OpenAI when the shell runs in
// a container.
func (c responsesCallContext) shellCall(item responsesShellCallItem) model.ToolCall {
	return model.ToolCall{
		ToolCallID: item.CallID,
		ToolName:   c.toolName("shell"),
		Input: string(responsesEncodeJSON(ShellInput{
			Action: ShellAction(item.Action),
		})),
		ProviderExecuted: c.isShellProviderExecuted,
		ProviderMetadata: c.toolCallMetadata(item.ID),
	}
}

// shellResult returns the output of a hosted shell call.
func (c responsesCallContext) shellResult(item responsesShellCallOutputItem) model.ToolResult {
	output := ShellOutput{
		MaxOutputLength: item.MaxOutputLength,
		Output:          make([]ShellCommandOutput, 0, len(item.Output)),
	}

	for _, entry := range item.Output {
		outcome := ShellOutcome{
			Type: "timeout",
		}

		if entry.Outcome.Type == "exit" {
			outcome = ShellOutcome{
				Type:     "exit",
				ExitCode: entry.Outcome.ExitCode,
			}
		}

		output.Output = append(output.Output, ShellCommandOutput{
			Stdout:  entry.Stdout,
			Stderr:  entry.Stderr,
			Outcome: outcome,
		})
	}

	return model.ToolResult{
		ToolCallID: item.CallID,
		ToolName:   c.toolName("shell"),
		Result:     responsesEncodeJSON(output),
	}
}

// programCall returns the provider-executed call of a hosted program.
func (c responsesCallContext) programCall(item responsesProgramItem) model.ToolCall {
	return model.ToolCall{
		ToolCallID: item.CallID,
		ToolName:   c.toolName("programmatic_tool_calling"),
		Input: string(responsesEncodeJSON(ProgrammaticToolCallingInput{
			Code:        item.Code,
			Fingerprint: item.Fingerprint,
		})),
		ProviderExecuted: true,
		ProviderMetadata: c.toolCallMetadata(item.ID),
	}
}

// programResult returns the result of a hosted program.
func (c responsesCallContext) programResult(item responsesProgramOutputItem) model.ToolResult {
	return model.ToolResult{
		ToolCallID: item.CallID,
		ToolName:   c.toolName("programmatic_tool_calling"),
		Result: responsesEncodeJSON(ProgrammaticToolCallingResult{
			Result: item.Result,
			Status: item.Status,
		}),
		ProviderMetadata: c.itemMetadata(item.ID),
	}
}

// customToolCall returns a custom tool call. Its text input is sent as a
// JSON string.
func (c responsesCallContext) customToolCall(item responsesCustomToolCallItem, async *bool) model.ToolCall {
	return model.ToolCall{
		ToolCallID: item.CallID,
		ToolName:   c.toolName(item.Name),
		Input:      string(responsesEncodeJSON(item.Input)),
		ProviderMetadata: c.metadata(ResponsesToolCallMetadata{
			ItemID: item.ID,
			Async:  async,
		}),
	}
}

// functionCall returns a function call.
func (c responsesCallContext) functionCall(item responsesFunctionCallItem, async *bool) model.ToolCall {
	return model.ToolCall{
		ToolCallID:       item.CallID,
		ToolName:         item.Name,
		Input:            item.Arguments,
		ProviderMetadata: c.functionCallMetadata(item, async),
	}
}

// computerCall returns a computer call for the application.
func (c responsesCallContext) computerCall(item responsesComputerCallItem, callID string) model.ToolCall {
	return model.ToolCall{
		ToolCallID:       callID,
		ToolName:         c.toolName("computer"),
		Input:            string(responsesEncodeJSON(responsesComputerInputOf(item))),
		ProviderMetadata: c.toolCallMetadata(item.ID),
	}
}

// computerUseCall and computerUseResult represent a legacy computer use
// call, which has no call ID and is executed by OpenAI.
func (c responsesCallContext) computerUseCall(itemID string) model.ToolCall {
	return model.ToolCall{
		ToolCallID:       itemID,
		ToolName:         c.toolName("computer_use"),
		ProviderExecuted: true,
	}
}

func (c responsesCallContext) computerUseResult(item responsesComputerCallItem) model.ToolResult {
	return model.ToolResult{
		ToolCallID: item.ID,
		ToolName:   c.toolName("computer_use"),
		Result: responsesEncodeJSON(ComputerUseResult{
			Type:   "computer_use_tool_result",
			Status: item.Status,
		}),
	}
}

// applyPatchCall returns an apply patch call for the application.
func (c responsesCallContext) applyPatchCall(item responsesApplyPatchCallItem, toolCallID string) model.ToolCall {
	return model.ToolCall{
		ToolCallID: toolCallID,
		ToolName:   c.toolName("apply_patch"),
		Input: string(responsesEncodeJSON(ApplyPatchInput{
			CallID:    item.CallID,
			Operation: item.Operation,
		})),
		ProviderMetadata: c.toolCallMetadata(item.ID),
	}
}

// toolSearchCall returns a tool search call. Hosted searches are executed
// by OpenAI and have no call ID in their input.
func (c responsesCallContext) toolSearchCall(item responsesToolSearchCallItem, toolCallID string, inputCallID *string) model.ToolCall {
	return model.ToolCall{
		ToolCallID: toolCallID,
		ToolName:   c.toolName("tool_search"),
		Input: string(responsesEncodeJSON(ToolSearchInput{
			Arguments: responsesNonNullJSON(item.Arguments),
			CallID:    inputCallID,
		})),
		ProviderExecuted: item.Execution == "server",
		ProviderMetadata: c.toolCallMetadata(item.ID),
	}
}

// toolSearchResult returns the tools a tool search loaded.
func (c responsesCallContext) toolSearchResult(item responsesToolSearchOutputItem, toolCallID string) model.ToolResult {
	return model.ToolResult{
		ToolCallID: toolCallID,
		ToolName:   c.toolName("tool_search"),
		Result: responsesEncodeJSON(ToolSearchOutput{
			Tools: responsesNonNilTools(item.Tools),
		}),
		ProviderMetadata: c.itemMetadata(item.ID),
	}
}

// mcpCall and mcpResult represent a remote MCP call. MCP tools are dynamic
// because they are discovered from the server.
func (c responsesCallContext) mcpCall(toolCallID, name, arguments string) model.ToolCall {
	return model.ToolCall{
		ToolCallID:       toolCallID,
		ToolName:         "mcp." + name,
		Input:            arguments,
		ProviderExecuted: true,
		Dynamic:          true,
	}
}

func (c responsesCallContext) mcpResult(item responsesMCPCallItem, toolCallID string) model.ToolResult {
	result := MCPResult{
		Type:        "call",
		ServerLabel: item.ServerLabel,
		Name:        item.Name,
		Arguments:   item.Arguments,
		Output:      item.Output,
		Error:       responsesNonNullJSON(item.Error),
	}

	return model.ToolResult{
		ToolCallID:       toolCallID,
		ToolName:         "mcp." + item.Name,
		Result:           responsesEncodeJSON(result),
		ProviderMetadata: c.itemMetadata(item.ID),
	}
}

// mcpToolCallID returns the tool call ID of an MCP call. Calls that needed
// approval reuse the ID created for the approval request.
func (c responsesCallContext) mcpToolCallID(item responsesMCPCallItem, streamed map[string]string) string {
	if item.ApprovalRequestID == nil {
		return item.ID
	}

	approvalID := *item.ApprovalRequestID

	if toolCallID, ok := streamed[approvalID]; ok {
		return toolCallID
	}

	if toolCallID, ok := c.approvalToolCallIDs[approvalID]; ok {
		return toolCallID
	}

	return item.ID
}

// mcpApproval returns the dynamic call and approval request of an MCP
// approval request item, with a new tool call ID.
func (c responsesCallContext) mcpApproval(item responsesMCPApprovalRequestItem) (model.ToolCall, model.ToolApprovalRequest) {
	toolCallID := c.generateID()

	approvalID := item.ID
	if item.ApprovalRequestID != nil {
		approvalID = *item.ApprovalRequestID
	}

	call := c.mcpCall(toolCallID, item.Name, item.Arguments)
	request := model.ToolApprovalRequest{
		ApprovalID: approvalID,
		ToolCallID: toolCallID,
	}

	return call, request
}

// annotationSource converts an annotation into a source with the given ID.
func (c responsesCallContext) annotationSource(annotation ResponsesAnnotation, sourceID string) model.Source {
	if annotation.Type == ResponsesURLCitation {
		return model.Source{
			SourceType: model.SourceURL,
			ID:         sourceID,
			URL:        annotation.URL,
			Title:      annotation.Title,
		}
	}

	filename := annotation.Filename
	mediaType := "text/plain"

	if annotation.Type == ResponsesFilePath {
		filename = annotation.FileID
		mediaType = "application/octet-stream"
	}

	metadata := ResponsesSourceMetadata{
		Type:   string(annotation.Type),
		FileID: annotation.FileID,
	}

	if annotation.Type == ResponsesContainerFileCitation {
		metadata.ContainerID = annotation.ContainerID
	} else {
		metadata.Index = annotation.Index
	}

	return model.Source{
		SourceType:       model.SourceDocument,
		ID:               sourceID,
		MediaType:        mediaType,
		Title:            filename,
		Filename:         filename,
		ProviderMetadata: c.metadata(metadata),
	}
}

// citations converts the citation annotations of a text. File paths are
// not citations. Sources are identified by URL or file ID.
func (c responsesCallContext) citations(annotations []ResponsesAnnotation) []model.Citation {
	var citations []model.Citation

	for _, annotation := range annotations {
		if annotation.Type == ResponsesFilePath {
			continue
		}

		sourceID := annotation.FileID
		if annotation.Type == ResponsesURLCitation {
			sourceID = annotation.URL
		}

		citations = append(citations, model.Citation{
			Source:     c.annotationSource(annotation, sourceID),
			StartIndex: annotation.StartIndex,
			EndIndex:   annotation.EndIndex,
		})
	}

	return citations
}

// textMetadata describes a message's text.
func (c responsesCallContext) textMetadata(itemID, phase string, annotations []ResponsesAnnotation) model.ProviderMetadata {
	metadata := ResponsesTextMetadata{
		ItemID: itemID,
		Phase:  phase,
	}

	if len(annotations) > 0 {
		metadata.Annotations = annotations
	}

	return c.metadata(metadata)
}

// responsesWebSearchResultOf converts a web search action into a result.
func responsesWebSearchResultOf(action *responsesWebSearchAction) WebSearchResult {
	if action == nil {
		return WebSearchResult{}
	}

	switch action.Type {
	case "search":
		return WebSearchResult{
			Action: &WebSearchAction{
				Type:    WebSearchActionSearch,
				Query:   responsesStringValue(action.Query),
				Queries: action.Queries,
			},
			Sources: action.Sources,
		}
	case "open_page":
		return WebSearchResult{
			Action: &WebSearchAction{
				Type: WebSearchActionOpenPage,
				URL:  responsesStringValue(action.URL),
			},
		}
	case "find_in_page":
		return WebSearchResult{
			Action: &WebSearchAction{
				Type:    WebSearchActionFindInPage,
				URL:     responsesStringValue(action.URL),
				Pattern: responsesStringValue(action.Pattern),
			},
		}
	default:
		return WebSearchResult{}
	}
}

// responsesComputerInputOf converts a computer call into its tool input.
func responsesComputerInputOf(item responsesComputerCallItem) ComputerInput {
	actions := item.Actions
	if actions == nil && item.Action != nil {
		actions = []responsesComputerAction{
			*item.Action,
		}
	}

	input := ComputerInput{
		Actions:             make([]ComputerAction, 0, len(actions)),
		PendingSafetyChecks: item.PendingSafetyChecks,
		Status:              item.Status,
	}

	for _, action := range actions {
		input.Actions = append(input.Actions, ComputerAction(action))
	}

	if input.PendingSafetyChecks == nil {
		input.PendingSafetyChecks = []ComputerSafetyCheck{}
	}

	return input
}

func responsesToolCallerOf(caller *responsesCaller) *ResponsesToolCaller {
	if caller == nil {
		return nil
	}

	return &ResponsesToolCaller{
		Type:     caller.Type,
		CallerID: caller.CallerID,
	}
}

func responsesStringValue(value *string) string {
	if value == nil {
		return ""
	}

	return *value
}

// responsesNonNullJSON returns nil for absent or null JSON.
func responsesNonNullJSON(value json.RawMessage) json.RawMessage {
	if len(value) == 0 || isJSONNull(value) {
		return nil
	}

	return value
}

// responsesHasURLSources reports whether a web search action lists URL
// sources.
func responsesHasURLSources(item responsesWebSearchCallItem) bool {
	if item.Action == nil || item.Action.Type != "search" {
		return false
	}

	for _, source := range item.Action.Sources {
		if source.Type == "url" {
			return true
		}
	}

	return false
}
