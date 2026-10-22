package anthropic

import (
	"encoding/json"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/internal/toolnames"
	"github.com/aarock1234/ai/internal/toolresult"
	"github.com/aarock1234/ai/pkg/model"
)

// jsonToolName is the synthetic tool that carries JSON output when native
// structured output is not used.
const jsonToolName = "json"

// contentConverter turns response content blocks into output content. It
// keeps the state that later blocks depend on, so one converter serves one
// response or stream.
type contentConverter struct {
	toolNames  toolnames.Mapping
	generateID func() string

	// documents are the documents citations refer to by index. Fetched web
	// pages are appended as their results arrive.
	documents []citationDocument

	// mcpCalls are the MCP tool calls by ID, for naming their results.
	mcpCalls map[string]model.ToolCall

	// serverToolCalls maps tool search call IDs to the tool search variant.
	serverToolCalls map[string]string

	usesJSONResponseTool bool

	// markCodeExecutionDynamic marks code execution calls as dynamic when
	// dynamic web tools implicitly provision code execution that the call
	// did not declare.
	markCodeExecutionDynamic bool
}

func newContentConverter(prepared *preparedRequest, documents []citationDocument, generateID func() string) *contentConverter {
	return &contentConverter{
		toolNames:                prepared.toolNames,
		generateID:               generateID,
		documents:                documents,
		mcpCalls:                 make(map[string]model.ToolCall),
		serverToolCalls:          make(map[string]string),
		usesJSONResponseTool:     prepared.usesJSONResponseTool,
		markCodeExecutionDynamic: hasDynamicWebToolWithoutCodeExecution(prepared.body.Tools),
	}
}

// hasDynamicWebToolWithoutCodeExecution reports whether the tools include a
// dynamic filtering web tool but no code execution tool.
func hasDynamicWebToolWithoutCodeExecution(tools []requestTool) bool {
	hasDynamicWebTool := false

	for _, tool := range tools {
		switch tool.toolType() {
		case "web_fetch_20260209", "web_fetch_20260318", "web_search_20260209", "web_search_20260318":
			hasDynamicWebTool = true
		case "code_execution_20250522", "code_execution_20250825", "code_execution_20260120":
			return false
		}
	}

	return hasDynamicWebTool
}

// responseContent converts the content of a non-streaming response. It
// reports whether the JSON response tool produced the output.
func (c *contentConverter) responseContent(blocks []responseBlock) ([]model.Content, bool) {
	var content []model.Content
	jsonFromTool := false
	hasWebSearchResults := containsWebSearchResults(blocks)

	for _, block := range blocks {
		switch typed := block.(type) {
		case *responseText:
			if !c.usesJSONResponseTool {
				content = append(content, c.textContent(typed, hasWebSearchResults)...)
			}

		case *responseThinking:
			content = append(content, model.Reasoning{
				Text:             typed.Thinking,
				ProviderMetadata: signatureMetadata(typed.Signature),
			})

		case *responseRedactedThinking:
			content = append(content, model.Reasoning{
				ProviderMetadata: redactedMetadata(typed.Data),
			})

		case *responseContainerUpload:
			content = append(content, model.Custom{
				Kind: containerUploadKind,
				ProviderMetadata: model.ProviderMetadata{
					Namespace: ContainerUploadMetadata{
						FileID: typed.FileID,
					},
				},
			})

		case *responseCompaction:
			if valueOrZero(typed.Content) != "" {
				content = append(content, model.Text{
					Text:             *typed.Content,
					ProviderMetadata: compactionMetadata(typed.Signature),
				})
			}

		case *responseToolUse:
			if c.isJSONResponseTool(typed.Name) {
				jsonFromTool = true
				content = append(content, model.Text{
					Text: compactInput(typed.Input),
				})

				continue
			}

			content = append(content, c.toolUseCall(typed))

		case *responseServerToolUse:
			if call, ok := c.serverToolCall(typed); ok {
				content = append(content, call)
			}

		case *responseMCPToolUse:
			content = append(content, c.mcpToolCall(typed))

		case *responseMCPToolResult:
			content = append(content, c.mcpToolResult(typed))

		case *responseWebFetchToolResult:
			content = append(content, c.webFetchResult(typed))

		case *responseWebSearchToolResult:
			result, sources := c.webSearchResult(typed)
			content = append(content, result)

			for _, source := range sources {
				content = append(content, source)
			}

		case *responseCodeExecutionToolResult:
			content = append(content, c.codeExecutionResult(typed))

		case *responseSubtoolResult:
			content = append(content, c.subtoolResult(typed))

		case *responseToolSearchToolResult:
			content = append(content, c.toolSearchResult(typed))

		case *responseAdvisorToolResult:
			content = append(content, c.advisorResult(typed))

		case *responseFallback:
			content = append(content, fallbackContent(typed))
		}
	}

	return content, jsonFromTool
}

func containsWebSearchResults(blocks []responseBlock) bool {
	for _, block := range blocks {
		result, ok := block.(*responseWebSearchToolResult)
		if ok && len(result.Content.Results) > 0 {
			return true
		}
	}

	return false
}

// textContent returns text with its citations, followed by a source for
// each citation. Web search citations get no source when the response has
// search results, which announce their own sources.
func (c *contentConverter) textContent(block *responseText, hasWebSearchResults bool) []model.Content {
	content := []model.Content{
		model.Text{
			Text:             block.Text,
			Citations:        mapCitations(block.Citations, c.documents),
			ProviderMetadata: citationsMetadata(block.Citations),
		},
	}

	for _, citation := range block.Citations {
		if citation.Type == CitationWebSearchResultLocation && hasWebSearchResults {
			continue
		}

		if source, ok := citationSource(citation, c.documents, c.generateID); ok {
			content = append(content, source)
		}
	}

	return content
}

func (c *contentConverter) isJSONResponseTool(name string) bool {
	return c.usesJSONResponseTool && name == jsonToolName
}

// toolUseCall converts a client tool call. Toolset member calls become
// calls of the toolset tool, with the member name as the input's action.
func (c *contentConverter) toolUseCall(block *responseToolUse) model.ToolCall {
	toolsetName := valueOrZero(block.ToolsetName)
	if toolsetName == "" {
		return model.ToolCall{
			ToolCallID:       block.ID,
			ToolName:         block.Name,
			Input:            compactInput(block.Input),
			ProviderMetadata: callerMetadata(block.Caller),
		}
	}

	return model.ToolCall{
		ToolCallID: block.ID,
		ToolName:   c.toolNames.ToApplication(toolsetName),
		Input:      string(toolsetMemberInput(block.Name, block.Input)),
		ProviderMetadata: model.ProviderMetadata{
			Namespace: ToolCallMetadata{
				ToolsetName: toolsetName,
				Caller:      toolCallerOf(block.Caller),
			},
		},
	}
}

// toolsetMemberInput injects the member name as the input's action, the
// input shape of the older computer tool versions.
func toolsetMemberInput(memberName string, input json.RawMessage) json.RawMessage {
	object, err := jsonobject.Parse(input)
	if err != nil {
		object = jsonobject.Object{}
	}

	withAction, err := object.WithFirst("action", jsonobject.EncodeString(memberName)).Encode()
	if err != nil {
		return input
	}

	return withAction
}

// serverToolCall converts a provider-executed tool call. Code execution
// subtool calls are reported as code execution calls whose input names the
// subtool in its type.
func (c *contentConverter) serverToolCall(block *responseServerToolUse) (model.ToolCall, bool) {
	call := model.ToolCall{
		ToolCallID:       block.ID,
		ProviderExecuted: true,
		ProviderMetadata: callerMetadata(block.Caller),
	}

	switch block.Name {
	case nameTextEditorCodeExecution, nameBashCodeExecution:
		call.ToolName = c.toolNames.ToApplication(nameCodeExecution)
		call.Input = subtoolInput(block.Name, block.Input)
		call.Dynamic = c.markCodeExecutionDynamic

	case nameCodeExecution:
		call.ToolName = c.toolNames.ToApplication(nameCodeExecution)
		call.Input = programmaticInput(compactInput(block.Input))
		call.Dynamic = c.markCodeExecutionDynamic

	case nameWebSearch, nameWebFetch, nameAdvisor:
		call.ToolName = c.toolNames.ToApplication(block.Name)
		call.Input = compactInput(block.Input)

	case nameToolSearchRegex, nameToolSearchBM25:
		c.serverToolCalls[block.ID] = block.Name
		call.ToolName = c.toolNames.ToApplication(block.Name)
		call.Input = compactInput(block.Input)

	default:
		return model.ToolCall{}, false
	}

	return call, true
}

// programmaticInput marks code execution input of the form {"code": ...}
// as a programmatic tool call by adding its type.
func programmaticInput(input string) string {
	object, err := jsonobject.Parse([]byte(input))
	if err != nil || !object.Has("code") || object.Has("type") {
		return input
	}

	typed, err := object.WithFirst("type", jsonobject.EncodeString(programmaticCallType)).Encode()
	if err != nil {
		return input
	}

	return string(typed)
}

// subtoolInput names a code execution subtool in the type of its input.
// Input that cannot be encoded again is reported unchanged.
func subtoolInput(name string, input json.RawMessage) string {
	typed, err := inputObject(input).WithFirst("type", jsonobject.EncodeString(name)).Encode()
	if err != nil {
		return compactInput(input)
	}

	return string(typed)
}

func (c *contentConverter) mcpToolCall(block *responseMCPToolUse) model.ToolCall {
	call := model.ToolCall{
		ToolCallID:       block.ID,
		ToolName:         block.Name,
		Input:            compactInput(block.Input),
		ProviderExecuted: true,
		Dynamic:          true,
		ProviderMetadata: model.ProviderMetadata{
			Namespace: ToolCallMetadata{
				Type:       mcpToolUseType,
				ServerName: block.ServerName,
			},
		},
	}

	c.mcpCalls[block.ID] = call

	return call
}

// mcpToolResult converts an MCP tool result, named after its call.
func (c *contentConverter) mcpToolResult(block *responseMCPToolResult) model.ToolResult {
	call := c.mcpCalls[block.ToolUseID]

	return model.ToolResult{
		ToolCallID:       block.ToolUseID,
		ToolName:         call.ToolName,
		Result:           nonNullResult(block.Content),
		IsError:          block.IsError,
		Dynamic:          true,
		ProviderMetadata: call.ProviderMetadata,
	}
}

func (c *contentConverter) webFetchResult(block *responseWebFetchToolResult) model.ToolResult {
	result := model.ToolResult{
		ToolCallID:       block.ToolUseID,
		ToolName:         c.toolNames.ToApplication(nameWebFetch),
		ProviderMetadata: callerMetadata(block.Caller),
	}

	fetched := block.Content
	if fetched.Type == resultWebFetchError {
		result.IsError = true
		result.Result = mustEncodeJSON(ToolResultError{
			Type:      resultWebFetchError,
			ErrorCode: fetched.ErrorCode,
		})

		return result
	}

	title := fetched.URL
	if fetched.Content.Title != nil {
		title = *fetched.Content.Title
	}

	c.documents = append(c.documents, citationDocument{
		title:     title,
		mediaType: fetched.Content.Source.MediaType,
	})

	result.Result = mustEncodeJSON(WebFetchResult{
		Type:        resultWebFetch,
		URL:         fetched.URL,
		RetrievedAt: fetched.RetrievedAt,
		Content: WebFetchDocument{
			Type:      fetched.Content.Type,
			Title:     fetched.Content.Title,
			Citations: fetched.Content.Citations,
			Source: WebFetchSource{
				Type:      fetched.Content.Source.Type,
				MediaType: fetched.Content.Source.MediaType,
				Data:      fetched.Content.Source.Data,
			},
		},
	})

	return result
}

// webSearchResult converts web search results, returning a URL source for
// each result.
func (c *contentConverter) webSearchResult(block *responseWebSearchToolResult) (model.ToolResult, []model.Source) {
	result := model.ToolResult{
		ToolCallID:       block.ToolUseID,
		ToolName:         c.toolNames.ToApplication(nameWebSearch),
		ProviderMetadata: callerMetadata(block.Caller),
	}

	if block.Content.Error != nil {
		result.IsError = true
		result.Result = mustEncodeJSON(ToolResultError{
			Type:      resultWebSearchError,
			ErrorCode: block.Content.Error.ErrorCode,
		})

		return result, nil
	}

	results := make([]WebSearchResult, 0, len(block.Content.Results))
	sources := make([]model.Source, 0, len(block.Content.Results))

	for _, entry := range block.Content.Results {
		results = append(results, WebSearchResult(entry))

		sources = append(sources, model.Source{
			SourceType: model.SourceURL,
			ID:         c.generateID(),
			URL:        entry.URL,
			Title:      valueOrZero(entry.Title),
			ProviderMetadata: model.ProviderMetadata{
				Namespace: WebSearchSourceMetadata{
					PageAge: entry.PageAge,
				},
			},
		})
	}

	result.Result = mustEncodeJSON(results)

	return result, sources
}

// codeExecutionResult converts a result of code execution 20250522.
func (c *contentConverter) codeExecutionResult(block *responseCodeExecutionToolResult) model.ToolResult {
	result := model.ToolResult{
		ToolCallID: block.ToolUseID,
		ToolName:   c.toolNames.ToApplication(nameCodeExecution),
	}

	executed := block.Content

	switch executed.Type {
	case resultCodeExecution:
		result.Result = mustEncodeJSON(CodeExecutionResult{
			Type:       executed.Type,
			Stdout:     executed.Stdout,
			Stderr:     executed.Stderr,
			ReturnCode: executed.ReturnCode,
			Content:    executed.Content,
		})

	case resultEncryptedCodeExecution:
		result.Result = mustEncodeJSON(EncryptedCodeExecutionResult{
			Type:            executed.Type,
			EncryptedStdout: executed.EncryptedStdout,
			Stderr:          executed.Stderr,
			ReturnCode:      executed.ReturnCode,
			Content:         executed.Content,
		})

	default:
		result.IsError = true
		result.Result = mustEncodeJSON(ToolResultError{
			Type:      resultCodeExecutionError,
			ErrorCode: executed.ErrorCode,
		})
	}

	return result
}

// subtoolResult converts a bash or text editor result of code execution
// 20250825, which keeps the API's shape.
func (c *contentConverter) subtoolResult(block *responseSubtoolResult) model.ToolResult {
	return model.ToolResult{
		ToolCallID: block.ToolUseID,
		ToolName:   c.toolNames.ToApplication(nameCodeExecution),
		Result:     mustEncodeJSON(block.Content),
	}
}

func (c *contentConverter) toolSearchResult(block *responseToolSearchToolResult) model.ToolResult {
	result := model.ToolResult{
		ToolCallID: block.ToolUseID,
		ToolName:   c.toolNames.ToApplication(c.toolSearchVariant(block.ToolUseID)),
	}

	if block.Content.Type == resultToolSearchError {
		result.IsError = true
		result.Result = mustEncodeJSON(ToolResultError{
			Type:      resultToolSearchError,
			ErrorCode: block.Content.ErrorCode,
		})

		return result
	}

	references := make([]ToolReference, 0, len(block.Content.ToolReferences))
	for _, reference := range block.Content.ToolReferences {
		references = append(references, ToolReference{
			Type:     reference.Type,
			ToolName: reference.ToolName,
		})
	}

	result.Result = mustEncodeJSON(references)

	return result
}

// toolSearchVariant returns the tool search variant of a call: the one that
// made it, or else the variant the call's tools declare, preferring BM25.
func (c *contentConverter) toolSearchVariant(toolCallID string) string {
	if variant, found := c.serverToolCalls[toolCallID]; found {
		return variant
	}

	if c.toolNames.ToApplication(nameToolSearchBM25) != nameToolSearchBM25 {
		return nameToolSearchBM25
	}

	return nameToolSearchRegex
}

func (c *contentConverter) advisorResult(block *responseAdvisorToolResult) model.ToolResult {
	result := model.ToolResult{
		ToolCallID: block.ToolUseID,
		ToolName:   c.toolNames.ToApplication(nameAdvisor),
	}

	advice := block.Content

	switch advice.Type {
	case resultAdvisor:
		result.Result = mustEncodeJSON(AdvisorResult{
			Type:       resultAdvisor,
			Text:       new(valueOrZero(advice.Text)),
			StopReason: advice.StopReason,
		})

	case resultAdvisorRedacted:
		result.Result = mustEncodeJSON(AdvisorResult{
			Type:             resultAdvisorRedacted,
			EncryptedContent: new(valueOrZero(advice.EncryptedContent)),
			StopReason:       advice.StopReason,
		})

	default:
		result.IsError = true
		result.Result = mustEncodeJSON(ToolResultError{
			Type:      resultAdvisorError,
			ErrorCode: valueOrZero(advice.ErrorCode),
		})
	}

	return result
}

func fallbackContent(block *responseFallback) model.Custom {
	return model.Custom{
		Kind: fallbackKind,
		ProviderMetadata: model.ProviderMetadata{
			Namespace: FallbackMetadata{
				Type: blockFallback,
				From: block.From,
				To:   block.To,
			},
		},
	}
}

func signatureMetadata(signature string) model.ProviderMetadata {
	return model.ProviderMetadata{
		Namespace: ReasoningMetadata{
			Signature: signature,
		},
	}
}

func redactedMetadata(data string) model.ProviderMetadata {
	return model.ProviderMetadata{
		Namespace: ReasoningMetadata{
			RedactedData: data,
		},
	}
}

func compactionMetadata(signature *string) model.ProviderMetadata {
	return model.ProviderMetadata{
		Namespace: TextMetadata{
			Type:      compactionTextType,
			Signature: valueOrZero(signature),
		},
	}
}

// callerMetadata returns the metadata of a programmatic tool call's caller,
// or nil for calls without one.
func callerMetadata(caller *wireCaller) model.ProviderMetadata {
	if caller == nil {
		return nil
	}

	return model.ProviderMetadata{
		Namespace: ToolCallMetadata{
			Caller: toolCallerOf(caller),
		},
	}
}

func toolCallerOf(caller *wireCaller) *ToolCaller {
	if caller == nil {
		return nil
	}

	return &ToolCaller{
		Type:   caller.Type,
		ToolID: caller.ToolID,
	}
}

// inputObject returns tool input as an object; absent or non-object input
// is an empty object.
func inputObject(input json.RawMessage) jsonobject.Object {
	object, err := jsonobject.Parse(input)
	if err != nil {
		return jsonobject.Object{}
	}

	return object
}

// compactInput returns tool input as compact JSON text. Absent input is an
// empty object.
func compactInput(input json.RawMessage) string {
	if len(input) == 0 || isJSONNull(input) {
		return "{}"
	}

	compact, err := toolresult.CompactJSON(input)
	if err != nil {
		return string(input)
	}

	return compact
}

// nonNullResult returns a result value, using an empty object for null,
// since results must not be null.
func nonNullResult(value json.RawMessage) json.RawMessage {
	if len(value) == 0 || isJSONNull(value) {
		return json.RawMessage("{}")
	}

	return value
}
