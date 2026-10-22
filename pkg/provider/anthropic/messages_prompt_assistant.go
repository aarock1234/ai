package anthropic

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	mcpToolUseType       = "mcp-tool-use"
	compactionTextType   = "compaction"
	fallbackKind         = "anthropic.fallback"
	containerUploadKind  = "anthropic.container_upload"
	messageStartKind     = "anthropic.message_start"
	programmaticCallType = "programmatic-tool-call"
)

// assistantGroup combines consecutive assistant messages into one assistant
// message.
func (c *promptConverter) assistantGroup(group messageGroup, isLastGroup bool) error {
	content := []requestBlock{}
	mcpToolUseIDs := make(map[string]bool)

	for messageIndex, message := range group.messages {
		assistant := message.(model.AssistantMessage)
		isLastMessage := messageIndex == len(group.messages)-1

		for partIndex, part := range assistant.Content {
			isLastPart := partIndex == len(assistant.Content)-1

			cacheControl := c.partCacheControl(
				assistantPartOptions(part),
				cacheAssistantMessagePart,
				assistant.ProviderOptions,
				cacheAssistantMessage,
				isLastPart,
			)

			position := assistantPosition{
				isFinalPart:  isLastGroup && isLastMessage && isLastPart,
				cacheControl: cacheControl,
			}

			block, err := c.assistantPart(part, position, mcpToolUseIDs)
			if err != nil {
				return err
			}

			if block != nil {
				content = append(content, block)
			}
		}
	}

	if len(content) == 0 {
		return nil
	}

	c.result.messages = append(c.result.messages, requestMessage{
		Role:    roleAssistant,
		Content: moveToolUseBlocksToEnd(content),
	})

	return nil
}

// assistantPosition describes where an assistant part sits.
type assistantPosition struct {
	// isFinalPart marks the last part of the prompt's final assistant
	// message.
	isFinalPart bool

	cacheControl *CacheControl
}

func assistantPartOptions(part model.AssistantPart) model.ProviderOptions {
	switch typed := part.(type) {
	case model.TextPart:
		return typed.ProviderOptions
	case model.FilePart:
		return typed.ProviderOptions
	case model.CustomPart:
		return typed.ProviderOptions
	case model.ReasoningPart:
		return typed.ProviderOptions
	case model.ReasoningFilePart:
		return typed.ProviderOptions
	case model.ToolCallPart:
		return typed.ProviderOptions
	case model.ToolResultPart:
		return typed.ProviderOptions
	default:
		return nil
	}
}

// assistantPart converts one assistant part. It returns nil for parts that
// are dropped; file parts have no assistant representation.
func (c *promptConverter) assistantPart(
	part model.AssistantPart,
	position assistantPosition,
	mcpToolUseIDs map[string]bool,
) (requestBlock, error) {
	switch typed := part.(type) {
	case model.TextPart:
		return c.assistantText(typed, position)
	case model.ReasoningPart:
		return c.assistantReasoning(typed)
	case model.CustomPart:
		return c.assistantCustom(typed)
	case model.ToolCallPart:
		return c.assistantToolCall(typed, position.cacheControl, mcpToolUseIDs)
	case model.ToolResultPart:
		return c.assistantToolResult(typed, position.cacheControl, mcpToolUseIDs)
	default:
		return nil, nil
	}
}

// assistantText replays text, or a compaction summary when the text's
// metadata marks it as one.
func (c *promptConverter) assistantText(part model.TextPart, position assistantPosition) (requestBlock, error) {
	metadata, _, err := model.ProviderValue[TextMetadata](part.ProviderOptions, Namespace)
	if err != nil {
		return nil, err
	}

	if metadata.Type != compactionTextType {
		return &textBlock{
			Type:         blockText,
			Text:         trimTrailingAssistantText(part.Text, position.isFinalPart),
			Citations:    metadata.Citations,
			CacheControl: position.cacheControl,
		}, nil
	}

	if part.Text == "" {
		return nil, nil
	}

	if metadata.Signature != "" {
		c.result.betas.add("compact-2026-09-04")
	}

	return &compactionBlock{
		Type:         blockCompaction,
		Content:      part.Text,
		Signature:    metadata.Signature,
		CacheControl: position.cacheControl,
	}, nil
}

// assistantReasoning replays a thinking or redacted thinking block from its
// metadata. Thinking blocks cannot carry cache breakpoints.
func (c *promptConverter) assistantReasoning(part model.ReasoningPart) (requestBlock, error) {
	if !c.sendReasoning {
		c.warnings.other("sending reasoning content is disabled for this model")

		return nil, nil
	}

	metadata, found, err := model.ProviderValue[ReasoningMetadata](part.ProviderOptions, Namespace)
	if err != nil {
		return nil, err
	}

	switch {
	case found && metadata.Signature != "":
		c.cache.cacheControl(part.ProviderOptions, cacheThinkingBlock)

		return &thinkingBlock{
			Type:      blockThinking,
			Thinking:  part.Text,
			Signature: metadata.Signature,
		}, nil

	case found && metadata.RedactedData != "":
		c.cache.cacheControl(part.ProviderOptions, cacheRedactedThinkingBlock)

		return &redactedThinkingBlock{
			Type: blockRedactedThinking,
			Data: metadata.RedactedData,
		}, nil

	default:
		c.warnings.other("unsupported reasoning metadata")

		return nil, nil
	}
}

// assistantCustom replays a server-side fallback marker.
func (c *promptConverter) assistantCustom(part model.CustomPart) (requestBlock, error) {
	if part.Kind != fallbackKind {
		return nil, nil
	}

	metadata, _, err := model.ProviderValue[FallbackMetadata](part.ProviderOptions, Namespace)
	if err != nil {
		return nil, err
	}

	isComplete := metadata.Type == blockFallback && metadata.From.Model != "" && metadata.To.Model != ""
	if !isComplete {
		c.warnings.other("anthropic fallback metadata must include from.model and to.model")

		return nil, nil
	}

	c.result.betas.add("server-side-fallback-2026-06-01")

	return &fallbackBlock{
		Type: blockFallback,
		From: metadata.From,
		To:   metadata.To,
	}, nil
}

func (c *promptConverter) assistantToolCall(
	part model.ToolCallPart,
	cacheControl *CacheControl,
	mcpToolUseIDs map[string]bool,
) (requestBlock, error) {
	metadata, _, err := model.ProviderValue[ToolCallMetadata](part.ProviderOptions, Namespace)
	if err != nil {
		return nil, err
	}

	caller := wireCallerOf(metadata.Caller)

	if part.ProviderExecuted {
		return c.serverToolCall(part, metadata, caller, cacheControl, mcpToolUseIDs), nil
	}

	toolsetName, err := c.toolsetName(part.ToolName, part.ProviderOptions)
	if err != nil {
		return nil, err
	}

	if toolsetName == "" {
		return &toolUseBlock{
			Type:         blockToolUse,
			ID:           part.ToolCallID,
			Name:         part.ToolName,
			Input:        toolInput(part.Input),
			Caller:       caller,
			CacheControl: cacheControl,
		}, nil
	}

	// A toolset member call names the member in the input's action.
	input := toolInput(part.Input)
	object, err := jsonobject.Parse(input)
	if err != nil {
		return nil, fmt.Errorf("%w: tool call input: %w", model.ErrInvalidPrompt, err)
	}

	memberName, hasAction := object.GetString("action")
	if !hasAction {
		c.warnings.other(fmt.Sprintf("toolset tool call for tool %s is missing the action", part.ToolName))
		memberName = toolsetName
	}

	return &toolUseBlock{
		Type:         blockToolUse,
		ID:           part.ToolCallID,
		Name:         memberName,
		ToolsetName:  toolsetName,
		Input:        input,
		Caller:       caller,
		CacheControl: cacheControl,
	}, nil
}

// serverToolCall replays a provider-executed tool call. Code execution
// subtool calls carry their subtool in the input's type, which is removed
// again for the API.
func (c *promptConverter) serverToolCall(
	part model.ToolCallPart,
	metadata ToolCallMetadata,
	caller *wireCaller,
	cacheControl *CacheControl,
	mcpToolUseIDs map[string]bool,
) requestBlock {
	if metadata.Type == mcpToolUseType {
		mcpToolUseIDs[part.ToolCallID] = true

		if metadata.ServerName == "" {
			c.warnings.other("mcp tool use server name is required and must be a string")

			return nil
		}

		return &mcpToolUseBlock{
			Type:         blockMCPToolUse,
			ID:           part.ToolCallID,
			Name:         part.ToolName,
			Input:        toolInput(part.Input),
			ServerName:   metadata.ServerName,
			CacheControl: cacheControl,
		}
	}

	providerName := c.toolNames.ToProvider(part.ToolName)
	serverToolUse := func(name string, input json.RawMessage) requestBlock {
		return &serverToolUseBlock{
			Type:         blockServerToolUse,
			ID:           part.ToolCallID,
			Name:         name,
			Input:        input,
			Caller:       caller,
			CacheControl: cacheControl,
		}
	}

	if providerName == nameCodeExecution {
		if name, input, found := codeExecutionSubtool(part.Input); found {
			return serverToolUse(name, input)
		}
	}

	switch providerName {
	case nameCodeExecution, nameWebFetch, nameWebSearch, nameToolSearchRegex, nameToolSearchBM25:
		return serverToolUse(providerName, toolInput(part.Input))

	case nameAdvisor:
		// The advisor's input is always empty.
		return serverToolUse(nameAdvisor, json.RawMessage("{}"))

	default:
		c.warnings.other(fmt.Sprintf("provider executed tool call for tool %s is not supported", part.ToolName))

		return nil
	}
}

// codeExecutionSubtool returns the server tool name and API input of a code
// execution call whose input type names a subtool or a programmatic call.
// It reports false for other input, which is replayed unchanged.
func codeExecutionSubtool(raw json.RawMessage) (string, json.RawMessage, bool) {
	input, found := parseJSONObjectOrNil(raw)
	if !found {
		return "", nil, false
	}

	inputType, _ := input.GetString("type")

	var name string

	switch inputType {
	case nameBashCodeExecution, nameTextEditorCodeExecution:
		name = inputType
	case programmaticCallType:
		name = nameCodeExecution
	default:
		return "", nil, false
	}

	untyped, err := input.Without("type").Encode()
	if err != nil {
		return "", nil, false
	}

	return name, untyped, true
}

// parseJSONObjectOrNil parses raw as an object, reporting false for other
// values.
func parseJSONObjectOrNil(raw json.RawMessage) (jsonobject.Object, bool) {
	object, err := jsonobject.Parse(raw)
	if err != nil {
		return nil, false
	}

	return object, true
}

// wireCallerOf converts caller metadata. Code execution callers need the ID
// of their code execution call.
func wireCallerOf(caller *ToolCaller) *wireCaller {
	if caller == nil {
		return nil
	}

	switch caller.Type {
	case CallerCodeExecution20250825, CallerCodeExecution20260120:
		if caller.ToolID == "" {
			return nil
		}

		return &wireCaller{
			Type:   caller.Type,
			ToolID: caller.ToolID,
		}

	case CallerDirect:
		return &wireCaller{
			Type: CallerDirect,
		}

	default:
		return nil
	}
}

// moveToolUseBlocksToEnd moves tool_use blocks after the other blocks of
// their segment. Thinking, redacted thinking and fallback blocks delimit
// segments and keep their positions.
func moveToolUseBlocksToEnd(content []requestBlock) []requestBlock {
	result := make([]requestBlock, 0, len(content))
	var others []requestBlock
	var toolUses []requestBlock

	flush := func() {
		result = append(result, others...)
		result = append(result, toolUses...)
		others = nil
		toolUses = nil
	}

	for _, block := range content {
		switch block.(type) {
		case *thinkingBlock, *redactedThinkingBlock, *fallbackBlock:
			flush()
			result = append(result, block)
		case *toolUseBlock:
			toolUses = append(toolUses, block)
		default:
			others = append(others, block)
		}
	}

	flush()

	return result
}

// removeOrphanedCallers removes caller metadata that refers to a code
// execution call no longer in the history, which pruning can cause. Only
// history before the last user message changes: tool results do not end a
// turn, and an active code execution needs its callers intact.
func (c *promptConverter) removeOrphanedCallers() {
	codeExecutionIDs := make(map[string]bool)

	for _, message := range c.result.messages {
		if message.Role != roleAssistant {
			continue
		}

		for _, block := range message.Content {
			serverToolUse, ok := block.(*serverToolUseBlock)
			if ok && serverToolUse.Name == nameCodeExecution {
				codeExecutionIDs[serverToolUse.ID] = true
			}
		}
	}

	warned := make(map[string]bool)

	for index := 0; index < c.lastUserMessageIndex; index++ {
		message := c.result.messages[index]
		if message.Role != roleAssistant {
			continue
		}

		for _, block := range message.Content {
			caller, toolCallID := blockCaller(block)
			if caller == nil || *caller == nil {
				continue
			}

			current := *caller
			if current.Type == CallerDirect || codeExecutionIDs[current.ToolID] {
				continue
			}

			*caller = nil

			if warned[toolCallID] {
				continue
			}

			warned[toolCallID] = true
			c.warnings.other(fmt.Sprintf(
				"Omitted caller metadata for tool %s because source code execution tool %s is missing from the conversation history.",
				toolCallID,
				current.ToolID,
			))
		}
	}
}

// blockCaller returns the caller field of blocks that have one, with the ID
// of the tool call the block belongs to.
func blockCaller(block requestBlock) (**wireCaller, string) {
	switch typed := block.(type) {
	case *toolUseBlock:
		return &typed.Caller, typed.ID
	case *serverToolUseBlock:
		return &typed.Caller, typed.ID
	case *serverToolResultBlock:
		return &typed.Caller, typed.ToolUseID
	default:
		return nil, ""
	}
}
