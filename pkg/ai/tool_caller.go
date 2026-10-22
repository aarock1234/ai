package ai

import (
	"fmt"
	"maps"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// DirectToolCall is the caller name, in ToolCallers, that lets the model call
// a tool directly.
const DirectToolCall = "AI_SDK_DIRECT_TOOL_CALL"

// ToolCaller lets a tool call other tools programmatically, for example a
// code execution tool that calls tools from the code it runs. It is a
// LocalToolCaller or a ProviderToolCaller. Set it as Tool.Caller and list
// the caller's name for the tools it may call in ToolCallers.
type ToolCaller interface {
	isToolCaller()
}

// LocalToolCaller is a caller that the application runs.
type LocalToolCaller struct {
	// Bind returns the caller tool bound to the tools it may call. It is
	// called for every step; the returned tool keeps the caller's name.
	Bind func(tools []Tool) Tool

	// PrepareModelMessage optionally describes the tools available through
	// the caller. When set, the model sees the unbound caller tool, so that
	// its declaration stays stable, and the returned text is added to the
	// conversation as a user message unless it was already announced. An
	// empty text adds nothing.
	PrepareModelMessage func(tools []Tool) string
}

// ProviderToolCaller is a provider-executed caller, such as a provider's
// code execution tool that calls client tools.
type ProviderToolCaller struct {
	// PrepareProviderOptions returns the provider options of a tool that
	// the provider may call through this caller.
	PrepareProviderOptions func(options model.ProviderOptions) model.ProviderOptions
}

func (LocalToolCaller) isToolCaller()    {}
func (ProviderToolCaller) isToolCaller() {}

// resolvedToolCallers maps tool names to their validated callers. A tool
// without an entry may be called directly; an empty entry permits no
// caller.
type resolvedToolCallers map[string][]string

// resolveToolCallers validates the ToolCallers option against the tools.
// Every named tool must exist, so ToolCallers without tools is an error.
func resolveToolCallers(tools []Tool, toolCallers map[string][]string) (resolvedToolCallers, error) {
	if toolCallers == nil {
		return nil, nil
	}

	if err := validateCallerFunctions(tools); err != nil {
		return nil, err
	}

	resolved := make(resolvedToolCallers, len(toolCallers))

	for _, toolName := range slices.Sorted(maps.Keys(toolCallers)) {
		if _, found := findTool(tools, toolName); !found {
			return nil, invalidToolCallers(fmt.Sprintf("unknown tool %q", toolName))
		}

		callers := toolCallers[toolName]

		for _, caller := range callers {
			if !isValidToolCaller(tools, caller) {
				return nil, invalidToolCallers(fmt.Sprintf("tool %q contains an invalid caller", toolName))
			}
		}

		resolved[toolName] = slices.Clone(callers)
	}

	return resolved, nil
}

// isValidToolCaller reports whether a caller name is DirectToolCall or the
// name of a tool with a Caller.
func isValidToolCaller(tools []Tool, caller string) bool {
	if caller == DirectToolCall {
		return true
	}

	tool, found := findTool(tools, caller)

	return found && tool.Caller != nil
}

// validateCallerFunctions rejects callers without the function that routes
// tools through them.
func validateCallerFunctions(tools []Tool) error {
	for _, tool := range tools {
		if !hasCallerFunction(tool.Caller) {
			return &InvalidArgumentError{
				Parameter: "tools." + tool.Name + ".caller",
				Message:   "local callers need Bind and provider callers need PrepareProviderOptions",
			}
		}
	}

	return nil
}

// hasCallerFunction reports whether a caller, if any, has the function its
// kind requires.
func hasCallerFunction(caller ToolCaller) bool {
	switch typed := caller.(type) {
	case LocalToolCaller:
		return typed.Bind != nil
	case ProviderToolCaller:
		return typed.PrepareProviderOptions != nil
	default:
		return true
	}
}

func invalidToolCallers(message string) error {
	return &InvalidArgumentError{
		Parameter: "toolCallers",
		Message:   message,
	}
}

// callersOf returns the callers of a tool. A tool without an entry may only
// be called directly.
func (c resolvedToolCallers) callersOf(toolName string) []string {
	if callers, ok := c[toolName]; ok {
		return callers
	}

	return []string{
		DirectToolCall,
	}
}

// toolCallerPreparation is the outcome of applying the tool callers to the
// tools of a step.
type toolCallerPreparation struct {
	// executionTools are the tools available for execution, with local
	// callers bound to the tools they may call.
	executionTools []Tool

	// modelTools are the tools declared to the model.
	modelTools []Tool

	// messages describe the tools available through local callers.
	messages []model.UserMessage
}

// prepareToolsForToolCallers routes the tools of a step through their
// callers: tools that are only callable through a local caller are hidden
// from the model, tools callable by a provider caller receive its provider
// options, and local callers are bound to their tools.
func prepareToolsForToolCallers(tools []Tool, callers resolvedToolCallers) toolCallerPreparation {
	if callers == nil {
		return toolCallerPreparation{
			executionTools: tools,
			modelTools:     tools,
		}
	}

	routing := routeToolCalls(tools, callers)
	bound := bindLocalCallers(routing)

	modelTools := make([]Tool, 0, len(tools))

	for _, tool := range routing.prepared {
		if routing.hidden[tool.Name] {
			continue
		}

		if replacement, ok := bound.modelReplacements[tool.Name]; ok {
			tool = replacement
		}

		modelTools = append(modelTools, tool)
	}

	return toolCallerPreparation{
		executionTools: bound.executionTools,
		modelTools:     modelTools,
		messages:       bound.messages,
	}
}

// toolCallRouting records how the tools of a step may be called.
type toolCallRouting struct {
	// prepared are the tools with the provider options of their provider
	// callers, in declaration order.
	prepared []Tool

	// hidden holds the tools the model may not call directly.
	hidden map[string]bool

	// localTools are the tools of each local caller, in declaration order.
	localTools map[string][]Tool
}

func routeToolCalls(tools []Tool, callers resolvedToolCallers) toolCallRouting {
	routing := toolCallRouting{
		prepared:   slices.Clone(tools),
		hidden:     make(map[string]bool),
		localTools: make(map[string][]Tool),
	}

	for index, tool := range routing.prepared {
		callerNames, configured := callers[tool.Name]
		if !configured {
			continue
		}

		availableDirectly := false
		availableToProvider := false

		for _, callerName := range callerNames {
			if callerName == DirectToolCall {
				availableDirectly = true

				continue
			}

			callerTool, found := findTool(routing.prepared, callerName)
			if !found {
				continue
			}

			switch caller := callerTool.Caller.(type) {
			case ProviderToolCaller:
				availableToProvider = true
				tool.ProviderOptions = caller.PrepareProviderOptions(tool.ProviderOptions)

			case LocalToolCaller:
				routing.localTools[callerName] = append(routing.localTools[callerName], tool)
			}
		}

		routing.prepared[index] = tool

		if !availableDirectly && !availableToProvider {
			routing.hidden[tool.Name] = true
		}
	}

	return routing
}

// boundToolCallers holds the local callers bound to their tools.
type boundToolCallers struct {
	executionTools []Tool

	// modelReplacements are bound callers that the model sees in place of
	// the unbound caller.
	modelReplacements map[string]Tool

	messages []model.UserMessage
}

func bindLocalCallers(routing toolCallRouting) boundToolCallers {
	bound := boundToolCallers{
		executionTools:    slices.Clone(routing.prepared),
		modelReplacements: make(map[string]Tool),
	}

	for index, callerTool := range bound.executionTools {
		caller, isLocal := callerTool.Caller.(LocalToolCaller)
		if !isLocal {
			continue
		}

		callable := routing.localTools[callerTool.Name]

		boundCaller := caller.Bind(callable)
		boundCaller.Name = callerTool.Name
		bound.executionTools[index] = boundCaller

		if routing.hidden[callerTool.Name] {
			continue
		}

		if caller.PrepareModelMessage == nil {
			bound.modelReplacements[callerTool.Name] = boundCaller

			continue
		}

		if text := caller.PrepareModelMessage(callable); text != "" {
			bound.messages = append(bound.messages, textUserMessage(text))
		}
	}

	return bound
}

// appendToolCallerMessages appends the caller messages that the
// conversation has not announced yet. A message counts as announced when it
// repeats the latest text-only user message or an earlier addition.
func appendToolCallerMessages(messages []model.Message, additions []model.UserMessage) []model.Message {
	if len(additions) == 0 {
		return messages
	}

	announced := make(map[string]bool)
	if latest, ok := latestUserText(messages); ok {
		announced[latest] = true
	}

	var appended []model.Message

	for _, addition := range additions {
		text, ok := userMessageText(addition)
		if !ok || announced[text] {
			continue
		}

		announced[text] = true
		appended = append(appended, addition)
	}

	if len(appended) == 0 {
		return messages
	}

	return slices.Concat(messages, appended)
}

// latestUserText returns the text of the last text-only user message.
func latestUserText(messages []model.Message) (string, bool) {
	for _, message := range slices.Backward(messages) {
		user, ok := message.(model.UserMessage)
		if !ok {
			continue
		}

		if text, isText := userMessageText(user); isText {
			return text, true
		}
	}

	return "", false
}

// userMessageText returns the text of a user message that consists of a
// single text part, the Go form of upstream's string content.
func userMessageText(message model.UserMessage) (string, bool) {
	if len(message.Content) != 1 {
		return "", false
	}

	text, ok := message.Content[0].(model.TextPart)
	if !ok {
		return "", false
	}

	return text.Text, true
}
