package harness

import (
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// mergeTools returns the built-in tools followed by the application tools.
// An application tool replaces the built-in tool of the same name in place.
func mergeTools(builtins []BuiltinTool, userTools []ai.Tool) []ai.Tool {
	merged := make([]ai.Tool, 0, len(builtins)+len(userTools))

	for _, builtin := range builtins {
		if override, found := findTool(userTools, builtin.Name); found {
			merged = append(merged, *override)

			continue
		}

		merged = append(merged, builtin.Tool)
	}

	for _, tool := range userTools {
		if !hasTool(merged, tool.Name) {
			merged = append(merged, tool)
		}
	}

	return merged
}

// toolFilteringInput describes the tool filtering of a turn.
type toolFilteringInput struct {
	harness       Harness
	userTools     []ai.Tool
	allTools      []ai.Tool
	activeTools   []string
	inactiveTools []string
}

// resolvedToolFiltering is the application tools the runtime may call and
// the filtering of its built-in tools.
type resolvedToolFiltering struct {
	activeUserTools      []ai.Tool
	builtinToolFiltering *BuiltinToolFiltering
}

// resolveToolFiltering applies ActiveTools or InactiveTools to the
// application and built-in tools.
func resolveToolFiltering(input toolFilteringInput) (resolvedToolFiltering, error) {
	if input.activeTools != nil && input.inactiveTools != nil {
		return resolvedToolFiltering{}, errors.New("harness agent: pass either ActiveTools or InactiveTools, not both")
	}

	activeTools := dedupeNames(input.activeTools)
	inactiveTools := dedupeNames(input.inactiveTools)

	requested := activeTools
	if requested == nil {
		requested = inactiveTools
	}

	if err := validateToolNames(requested, input.allTools); err != nil {
		return resolvedToolFiltering{}, err
	}

	isActive := func(name string) bool {
		switch {
		case activeTools != nil:
			return slices.Contains(activeTools, name)
		case inactiveTools != nil:
			return !slices.Contains(inactiveTools, name)
		default:
			return true
		}
	}

	var activeUserTools []ai.Tool
	for _, tool := range input.userTools {
		if isActive(tool.Name) {
			activeUserTools = append(activeUserTools, tool)
		}
	}

	filtering := builtinFiltering(input.harness, activeTools, isActive)

	if filtering != nil && !supportsBuiltinToolFiltering(input.harness) && !supportsBuiltinToolApprovals(input.harness) {
		return resolvedToolFiltering{}, &CapabilityUnsupportedError{
			Message:   fmt.Sprintf("Harness '%s' does not support built-in tool filtering controls.", input.harness.HarnessID()),
			HarnessID: input.harness.HarnessID(),
		}
	}

	return resolvedToolFiltering{
		activeUserTools:      activeUserTools,
		builtinToolFiltering: filtering,
	}, nil
}

// builtinFiltering returns the filtering of the built-in tools, or nil when
// every built-in tool stays active. Active tool lists become allow lists and
// inactive tool lists deny lists.
func builtinFiltering(harness Harness, activeTools []string, isActive func(name string) bool) *BuiltinToolFiltering {
	var kept []string
	var disabled []string

	for _, tool := range harness.BuiltinTools() {
		if isActive(tool.Name) {
			kept = append(kept, tool.Name)
		} else {
			disabled = append(disabled, tool.Name)
		}
	}

	if len(disabled) == 0 {
		return nil
	}

	if activeTools != nil {
		return &BuiltinToolFiltering{
			Mode:      FilterAllow,
			ToolNames: nonNil(kept),
		}
	}

	return &BuiltinToolFiltering{
		Mode:      FilterDeny,
		ToolNames: disabled,
	}
}

// validateToolNames fails with *ai.NoSuchToolError for unknown names.
func validateToolNames(names []string, tools []ai.Tool) error {
	for _, name := range names {
		if !hasTool(tools, name) {
			return &ai.NoSuchToolError{
				ToolName:       name,
				AvailableTools: toolNames(tools),
			}
		}
	}

	return nil
}

// toolSpecsOf describes application tools for the runtime. Built-in tools
// are known to the runtime and never declared.
func toolSpecsOf(tools []ai.Tool) []ToolSpec {
	specs := make([]ToolSpec, 0, len(tools))

	for _, tool := range tools {
		specs = append(specs, ToolSpec{
			Name:        tool.Name,
			Description: tool.Description,
			InputSchema: tool.InputSchema,
		})
	}

	return specs
}

// supportsBuiltinToolApprovals reports the optional harness capability.
func supportsBuiltinToolApprovals(harness Harness) bool {
	supporter, ok := harness.(BuiltinToolApprovalSupporter)

	return ok && supporter.SupportsBuiltinToolApprovals()
}

// supportsBuiltinToolFiltering reports the optional harness capability.
func supportsBuiltinToolFiltering(harness Harness) bool {
	supporter, ok := harness.(BuiltinToolFilteringSupporter)

	return ok && supporter.SupportsBuiltinToolFiltering()
}

// resolvePermissionMode applies the default permission mode.
func resolvePermissionMode(mode PermissionMode) PermissionMode {
	if mode == "" {
		return PermissionAllowAll
	}

	return mode
}

// normalizeHeaders copies the configured headers, rejecting the headers the
// harness manages. It returns nil when there are none.
func normalizeHeaders(headers http.Header) (http.Header, error) {
	if len(headers) == 0 {
		return nil, nil
	}

	for name := range headers {
		if slices.Contains(managedHeaders, strings.ToLower(name)) {
			return nil, fmt.Errorf("harness agent: headers must not include the managed header %s", strings.ToLower(name))
		}
	}

	return headers.Clone(), nil
}

// continuationOf returns the host input of a trailing tool message, or nil
// when the call starts a fresh turn.
func continuationOf(prompt string, messages []model.Message) (*turnContinuation, error) {
	if prompt != "" || len(messages) == 0 {
		return nil, nil
	}

	approvals, err := CollectToolApprovalContinuations(messages)
	if err != nil {
		return nil, err
	}

	results := CollectToolResultContinuations(messages)

	if len(approvals) == 0 && len(results) == 0 {
		return nil, nil
	}

	return &turnContinuation{
		approvals: approvals,
		results:   results,
	}, nil
}

// promptOf reduces a call's input to the user message of the turn: a text
// prompt, or the last user message. The session owns earlier turns, so they
// are never replayed.
func promptOf(prompt string, messages []model.Message) (model.UserMessage, error) {
	if prompt != "" {
		return TextPrompt(prompt), nil
	}

	if len(messages) == 0 {
		return model.UserMessage{}, errors.New("harness agent: either a prompt or messages are required")
	}

	for index := len(messages) - 1; index >= 0; index-- {
		switch typed := messages[index].(type) {
		case model.UserMessage:
			return typed, nil
		case *model.UserMessage:
			if typed != nil {
				return *typed, nil
			}
		}
	}

	return model.UserMessage{}, errors.New("harness agent: messages must contain at least one user message")
}

// dedupeNames removes repeated names, keeping nil as unset.
func dedupeNames(names []string) []string {
	if names == nil {
		return nil
	}

	unique := make([]string, 0, len(names))
	for _, name := range names {
		if !slices.Contains(unique, name) {
			unique = append(unique, name)
		}
	}

	return unique
}

// nonNil returns an empty slice instead of nil.
func nonNil(values []string) []string {
	if values == nil {
		return []string{}
	}

	return values
}
