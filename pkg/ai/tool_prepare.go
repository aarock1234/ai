package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// validateTools reports the first construction error recorded in a tool and
// rejects tools that cannot be declared: missing or duplicate names, unknown
// kinds and provider tools without a provider definition.
func validateTools(tools []Tool) error {
	seen := make(map[string]bool, len(tools))

	for index, tool := range tools {
		if tool.err != nil {
			return tool.err
		}

		if tool.Name == "" {
			return invalidToolsArgument(index, "tool name must not be empty")
		}

		if seen[tool.Name] {
			return invalidToolsArgument(index, fmt.Sprintf("duplicate tool name %q", tool.Name))
		}

		seen[tool.Name] = true

		switch tool.kind() {
		case ToolKindFunction, ToolKindDynamic:
		case ToolKindProvider:
			if tool.Provider == nil {
				return invalidToolsArgument(index, fmt.Sprintf("provider tool %q has no provider definition", tool.Name))
			}
		default:
			return invalidToolsArgument(index, fmt.Sprintf("tool %q has unknown kind %q", tool.Name, tool.Kind))
		}

		if !isCompleteToolCaller(tool.Caller) {
			return invalidToolsArgument(index, fmt.Sprintf("tool %q has a caller without its function", tool.Name))
		}
	}

	return nil
}

// isCompleteToolCaller reports whether a tool caller has the function its
// kind requires. A nil caller is complete.
func isCompleteToolCaller(caller ToolCaller) bool {
	switch typed := caller.(type) {
	case LocalToolCaller:
		return typed.Bind != nil
	case ProviderToolCaller:
		return typed.PrepareProviderOptions != nil
	default:
		return true
	}
}

func invalidToolsArgument(index int, message string) error {
	return &InvalidArgumentError{
		Parameter: "tools",
		Value:     index,
		Message:   message,
	}
}

// FilterActiveTools returns the tools named in activeTools, in declaration
// order. A nil activeTools applies no restriction; an empty, non-nil one
// disables every tool. It is upstream's experimental_filterActiveTools.
func FilterActiveTools(tools []Tool, activeTools []string) []Tool {
	if activeTools == nil {
		return tools
	}

	active := make([]Tool, 0, len(activeTools))
	for _, tool := range tools {
		if slices.Contains(activeTools, tool.Name) {
			active = append(active, tool)
		}
	}

	return active
}

// orderTools applies a tool order: the named tools come first in the given
// order, followed by the remaining tools sorted by name. A nil order keeps
// declaration order.
func orderTools(tools []Tool, order []string) []Tool {
	if order == nil {
		return tools
	}

	listed := make([]Tool, 0, len(order))
	unlisted := make([]Tool, 0, len(tools))

	for _, tool := range tools {
		if slices.Contains(order, tool.Name) {
			listed = append(listed, tool)
		} else {
			unlisted = append(unlisted, tool)
		}
	}

	slices.SortStableFunc(listed, func(first, second Tool) int {
		return slices.Index(order, first.Name) - slices.Index(order, second.Name)
	})

	slices.SortStableFunc(unlisted, func(first, second Tool) int {
		switch {
		case first.Name < second.Name:
			return -1
		case first.Name > second.Name:
			return 1
		default:
			return 0
		}
	})

	return append(listed, unlisted...)
}

// prepareTools converts the tools of a step into model tool declarations,
// in the given tool order; a nil order keeps declaration order. It returns
// nil when there are no tools, so that the call declares none.
func prepareTools(tools []Tool, toolOrder []string) []model.Tool {
	if len(tools) == 0 {
		return nil
	}

	ordered := orderTools(tools, toolOrder)

	declarations := make([]model.Tool, 0, len(ordered))
	for _, tool := range ordered {
		declarations = append(declarations, declareTool(tool))
	}

	return declarations
}

func declareTool(tool Tool) model.Tool {
	if tool.kind() == ToolKindProvider {
		return model.ProviderTool{
			ID:   tool.Provider.ID,
			Name: tool.Name,
			Args: tool.Provider.Args,
		}
	}

	return model.FunctionTool{
		Name:            tool.Name,
		Description:     tool.Description,
		InputSchema:     tool.InputSchema,
		InputExamples:   tool.InputExamples,
		Strict:          tool.Strict,
		ProviderOptions: tool.ProviderOptions,
	}
}

// prepareToolChoice returns the tool choice for a call. Nil selects automatic
// tool choice.
func prepareToolChoice(choice *model.ToolChoice) *model.ToolChoice {
	if choice == nil {
		return &model.ToolChoice{
			Type: model.ToolChoiceAuto,
		}
	}

	prepared := *choice

	return &prepared
}

// findTool returns the tool with the given name.
func findTool(tools []Tool, name string) (Tool, bool) {
	index := slices.IndexFunc(tools, func(tool Tool) bool {
		return tool.Name == name
	})
	if index < 0 {
		return Tool{}, false
	}

	return tools[index], true
}

// toolOutputErrorMode selects how a tool output is reported to the model.
type toolOutputErrorMode string

const (
	toolOutputSucceeded toolOutputErrorMode = "none"
	toolOutputErrorText toolOutputErrorMode = "text"
	toolOutputErrorJSON toolOutputErrorMode = "json"
)

// toolModelOutputRequest is one tool outcome to convert for the model.
type toolModelOutputRequest struct {
	toolCallID string
	input      json.RawMessage
	output     any

	// tool is nil when the tool is unknown, for example for a
	// provider-executed tool.
	tool *Tool

	errorMode toolOutputErrorMode
}

// createToolModelOutput converts a tool output or failure into the result
// sent to the model. Failures become error text or error JSON; successful
// outputs go through the tool's ToModelOutput when set, and otherwise become
// text for strings and JSON for everything else.
func createToolModelOutput(ctx context.Context, request toolModelOutputRequest) (model.ToolResultOutput, error) {
	switch request.errorMode {
	case toolOutputErrorText:
		return model.ErrorTextOutput{
			Value: errorMessage(request.output),
		}, nil

	case toolOutputErrorJSON:
		value, err := encodeToolOutput(request.output)
		if err != nil {
			return nil, err
		}

		return model.ErrorJSONOutput{
			Value: value,
		}, nil
	}

	if request.tool != nil && request.tool.ToModelOutput != nil {
		return request.tool.ToModelOutput(ctx, ToolOutputEvent[json.RawMessage, any]{
			ToolCallID: request.toolCallID,
			Input:      request.input,
			Output:     request.output,
		})
	}

	if text, ok := request.output.(string); ok {
		return model.TextOutput{
			Value: text,
		}, nil
	}

	value, err := encodeToolOutput(request.output)
	if err != nil {
		return nil, err
	}

	return model.JSONOutput{
		Value: value,
	}, nil
}

// encodeToolOutput encodes an output as JSON. Nil encodes as null, and
// errors encode as their message.
func encodeToolOutput(output any) (json.RawMessage, error) {
	if err, ok := output.(error); ok {
		output = err.Error()
	}

	encoded, err := json.Marshal(output)
	if err != nil {
		return nil, fmt.Errorf("encode tool output: %w", err)
	}

	return encoded, nil
}

// errorMessage describes a failure value for the model.
func errorMessage(value any) string {
	switch typed := value.(type) {
	case nil:
		return "unknown error"
	case string:
		return typed
	case error:
		return typed.Error()
	}

	encoded, err := json.Marshal(value)
	if err != nil {
		return fmt.Sprint(value)
	}

	return string(encoded)
}
