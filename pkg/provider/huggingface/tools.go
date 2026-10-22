package huggingface

import (
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// preparedTools are the tools and tool choice of a request.
type preparedTools struct {
	tools      []functionTool
	toolChoice *toolChoice
	warnings   []model.Warning
}

// prepareTools converts function tools and the tool choice. Provider tools
// are not supported, and tool choice "none" is ignored.
func prepareTools(tools []model.Tool, choice *model.ToolChoice) (preparedTools, error) {
	if len(tools) == 0 {
		return preparedTools{}, nil
	}

	var prepared preparedTools

	for _, tool := range tools {
		switch tool := tool.(type) {
		case model.FunctionTool:
			prepared.tools = append(prepared.tools, functionTool{
				Type:        "function",
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.InputSchema,
			})

		case model.ProviderTool:
			prepared.warnings = append(prepared.warnings, model.UnsupportedWarning("provider-defined tool "+tool.ID, ""))
		}
	}

	if choice == nil {
		return prepared, nil
	}

	switch choice.Type {
	case model.ToolChoiceAuto, model.ToolChoiceRequired:
		prepared.toolChoice = &toolChoice{
			mode: string(choice.Type),
		}

	case model.ToolChoiceNone:
		// The Responses API has no "none" choice; the tools are still sent.

	case model.ToolChoiceTool:
		prepared.toolChoice = &toolChoice{
			function: choice.ToolName,
		}

	default:
		return preparedTools{}, fmt.Errorf("%w: tool choice type %q", model.ErrUnsupported, choice.Type)
	}

	return prepared, nil
}
