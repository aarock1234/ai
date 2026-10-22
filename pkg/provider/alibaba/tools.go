package alibaba

import (
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// preparedTools are the wire tools, tool choice and warnings of a call.
type preparedTools struct {
	Tools      []tool
	ToolChoice *toolChoice
	Warnings   []model.Warning
}

// prepareTools converts the call's function tools. Alibaba has no provider
// tools.
func prepareTools(tools []model.Tool, choice *model.ToolChoice) (preparedTools, error) {
	var prepared preparedTools
	if len(tools) == 0 {
		return prepared, nil
	}

	prepared.Tools = []tool{}
	for _, definition := range tools {
		switch definition := definition.(type) {
		case model.FunctionTool:
			prepared.Tools = append(prepared.Tools, tool{
				Type: functionType,
				Function: toolFunction{
					Name:        definition.Name,
					Description: definition.Description,
					Parameters:  definition.InputSchema,
					Strict:      definition.Strict,
				},
			})

		case model.ProviderTool:
			prepared.Warnings = append(prepared.Warnings, model.UnsupportedWarning("provider-defined tool "+definition.ID, ""))

		default:
			return prepared, fmt.Errorf("%w: tool type %T", model.ErrUnsupported, definition)
		}
	}

	if choice == nil {
		return prepared, nil
	}

	switch choice.Type {
	case model.ToolChoiceAuto, model.ToolChoiceNone, model.ToolChoiceRequired:
		prepared.ToolChoice = &toolChoice{
			Mode: string(choice.Type),
		}

	case model.ToolChoiceTool:
		prepared.ToolChoice = &toolChoice{
			FunctionName: choice.ToolName,
		}

	default:
		return prepared, fmt.Errorf("%w: tool choice type %q", model.ErrUnsupported, choice.Type)
	}

	return prepared, nil
}
