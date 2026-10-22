package openaicompatible

import (
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// preparedTools are the wire tools, tool choice and warnings of a call.
type preparedTools struct {
	Tools      []chatTool
	ToolChoice *chatToolChoice
	Warnings   []model.Warning
}

// prepareTools converts function tools. Provider tools are not supported and
// produce warnings.
func prepareTools(tools []model.Tool, toolChoice *model.ToolChoice) (preparedTools, error) {
	var prepared preparedTools
	if len(tools) == 0 {
		return prepared, nil
	}

	prepared.Tools = []chatTool{}
	for _, tool := range tools {
		switch tool := tool.(type) {
		case model.FunctionTool:
			prepared.Tools = append(prepared.Tools, chatTool{
				Type: "function",
				Function: chatToolDefinition{
					Name:        tool.Name,
					Description: tool.Description,
					Parameters:  tool.InputSchema,
					Strict:      tool.Strict,
				},
			})

		case model.ProviderTool:
			prepared.Warnings = append(prepared.Warnings, model.UnsupportedWarning(
				"provider-defined tool "+tool.ID,
				"",
			))

		default:
			return prepared, fmt.Errorf("%w: tool type %T", model.ErrUnsupported, tool)
		}
	}

	if toolChoice == nil {
		return prepared, nil
	}

	choice, err := convertToolChoice(*toolChoice)
	if err != nil {
		return prepared, err
	}

	prepared.ToolChoice = &choice

	return prepared, nil
}

func convertToolChoice(toolChoice model.ToolChoice) (chatToolChoice, error) {
	switch toolChoice.Type {
	case model.ToolChoiceAuto, model.ToolChoiceNone, model.ToolChoiceRequired:
		return chatToolChoice{
			Mode: string(toolChoice.Type),
		}, nil

	case model.ToolChoiceTool:
		return chatToolChoice{
			FunctionName: toolChoice.ToolName,
		}, nil

	default:
		return chatToolChoice{}, fmt.Errorf("%w: tool choice type %q", model.ErrUnsupported, toolChoice.Type)
	}
}
