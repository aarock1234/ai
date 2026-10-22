package deepseek

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

// prepareTools converts the call's function tools. Strict tool calls need
// the beta API, and in strict mode every function tool must be strict.
func prepareTools(tools []model.Tool, choice *model.ToolChoice, supportsStrictToolCalls bool) (preparedTools, error) {
	var prepared preparedTools
	if len(tools) == 0 {
		return prepared, nil
	}

	if err := validateStrictTools(tools, supportsStrictToolCalls); err != nil {
		return prepared, err
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
		prepared.Warnings = append(prepared.Warnings, model.UnsupportedWarning(fmt.Sprintf("tool choice type: %s", choice.Type), ""))
	}

	return prepared, nil
}

// validateStrictTools rejects strict tools outside the beta API and a mix of
// strict and non-strict tools.
func validateStrictTools(tools []model.Tool, supportsStrictToolCalls bool) error {
	hasStrict := false
	hasNonStrict := false

	for _, definition := range tools {
		function, ok := definition.(model.FunctionTool)
		if !ok {
			continue
		}

		if function.Strict != nil && *function.Strict {
			hasStrict = true
		} else {
			hasNonStrict = true
		}
	}

	if !hasStrict {
		return nil
	}

	if !supportsStrictToolCalls {
		return fmt.Errorf("%w: deepseek strict tool calls require a beta base URL ending in /beta", model.ErrUnsupported)
	}

	if hasNonStrict {
		return fmt.Errorf(
			"%w: deepseek strict mode requires every function tool in the request to set strict: true",
			model.ErrUnsupported,
		)
	}

	return nil
}
