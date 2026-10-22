package cohere

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// toolChoice is a Cohere tool choice. Empty lets the model decide.
type toolChoice string

const (
	toolChoiceNone     toolChoice = "NONE"
	toolChoiceRequired toolChoice = "REQUIRED"
)

type tool struct {
	Type     string       `json:"type"`
	Function toolFunction `json:"function"`
}

type toolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
}

// preparedTools are the wire tools, tool choice and warnings of a call.
type preparedTools struct {
	Tools      []tool
	ToolChoice toolChoice
	Warnings   []model.Warning
}

// prepareTools converts function tools. Cohere cannot force a specific tool,
// so a tool choice keeps only that tool and requires a call.
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
				Type: "function",
				Function: toolFunction{
					Name:        definition.Name,
					Description: definition.Description,
					Parameters:  definition.InputSchema,
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
	case model.ToolChoiceAuto:
		// Letting the model decide is Cohere's default.

	case model.ToolChoiceNone:
		prepared.ToolChoice = toolChoiceNone

	case model.ToolChoiceRequired:
		prepared.ToolChoice = toolChoiceRequired

	case model.ToolChoiceTool:
		prepared.Tools = onlyTool(prepared.Tools, choice.ToolName)
		prepared.ToolChoice = toolChoiceRequired

	default:
		return prepared, fmt.Errorf("%w: tool choice type %q", model.ErrUnsupported, choice.Type)
	}

	return prepared, nil
}

func onlyTool(tools []tool, name string) []tool {
	kept := []tool{}
	for _, candidate := range tools {
		if candidate.Function.Name == name {
			kept = append(kept, candidate)
		}
	}

	return kept
}
