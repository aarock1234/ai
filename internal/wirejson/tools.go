package wirejson

import (
	"encoding/json"

	"github.com/aarock1234/ai/pkg/model"
)

// ToolDefinition is an encoded tool declaration. Function tools have a
// name, description and input schema; provider tools have an ID, name and
// arguments; extension tools, which the provider executed without a
// declaration, have only a name.
type ToolDefinition struct {
	Type            string                `json:"type"`
	ID              string                `json:"id,omitempty"`
	Name            string                `json:"name"`
	Description     string                `json:"description,omitempty"`
	InputSchema     json.RawMessage       `json:"inputSchema,omitempty"`
	InputExamples   []ToolInputExample    `json:"inputExamples,omitempty"`
	Strict          *bool                 `json:"strict,omitempty"`
	Args            json.RawMessage       `json:"args,omitempty"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

// ToolInputExample is an example input of a function tool.
type ToolInputExample struct {
	Input json.RawMessage `json:"input"`
}

// ToolChoice is an encoded tool choice.
type ToolChoice struct {
	Type     model.ToolChoiceType `json:"type"`
	ToolName string               `json:"toolName,omitempty"`
}

// EncodeTools encodes the tools sent to a model.
func EncodeTools(tools []model.Tool) []ToolDefinition {
	definitions := make([]ToolDefinition, 0, len(tools))

	for _, tool := range tools {
		definitions = append(definitions, toolDefinition(tool))
	}

	return definitions
}

func toolDefinition(tool model.Tool) ToolDefinition {
	switch typed := tool.(type) {
	case model.FunctionTool:
		return ToolDefinition{
			Type:            "function",
			Name:            typed.Name,
			Description:     typed.Description,
			InputSchema:     typed.InputSchema,
			InputExamples:   toolInputExamples(typed.InputExamples),
			Strict:          typed.Strict,
			ProviderOptions: typed.ProviderOptions,
		}

	case model.ProviderTool:
		return ToolDefinition{
			Type: "provider",
			ID:   typed.ID,
			Name: typed.Name,
			Args: typed.Args,
		}

	case nil:
		return ToolDefinition{
			Type: "unknown",
		}

	default:
		return ToolDefinition{
			Type: "unknown",
			Name: tool.ToolName(),
		}
	}
}

func toolInputExamples(examples []json.RawMessage) []ToolInputExample {
	if len(examples) == 0 {
		return nil
	}

	converted := make([]ToolInputExample, 0, len(examples))

	for _, example := range examples {
		converted = append(converted, ToolInputExample{
			Input: example,
		})
	}

	return converted
}

// EncodeToolChoice encodes a tool choice.
func EncodeToolChoice(choice model.ToolChoice) ToolChoice {
	return ToolChoice{
		Type:     choice.Type,
		ToolName: choice.ToolName,
	}
}
