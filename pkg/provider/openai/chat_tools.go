package openai

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// chatEmptyObjectSchema is sent for function tools without an input schema.
var chatEmptyObjectSchema = json.RawMessage(`{"type":"object","properties":{}}`)

// chatPreparedTools are the tool settings of a Chat Completions request.
type chatPreparedTools struct {
	tools      []chatTool
	toolChoice *chatToolChoice
	warnings   []model.Warning
}

// chatPrepareTools converts function tools and the tool choice. Chat
// Completions has no provider-defined tools, so those produce warnings.
func chatPrepareTools(tools []model.Tool, toolChoice *model.ToolChoice) (chatPreparedTools, error) {
	var prepared chatPreparedTools

	if len(tools) == 0 {
		return prepared, nil
	}

	prepared.tools = make([]chatTool, 0, len(tools))

	for _, tool := range tools {
		function, ok := tool.(model.FunctionTool)
		if !ok {
			prepared.warnings = append(prepared.warnings, model.UnsupportedWarning("tool type: provider", ""))

			continue
		}

		converted, warnings, err := chatFunctionTool(function)
		if err != nil {
			return chatPreparedTools{}, err
		}

		prepared.tools = append(prepared.tools, converted)
		prepared.warnings = append(prepared.warnings, warnings...)
	}

	choice, err := chatConvertToolChoice(toolChoice)
	if err != nil {
		return chatPreparedTools{}, err
	}

	prepared.toolChoice = choice

	return prepared, nil
}

func chatFunctionTool(tool model.FunctionTool) (chatTool, []model.Warning, error) {
	schema := tool.InputSchema
	if len(schema) == 0 {
		schema = chatEmptyObjectSchema
	}

	parameters, warnings, err := normalizeJSONSchema(schema)
	if err != nil {
		return chatTool{}, nil, fmt.Errorf("tool %q input schema: %w", tool.Name, err)
	}

	return chatTool{
		Type: "function",
		Function: chatToolFunction{
			Name:        tool.Name,
			Description: tool.Description,
			Parameters:  parameters,
			Strict:      tool.Strict,
		},
	}, warnings, nil
}

func chatConvertToolChoice(toolChoice *model.ToolChoice) (*chatToolChoice, error) {
	if toolChoice == nil {
		return nil, nil
	}

	switch toolChoice.Type {
	case model.ToolChoiceAuto, model.ToolChoiceNone, model.ToolChoiceRequired:
		return &chatToolChoice{
			mode: toolChoice.Type,
		}, nil

	case model.ToolChoiceTool:
		if toolChoice.ToolName == "" {
			return nil, fmt.Errorf("%w: tool choice requires a tool name", model.ErrInvalidArgument)
		}

		return &chatToolChoice{
			mode:         toolChoice.Type,
			functionName: toolChoice.ToolName,
		}, nil

	default:
		return nil, fmt.Errorf("%w: tool choice type %q", model.ErrUnsupported, toolChoice.Type)
	}
}
