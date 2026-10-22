package ai

import (
	"context"
	"encoding/json"

	"github.com/aarock1234/ai/pkg/model"
)

// toolOutputFields are the parts of a tool result or error that the model
// receives.
type toolOutputFields struct {
	toolCallID       string
	toolName         string
	input            json.RawMessage
	output           any
	errorMode        toolOutputErrorMode
	providerMetadata model.ProviderMetadata
}

// toolOutputPart converts a tool result or error into the part that reports
// it to the model. Errors of provider-executed tools are sent as JSON values;
// errors of local tools are sent as text.
func toolOutputPart(ctx context.Context, output ToolOutput, tools []Tool) (model.ToolResultPart, error) {
	fields := toolOutputFieldsOf(output)

	modelOutput, err := createToolModelOutput(ctx, toolModelOutputRequest{
		toolCallID: fields.toolCallID,
		input:      fields.input,
		output:     fields.output,
		tool:       lookupTool(tools, fields.toolName),
		errorMode:  fields.errorMode,
	})
	if err != nil {
		return model.ToolResultPart{}, err
	}

	return model.ToolResultPart{
		ToolCallID:      fields.toolCallID,
		ToolName:        fields.toolName,
		Output:          modelOutput,
		ProviderOptions: fields.providerMetadata.Options(),
	}, nil
}

func toolOutputFieldsOf(output ToolOutput) toolOutputFields {
	switch typed := output.(type) {
	case ToolResult:
		return toolOutputFields{
			toolCallID:       typed.ToolCallID,
			toolName:         typed.ToolName,
			input:            typed.Input,
			output:           typed.Output,
			errorMode:        toolOutputSucceeded,
			providerMetadata: typed.ProviderMetadata,
		}

	case ToolError:
		return toolErrorFields(typed)

	default:
		return toolOutputFields{}
	}
}

func toolErrorFields(toolError ToolError) toolOutputFields {
	fields := toolOutputFields{
		toolCallID:       toolError.ToolCallID,
		toolName:         toolError.ToolName,
		input:            toolError.Input,
		output:           toolError.Error,
		errorMode:        toolOutputErrorText,
		providerMetadata: toolError.ProviderMetadata,
	}

	if toolError.ProviderExecuted {
		fields.output = toolErrorValue(toolError.Error)
		fields.errorMode = toolOutputErrorJSON
	}

	return fields
}
