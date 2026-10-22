package google

import (
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// assistantConversion tracks one assistant message. Gemini 3 returns a single
// signature for parallel function calls, on the first call only.
type assistantConversion struct {
	hasSignedFunctionCall bool
}

func (c *promptConverter) convertAssistantMessage(message model.AssistantMessage) error {
	var conversion assistantConversion
	parts := make([]part, 0, len(message.Content))

	for _, assistantPart := range message.Content {
		converted, ok, err := c.convertAssistantPart(assistantPart, &conversion)
		if err != nil {
			return err
		}

		if ok {
			parts = append(parts, converted)
		}
	}

	c.contents = append(c.contents, content{
		Role:  roleModel,
		Parts: parts,
	})

	return nil
}

// convertAssistantPart returns the model part of an assistant part. It
// reports false for parts that Gemini does not replay, such as empty text and
// content of other providers.
func (c *promptConverter) convertAssistantPart(assistantPart model.AssistantPart, conversion *assistantConversion) (part, bool, error) {
	switch assistantPart := assistantPart.(type) {
	case model.TextPart:
		return c.convertAssistantText(assistantPart.Text, false, assistantPart.ProviderOptions)

	case model.ReasoningPart:
		return c.convertAssistantText(assistantPart.Text, true, assistantPart.ProviderOptions)

	case model.ReasoningFilePart:
		return c.convertReasoningFile(assistantPart)

	case model.FilePart:
		return c.convertAssistantFile(assistantPart)

	case model.ToolCallPart:
		return c.convertToolCall(assistantPart, conversion)

	case model.ToolResultPart:
		return c.convertProviderToolResult(assistantPart)

	case model.CustomPart:
		return part{}, false, nil

	default:
		return part{}, false, fmt.Errorf("%w: unsupported assistant part %T", model.ErrInvalidPrompt, assistantPart)
	}
}

func (c *promptConverter) convertAssistantText(text string, thought bool, options model.ProviderOptions) (part, bool, error) {
	if text == "" {
		return part{}, false, nil
	}

	metadata, err := c.partMetadata(options)
	if err != nil {
		return part{}, false, err
	}

	return part{
		Text:             new(text),
		Thought:          thought,
		ThoughtSignature: metadata.ThoughtSignature,
	}, true, nil
}

func (c *promptConverter) convertReasoningFile(file model.ReasoningFilePart) (part, bool, error) {
	metadata, err := c.partMetadata(file.ProviderOptions)
	if err != nil {
		return part{}, false, err
	}

	switch data := file.Data.(type) {
	case model.FileBytes:
		return part{
			InlineData: &InlineData{
				MimeType: file.MediaType,
				Data:     base64.StdEncoding.EncodeToString(data.Data),
			},
			Thought:          true,
			ThoughtSignature: metadata.ThoughtSignature,
		}, true, nil

	case model.FileURL:
		return part{}, false, fmt.Errorf("%w: file data URLs in assistant messages are not supported", model.ErrUnsupported)

	default:
		return part{}, false, fmt.Errorf("%w: unsupported reasoning file data %T", model.ErrInvalidPrompt, file.Data)
	}
}

func (c *promptConverter) convertAssistantFile(file model.FilePart) (part, bool, error) {
	metadata, err := c.partMetadata(file.ProviderOptions)
	if err != nil {
		return part{}, false, err
	}

	converted := part{
		Thought:          metadata.Thought,
		ThoughtSignature: metadata.ThoughtSignature,
	}

	switch data := file.Data.(type) {
	case model.FileURL:
		return part{}, false, fmt.Errorf("%w: file data URLs in assistant messages are not supported", model.ErrUnsupported)

	case model.FileReference:
		if c.settings.isVertex {
			return part{}, false, errVertexFileReference
		}

		fileURI, ok := data.IDs[Namespace]
		if !ok {
			return part{}, false, fmt.Errorf("%w: file reference has no %s file ID", model.ErrInvalidPrompt, Namespace)
		}

		converted.FileData = &FileData{
			MimeType: file.MediaType,
			FileURI:  fileURI,
		}

	case model.FileText:
		converted.InlineData = textInlineData(file.MediaType, data)

	case model.FileBytes:
		converted.InlineData = &InlineData{
			MimeType: file.MediaType,
			Data:     base64.StdEncoding.EncodeToString(data.Data),
		}

	default:
		return part{}, false, fmt.Errorf("%w: unsupported file data %T", model.ErrInvalidPrompt, file.Data)
	}

	return converted, true, nil
}

// convertToolCall replays a code execution call, a server-side tool call or
// a function call. Gemini 3 rejects unsigned function calls, so they receive
// the documented skip-validator signature unless an earlier call of the same
// response was signed.
func (c *promptConverter) convertToolCall(call model.ToolCallPart, conversion *assistantConversion) (part, bool, error) {
	if call.ProviderExecuted && c.settings.toolNames.ToProvider(call.ToolName) == codeExecutionToolName {
		input, err := decodeCodeExecutionInput(call.Input)
		if err != nil {
			return part{}, false, err
		}

		return part{
			ExecutableCode: &input,
		}, true, nil
	}

	metadata, err := c.partMetadata(call.ProviderOptions)
	if err != nil {
		return part{}, false, err
	}

	isServerToolCall := metadata.isServerTool()
	signature := metadata.ThoughtSignature

	needsSentinel := signature == "" && c.settings.isGemini3
	if !isServerToolCall && conversion.hasSignedFunctionCall {
		needsSentinel = false
	}

	if needsSentinel {
		signature = skipThoughtSignatureValidator
		c.unsignedToolNames = append(c.unsignedToolNames, call.ToolName)
	}

	if !isServerToolCall && metadata.ThoughtSignature != "" {
		conversion.hasSignedFunctionCall = true
	}

	if isServerToolCall {
		return part{
			ToolCall: &serverToolCall{
				ToolType: metadata.ServerToolType,
				Args:     call.Input,
				ID:       metadata.ServerToolCallID,
			},
			ThoughtSignature: signature,
		}, true, nil
	}

	return part{
		FunctionCall: &functionCall{
			ID:   call.ToolCallID,
			Name: call.ToolName,
			Args: call.Input,
		},
		ThoughtSignature: signature,
	}, true, nil
}

// convertProviderToolResult replays a code execution result or a
// server-side tool response. Other provider-executed results are dropped.
func (c *promptConverter) convertProviderToolResult(result model.ToolResultPart) (part, bool, error) {
	jsonOutput, isJSON := result.Output.(model.JSONOutput)

	if isJSON && c.settings.toolNames.ToProvider(result.ToolName) == codeExecutionToolName {
		output, err := decodeCodeExecutionOutput(jsonOutput.Value)
		if err != nil {
			return part{}, false, err
		}

		return part{
			CodeExecutionResult: &output,
		}, true, nil
	}

	metadata, err := c.partMetadata(result.ProviderOptions)
	if err != nil {
		return part{}, false, err
	}

	if !metadata.isServerTool() {
		return part{}, false, nil
	}

	return serverToolResponsePart(result.Output, metadata), true, nil
}

func serverToolResponsePart(output model.ToolResultOutput, metadata PartMetadata) part {
	response := json.RawMessage("{}")
	if jsonOutput, ok := output.(model.JSONOutput); ok {
		response = jsonOutput.Value
	}

	return part{
		ToolResponse: &serverToolResponse{
			ToolType: metadata.ServerToolType,
			Response: response,
			ID:       metadata.ServerToolCallID,
		},
		ThoughtSignature: metadata.ThoughtSignature,
	}
}

func decodeCodeExecutionInput(input json.RawMessage) (CodeExecutionInput, error) {
	var fields struct {
		Language *string `json:"language"`
		Code     *string `json:"code"`
	}

	if err := json.Unmarshal(input, &fields); err != nil {
		return CodeExecutionInput{}, fmt.Errorf("%w: code execution input: %w", model.ErrInvalidPrompt, err)
	}

	if fields.Language == nil || fields.Code == nil {
		return CodeExecutionInput{}, fmt.Errorf("%w: code execution input requires language and code", model.ErrInvalidPrompt)
	}

	return CodeExecutionInput{
		Language: *fields.Language,
		Code:     *fields.Code,
	}, nil
}

func decodeCodeExecutionOutput(output json.RawMessage) (CodeExecutionOutput, error) {
	var fields struct {
		Outcome *string `json:"outcome"`
		Output  *string `json:"output"`
	}

	if err := json.Unmarshal(output, &fields); err != nil {
		return CodeExecutionOutput{}, fmt.Errorf("%w: code execution output: %w", model.ErrInvalidPrompt, err)
	}

	if fields.Outcome == nil || fields.Output == nil {
		return CodeExecutionOutput{}, fmt.Errorf("%w: code execution output requires outcome and output", model.ErrInvalidPrompt)
	}

	return CodeExecutionOutput{
		Outcome: *fields.Outcome,
		Output:  *fields.Output,
	}, nil
}
