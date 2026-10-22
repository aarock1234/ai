package openai

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/internal/toolresult"
	"github.com/aarock1234/ai/pkg/model"
)

// tool converts tool results and approval responses.
func (c *responsesInputConverter) tool(message model.ToolMessage) error {
	for _, part := range message.Content {
		var err error

		switch part := part.(type) {
		case model.ToolApprovalResponsePart:
			c.approvalResponse(part)
		case model.ToolResultPart:
			err = c.toolResult(part)
		}

		if err != nil {
			return err
		}
	}

	return nil
}

func (c *responsesInputConverter) approvalResponse(part model.ToolApprovalResponsePart) {
	if c.processedApprovalIDs[part.ApprovalID] {
		return
	}

	c.processedApprovalIDs[part.ApprovalID] = true

	if c.options.store && !c.options.hasConversation && !c.options.hasPreviousResponseID {
		c.addReference(part.ApprovalID)
	}

	c.add(responsesMCPApprovalResponseInput{
		Type:              responsesMCPApprovalResponse,
		ApprovalRequestID: part.ApprovalID,
		Approve:           part.Approved,
	})
}

// toolResult converts the result of a client-executed tool.
func (c *responsesInputConverter) toolResult(part model.ToolResultPart) error {
	options, err := responsesProviderValue[ResponsesToolResultOptions](part.ProviderOptions, c.options.namespace)
	if err != nil {
		return err
	}

	if err := options.validate(); err != nil {
		return err
	}

	handled, err := c.parallelResult(options.ParallelToolCall)
	if handled || err != nil {
		return err
	}

	// A denial reported through an approval response was already sent.
	if denied, ok := part.Output.(model.ExecutionDeniedOutput); ok {
		outputOptions, err := responsesProviderValue[ResponsesToolOutputOptions](denied.ProviderOptions, Namespace)
		if err != nil {
			return err
		}

		if outputOptions.ApprovalID != "" {
			return nil
		}
	}

	handled, err = c.providerDefinedResult(part)
	if handled || err != nil {
		return err
	}

	providerName := c.options.toolNames.ToProvider(part.ToolName)
	if c.options.customToolNames[providerName] {
		return c.customToolResult(part)
	}

	return c.functionResult(part, options)
}

// providerDefinedResult converts JSON results of provider-defined tools
// into their dedicated output items. It reports whether it handled the
// result.
func (c *responsesInputConverter) providerDefinedResult(part model.ToolResultPart) (bool, error) {
	output, isJSON := part.Output.(model.JSONOutput)
	if !isJSON {
		return false, nil
	}

	providerName := c.options.toolNames.ToProvider(part.ToolName)

	switch {
	case c.isToolSearch(part.ToolName):
		decoded, err := responsesDecodeToolOutput[ToolSearchOutput](output)
		if err != nil {
			return true, err
		}

		c.add(responsesToolSearchOutputInput{
			Type:      responsesToolSearchOutput,
			Execution: "client",
			CallID:    new(part.ToolCallID),
			Status:    "completed",
			Tools:     responsesNonNilTools(decoded.Tools),
		})

	case c.options.hasLocalShellTool && providerName == "local_shell":
		decoded, err := responsesDecodeToolOutput[LocalShellOutput](output)
		if err != nil {
			return true, err
		}

		c.add(responsesLocalShellCallOutputInput{
			Type:   responsesLocalShellCallOutput,
			CallID: part.ToolCallID,
			Output: decoded.Output,
		})

	case c.options.hasShellTool && providerName == "shell":
		return true, c.shellOutput(part)

	case c.options.hasApplyPatchTool && part.ToolName == "apply_patch":
		return true, c.applyPatchOutput(part.ToolCallID, output)

	case c.options.hasComputerTool && providerName == "computer":
		return true, c.computerOutput(part.ToolCallID, output)

	default:
		return false, nil
	}

	return true, nil
}

func (c *responsesInputConverter) applyPatchOutput(callID string, output model.JSONOutput) error {
	decoded, err := responsesDecodeToolOutput[ApplyPatchOutput](output)
	if err != nil {
		return err
	}

	if err := responsesCheckEnum("apply patch status", decoded.Status, ApplyPatchCompleted, ApplyPatchFailed); err != nil {
		return err
	}

	c.add(responsesApplyPatchCallOutputInput{
		Type:   responsesApplyPatchCallOutput,
		CallID: callID,
		Status: string(decoded.Status),
		Output: decoded.Output,
	})

	return nil
}

func (c *responsesInputConverter) computerOutput(callID string, output model.JSONOutput) error {
	decoded, err := responsesDecodeToolOutput[ComputerOutput](output)
	if err != nil {
		return err
	}

	c.add(responsesComputerCallOutputInput{
		Type:   responsesComputerCallOutput,
		CallID: callID,
		Output: responsesComputerScreenshot{
			Type:     "computer_screenshot",
			ImageURL: decoded.Output.ImageURL,
			FileID:   decoded.Output.FileID,
			Detail:   decoded.Output.Detail,
		},
		AcknowledgedSafetyChecks: decoded.AcknowledgedSafetyChecks,
	})

	return nil
}

// customToolResult converts the result of a custom tool.
func (c *responsesInputConverter) customToolResult(part model.ToolResultPart) error {
	output, err := c.outputContent(part, false, "custom tool")
	if err != nil {
		return err
	}

	c.add(responsesCustomToolCallOutputInput{
		Type:   responsesCustomToolCallOutput,
		CallID: part.ToolCallID,
		Output: output,
	})

	return nil
}

// functionResult converts the result of a function tool.
func (c *responsesInputConverter) functionResult(part model.ToolResultPart, options ResponsesToolResultOptions) error {
	_, denied := part.Output.(model.ExecutionDeniedOutput)
	isProgrammatic := options.Caller != nil && options.Caller.Type == ResponsesCallerProgram
	if denied && (isProgrammatic || c.programmaticToolCallIDs[part.ToolCallID]) {
		return fmt.Errorf("%w: execution-denied results for programmatic tool calls", model.ErrUnsupported)
	}

	output, err := c.outputContent(part, true, "tool")
	if err != nil {
		return err
	}

	c.add(responsesFunctionCallOutputInput{
		Type:   responsesFunctionCallOutput,
		CallID: part.ToolCallID,
		Output: output,
		Caller: responsesCallerOf(options.Caller),
	})

	if len(options.AdditionalTools) > 0 {
		c.add(responsesAdditionalToolsInput{
			Type:  responsesAdditionalTools,
			Role:  "developer",
			Tools: options.AdditionalTools,
		})
	}

	return nil
}

// outputContent converts a tool result into output content. Function
// results may refer to uploaded files; custom tool results may not.
func (c *responsesInputConverter) outputContent(part model.ToolResultPart, isFunction bool, kind string) (responsesContent, error) {
	if content, ok := part.Output.(model.ContentOutput); ok {
		parts, err := c.toolContentParts(content.Value, isFunction, kind)
		if err != nil {
			return responsesContent{}, err
		}

		return responsesContent{
			Parts: parts,
		}, nil
	}

	quoteText := isFunction && c.options.outputSchemaToolNames[part.ToolName]

	text, err := responsesScalarOutput(part.Output, quoteText)
	if err != nil {
		return responsesContent{}, err
	}

	breakpoint, err := c.scalarBreakpoint(part.Output, part.ProviderOptions)
	if err != nil {
		return responsesContent{}, err
	}

	return responsesTextContent(text, breakpoint), nil
}

// scalarBreakpoint returns the cache breakpoint of a non-content output,
// set on the output or on the tool result part.
func (c *responsesInputConverter) scalarBreakpoint(
	output model.ToolResultOutput,
	partOptions model.ProviderOptions,
) (*ResponsesPromptCacheBreakpoint, error) {
	if _, ok := output.(model.ContentOutput); ok {
		return nil, nil
	}

	outputOptions, err := responsesProviderValue[ResponsesToolOutputOptions](responsesOutputOptions(output), c.options.namespace)
	if err != nil {
		return nil, err
	}

	if outputOptions.PromptCacheBreakpoint != nil {
		return outputOptions.PromptCacheBreakpoint, nil
	}

	resultOptions, err := responsesProviderValue[ResponsesToolResultOptions](partOptions, c.options.namespace)
	if err != nil {
		return nil, err
	}

	return resultOptions.PromptCacheBreakpoint, nil
}

// toolContentParts converts multipart tool output. Unsupported items are
// skipped with a warning.
func (c *responsesInputConverter) toolContentParts(items []model.ToolResultContent, allowReferences bool, kind string) ([]responsesContentPart, error) {
	parts := []responsesContentPart{}

	for _, item := range items {
		switch item := item.(type) {
		case model.ToolResultText:
			options, err := responsesProviderValue[ResponsesPartOptions](item.ProviderOptions, c.options.namespace)
			if err != nil {
				return nil, err
			}

			parts = append(parts, responsesInputText{
				Type:                  "input_text",
				Text:                  item.Text,
				PromptCacheBreakpoint: options.PromptCacheBreakpoint,
			})

		case model.ToolResultFile:
			part, err := c.toolFile(item, allowReferences)
			if err != nil {
				return nil, err
			}

			if part == nil {
				c.warnOther(fmt.Sprintf(
					"unsupported %s content part type: file with data type: %s",
					kind,
					responsesFileDataType(item.Data),
				))

				continue
			}

			parts = append(parts, part)

		default:
			c.warnOther(fmt.Sprintf("unsupported %s content part type: custom", kind))
		}
	}

	return parts, nil
}

// toolFile converts a file in tool output. It returns nil for data it
// cannot send.
func (c *responsesInputConverter) toolFile(item model.ToolResultFile, allowReferences bool) (responsesContentPart, error) {
	options, err := responsesProviderValue[ResponsesPartOptions](item.ProviderOptions, c.options.namespace)
	if err != nil {
		return nil, err
	}

	isImage := mediatype.TopLevel(item.MediaType) == "image"

	switch data := item.Data.(type) {
	case model.FileReference:
		if !allowReferences {
			return nil, nil
		}

		fileID, err := c.fileID(data)
		if err != nil {
			return nil, err
		}

		return responsesFileByID(fileID, isImage, options), nil

	case model.FileBytes:
		mediaType, err := mediatype.ResolveFile(item.MediaType, data)
		if err != nil {
			return nil, err
		}

		if isImage {
			return responsesInputImage{
				Type:                  "input_image",
				ImageURL:              dataurl.Encode(mediaType, data.Data),
				Detail:                options.ImageDetail,
				PromptCacheBreakpoint: options.PromptCacheBreakpoint,
			}, nil
		}

		return responsesInputFile{
			Type:                  "input_file",
			Filename:              responsesFirstNonEmpty(item.Filename, "data"),
			FileData:              dataurl.Encode(mediaType, data.Data),
			PromptCacheBreakpoint: options.PromptCacheBreakpoint,
		}, nil

	case model.FileURL:
		fileURL, err := responsesURLString(data)
		if err != nil {
			return nil, err
		}

		return responsesFileByURL(fileURL, isImage, options), nil

	default:
		return nil, nil
	}
}

// responsesScalarOutput returns a non-content tool output as text. Errors
// become a JSON object with an "error" member. quoteText encodes text as a
// JSON string, which functions with an output schema require.
func responsesScalarOutput(output model.ToolResultOutput, quoteText bool) (string, error) {
	switch output := output.(type) {
	case model.TextOutput:
		return responsesMaybeQuote(output.Value, quoteText), nil

	case model.ErrorTextOutput:
		return string(responsesEncodeJSON(responsesErrorOutput{
			Error: responsesEncodeJSON(output.Value),
		})), nil

	case model.ErrorJSONOutput:
		value, err := toolresult.CompactJSON(output.Value)
		if err != nil {
			return "", err
		}

		return string(responsesEncodeJSON(responsesErrorOutput{
			Error: json.RawMessage(value),
		})), nil

	case model.ExecutionDeniedOutput:
		reason := responsesFirstNonEmpty(output.Reason, toolresult.DeniedText)

		return responsesMaybeQuote(reason, quoteText), nil

	case model.JSONOutput:
		return toolresult.CompactJSON(output.Value)

	default:
		return "", fmt.Errorf("%w: unsupported tool output %T", model.ErrInvalidPrompt, output)
	}
}

// responsesErrorOutput is the text sent for failed tool results.
type responsesErrorOutput struct {
	Error json.RawMessage `json:"error"`
}

// responsesOutputOptions returns the provider options of a tool output.
func responsesOutputOptions(output model.ToolResultOutput) model.ProviderOptions {
	switch output := output.(type) {
	case model.TextOutput:
		return output.ProviderOptions
	case model.JSONOutput:
		return output.ProviderOptions
	case model.ExecutionDeniedOutput:
		return output.ProviderOptions
	case model.ErrorTextOutput:
		return output.ProviderOptions
	case model.ErrorJSONOutput:
		return output.ProviderOptions
	default:
		return nil
	}
}

// responsesFileDataType names a file data kind for warnings.
func responsesFileDataType(data model.FileData) string {
	switch data.(type) {
	case model.FileBytes:
		return "data"
	case model.FileURL:
		return "url"
	case model.FileReference:
		return "reference"
	case model.FileText:
		return "text"
	default:
		return fmt.Sprintf("%T", data)
	}
}

// responsesDecodeToolOutput decodes a JSON tool result.
func responsesDecodeToolOutput[T any](output model.JSONOutput) (T, error) {
	var value T
	if err := json.Unmarshal(output.Value, &value); err != nil {
		return value, fmt.Errorf("%w: decode %T tool output: %w", model.ErrInvalidPrompt, value, err)
	}

	return value, nil
}

func responsesMaybeQuote(text string, quote bool) string {
	if !quote {
		return text
	}

	return `"` + responsesEscapeJSONString(text) + `"`
}
