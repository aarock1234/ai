package ui

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"net/url"
	"strings"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// unavailableToolOutput replaces the outputs of tools that are no longer
// available, so that their unvalidated outputs are not sent to the model.
const unavailableToolOutput = "Tool output omitted because the tool is no longer available."

// deniedToolOutput describes a denied tool call without a reason.
const deniedToolOutput = "Tool call execution denied."

// ConvertToModelMessagesOptions configure ConvertToModelMessages.
type ConvertToModelMessagesOptions struct {
	// Tools are the tools of the generation. Their ToModelOutput functions
	// convert tool outputs for the model.
	Tools []ai.Tool

	// IgnoreIncompleteToolCalls leaves out tool parts without a final
	// output, denial or approval response.
	IgnoreIncompleteToolCalls bool

	// ConvertDataPart converts data parts into a model.TextPart or
	// model.FilePart. A nil part, or a nil function, leaves the data part out.
	ConvertDataPart func(ctx context.Context, part DataPart) (model.UserPart, error)
}

// ConvertToModelMessages converts UI messages, such as those received from
// useChat, into model messages for GenerateText and StreamText. Assistant
// messages become an assistant message per step, each followed by a tool
// message with the step's tool results and approval responses.
//
// Tool outputs are passed to ToModelOutput as json.RawMessage, except JSON
// strings, which are passed as string.
func ConvertToModelMessages(ctx context.Context, messages []Message, options ConvertToModelMessagesOptions) ([]model.Message, error) {
	converter := messageConverter{
		options: options,
	}

	var converted []model.Message

	for index, message := range filterToolParts(messages, options.IgnoreIncompleteToolCalls) {
		modelMessages, err := converter.convert(ctx, message)
		if err != nil {
			return nil, fmt.Errorf("convert message %d: %w", index, err)
		}

		converted = append(converted, modelMessages...)
	}

	return converted, nil
}

// filterToolParts removes approval requests that a later user message
// superseded and, when ignoreIncomplete is set, tool parts without a final
// state.
func filterToolParts(messages []Message, ignoreIncomplete bool) []Message {
	lastUserIndex := -1
	for index, message := range messages {
		if message.Role == RoleUser {
			lastUserIndex = index
		}
	}

	if !ignoreIncomplete && lastUserIndex <= 0 {
		return messages
	}

	filtered := make([]Message, 0, len(messages))

	for index, message := range messages {
		var parts []Part

		for _, part := range message.Parts {
			tool, isTool := part.(ToolPart)
			if !isTool {
				parts = append(parts, part)

				continue
			}

			superseded := tool.State == ToolApprovalRequested && index < lastUserIndex
			incomplete := ignoreIncomplete && !isCompleteToolPart(tool)

			if !superseded && !incomplete {
				parts = append(parts, part)
			}
		}

		message.Parts = parts
		filtered = append(filtered, message)
	}

	return filtered
}

// isCompleteToolPart reports whether a tool part has a final output, a
// denial or an approval response.
func isCompleteToolPart(part ToolPart) bool {
	switch part.State {
	case ToolApprovalResponded, ToolOutputError, ToolOutputDenied:
		return true
	case ToolOutputAvailable:
		return !part.Preliminary
	default:
		return false
	}
}

// messageConverter converts single UI messages.
type messageConverter struct {
	options ConvertToModelMessagesOptions
}

func (c messageConverter) convert(ctx context.Context, message Message) ([]model.Message, error) {
	switch message.Role {
	case RoleSystem:
		return []model.Message{
			systemMessage(message),
		}, nil

	case RoleUser:
		userMessage, err := c.userMessage(ctx, message)
		if err != nil {
			return nil, err
		}

		return []model.Message{
			userMessage,
		}, nil

	case RoleAssistant:
		return c.assistantMessages(ctx, message)

	default:
		return nil, fmt.Errorf("unsupported role %q", message.Role)
	}
}

// systemMessage joins the text parts and merges their provider metadata.
func systemMessage(message Message) model.SystemMessage {
	var text strings.Builder

	var options model.ProviderOptions

	for _, part := range message.Parts {
		textPart, ok := part.(TextPart)
		if !ok {
			continue
		}

		text.WriteString(textPart.Text)

		if len(textPart.ProviderMetadata) > 0 {
			if options == nil {
				options = make(model.ProviderOptions)
			}

			maps.Copy(options, textPart.ProviderMetadata)
		}
	}

	return model.SystemMessage{
		Content:         text.String(),
		ProviderOptions: options,
	}
}

func (c messageConverter) userMessage(ctx context.Context, message Message) (model.UserMessage, error) {
	content := make([]model.UserPart, 0, len(message.Parts))

	for index, part := range message.Parts {
		var converted model.UserPart

		switch typed := part.(type) {
		case TextPart:
			converted = model.TextPart{
				Text:            typed.Text,
				ProviderOptions: typed.ProviderMetadata.Options(),
			}

		case FilePart:
			filePart, err := modelFilePart(typed)
			if err != nil {
				return model.UserMessage{}, fmt.Errorf("part %d: %w", index, err)
			}

			converted = filePart

		case DataPart:
			dataPart, err := c.convertDataPart(ctx, typed)
			if err != nil {
				return model.UserMessage{}, fmt.Errorf("part %d: %w", index, err)
			}

			converted = dataPart
		}

		if converted != nil {
			content = append(content, converted)
		}
	}

	return model.UserMessage{
		Content: content,
	}, nil
}

func (c messageConverter) convertDataPart(ctx context.Context, part DataPart) (model.UserPart, error) {
	if c.options.ConvertDataPart == nil {
		return nil, nil
	}

	converted, err := c.options.ConvertDataPart(ctx, part)
	if err != nil {
		return nil, fmt.Errorf("convert data part %q: %w", part.Name, err)
	}

	return converted, nil
}

// modelFilePart converts a file part. A provider reference takes precedence
// over the URL.
func modelFilePart(part FilePart) (model.FilePart, error) {
	data, err := fileData(part.URL, part.ProviderReference)
	if err != nil {
		return model.FilePart{}, err
	}

	return model.FilePart{
		Filename:        part.Filename,
		MediaType:       part.MediaType,
		Data:            data,
		ProviderOptions: part.ProviderMetadata.Options(),
	}, nil
}

func fileData(fileURL string, reference map[string]string) (model.FileData, error) {
	if reference != nil {
		return model.FileReference{
			IDs: reference,
		}, nil
	}

	parsed, err := url.Parse(fileURL)
	if err != nil {
		return nil, fmt.Errorf("parse file url: %w", err)
	}

	return model.FileURL{
		URL: parsed,
	}, nil
}

// assistantMessages converts the steps of an assistant message, separated
// by step-start parts.
func (c messageConverter) assistantMessages(ctx context.Context, message Message) ([]model.Message, error) {
	var messages []model.Message
	var block []Part

	flush := func() error {
		converted, err := c.assistantBlock(ctx, block)
		if err != nil {
			return err
		}

		messages = append(messages, converted...)
		block = nil

		return nil
	}

	for _, part := range message.Parts {
		switch part.(type) {
		case StepStartPart:
			if err := flush(); err != nil {
				return nil, err
			}

		case SourceURLPart, SourceDocumentPart:
			// Sources are not sent back to the model.

		default:
			block = append(block, part)
		}
	}

	if err := flush(); err != nil {
		return nil, err
	}

	return messages, nil
}

// assistantBlock converts the parts of one step into an assistant message
// and a tool message.
func (c messageConverter) assistantBlock(ctx context.Context, block []Part) ([]model.Message, error) {
	if len(block) == 0 {
		return nil, nil
	}

	var messages []model.Message

	content, err := c.assistantContent(ctx, block)
	if err != nil {
		return nil, err
	}

	if len(content) > 0 {
		messages = append(messages, model.AssistantMessage{
			Content: content,
		})
	}

	toolContent, err := c.toolContent(ctx, block)
	if err != nil {
		return nil, err
	}

	if len(toolContent) > 0 {
		messages = append(messages, model.ToolMessage{
			Content: toolContent,
		})
	}

	return messages, nil
}

func (c messageConverter) assistantContent(ctx context.Context, block []Part) ([]model.AssistantPart, error) {
	var content []model.AssistantPart

	for _, part := range block {
		switch typed := part.(type) {
		case TextPart:
			content = append(content, model.TextPart{
				Text:            typed.Text,
				ProviderOptions: typed.ProviderMetadata.Options(),
			})

		case CustomPart:
			content = append(content, model.CustomPart{
				Kind:            typed.Kind,
				ProviderOptions: typed.ProviderMetadata.Options(),
			})

		case FilePart:
			filePart, err := modelFilePart(typed)
			if err != nil {
				return nil, err
			}

			content = append(content, filePart)

		case ReasoningFilePart:
			data, err := fileData(typed.URL, nil)
			if err != nil {
				return nil, err
			}

			content = append(content, model.ReasoningFilePart{
				MediaType:       typed.MediaType,
				Data:            data,
				ProviderOptions: typed.ProviderMetadata.Options(),
			})

		case ReasoningPart:
			content = append(content, model.ReasoningPart{
				Text:            typed.Text,
				ProviderOptions: typed.ProviderMetadata.Options(),
			})

		case ToolPart:
			toolParts, err := c.assistantToolContent(ctx, typed)
			if err != nil {
				return nil, err
			}

			content = append(content, toolParts...)

		case DataPart:
			converted, err := c.convertDataPart(ctx, typed)
			if err != nil {
				return nil, err
			}

			if assistantPart, ok := converted.(model.AssistantPart); ok {
				content = append(content, assistantPart)
			}
		}
	}

	return content, nil
}

// assistantToolContent returns the tool call of a tool part, its approval
// request and, for provider-executed tools, its result.
func (c messageConverter) assistantToolContent(ctx context.Context, part ToolPart) ([]model.AssistantPart, error) {
	if part.State == ToolInputStreaming {
		return nil, nil
	}

	callMetadata := part.CallProviderMetadata
	if callMetadata == nil && part.State == ToolOutputError {
		callMetadata = part.ResultProviderMetadata
	}

	input := part.Input
	if part.State == ToolOutputError && input == nil {
		input = part.RawInput
	}

	content := []model.AssistantPart{
		model.ToolCallPart{
			ToolCallID:       part.ToolCallID,
			ToolName:         part.ToolName,
			Input:            input,
			ProviderExecuted: part.ProviderExecuted,
			ProviderOptions:  callMetadata.Options(),
		},
	}

	if part.Approval != nil {
		content = append(content, model.ToolApprovalRequestPart{
			ApprovalID:       part.Approval.ID,
			ToolCallID:       part.ToolCallID,
			Reason:           part.Approval.RequestReason,
			IsAutomatic:      part.Approval.IsAutomatic,
			Signature:        part.Approval.Signature,
			InputSchemaInput: part.Approval.InputSchemaInput,
		})
	}

	if !part.ProviderExecuted || !part.hasOutputState() {
		return content, nil
	}

	resultMetadata := part.ResultProviderMetadata
	if resultMetadata == nil {
		resultMetadata = part.CallProviderMetadata
	}

	output, err := c.toolOutput(ctx, part, toolOutputErrorJSON)
	if err != nil {
		return nil, err
	}

	return append(content, model.ToolResultPart{
		ToolCallID:      part.ToolCallID,
		ToolName:        part.ToolName,
		Output:          output,
		ProviderOptions: resultMetadata.Options(),
	}), nil
}

// toolContent returns the approval responses and the results of the tool
// parts of a step. Results of provider-executed tools are part of the
// assistant message instead.
func (c messageConverter) toolContent(ctx context.Context, block []Part) ([]model.ToolPart, error) {
	var content []model.ToolPart

	for _, part := range block {
		tool, ok := part.(ToolPart)
		if !ok {
			continue
		}

		hasResponse := tool.Approval != nil && tool.Approval.Approved != nil
		if tool.ProviderExecuted && !hasResponse {
			continue
		}

		if hasResponse {
			content = append(content, approvalResponseContent(tool)...)
		}

		if tool.ProviderExecuted {
			continue
		}

		result, ok, err := c.toolResult(ctx, tool)
		if err != nil {
			return nil, err
		}

		if ok {
			content = append(content, result)
		}
	}

	return content, nil
}

// approvalResponseContent returns the approval response of a tool part and,
// for a denial that has not been reported yet, an execution-denied result.
func approvalResponseContent(part ToolPart) []model.ToolPart {
	approval := part.Approval

	content := []model.ToolPart{
		model.ToolApprovalResponsePart{
			ApprovalID:       approval.ID,
			Approved:         *approval.Approved,
			Reason:           approval.Reason,
			ProviderExecuted: part.ProviderExecuted,
		},
	}

	if part.State != ToolApprovalResponded || *approval.Approved {
		return content
	}

	return append(content, model.ToolResultPart{
		ToolCallID: part.ToolCallID,
		ToolName:   part.ToolName,
		Output: model.ExecutionDeniedOutput{
			Reason: approval.Reason,
		},
		ProviderOptions: part.CallProviderMetadata.Options(),
	})
}

// toolResult returns the result of an application tool part in a final
// state.
func (c messageConverter) toolResult(ctx context.Context, part ToolPart) (model.ToolResultPart, bool, error) {
	var output model.ToolResultOutput

	switch part.State {
	case ToolOutputDenied:
		reason := deniedToolOutput
		if part.Approval != nil && part.Approval.Reason != "" {
			reason = part.Approval.Reason
		}

		output = model.ErrorTextOutput{
			Value: reason,
		}

	case ToolOutputAvailable, ToolOutputError:
		converted, err := c.toolOutput(ctx, part, toolOutputErrorText)
		if err != nil {
			return model.ToolResultPart{}, false, err
		}

		output = converted

	default:
		return model.ToolResultPart{}, false, nil
	}

	return model.ToolResultPart{
		ToolCallID:      part.ToolCallID,
		ToolName:        part.ToolName,
		Output:          output,
		ProviderOptions: part.CallProviderMetadata.Options(),
	}, true, nil
}

// toolOutputErrorMode selects how a failed tool call is reported.
type toolOutputErrorMode int

const (
	toolOutputErrorText toolOutputErrorMode = iota
	toolOutputErrorJSON
)

// toolOutput converts the output or failure of a tool part for the model.
func (c messageConverter) toolOutput(ctx context.Context, part ToolPart, errorMode toolOutputErrorMode) (model.ToolResultOutput, error) {
	if part.State == ToolOutputError {
		if errorMode == toolOutputErrorText {
			return model.ErrorTextOutput{
				Value: part.ErrorText,
			}, nil
		}

		value, err := encodeJSON(part.ErrorText)
		if err != nil {
			return nil, fmt.Errorf("encode tool error: %w", err)
		}

		return model.ErrorJSONOutput{
			Value: value,
		}, nil
	}

	tool, found := findTool(c.options.Tools, part.ToolName)
	if !found && part.Unavailable {
		return model.TextOutput{
			Value: unavailableToolOutput,
		}, nil
	}

	output := part.Output
	if output == nil {
		output = json.RawMessage("null")
	}

	if found && tool.ToModelOutput != nil {
		converted, err := tool.ToModelOutput(ctx, ai.ToolOutputEvent[json.RawMessage, any]{
			ToolCallID: part.ToolCallID,
			Input:      part.Input,
			Output:     toolOutputValue(output),
		})
		if err != nil {
			return nil, fmt.Errorf("convert output of tool %q: %w", part.ToolName, err)
		}

		return converted, nil
	}

	if text, ok := toolOutputValue(output).(string); ok {
		return model.TextOutput{
			Value: text,
		}, nil
	}

	return model.JSONOutput{
		Value: output,
	}, nil
}

// toolOutputValue returns a JSON string output as a string and any other
// output as raw JSON.
func toolOutputValue(output json.RawMessage) any {
	var text string
	if err := json.Unmarshal(output, &text); err == nil {
		return text
	}

	return output
}

func findTool(tools []ai.Tool, name string) (ai.Tool, bool) {
	for _, tool := range tools {
		if tool.Name == name {
			return tool, true
		}
	}

	return ai.Tool{}, false
}
