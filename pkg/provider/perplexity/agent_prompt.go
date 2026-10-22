package perplexity

import (
	"fmt"
	"strings"

	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/internal/toolresult"
	"github.com/aarock1234/ai/pkg/model"
)

// Input item types and roles.
const (
	inputMessageType   = "message"
	inputFunctionCall  = "function_call"
	inputFunctionReply = "function_call_output"

	partInputText  = "input_text"
	partInputImage = "input_image"
)

// convertPrompt converts the prompt into Agent API input items.
func convertPrompt(prompt []model.Message) ([]inputItem, []model.Warning, error) {
	input := []inputItem{}
	var warnings []model.Warning

	for _, message := range prompt {
		var items []inputItem
		var messageWarnings []model.Warning
		var err error

		switch typed := message.(type) {
		case model.SystemMessage:
			items = []inputItem{
				textMessage(string(model.RoleSystem), typed.Content),
			}

		case model.UserMessage:
			items, err = convertUserMessage(typed)

		case model.AssistantMessage:
			items, messageWarnings, err = convertAssistantMessage(typed)

		case model.ToolMessage:
			items, err = convertToolMessage(typed)

		case nil:
			err = fmt.Errorf("%w: message is nil", model.ErrInvalidPrompt)

		default:
			err = fmt.Errorf("%w: unsupported role %q", model.ErrInvalidPrompt, message.Role())
		}

		if err != nil {
			return nil, nil, err
		}

		input = append(input, items...)
		warnings = append(warnings, messageWarnings...)
	}

	return input, warnings, nil
}

func textMessage(role string, text string) messageInput {
	return messageInput{
		Type: inputMessageType,
		Role: role,
		Content: inputContent{
			text: text,
		},
	}
}

// convertUserMessage sends text-only content as one string and other
// content as parts.
func convertUserMessage(message model.UserMessage) ([]inputItem, error) {
	parts := make([]inputPart, 0, len(message.Content))
	textOnly := true

	for _, part := range message.Content {
		converted, err := convertUserPart(part)
		if err != nil {
			return nil, err
		}

		if converted.Type != partInputText {
			textOnly = false
		}

		parts = append(parts, converted)
	}

	if !textOnly {
		return []inputItem{
			messageInput{
				Type: inputMessageType,
				Role: string(model.RoleUser),
				Content: inputContent{
					parts: parts,
				},
			},
		}, nil
	}

	var text strings.Builder
	for _, part := range parts {
		text.WriteString(part.Text)
	}

	return []inputItem{
		textMessage(string(model.RoleUser), text.String()),
	}, nil
}

func convertUserPart(part model.UserPart) (inputPart, error) {
	switch typed := part.(type) {
	case model.TextPart:
		return inputPart{
			Type: partInputText,
			Text: typed.Text,
		}, nil

	case model.FilePart:
		return convertImagePart(typed)

	default:
		return inputPart{}, fmt.Errorf("%w: user part %T", model.ErrUnsupported, part)
	}
}

// convertImagePart sends images as URLs or data URLs. Other files are not
// supported.
func convertImagePart(part model.FilePart) (inputPart, error) {
	if mediatype.TopLevel(part.MediaType) != "image" {
		return inputPart{}, fmt.Errorf("%w: file part media type %s", model.ErrUnsupported, part.MediaType)
	}

	switch data := part.Data.(type) {
	case model.FileURL:
		if data.URL == nil {
			return inputPart{}, fmt.Errorf("%w: image URL is missing", model.ErrInvalidPrompt)
		}

		return inputPart{
			Type:     partInputImage,
			ImageURL: data.URL.String(),
		}, nil

	case model.FileBytes:
		mediaType, err := mediatype.ResolveFile(part.MediaType, data)
		if err != nil {
			return inputPart{}, err
		}

		return inputPart{
			Type:     partInputImage,
			ImageURL: dataurl.Encode(mediaType, data.Data),
		}, nil

	case model.FileReference:
		return inputPart{}, fmt.Errorf("%w: file parts with provider references", model.ErrUnsupported)

	case model.FileText:
		return inputPart{}, fmt.Errorf("%w: text file parts", model.ErrUnsupported)

	default:
		return inputPart{}, fmt.Errorf("%w: file part data type %T", model.ErrUnsupported, data)
	}
}

// convertAssistantMessage sends the joined text first, followed by the tool
// calls and provider-executed tool results in order.
func convertAssistantMessage(message model.AssistantMessage) ([]inputItem, []model.Warning, error) {
	var items []inputItem
	var warnings []model.Warning

	var text strings.Builder
	for _, part := range message.Content {
		if textPart, ok := part.(model.TextPart); ok {
			text.WriteString(textPart.Text)
		}
	}

	if text.Len() > 0 {
		items = append(items, textMessage(string(model.RoleAssistant), text.String()))
	}

	for _, part := range message.Content {
		switch typed := part.(type) {
		case model.TextPart, model.ToolApprovalRequestPart:
			// Text is sent first; approval requests are not sent.

		case model.ToolCallPart:
			item, err := functionCallOf(typed)
			if err != nil {
				return nil, nil, err
			}

			items = append(items, item)

		case model.ToolResultPart:
			item, err := functionOutputOf(typed)
			if err != nil {
				return nil, nil, err
			}

			items = append(items, item)

		case model.ReasoningPart:
			warnings = append(warnings, model.UnsupportedWarning("reasoning content in prompt", ""))

		case model.FilePart:
			return nil, nil, fmt.Errorf("%w: assistant file parts", model.ErrUnsupported)

		case model.ReasoningFilePart:
			return nil, nil, fmt.Errorf("%w: assistant reasoning-file parts", model.ErrUnsupported)

		case model.CustomPart:
			return nil, nil, fmt.Errorf("%w: assistant custom parts", model.ErrUnsupported)

		default:
			return nil, nil, fmt.Errorf("%w: assistant part %T", model.ErrUnsupported, part)
		}
	}

	return items, warnings, nil
}

func functionCallOf(part model.ToolCallPart) (functionCallInput, error) {
	input := part.Input
	if isAbsent(input) {
		input = []byte("{}")
	}

	arguments, err := toolresult.CompactJSON(input)
	if err != nil {
		return functionCallInput{}, err
	}

	signature, err := thoughtSignature(part.ProviderOptions)
	if err != nil {
		return functionCallInput{}, err
	}

	return functionCallInput{
		Type:             inputFunctionCall,
		CallID:           part.ToolCallID,
		Name:             part.ToolName,
		Arguments:        arguments,
		ThoughtSignature: signature,
	}, nil
}

func functionOutputOf(part model.ToolResultPart) (functionOutputInput, error) {
	output, err := serializeToolOutput(part.Output)
	if err != nil {
		return functionOutputInput{}, err
	}

	signature, err := thoughtSignature(part.ProviderOptions)
	if err != nil {
		return functionOutputInput{}, err
	}

	return functionOutputInput{
		Type:             inputFunctionReply,
		CallID:           part.ToolCallID,
		Name:             part.ToolName,
		Output:           output,
		ThoughtSignature: signature,
	}, nil
}

// convertToolMessage sends each tool result. Approval responses are not
// supported.
func convertToolMessage(message model.ToolMessage) ([]inputItem, error) {
	var items []inputItem

	for _, part := range message.Content {
		result, ok := part.(model.ToolResultPart)
		if !ok {
			return nil, fmt.Errorf("%w: tool approval responses", model.ErrUnsupported)
		}

		item, err := functionOutputOf(result)
		if err != nil {
			return nil, err
		}

		items = append(items, item)
	}

	return items, nil
}

// serializeToolOutput returns a tool result as text. Multipart results may
// contain text only.
func serializeToolOutput(output model.ToolResultOutput) (string, error) {
	content, ok := output.(model.ContentOutput)
	if !ok {
		return toolresult.Text(output)
	}

	var text strings.Builder
	for _, item := range content.Value {
		textItem, ok := item.(model.ToolResultText)
		if !ok {
			return "", fmt.Errorf("%w: file and custom tool result content", model.ErrUnsupported)
		}

		text.WriteString(textItem.Text)
	}

	return text.String(), nil
}

// thoughtSignature reads the thought signature replayed under the
// "perplexity" namespace.
func thoughtSignature(options model.ProviderOptions) (string, error) {
	metadata, _, err := model.ProviderValue[ToolCallMetadata](options, Namespace)
	if err != nil {
		return "", err
	}

	return metadata.ThoughtSignature, nil
}
