package xai

import (
	"fmt"

	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/internal/toolresult"
	"github.com/aarock1234/ai/pkg/model"
)

// Input roles, item types and part types.
const (
	roleSystem    = "system"
	roleUser      = "user"
	roleAssistant = "assistant"

	inputFunctionCall   = "function_call"
	inputFunctionOutput = "function_call_output"
	inputReasoning      = "reasoning"

	partInputText  = "input_text"
	partInputImage = "input_image"
	partInputFile  = "input_file"

	statusCompleted     = "completed"
	summaryTextType     = "summary_text"
	deniedToolExecution = "tool execution denied"
)

// convertInput converts the prompt into Responses API input items.
func convertInput(prompt []model.Message) ([]inputItem, []model.Warning, error) {
	input := []inputItem{}
	var warnings []model.Warning

	for _, message := range prompt {
		switch message := message.(type) {
		case model.SystemMessage:
			input = append(input, systemInput{
				Role:    roleSystem,
				Content: message.Content,
			})

		case model.UserMessage:
			item, err := convertUserMessage(message)
			if err != nil {
				return nil, nil, err
			}

			input = append(input, item)

		case model.AssistantMessage:
			items, messageWarnings, err := convertAssistantMessage(message)
			if err != nil {
				return nil, nil, err
			}

			input = append(input, items...)
			warnings = append(warnings, messageWarnings...)

		case model.ToolMessage:
			items, err := convertToolMessage(message)
			if err != nil {
				return nil, nil, err
			}

			input = append(input, items...)

		default:
			warnings = append(warnings, model.Warning{
				Type:    model.WarningOther,
				Message: "unsupported message role",
			})
		}
	}

	return input, warnings, nil
}

func convertUserMessage(message model.UserMessage) (userInput, error) {
	parts := []inputPart{}

	for _, part := range message.Content {
		switch part := part.(type) {
		case model.TextPart:
			parts = append(parts, inputPart{
				Type: partInputText,
				Text: new(part.Text),
			})

		case model.FilePart:
			converted, err := convertFilePart(part)
			if err != nil {
				return userInput{}, err
			}

			parts = append(parts, converted)
		}
	}

	return userInput{
		Role:    roleUser,
		Content: parts,
	}, nil
}

// convertFilePart sends uploaded files by ID, images by URL or inline, and
// other files by URL. Inline non-image files need the Files API.
func convertFilePart(part model.FilePart) (inputPart, error) {
	switch data := part.Data.(type) {
	case model.FileReference:
		fileID, ok := data.IDs[Namespace]
		if !ok {
			return inputPart{}, fmt.Errorf("%w: file reference has no %s file ID", model.ErrInvalidPrompt, Namespace)
		}

		return inputPart{
			Type:   partInputFile,
			FileID: fileID,
		}, nil

	case model.FileText:
		return inputPart{}, fmt.Errorf("%w: text file parts", model.ErrUnsupported)

	case model.FileURL, model.FileBytes:
		if mediatype.TopLevel(part.MediaType) == "image" {
			return imagePart(part)
		}

		location, ok := data.(model.FileURL)
		if !ok {
			return inputPart{}, fmt.Errorf(
				"%w: file part media type %s as inline data (xAI Responses requires a URL or a Files API reference for non-image files)",
				model.ErrUnsupported,
				part.MediaType,
			)
		}

		return inputPart{
			Type:    partInputFile,
			FileURL: location.URL.String(),
		}, nil

	default:
		return inputPart{}, fmt.Errorf("%w: file part data type %T", model.ErrUnsupported, data)
	}
}

func imagePart(part model.FilePart) (inputPart, error) {
	location, err := imageLocation(part.MediaType, part.Data)
	if err != nil {
		return inputPart{}, err
	}

	partOptions, err := readFilePartOptions(part.ProviderOptions)
	if err != nil {
		return inputPart{}, err
	}

	return inputPart{
		Type:     partInputImage,
		ImageURL: location,
		Detail:   partOptions.ImageDetail,
	}, nil
}

// imageLocation returns an image URL, or a data URL of inline bytes.
func imageLocation(mediaType string, data model.FileData) (string, error) {
	if location, ok := data.(model.FileURL); ok {
		return location.URL.String(), nil
	}

	inline, ok := data.(model.FileBytes)
	if !ok {
		return "", fmt.Errorf("%w: file part data type %T", model.ErrUnsupported, data)
	}

	resolved, err := mediatype.ResolveFile(mediaType, inline)
	if err != nil {
		return "", err
	}

	return dataurl.Encode(resolved, inline.Data), nil
}

// convertAssistantMessage sends text, local tool calls and replayable
// reasoning. Provider-executed calls and tool results are not sent.
func convertAssistantMessage(message model.AssistantMessage) ([]inputItem, []model.Warning, error) {
	var items []inputItem
	var warnings []model.Warning

	for _, part := range message.Content {
		switch part := part.(type) {
		case model.TextPart:
			items = append(items, assistantInput{
				Role:    roleAssistant,
				Content: part.Text,
				ID:      readReplayedItem(part.ProviderOptions).ItemID,
			})

		case model.ToolCallPart:
			if part.ProviderExecuted {
				continue
			}

			item, err := functionCallItem(part)
			if err != nil {
				return nil, nil, err
			}

			items = append(items, item)

		case model.ToolResultPart, model.ToolApprovalRequestPart:

		case model.ReasoningPart:
			item, ok := reasoningItem(part)
			if !ok {
				warnings = append(warnings, model.Warning{
					Type:    model.WarningOther,
					Message: "Reasoning parts without itemId or encrypted content cannot be sent back to xAI. Skipping.",
				})

				continue
			}

			items = append(items, item)

		case model.ReasoningFilePart:
			warnings = append(warnings, unsupportedAssistantPart("reasoning-file"))

		case model.CustomPart:
			warnings = append(warnings, unsupportedAssistantPart("custom"))

		case model.FilePart:
			warnings = append(warnings, unsupportedAssistantPart("file"))
		}
	}

	return items, warnings, nil
}

func unsupportedAssistantPart(partType string) model.Warning {
	return model.Warning{
		Type:    model.WarningOther,
		Message: fmt.Sprintf("xAI Responses API does not support %s in assistant messages", partType),
	}
}

func functionCallItem(part model.ToolCallPart) (functionCallInput, error) {
	arguments, err := toolresult.CompactJSON(part.Input)
	if err != nil {
		return functionCallInput{}, err
	}

	itemID := part.ToolCallID
	if replayed := readReplayedItem(part.ProviderOptions).ItemID; replayed != nil {
		itemID = *replayed
	}

	return functionCallInput{
		Type:      inputFunctionCall,
		ID:        itemID,
		CallID:    part.ToolCallID,
		Name:      part.ToolName,
		Arguments: arguments,
		Status:    statusCompleted,
	}, nil
}

// reasoningItem replays reasoning that has an item ID or encrypted content.
func reasoningItem(part model.ReasoningPart) (reasoningInput, bool) {
	replayed := readReplayedItem(part.ProviderOptions)
	if replayed.ItemID == nil && replayed.ReasoningEncryptedContent == nil {
		return reasoningInput{}, false
	}

	item := reasoningInput{
		Type:             inputReasoning,
		Summary:          []summaryText{},
		Status:           statusCompleted,
		EncryptedContent: replayed.ReasoningEncryptedContent,
	}

	if replayed.ItemID != nil {
		item.ID = *replayed.ItemID
	}

	if part.Text != "" {
		item.Summary = append(item.Summary, summaryText{
			Type: summaryTextType,
			Text: part.Text,
		})
	}

	return item, true
}

// convertToolMessage sends each tool result as a function call output.
// Approval responses are skipped.
func convertToolMessage(message model.ToolMessage) ([]inputItem, error) {
	var items []inputItem

	for _, part := range message.Content {
		result, ok := part.(model.ToolResultPart)
		if !ok {
			continue
		}

		output, err := functionOutputOf(result.Output)
		if err != nil {
			return nil, err
		}

		items = append(items, functionOutputInput{
			Type:   inputFunctionOutput,
			CallID: result.ToolCallID,
			Output: output,
		})
	}

	return items, nil
}

// functionOutputOf sends multipart results as text and image parts and other
// results as text.
func functionOutputOf(output model.ToolResultOutput) (functionOutput, error) {
	switch output := output.(type) {
	case model.ExecutionDeniedOutput:
		reason := output.Reason
		if reason == "" {
			reason = deniedToolExecution
		}

		return functionOutput{
			text: reason,
		}, nil

	case model.ContentOutput:
		return contentOutput(output.Value)

	default:
		text, err := toolresult.Text(output)
		if err != nil {
			return functionOutput{}, err
		}

		return functionOutput{
			text: text,
		}, nil
	}
}

// contentOutput keeps text and URL or inline images; other items are
// omitted.
func contentOutput(items []model.ToolResultContent) (functionOutput, error) {
	parts := []inputPart{}

	for _, item := range items {
		switch item := item.(type) {
		case model.ToolResultText:
			parts = append(parts, inputPart{
				Type: partInputText,
				Text: new(item.Text),
			})

		case model.ToolResultFile:
			if mediatype.TopLevel(item.MediaType) != "image" {
				continue
			}

			switch item.Data.(type) {
			case model.FileURL, model.FileBytes:
			default:
				continue
			}

			location, err := imageLocation(item.MediaType, item.Data)
			if err != nil {
				return functionOutput{}, err
			}

			parts = append(parts, inputPart{
				Type:     partInputImage,
				ImageURL: location,
			})
		}
	}

	return functionOutput{
		parts: parts,
	}, nil
}
