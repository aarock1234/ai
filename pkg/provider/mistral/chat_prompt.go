package mistral

import (
	"fmt"
	"strings"

	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/internal/toolresult"
	"github.com/aarock1234/ai/pkg/model"
)

// convertChatPrompt converts the prompt into chat messages. A trailing
// assistant message is sent as a prefix that the model continues.
func convertChatPrompt(prompt []model.Message) ([]chatMessage, error) {
	messages := make([]chatMessage, 0, len(prompt))

	for index, message := range prompt {
		isLast := index == len(prompt)-1

		switch message := message.(type) {
		case model.SystemMessage:
			messages = append(messages, systemMessage{
				Role:    "system",
				Content: message.Content,
			})

		case model.UserMessage:
			content, err := userContent(message)
			if err != nil {
				return nil, err
			}

			messages = append(messages, userMessage{
				Role:    "user",
				Content: content,
			})

		case model.AssistantMessage:
			converted, err := convertAssistantMessage(message, isLast)
			if err != nil {
				return nil, err
			}

			messages = append(messages, converted)

		case model.ToolMessage:
			converted, err := convertToolMessage(message)
			if err != nil {
				return nil, err
			}

			messages = append(messages, converted...)

		default:
			return nil, fmt.Errorf("%w: unsupported message role %q", model.ErrInvalidPrompt, message.Role())
		}
	}

	return messages, nil
}

// userContent converts text, images and PDF documents.
func userContent(message model.UserMessage) ([]contentPart, error) {
	parts := make([]contentPart, 0, len(message.Content))

	for _, part := range message.Content {
		switch part := part.(type) {
		case model.TextPart:
			parts = append(parts, textPart(part.Text))

		case model.FilePart:
			converted, err := fileContentPart(part)
			if err != nil {
				return nil, err
			}

			parts = append(parts, converted)

		default:
			return nil, fmt.Errorf("%w: user part %T", model.ErrUnsupported, part)
		}
	}

	return parts, nil
}

// fileContentPart sends images as image URLs and PDFs as document URLs.
func fileContentPart(part model.FilePart) (contentPart, error) {
	switch part.Data.(type) {
	case model.FileReference:
		return contentPart{}, fmt.Errorf("%w: file parts with provider references", model.ErrUnsupported)

	case model.FileText:
		return contentPart{}, fmt.Errorf("%w: text file parts", model.ErrUnsupported)
	}

	if mediatype.TopLevel(part.MediaType) == "image" {
		address, err := fileAddress(part)
		if err != nil {
			return contentPart{}, err
		}

		return contentPart{
			Type:     contentImageURL,
			ImageURL: address,
		}, nil
	}

	if err := requirePDF(part); err != nil {
		return contentPart{}, err
	}

	address, err := fileAddress(part)
	if err != nil {
		return contentPart{}, err
	}

	return contentPart{
		Type:        contentDocumentURL,
		DocumentURL: address,
	}, nil
}

// requirePDF rejects files other than PDF documents. The type of inline bytes
// may be detected from their content.
func requirePDF(part model.FilePart) error {
	mediaType := part.MediaType
	if _, ok := part.Data.(model.FileBytes); ok {
		resolved, err := mediatype.ResolveFile(part.MediaType, part.Data)
		if err != nil {
			return err
		}

		mediaType = resolved
	}

	if mediaType != "application/pdf" {
		return fmt.Errorf("%w: only images and PDF file parts are supported", model.ErrUnsupported)
	}

	return nil
}

// fileAddress returns the URL of a URL file or a data URL of inline bytes.
func fileAddress(part model.FilePart) (string, error) {
	switch data := part.Data.(type) {
	case model.FileURL:
		return data.URL.String(), nil

	case model.FileBytes:
		mediaType, err := mediatype.ResolveFile(part.MediaType, part.Data)
		if err != nil {
			return "", err
		}

		return dataurl.Encode(mediaType, data.Data), nil

	default:
		return "", fmt.Errorf("%w: file part data type %T", model.ErrUnsupported, data)
	}
}

// convertAssistantMessage sends plain text, or text and thinking parts when
// the message has reasoning.
func convertAssistantMessage(message model.AssistantMessage, isLast bool) (chatMessage, error) {
	var text strings.Builder
	var parts []contentPart
	var toolCalls []messageToolCall
	hasReasoning := false

	for _, part := range message.Content {
		switch part := part.(type) {
		case model.TextPart:
			text.WriteString(part.Text)
			parts = append(parts, textPart(part.Text))

		case model.ReasoningPart:
			hasReasoning = true
			parts = append(parts, thinkingPart(part.Text))

		case model.ToolCallPart:
			toolCall, err := convertToolCall(part)
			if err != nil {
				return nil, err
			}

			toolCalls = append(toolCalls, toolCall)

		default:
			return nil, fmt.Errorf("%w: unsupported content type in assistant message: %T", model.ErrUnsupported, part)
		}
	}

	content := assistantContent{
		Text: text.String(),
	}
	if hasReasoning {
		content.Parts = parts
	}

	converted := assistantMessage{
		Role:      "assistant",
		Content:   content,
		ToolCalls: toolCalls,
	}

	if isLast {
		converted.Prefix = new(true)
	}

	return converted, nil
}

func convertToolCall(part model.ToolCallPart) (messageToolCall, error) {
	arguments, err := toolresult.CompactJSON(part.Input)
	if err != nil {
		return messageToolCall{}, err
	}

	return messageToolCall{
		ID:   part.ToolCallID,
		Type: "function",
		Function: functionCall{
			Name:      part.ToolName,
			Arguments: arguments,
		},
	}, nil
}

// convertToolMessage returns one message per tool result. Approval responses
// are not sent.
func convertToolMessage(message model.ToolMessage) ([]chatMessage, error) {
	var messages []chatMessage

	for _, part := range message.Content {
		result, ok := part.(model.ToolResultPart)
		if !ok {
			continue
		}

		content, err := toolresult.Text(result.Output)
		if err != nil {
			return nil, err
		}

		messages = append(messages, toolMessage{
			Role:       "tool",
			Name:       result.ToolName,
			ToolCallID: result.ToolCallID,
			Content:    content,
		})
	}

	return messages, nil
}
