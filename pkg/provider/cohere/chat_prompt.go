package cohere

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/internal/toolresult"
	"github.com/aarock1234/ai/pkg/model"
)

// chatPrompt is a prompt converted for the chat API. Non-image files become
// documents for retrieval-augmented generation.
type chatPrompt struct {
	Messages  []chatMessage
	Documents []document
}

func convertChatPrompt(prompt []model.Message) (chatPrompt, error) {
	var converted chatPrompt

	for _, message := range prompt {
		switch message := message.(type) {
		case model.SystemMessage:
			converted.Messages = append(converted.Messages, systemMessage{
				Role:    "system",
				Content: message.Content,
			})

		case model.UserMessage:
			user, documents, err := convertUserMessage(message)
			if err != nil {
				return chatPrompt{}, err
			}

			converted.Messages = append(converted.Messages, user)
			converted.Documents = append(converted.Documents, documents...)

		case model.AssistantMessage:
			assistant, err := convertAssistantMessage(message)
			if err != nil {
				return chatPrompt{}, err
			}

			converted.Messages = append(converted.Messages, assistant)

		case model.ToolMessage:
			tools, err := convertToolMessage(message)
			if err != nil {
				return chatPrompt{}, err
			}

			converted.Messages = append(converted.Messages, tools...)

		default:
			return chatPrompt{}, fmt.Errorf("%w: unsupported message role %q", model.ErrInvalidPrompt, message.Role())
		}
	}

	return converted, nil
}

// convertUserMessage sends text and images as content and other files as
// documents. Without images, the content is plain text.
func convertUserMessage(message model.UserMessage) (userMessage, []document, error) {
	var parts []contentPart
	var documents []document
	hasImage := false

	for _, part := range message.Content {
		switch part := part.(type) {
		case model.TextPart:
			if part.Text != "" {
				parts = append(parts, textContentPart(part.Text))
			}

		case model.FilePart:
			if mediatype.TopLevel(part.MediaType) == "image" {
				image, err := imageContentPart(part)
				if err != nil {
					return userMessage{}, nil, err
				}

				hasImage = true
				parts = append(parts, image)

				continue
			}

			converted, err := fileDocument(part)
			if err != nil {
				return userMessage{}, nil, err
			}

			documents = append(documents, converted)

		default:
			return userMessage{}, nil, fmt.Errorf("%w: user part %T", model.ErrUnsupported, part)
		}
	}

	content := userContent{
		Parts: parts,
	}
	if !hasImage {
		content = userContent{
			Text: joinText(parts),
		}
	}

	return userMessage{
		Role:    "user",
		Content: content,
	}, documents, nil
}

func textContentPart(text string) contentPart {
	return contentPart{
		Type: "text",
		Text: &text,
	}
}

func joinText(parts []contentPart) string {
	var text strings.Builder
	for _, part := range parts {
		if part.Text != nil {
			text.WriteString(*part.Text)
		}
	}

	return text.String()
}

func imageContentPart(part model.FilePart) (contentPart, error) {
	address, err := imageAddress(part)
	if err != nil {
		return contentPart{}, err
	}

	partOptions, err := readImagePartOptions(part.ProviderOptions)
	if err != nil {
		return contentPart{}, err
	}

	return contentPart{
		Type: "image_url",
		ImageURL: &imageURL{
			URL:    address,
			Detail: partOptions.Detail,
		},
	}, nil
}

// imageAddress returns the URL of a URL image or a data URL of inline bytes.
func imageAddress(part model.FilePart) (string, error) {
	switch data := part.Data.(type) {
	case model.FileURL:
		return data.URL.String(), nil

	case model.FileBytes:
		mediaType, err := mediatype.ResolveFile(part.MediaType, data)
		if err != nil {
			return "", err
		}

		return dataurl.Encode(mediaType, data.Data), nil

	case model.FileReference:
		return "", fmt.Errorf("%w: image file parts with provider references", model.ErrUnsupported)

	case model.FileText:
		return "", fmt.Errorf("%w: image file parts with text data", model.ErrUnsupported)

	default:
		return "", fmt.Errorf("%w: file data %T", model.ErrUnsupported, data)
	}
}

// fileDocument sends a non-image file as a text document. Invalid UTF-8 is
// replaced, as a text decoder would.
func fileDocument(part model.FilePart) (document, error) {
	var text string

	switch data := part.Data.(type) {
	case model.FileText:
		text = data.Text

	case model.FileBytes:
		text = string(data.Data)
		if !utf8.ValidString(text) {
			text = strings.ToValidUTF8(text, string(utf8.RuneError))
		}

	case model.FileReference:
		return document{}, fmt.Errorf("%w: file parts with provider references", model.ErrUnsupported)

	case model.FileURL:
		return document{}, fmt.Errorf(
			"%w: file URL data: URLs should be downloaded before they reach the provider, which indicates a configuration issue",
			model.ErrUnsupported,
		)

	default:
		return document{}, fmt.Errorf("%w: file data %T", model.ErrUnsupported, data)
	}

	return document{
		Data: documentData{
			Text:  text,
			Title: part.Filename,
		},
	}, nil
}

// convertAssistantMessage sends the text, or the tool calls without text.
// Reasoning is not sent.
func convertAssistantMessage(message model.AssistantMessage) (assistantMessage, error) {
	var text strings.Builder
	var toolCalls []messageToolCall

	for _, part := range message.Content {
		switch part := part.(type) {
		case model.TextPart:
			text.WriteString(part.Text)

		case model.ToolCallPart:
			arguments, err := toolresult.CompactJSON(part.Input)
			if err != nil {
				return assistantMessage{}, err
			}

			toolCalls = append(toolCalls, messageToolCall{
				ID:   part.ToolCallID,
				Type: "function",
				Function: functionCall{
					Name:      part.ToolName,
					Arguments: arguments,
				},
			})
		}
	}

	converted := assistantMessage{
		Role:      "assistant",
		ToolCalls: toolCalls,
	}

	if len(toolCalls) == 0 {
		converted.Content = new(text.String())
	}

	return converted, nil
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
			Content:    content,
			ToolCallID: result.ToolCallID,
		})
	}

	return messages, nil
}
