package openaicompatible

import (
	"encoding/base64"
	"fmt"
	"maps"
	"strings"
	"unicode/utf8"

	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/internal/toolresult"
	"github.com/aarock1234/ai/pkg/model"
)

// promptSettings controls prompt conversion.
type promptSettings struct {
	// provider is the provider options name, used to resolve file
	// references.
	provider string

	// metadataKey is the namespace that holds tool call thought signatures.
	metadataKey string

	supportsMultiPartToolContent bool
}

// convertChatPrompt converts the prompt into chat messages.
func convertChatPrompt(prompt []model.Message, settings promptSettings) ([]chatMessage, error) {
	messages := make([]chatMessage, 0, len(prompt))

	for _, message := range prompt {
		switch message := message.(type) {
		case model.SystemMessage:
			converted, err := convertSystemMessage(message)
			if err != nil {
				return nil, err
			}

			messages = append(messages, converted)

		case model.UserMessage:
			converted, err := convertUserMessage(message, settings)
			if err != nil {
				return nil, err
			}

			messages = append(messages, converted)

		case model.AssistantMessage:
			converted, err := convertAssistantMessage(message, settings)
			if err != nil {
				return nil, err
			}

			messages = append(messages, converted)

		case model.ToolMessage:
			converted, err := convertToolMessage(message, settings)
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

// extraMembers returns the members of the "openaiCompatible" provider options,
// which are added to the converted message or part.
func extraMembers(options model.ProviderOptions) (messageExtra, error) {
	members, _, err := model.ProviderValue[messageExtra](options, Namespace)

	return members, err
}

func convertSystemMessage(message model.SystemMessage) (chatMessage, error) {
	extra, err := extraMembers(message.ProviderOptions)
	if err != nil {
		return nil, err
	}

	return systemMessage{
		Role:    "system",
		Content: message.Content,
		Extra:   extra,
	}, nil
}

// convertUserMessage sends a lone text part as plain text and anything else
// as content parts.
func convertUserMessage(message model.UserMessage, settings promptSettings) (chatMessage, error) {
	if len(message.Content) == 1 {
		if text, ok := message.Content[0].(model.TextPart); ok {
			extra, err := extraMembers(text.ProviderOptions)
			if err != nil {
				return nil, err
			}

			return userMessage{
				Role: "user",
				Content: messageContent{
					Text: text.Text,
				},
				Extra: extra,
			}, nil
		}
	}

	extra, err := extraMembers(message.ProviderOptions)
	if err != nil {
		return nil, err
	}

	parts := make([]contentPart, 0, len(message.Content))
	for _, part := range message.Content {
		converted, err := convertUserPart(part, settings.provider)
		if err != nil {
			return nil, err
		}

		parts = append(parts, converted)
	}

	return userMessage{
		Role: "user",
		Content: messageContent{
			Parts: parts,
		},
		Extra: extra,
	}, nil
}

func convertUserPart(part model.UserPart, provider string) (contentPart, error) {
	switch part := part.(type) {
	case model.TextPart:
		return textContentPart(part.Text, part.ProviderOptions)

	case model.FilePart:
		return fileContentPart(part, provider)

	default:
		return contentPart{}, fmt.Errorf("%w: user part %T", model.ErrUnsupported, part)
	}
}

func textContentPart(text string, options model.ProviderOptions) (contentPart, error) {
	extra, err := extraMembers(options)
	if err != nil {
		return contentPart{}, err
	}

	return contentPart{
		Type:  contentPartText,
		Text:  &text,
		Extra: extra,
	}, nil
}

// fileContentPart converts a file by its top-level media type: images,
// videos, audio, PDF documents and text.
func fileContentPart(part model.FilePart, provider string) (contentPart, error) {
	extra, err := extraMembers(part.ProviderOptions)
	if err != nil {
		return contentPart{}, err
	}

	converted, err := fileContent(part, provider)
	if err != nil {
		return contentPart{}, err
	}

	converted.Extra = extra

	return converted, nil
}

func fileContent(part model.FilePart, provider string) (contentPart, error) {
	switch data := part.Data.(type) {
	case model.FileReference:
		fileID, ok := data.IDs[provider]
		if !ok {
			return contentPart{}, fmt.Errorf("%w: file reference has no ID for provider %q", model.ErrInvalidPrompt, provider)
		}

		return contentPart{
			Type: contentPartFile,
			File: &fileAttachment{
				FileID: fileID,
			},
		}, nil

	case model.FileText:
		return contentPart{}, fmt.Errorf("%w: text file parts", model.ErrUnsupported)

	case model.FileURL, model.FileBytes:
		return mediaContent(part)

	default:
		return contentPart{}, fmt.Errorf("%w: file data %T", model.ErrUnsupported, data)
	}
}

// mediaContent converts a file given as a URL or inline bytes.
func mediaContent(part model.FilePart) (contentPart, error) {
	switch mediatype.TopLevel(part.MediaType) {
	case "image":
		address, err := fileAddress(part)
		if err != nil {
			return contentPart{}, err
		}

		return contentPart{
			Type: contentPartImageURL,
			ImageURL: &mediaURL{
				URL: address,
			},
		}, nil

	case "video":
		address, err := fileAddress(part)
		if err != nil {
			return contentPart{}, err
		}

		return contentPart{
			Type: contentPartVideoURL,
			VideoURL: &mediaURL{
				URL: address,
			},
		}, nil

	case "audio":
		return audioContent(part)

	case "application":
		return documentContent(part)

	case "text":
		return textFileContent(part)

	default:
		return contentPart{}, fmt.Errorf("%w: file part media type %s", model.ErrUnsupported, part.MediaType)
	}
}

func audioContent(part model.FilePart) (contentPart, error) {
	data, ok := part.Data.(model.FileBytes)
	if !ok {
		return contentPart{}, fmt.Errorf("%w: audio file parts with URLs", model.ErrUnsupported)
	}

	mediaType, err := mediatype.ResolveFile(part.MediaType, part.Data)
	if err != nil {
		return contentPart{}, err
	}

	format, ok := audioFormatOf(mediaType)
	if !ok {
		return contentPart{}, fmt.Errorf("%w: audio media type %s", model.ErrUnsupported, mediaType)
	}

	return contentPart{
		Type: contentPartInputAudio,
		InputAudio: &inputAudio{
			Data:   base64.StdEncoding.EncodeToString(data.Data),
			Format: format,
		},
	}, nil
}

func audioFormatOf(mediaType string) (audioFormat, bool) {
	switch mediaType {
	case "audio/wav":
		return audioFormatWAV, true
	case "audio/mp3", "audio/mpeg":
		return audioFormatMP3, true
	default:
		return "", false
	}
}

func documentContent(part model.FilePart) (contentPart, error) {
	data, ok := part.Data.(model.FileBytes)
	if !ok {
		return contentPart{}, fmt.Errorf("%w: PDF file parts with URLs", model.ErrUnsupported)
	}

	mediaType, err := mediatype.ResolveFile(part.MediaType, part.Data)
	if err != nil {
		return contentPart{}, err
	}

	if mediaType != "application/pdf" {
		return contentPart{}, fmt.Errorf("%w: file part media type %s", model.ErrUnsupported, mediaType)
	}

	filename := part.Filename
	if filename == "" {
		filename = "document.pdf"
	}

	return contentPart{
		Type: contentPartFile,
		File: &fileAttachment{
			Filename: filename,
			FileData: dataurl.Encode("application/pdf", data.Data),
		},
	}, nil
}

// textFileContent sends a text file as text: its decoded content, or its URL.
func textFileContent(part model.FilePart) (contentPart, error) {
	var text string

	switch data := part.Data.(type) {
	case model.FileURL:
		text = data.URL.String()

	case model.FileBytes:
		if !utf8.Valid(data.Data) {
			return contentPart{}, fmt.Errorf("%w: text file part is not valid UTF-8", model.ErrInvalidPrompt)
		}

		text = string(data.Data)
	}

	return contentPart{
		Type: contentPartText,
		Text: &text,
	}, nil
}

// fileAddress returns the URL of a URL file or a data URL of inline bytes.
func fileAddress(part model.FilePart) (string, error) {
	if data, ok := part.Data.(model.FileURL); ok {
		return data.URL.String(), nil
	}

	data, ok := part.Data.(model.FileBytes)
	if !ok {
		return "", fmt.Errorf("%w: file data %T", model.ErrUnsupported, part.Data)
	}

	mediaType, err := mediatype.ResolveFile(part.MediaType, part.Data)
	if err != nil {
		return "", err
	}

	return dataurl.Encode(mediaType, data.Data), nil
}

// convertAssistantMessage joins the text and reasoning parts and collects the
// tool calls. Other parts are not sent.
func convertAssistantMessage(message model.AssistantMessage, settings promptSettings) (chatMessage, error) {
	extra, err := extraMembers(message.ProviderOptions)
	if err != nil {
		return nil, err
	}

	var text string
	var reasoning strings.Builder
	var toolCalls []messageToolCall

	for _, part := range message.Content {
		switch part := part.(type) {
		case model.TextPart:
			text += part.Text

		case model.ReasoningPart:
			reasoning.WriteString(part.Text)

		case model.ToolCallPart:
			toolCall, err := convertToolCall(part, settings.metadataKey)
			if err != nil {
				return nil, err
			}

			toolCalls = append(toolCalls, toolCall)
		}
	}

	content := &text
	if len(toolCalls) > 0 && text == "" {
		content = nil
	}

	return assistantMessage{
		Role:             "assistant",
		Content:          content,
		ReasoningContent: reasoning.String(),
		ToolCalls:        toolCalls,
		Extra:            extra,
	}, nil
}

// convertToolCall converts a tool call and replays its Google thought
// signature.
func convertToolCall(part model.ToolCallPart, metadataKey string) (messageToolCall, error) {
	extra, err := extraMembers(part.ProviderOptions)
	if err != nil {
		return messageToolCall{}, err
	}

	signature, err := thoughtSignature(part.ProviderOptions, metadataKey)
	if err != nil {
		return messageToolCall{}, err
	}

	arguments, err := toolresult.CompactJSON(part.Input)
	if err != nil {
		return messageToolCall{}, err
	}

	toolCall := messageToolCall{
		ID:   part.ToolCallID,
		Type: "function",
		Function: functionCall{
			Name:      part.ToolName,
			Arguments: arguments,
		},
		Extra: extra,
	}

	if signature != "" {
		toolCall.Extra = maps.Clone(extra)
		delete(toolCall.Extra, "extra_content")

		toolCall.ExtraContent = &extraContent{
			Google: &googleExtraContent{
				ThoughtSignature: &signature,
			},
		}
	}

	return toolCall, nil
}

// thoughtSignature reads a tool call's thought signature from the metadata
// namespace, falling back to the "google" namespace.
func thoughtSignature(options model.ProviderOptions, metadataKey string) (string, error) {
	for _, namespace := range []string{metadataKey, "google"} {
		metadata, _, err := model.ProviderValue[ToolCallMetadata](options, namespace)
		if err != nil {
			return "", err
		}

		if metadata.ThoughtSignature != "" {
			return metadata.ThoughtSignature, nil
		}
	}

	return "", nil
}

// convertToolMessage returns one message per tool result. Approval responses
// are not sent.
func convertToolMessage(message model.ToolMessage, settings promptSettings) ([]chatMessage, error) {
	var messages []chatMessage

	for _, part := range message.Content {
		result, ok := part.(model.ToolResultPart)
		if !ok {
			continue
		}

		content, err := toolResultContent(result.Output, settings)
		if err != nil {
			return nil, err
		}

		extra, err := extraMembers(result.ProviderOptions)
		if err != nil {
			return nil, err
		}

		messages = append(messages, toolMessage{
			Role:       "tool",
			ToolCallID: result.ToolCallID,
			Content:    content,
			Extra:      extra,
		})
	}

	return messages, nil
}

// toolResultContent sends multipart content as content parts when the
// provider supports them, and every other output as text.
func toolResultContent(output model.ToolResultOutput, settings promptSettings) (messageContent, error) {
	content, isContent := output.(model.ContentOutput)
	if isContent && settings.supportsMultiPartToolContent {
		return multiPartToolContent(content, settings)
	}

	text, err := toolresult.Text(output)
	if err != nil {
		return messageContent{}, err
	}

	return messageContent{
		Text: text,
	}, nil
}

// multiPartToolContent converts multipart tool result content to content
// parts.
func multiPartToolContent(output model.ContentOutput, settings promptSettings) (messageContent, error) {
	parts := make([]contentPart, 0, len(output.Value))
	for _, item := range output.Value {
		part, err := toolContentPart(item, settings.provider)
		if err != nil {
			return messageContent{}, err
		}

		parts = append(parts, part)
	}

	return messageContent{
		Parts: parts,
	}, nil
}

func toolContentPart(item model.ToolResultContent, provider string) (contentPart, error) {
	switch item := item.(type) {
	case model.ToolResultText:
		return textContentPart(item.Text, item.ProviderOptions)

	case model.ToolResultFile:
		//nolint:staticcheck // tool result files and prompt file parts are distinct contracts that happen to share fields
		file := model.FilePart{
			Filename:        item.Filename,
			MediaType:       item.MediaType,
			Data:            item.Data,
			ProviderOptions: item.ProviderOptions,
		}

		return fileContentPart(file, provider)

	case model.ToolResultCustom:
		return contentPart{}, fmt.Errorf("%w: custom tool content parts", model.ErrUnsupported)

	default:
		return contentPart{}, fmt.Errorf("%w: tool content part %T", model.ErrUnsupported, item)
	}
}
