package moonshotai

import (
	"fmt"
	"slices"
	"strings"

	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/internal/toolresult"
	"github.com/aarock1234/ai/pkg/model"
)

// fileReferencePrefix starts the URL of a file uploaded to the Moonshot
// Files API.
const fileReferencePrefix = "ms://"

var supportedImageMediaTypes = []string{
	"image/jpeg",
	"image/png",
	"image/gif",
	"image/webp",
	"image/bmp",
	"image/heic",
	"image/heif",
}

var supportedVideoMediaTypes = []string{
	"video/mp4",
	"video/mpeg",
	"video/mov",
	"video/avi",
	"video/x-flv",
	"video/mpg",
	"video/webm",
	"video/wmv",
	"video/3gpp",
}

// promptSettings configures prompt conversion.
type promptSettings struct {
	modelID string

	// jsonObjectFormat reports that the call requests the JSON object
	// response format, which partial mode cannot be combined with.
	jsonObjectFormat bool
}

// convertPrompt converts the prompt into chat messages. Moonshot accepts
// text, image and video content only; other files fail here instead of
// being rejected by the API.
func convertPrompt(prompt []model.Message, settings promptSettings) ([]chatMessage, []model.Warning, error) {
	var messages []chatMessage
	var warnings []model.Warning

	for index, message := range prompt {
		options, err := readMessageOptions(messageProviderOptions(message))
		if err != nil {
			return nil, nil, err
		}

		if options.isPartial() && message.Role() != model.RoleAssistant {
			return nil, nil, fmt.Errorf("%w: Moonshot AI Partial Mode requires `partial: true` on an assistant message", model.ErrInvalidPrompt)
		}

		if options.Tools != nil && message.Role() != model.RoleSystem {
			return nil, nil, fmt.Errorf("%w: Moonshot dynamic tools must be configured on a system message", model.ErrInvalidPrompt)
		}

		converted, messageWarnings, err := convertMessage(message, options, messagePlacement{
			settings: settings,
			isLast:   index == len(prompt)-1,
		})
		if err != nil {
			return nil, nil, err
		}

		messages = append(messages, converted...)
		warnings = append(warnings, messageWarnings...)
	}

	return messages, warnings, nil
}

// messagePlacement describes where a message appears in the call.
type messagePlacement struct {
	settings promptSettings
	isLast   bool
}

func convertMessage(message model.Message, options MessageOptions, placement messagePlacement) ([]chatMessage, []model.Warning, error) {
	switch message := message.(type) {
	case model.SystemMessage:
		return convertSystemMessage(message, options, placement.settings.modelID)

	case model.UserMessage:
		converted, err := convertUserMessage(message, options)
		if err != nil {
			return nil, nil, err
		}

		return []chatMessage{
			converted,
		}, nil, nil

	case model.AssistantMessage:
		converted, err := convertAssistantMessage(message, options, placement)
		if err != nil {
			return nil, nil, err
		}

		return []chatMessage{
			converted,
		}, nil, nil

	case model.ToolMessage:
		return convertToolMessage(message, options)

	default:
		return nil, nil, fmt.Errorf("%w: unsupported role %q", model.ErrInvalidPrompt, message.Role())
	}
}

// convertSystemMessage converts instructions, or a dynamic tool message on
// Kimi K3 and custom models.
func convertSystemMessage(message model.SystemMessage, options MessageOptions, modelID string) ([]chatMessage, []model.Warning, error) {
	if len(options.Tools) == 0 {
		return []chatMessage{
			systemMessage{
				Role:    roleSystem,
				Content: message.Content,
				Name:    options.Name,
			},
		}, nil, nil
	}

	if message.Content != "" {
		return nil, nil, fmt.Errorf(
			"%w: A Moonshot dynamic-tool system message must use empty content because the API forbids content alongside tools",
			model.ErrInvalidPrompt,
		)
	}

	family := ModelFamily(modelID)
	if family != FamilyKimiK3 && family != FamilyUnknown {
		return nil, []model.Warning{
			model.UnsupportedWarning(
				fmt.Sprintf("dynamic tool loading for model \"%s\"", modelID),
				"Moonshot documents dynamic tool loading only for Kimi K3. The dynamic system message has been omitted.",
			),
		}, nil
	}

	tools := make([]functionTool, 0, len(options.Tools))
	for _, tool := range options.Tools {
		converted, err := convertFunctionTool(functionToolSpec{
			name:        tool.Name,
			description: tool.Description,
			inputSchema: tool.InputSchema,
			strict:      tool.Strict,
		})
		if err != nil {
			return nil, nil, err
		}

		tools = append(tools, converted)
	}

	return []chatMessage{
		systemToolsMessage{
			Role:  roleSystem,
			Tools: tools,
		},
	}, nil, nil
}

// convertUserMessage sends a single text part as a string and other content
// as parts.
func convertUserMessage(message model.UserMessage, options MessageOptions) (userMessage, error) {
	converted := userMessage{
		Role: roleUser,
		Name: options.Name,
	}

	if len(message.Content) == 1 {
		if text, ok := message.Content[0].(model.TextPart); ok {
			converted.Content = userContent{
				text: text.Text,
			}

			return converted, nil
		}
	}

	parts := make([]contentPart, 0, len(message.Content))
	for _, part := range message.Content {
		convertedPart, err := convertUserPart(part)
		if err != nil {
			return userMessage{}, err
		}

		parts = append(parts, convertedPart)
	}

	converted.Content = userContent{
		parts: parts,
	}

	return converted, nil
}

func convertUserPart(part model.UserPart) (contentPart, error) {
	switch part := part.(type) {
	case model.TextPart:
		return textPart{
			Type: partText,
			Text: part.Text,
		}, nil

	case model.FilePart:
		return convertFilePart(part)

	default:
		return nil, fmt.Errorf("%w: user part %T", model.ErrUnsupported, part)
	}
}

func convertFilePart(part model.FilePart) (contentPart, error) {
	topLevel := mediatype.TopLevel(part.MediaType)

	switch data := part.Data.(type) {
	case model.FileReference:
		return referencePart(part, data, topLevel)

	case model.FileText:
		if topLevel != "text" {
			return nil, unsupportedMediaType(part.MediaType)
		}

		return textPart{
			Type: partText,
			Text: data.Text,
		}, nil

	case model.FileURL, model.FileBytes:
		return inlinePart(part, topLevel)

	default:
		return nil, fmt.Errorf("%w: file part data type %T", model.ErrUnsupported, data)
	}
}

// referencePart sends an image or video uploaded to the Moonshot Files API
// by its ms:// URL.
func referencePart(part model.FilePart, reference model.FileReference, topLevel string) (contentPart, error) {
	if topLevel != "image" && topLevel != "video" {
		return nil, unsupportedMediaType(part.MediaType)
	}

	location, ok := reference.IDs[Namespace]
	if !ok {
		return nil, fmt.Errorf("%w: file reference has no %s file ID", model.ErrInvalidPrompt, Namespace)
	}

	if !strings.HasPrefix(location, fileReferencePrefix) {
		return nil, fmt.Errorf("%w: Moonshot file provider references without an ms:// URL", model.ErrUnsupported)
	}

	return mediaPart(topLevel, location), nil
}

// inlinePart sends images and videos as URLs or data URLs and text files as
// text.
func inlinePart(part model.FilePart, topLevel string) (contentPart, error) {
	switch topLevel {
	case "image":
		location, err := mediaLocation(part, supportedImageMediaTypes, topLevel)
		if err != nil {
			return nil, err
		}

		return mediaPart(topLevel, location), nil

	case "video":
		location, err := mediaLocation(part, supportedVideoMediaTypes, topLevel)
		if err != nil {
			return nil, err
		}

		return mediaPart(topLevel, location), nil

	case "text":
		return textPart{
			Type: partText,
			Text: inlineText(part.Data),
		}, nil

	default:
		return nil, unsupportedMediaType(part.MediaType)
	}
}

func mediaPart(topLevel string, location string) contentPart {
	if topLevel == "image" {
		return imagePart{
			Type: partImage,
			ImageURL: mediaURL{
				URL: location,
			},
		}
	}

	return videoPart{
		Type: partVideo,
		VideoURL: mediaURL{
			URL: location,
		},
	}
}

// mediaLocation returns the URL of a URL file, or a data URL of inline
// bytes, after checking the media type against the supported types. URL
// files may also declare only the top-level type.
func mediaLocation(part model.FilePart, supported []string, topLevel string) (string, error) {
	if location, ok := part.Data.(model.FileURL); ok {
		mediaType := part.MediaType
		declaresTopLevelOnly := mediaType == topLevel || mediaType == topLevel+"/*"

		if !slices.Contains(supported, mediaType) && !declaresTopLevelOnly {
			return "", unsupportedMediaType(mediaType)
		}

		return location.URL.String(), nil
	}

	mediaType, err := mediatype.ResolveFile(part.MediaType, part.Data)
	if err != nil {
		return "", err
	}

	if !slices.Contains(supported, mediaType) {
		return "", unsupportedMediaType(mediaType)
	}

	inline, ok := part.Data.(model.FileBytes)
	if !ok {
		return "", fmt.Errorf("%w: file part data type %T", model.ErrUnsupported, part.Data)
	}

	return dataurl.Encode(mediaType, inline.Data), nil
}

// inlineText returns a text file URL as text, or the decoded file content.
func inlineText(data model.FileData) string {
	switch data := data.(type) {
	case model.FileURL:
		return data.URL.String()
	case model.FileBytes:
		return string(data.Data)
	default:
		return ""
	}
}

func unsupportedMediaType(mediaType string) error {
	return fmt.Errorf("%w: file part media type %s", model.ErrUnsupported, mediaType)
}

// convertAssistantMessage joins the text and reasoning parts and lists the
// tool calls.
func convertAssistantMessage(message model.AssistantMessage, options MessageOptions, placement messagePlacement) (assistantMessage, error) {
	if options.isPartial() {
		if !placement.isLast {
			return assistantMessage{}, fmt.Errorf(
				"%w: Moonshot AI Partial Mode requires the partial assistant message to be the final message",
				model.ErrInvalidPrompt,
			)
		}

		if placement.settings.jsonObjectFormat {
			return assistantMessage{}, fmt.Errorf(
				"%w: Moonshot AI Partial Mode cannot be combined with JSON object response format",
				model.ErrInvalidPrompt,
			)
		}
	}

	var text strings.Builder
	var reasoning strings.Builder
	var toolCalls []messageToolCall

	for _, part := range message.Content {
		switch part := part.(type) {
		case model.TextPart:
			text.WriteString(part.Text)

		case model.ReasoningPart:
			reasoning.WriteString(part.Text)

		case model.ToolCallPart:
			arguments, err := toolresult.CompactJSON(part.Input)
			if err != nil {
				return assistantMessage{}, err
			}

			toolCalls = append(toolCalls, messageToolCall{
				ID:   part.ToolCallID,
				Type: functionType,
				Function: functionCall{
					Name:      part.ToolName,
					Arguments: arguments,
				},
			})
		}
	}

	content := new(text.String())
	if len(toolCalls) > 0 && *content == "" {
		content = nil
	}

	converted := assistantMessage{
		Role:             roleAssistant,
		Content:          content,
		Name:             options.Name,
		ReasoningContent: reasoning.String(),
		ToolCalls:        toolCalls,
	}

	if options.isPartial() {
		converted.Partial = new(true)
	}

	return converted, nil
}

// convertToolMessage sends each tool result as a tool message. Approval
// responses are skipped.
func convertToolMessage(message model.ToolMessage, options MessageOptions) ([]chatMessage, []model.Warning, error) {
	var warnings []model.Warning
	if options.Name != nil {
		warnings = append(warnings, model.UnsupportedWarning("message name on tool messages", ""))
	}

	var messages []chatMessage

	for _, part := range message.Content {
		result, ok := part.(model.ToolResultPart)
		if !ok {
			continue
		}

		content, err := toolresult.Text(result.Output)
		if err != nil {
			return nil, nil, err
		}

		messages = append(messages, toolMessage{
			Role:       roleTool,
			Content:    content,
			ToolCallID: result.ToolCallID,
		})
	}

	return messages, warnings, nil
}

// messageProviderOptions returns the provider options of a message.
func messageProviderOptions(message model.Message) model.ProviderOptions {
	switch message := message.(type) {
	case model.SystemMessage:
		return message.ProviderOptions
	case model.UserMessage:
		return message.ProviderOptions
	case model.AssistantMessage:
		return message.ProviderOptions
	case model.ToolMessage:
		return message.ProviderOptions
	default:
		return nil
	}
}
