package deepseek

import (
	_ "embed"
	"fmt"
	"regexp"
	"strings"

	"github.com/aarock1234/ai/internal/dataurl"
	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/internal/toolresult"
	"github.com/aarock1234/ai/pkg/model"
)

// schemaPlaceholder is replaced with the JSON schema in the schema
// instruction.
const schemaPlaceholder = "{{schema}}"

// maxImageURLLength is the longest image URL DeepSeek accepts.
const maxImageURLLength = 8192

var (
	//go:embed prompts/json_instruction.md
	jsonInstruction string

	//go:embed prompts/json_schema_instruction.md
	jsonSchemaInstruction string
)

// supportedImageTypes are the image media types DeepSeek accepts.
var supportedImageTypes = map[string]bool{
	"image/gif":  true,
	"image/jpeg": true,
	"image/jpg":  true,
	"image/png":  true,
	"image/webp": true,
}

// legacyModelPattern matches the IDs of models older than V4: legacy aliases
// and pre-V4 versions. Unknown IDs use the current protocol.
var legacyModelPattern = regexp.MustCompile(`(^|[/.])deepseek-(?:chat|reasoner|v[1-3](?:\.\d+)?)([-/:.]|$)`)

// isV4Model reports whether a model uses DeepSeek V4 behavior: thinking on
// by default and reasoning_content required on every assistant turn.
func isV4Model(modelID string) bool {
	return !legacyModelPattern.MatchString(modelID)
}

// promptSettings configures prompt conversion.
type promptSettings struct {
	responseFormat *model.ResponseFormat
	modelID        string
	namespace      string

	supportsAssistantPrefixCompletion bool
	supportsStructuredOutputs         bool
}

// promptConverter converts one prompt and collects its warnings.
type promptConverter struct {
	settings      promptSettings
	isV4          bool
	lastUserIndex int
	warnings      []model.Warning
}

// convertChatPrompt converts the prompt into DeepSeek chat messages. JSON
// responses without structured outputs are requested with a system
// instruction.
func convertChatPrompt(prompt []model.Message, settings promptSettings) ([]chatMessage, []model.Warning, error) {
	converter := &promptConverter{
		settings:      settings,
		isV4:          isV4Model(settings.modelID),
		lastUserIndex: lastUserIndex(prompt),
	}

	messages, err := converter.jsonInstructionMessages()
	if err != nil {
		return nil, nil, err
	}

	for index, message := range prompt {
		converted, err := converter.convertMessage(message, index, len(prompt))
		if err != nil {
			return nil, nil, err
		}

		messages = append(messages, converted...)
	}

	return messages, converter.warnings, nil
}

func lastUserIndex(prompt []model.Message) int {
	for index := len(prompt) - 1; index >= 0; index-- {
		if prompt[index].Role() == model.RoleUser {
			return index
		}
	}

	return -1
}

// jsonInstructionMessages returns the system instruction of a JSON response
// format that is not sent as a JSON schema response format.
func (c *promptConverter) jsonInstructionMessages() ([]chatMessage, error) {
	format := c.settings.responseFormat
	if format == nil || format.Type != model.ResponseFormatJSON {
		return nil, nil
	}

	if format.Schema == nil {
		return []chatMessage{
			systemMessage{
				Role:    "system",
				Content: strings.TrimSpace(jsonInstruction),
			},
		}, nil
	}

	if c.settings.supportsStructuredOutputs {
		return nil, nil
	}

	schema, err := toolresult.CompactJSON(format.Schema)
	if err != nil {
		return nil, fmt.Errorf("%w: invalid response format schema: %w", model.ErrInvalidArgument, err)
	}

	c.warnings = append(c.warnings, model.CompatibilityWarning(
		"responseFormat JSON schema",
		"JSON response schema is injected into the system message.",
	))

	instruction := strings.Replace(strings.TrimSpace(jsonSchemaInstruction), schemaPlaceholder, schema, 1)

	return []chatMessage{
		systemMessage{
			Role:    "system",
			Content: instruction,
		},
	}, nil
}

// convertMessage converts one message. Only an assistant message may be a
// prefix.
func (c *promptConverter) convertMessage(message model.Message, index int, count int) ([]chatMessage, error) {
	if message == nil {
		return nil, fmt.Errorf("%w: message is nil", model.ErrInvalidPrompt)
	}

	options, err := readMessageOptions(messageProviderOptions(message), c.settings.namespace)
	if err != nil {
		return nil, err
	}

	if options.hasPrefix() && message.Role() != model.RoleAssistant {
		return nil, fmt.Errorf("%w: deepseek assistant prefix completion requires prefix: true on an assistant message", model.ErrInvalidPrompt)
	}

	switch typed := message.(type) {
	case model.SystemMessage:
		return []chatMessage{
			systemMessage{
				Role:    "system",
				Content: typed.Content,
				Name:    options.Name,
			},
		}, nil

	case model.UserMessage:
		converted, err := c.convertUserMessage(typed, options)
		if err != nil {
			return nil, err
		}

		return []chatMessage{
			converted,
		}, nil

	case model.AssistantMessage:
		converted, err := c.convertAssistantMessage(typed, options, index, count)
		if err != nil {
			return nil, err
		}

		return []chatMessage{
			converted,
		}, nil

	case model.ToolMessage:
		if options.Name != nil {
			c.warnings = append(c.warnings, model.UnsupportedWarning("message name on tool messages", ""))
		}

		return c.convertToolMessage(typed)

	default:
		c.warnings = append(c.warnings, model.UnsupportedWarning("message role: "+string(message.Role()), ""))

		return nil, nil
	}
}

func messageProviderOptions(message model.Message) model.ProviderOptions {
	switch typed := message.(type) {
	case model.SystemMessage:
		return typed.ProviderOptions

	case model.UserMessage:
		return typed.ProviderOptions

	case model.AssistantMessage:
		return typed.ProviderOptions

	case model.ToolMessage:
		return typed.ProviderOptions

	default:
		return nil
	}
}

// convertUserMessage sends text-only messages as plain text and messages
// with images as content parts.
func (c *promptConverter) convertUserMessage(message model.UserMessage, options AssistantMessageOptions) (userMessage, error) {
	converted := userMessage{
		Role: "user",
		Name: options.Name,
	}

	if !hasUserImage(message.Content) {
		var text strings.Builder
		for _, part := range message.Content {
			if typed, ok := part.(model.TextPart); ok {
				text.WriteString(typed.Text)

				continue
			}

			c.warnings = append(c.warnings, unsupportedUserPart(part))
		}

		converted.Content = messageContent{
			Text: text.String(),
		}

		return converted, nil
	}

	parts := []contentPart{}
	for _, part := range message.Content {
		item, ok, err := c.userContentPart(part)
		if err != nil {
			return userMessage{}, err
		}

		if ok {
			parts = append(parts, item)
		}
	}

	converted.Content = messageContent{
		Parts: parts,
	}

	return converted, nil
}

// userContentPart converts one part of a message with images. Unsupported
// parts produce a warning and report false.
func (c *promptConverter) userContentPart(part model.UserPart) (contentPart, bool, error) {
	switch typed := part.(type) {
	case model.TextPart:
		return textPart(typed.Text), true, nil

	case model.FilePart:
		if !isImageFile(typed.MediaType, typed.Data) {
			break
		}

		converted, err := c.userImagePart(typed)
		if err != nil {
			return contentPart{}, false, err
		}

		return converted, true, nil
	}

	c.warnings = append(c.warnings, unsupportedUserPart(part))

	return contentPart{}, false, nil
}

// userImagePart sends uploaded images by file ID, image URLs as they are and
// inline images as data URLs or file data.
func (c *promptConverter) userImagePart(part model.FilePart) (contentPart, error) {
	partOptions, err := readFilePartOptions(part.ProviderOptions, c.settings.namespace)
	if err != nil {
		return contentPart{}, err
	}

	if reference, ok := part.Data.(model.FileReference); ok {
		return fileReferencePart(reference)
	}

	mediaType, err := resolveImageMediaType(part.MediaType, part.Data)
	if err != nil {
		return contentPart{}, err
	}

	switch data := part.Data.(type) {
	case model.FileURL:
		address, err := imageURLAddress(data)
		if err != nil {
			return contentPart{}, err
		}

		if partOptions.sendsFileData() {
			return contentPart{}, fmt.Errorf("%w: deepseek fileData image parts require inline data, not a URL", model.ErrInvalidPrompt)
		}

		return imagePart(address, partOptions.ImageDetail), nil

	case model.FileBytes:
		address := imageDataURL(mediaType, data.Data)

		if !partOptions.sendsFileData() {
			return imagePart(address, partOptions.ImageDetail), nil
		}

		if partOptions.ImageDetail != "" {
			return contentPart{}, fmt.Errorf("%w: deepseek imageDetail cannot be combined with fileData", model.ErrInvalidPrompt)
		}

		return contentPart{
			Type:     contentPartFile,
			FileData: address,
			Filename: part.Filename,
		}, nil

	default:
		return contentPart{}, fmt.Errorf("%w: file part data type %T", model.ErrUnsupported, data)
	}
}

// convertAssistantMessage joins the text and reasoning of an assistant
// message. Pre-V4 models must not receive reasoning from earlier turns; V4
// models require reasoning_content on every assistant turn.
func (c *promptConverter) convertAssistantMessage(
	message model.AssistantMessage,
	options AssistantMessageOptions,
	index int,
	count int,
) (assistantMessage, error) {
	if options.hasPrefix() {
		if index != count-1 {
			return assistantMessage{}, fmt.Errorf(
				"%w: deepseek assistant prefix completion requires the prefixed assistant message to be the final message",
				model.ErrInvalidPrompt,
			)
		}

		if !c.settings.supportsAssistantPrefixCompletion {
			return assistantMessage{}, fmt.Errorf(
				"%w: deepseek assistant prefix completion requires a beta base URL ending in /beta",
				model.ErrUnsupported,
			)
		}
	}

	var text strings.Builder
	var reasoning *string
	var toolCalls []messageToolCall

	isEarlierTurn := index <= c.lastUserIndex
	dropsReasoning := isEarlierTurn && !c.isV4

	for _, part := range message.Content {
		switch typed := part.(type) {
		case model.TextPart:
			text.WriteString(typed.Text)

		case model.ReasoningPart:
			if dropsReasoning {
				continue
			}

			reasoning = new(valueOrEmpty(reasoning) + typed.Text)

		case model.ToolCallPart:
			arguments, err := toolresult.CompactJSON(typed.Input)
			if err != nil {
				return assistantMessage{}, err
			}

			toolCalls = append(toolCalls, messageToolCall{
				ID:   typed.ToolCallID,
				Type: functionType,
				Function: functionCall{
					Name:      typed.ToolName,
					Arguments: arguments,
				},
			})
		}
	}

	if reasoning == nil && c.isV4 {
		reasoning = new("")
	}

	converted := assistantMessage{
		Role:             "assistant",
		Content:          text.String(),
		Name:             options.Name,
		ReasoningContent: reasoning,
		ToolCalls:        toolCalls,
	}

	if options.hasPrefix() {
		converted.Prefix = new(true)
	}

	return converted, nil
}

// convertToolMessage returns one message per tool result. Approval responses
// are not sent.
func (c *promptConverter) convertToolMessage(message model.ToolMessage) ([]chatMessage, error) {
	var messages []chatMessage

	for _, part := range message.Content {
		result, ok := part.(model.ToolResultPart)
		if !ok {
			continue
		}

		content, err := c.toolResultContent(result.Output)
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

// toolResultContent sends multipart results with images as content parts
// and every other result as text.
func (c *promptConverter) toolResultContent(output model.ToolResultOutput) (messageContent, error) {
	content, ok := output.(model.ContentOutput)
	if !ok || !hasToolResultImage(content.Value) {
		text, err := toolresult.Text(output)
		if err != nil {
			return messageContent{}, err
		}

		return messageContent{
			Text: text,
		}, nil
	}

	parts := []contentPart{}
	for _, item := range content.Value {
		converted, ok, err := c.toolResultPart(item)
		if err != nil {
			return messageContent{}, err
		}

		if ok {
			parts = append(parts, converted)
		}
	}

	return messageContent{
		Parts: parts,
	}, nil
}

// toolResultPart converts one item of a multipart tool result. Unsupported
// items produce a warning and report false.
func (c *promptConverter) toolResultPart(item model.ToolResultContent) (contentPart, bool, error) {
	switch typed := item.(type) {
	case model.ToolResultText:
		return textPart(typed.Text), true, nil

	case model.ToolResultFile:
		if !isImageFile(typed.MediaType, typed.Data) {
			break
		}

		converted, err := c.toolResultImage(typed)
		if err != nil {
			return contentPart{}, false, err
		}

		return converted, true, nil
	}

	feature := "tool result content part type: " + toolResultItemType(item)
	c.warnings = append(c.warnings, model.UnsupportedWarning(feature, ""))

	return contentPart{}, false, nil
}

// toolResultImage sends an uploaded image by file ID and other images as
// image URLs.
func (c *promptConverter) toolResultImage(item model.ToolResultFile) (contentPart, error) {
	if reference, ok := item.Data.(model.FileReference); ok {
		return fileReferencePart(reference)
	}

	partOptions, err := readFilePartOptions(item.ProviderOptions, c.settings.namespace)
	if err != nil {
		return contentPart{}, err
	}

	mediaType, err := resolveImageMediaType(item.MediaType, item.Data)
	if err != nil {
		return contentPart{}, err
	}

	switch data := item.Data.(type) {
	case model.FileURL:
		address, err := imageURLAddress(data)
		if err != nil {
			return contentPart{}, err
		}

		return imagePart(address, partOptions.ImageDetail), nil

	case model.FileBytes:
		address := imageDataURL(mediaType, data.Data)

		return imagePart(address, partOptions.ImageDetail), nil

	default:
		return contentPart{}, fmt.Errorf("%w: file part data type %T", model.ErrUnsupported, data)
	}
}

// hasUserImage reports whether a user message has an image passed by
// reference, URL or inline bytes.
func hasUserImage(parts []model.UserPart) bool {
	for _, part := range parts {
		file, ok := part.(model.FilePart)
		if ok && isImageFile(file.MediaType, file.Data) {
			return true
		}
	}

	return false
}

func hasToolResultImage(items []model.ToolResultContent) bool {
	for _, item := range items {
		file, ok := item.(model.ToolResultFile)
		if ok && isImageFile(file.MediaType, file.Data) {
			return true
		}
	}

	return false
}

// isImageFile reports an image passed by reference, URL or inline bytes.
func isImageFile(mediaType string, data model.FileData) bool {
	if mediatype.TopLevel(mediaType) != "image" {
		return false
	}

	_, isText := data.(model.FileText)

	return !isText
}

// resolveImageMediaType returns the full media type of an image and rejects
// formats that DeepSeek does not accept.
func resolveImageMediaType(declared string, data model.FileData) (string, error) {
	mediaType, err := mediatype.ResolveFile(declared, data)
	if err != nil {
		return "", err
	}

	if !supportedImageTypes[mediaType] {
		return "", fmt.Errorf(
			"%w: deepseek image media type %s: deepseek supports JPEG, PNG, GIF, and WebP image inputs",
			model.ErrUnsupported,
			mediaType,
		)
	}

	return mediaType, nil
}

// imageURLAddress returns the address of an image URL, rejecting missing
// and overlong URLs.
func imageURLAddress(file model.FileURL) (string, error) {
	if file.URL == nil {
		return "", fmt.Errorf("%w: deepseek image URL is missing", model.ErrInvalidPrompt)
	}

	address := file.URL.String()
	if len(address) > maxImageURLLength {
		return "", fmt.Errorf("%w: deepseek image URLs must not exceed %d characters", model.ErrInvalidPrompt, maxImageURLLength)
	}

	return address, nil
}

// imageDataURL returns a data URL of inline image bytes. DeepSeek expects
// "image/jpeg" for JPEG images.
func imageDataURL(mediaType string, data []byte) string {
	if mediaType == "image/jpg" {
		mediaType = "image/jpeg"
	}

	return dataurl.Encode(mediaType, data)
}

// fileReferencePart sends an uploaded file by its DeepSeek file ID.
func fileReferencePart(reference model.FileReference) (contentPart, error) {
	fileID, ok := reference.IDs[Namespace]
	if !ok {
		return contentPart{}, fmt.Errorf("%w: file reference has no %s file ID", model.ErrInvalidPrompt, Namespace)
	}

	return contentPart{
		Type:   contentPartFile,
		FileID: fileID,
	}, nil
}

func textPart(text string) contentPart {
	return contentPart{
		Type: contentPartText,
		Text: new(text),
	}
}

func imagePart(address string, detail ImageDetail) contentPart {
	return contentPart{
		Type: contentPartImageURL,
		ImageURL: &imageURL{
			URL:    address,
			Detail: detail,
		},
	}
}

func unsupportedUserPart(part model.UserPart) model.Warning {
	partType := "file"
	if _, ok := part.(model.TextPart); ok {
		partType = "text"
	}

	return model.UnsupportedWarning("user message part type: "+partType, "")
}

func toolResultItemType(item model.ToolResultContent) string {
	switch item.(type) {
	case model.ToolResultText:
		return "text"

	case model.ToolResultFile:
		return "file"

	default:
		return "custom"
	}
}
