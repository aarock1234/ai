package anthropic

import (
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/internal/toolresult"
	"github.com/aarock1234/ai/pkg/model"
)

func (c *promptConverter) toolMessage(message model.ToolMessage) ([]requestBlock, error) {
	blocks := make([]requestBlock, 0, len(message.Content))

	for index, part := range message.Content {
		result, ok := part.(model.ToolResultPart)
		if !ok {
			// Approval responses are handled by the application.
			continue
		}

		isLastPart := index == len(message.Content)-1

		block, err := c.toolResult(result, message.ProviderOptions, isLastPart)
		if err != nil {
			return nil, err
		}

		blocks = append(blocks, block)
	}

	return blocks, nil
}

func (c *promptConverter) toolResult(part model.ToolResultPart, messageOptions model.ProviderOptions, isLastPart bool) (requestBlock, error) {
	cacheControl := c.cache.cacheControl(part.ProviderOptions, cacheToolResultPart)
	if cacheControl == nil {
		cacheControl = c.cache.cacheControl(outputProviderOptions(part.Output), cacheToolResultOutput)
	}

	if cacheControl == nil && isLastPart {
		cacheControl = c.cache.cacheControl(messageOptions, cacheToolResultMessage)
	}

	content, err := c.toolResultContent(part.Output)
	if err != nil {
		return nil, err
	}

	toolsetName, err := c.toolsetName(part.ToolName, part.ProviderOptions)
	if err != nil {
		return nil, err
	}

	block := &toolResultBlock{
		Type:         blockToolResult,
		ToolUseID:    part.ToolCallID,
		ToolsetName:  toolsetName,
		Content:      content,
		CacheControl: cacheControl,
	}

	if isErrorOutput(part.Output) {
		block.IsError = new(true)
	}

	return block, nil
}

// outputProviderOptions returns the provider options of a tool result
// output, or of the first item of a multipart output that has any.
func outputProviderOptions(output model.ToolResultOutput) model.ProviderOptions {
	switch typed := output.(type) {
	case model.TextOutput:
		return typed.ProviderOptions
	case model.JSONOutput:
		return typed.ProviderOptions
	case model.ExecutionDeniedOutput:
		return typed.ProviderOptions
	case model.ErrorTextOutput:
		return typed.ProviderOptions
	case model.ErrorJSONOutput:
		return typed.ProviderOptions
	case model.ContentOutput:
		for _, item := range typed.Value {
			if options := toolResultItemOptions(item); options != nil {
				return options
			}
		}
	}

	return nil
}

func toolResultItemOptions(item model.ToolResultContent) model.ProviderOptions {
	switch typed := item.(type) {
	case model.ToolResultText:
		return typed.ProviderOptions
	case model.ToolResultFile:
		return typed.ProviderOptions
	case model.ToolResultCustom:
		return typed.ProviderOptions
	default:
		return nil
	}
}

func isErrorOutput(output model.ToolResultOutput) bool {
	switch output.(type) {
	case model.ErrorTextOutput, model.ErrorJSONOutput:
		return true
	default:
		return false
	}
}

func (c *promptConverter) toolResultContent(output model.ToolResultOutput) (toolResultContent, error) {
	switch typed := output.(type) {
	case model.TextOutput:
		return toolResultContent{
			Text: typed.Value,
		}, nil

	case model.ErrorTextOutput:
		return toolResultContent{
			Text: typed.Value,
		}, nil

	case model.ExecutionDeniedOutput:
		reason := typed.Reason
		if reason == "" {
			reason = toolresult.DeniedText
		}

		return toolResultContent{
			Text: reason,
		}, nil

	case model.JSONOutput:
		return jsonToolResultContent(typed.Value)

	case model.ErrorJSONOutput:
		return jsonToolResultContent(typed.Value)

	case model.ContentOutput:
		parts, err := c.toolResultParts(typed.Value)
		if err != nil {
			return toolResultContent{}, err
		}

		return toolResultContent{
			Parts: parts,
		}, nil

	default:
		return toolResultContent{}, fmt.Errorf("%w: unsupported tool result output %T", model.ErrInvalidPrompt, output)
	}
}

func jsonToolResultContent(value json.RawMessage) (toolResultContent, error) {
	text, err := toolresult.CompactJSON(value)
	if err != nil {
		return toolResultContent{}, err
	}

	return toolResultContent{
		Text: text,
	}, nil
}

// toolResultParts converts the items of a multipart tool result. Items the
// API cannot represent are dropped with a warning.
func (c *promptConverter) toolResultParts(items []model.ToolResultContent) ([]requestBlock, error) {
	parts := []requestBlock{}

	for _, item := range items {
		part, err := c.toolResultPart(item)
		if err != nil {
			return nil, err
		}

		if part != nil {
			parts = append(parts, part)
		}
	}

	return parts, nil
}

func (c *promptConverter) toolResultPart(item model.ToolResultContent) (requestBlock, error) {
	switch typed := item.(type) {
	case model.ToolResultText:
		return &textBlock{
			Type: blockText,
			Text: typed.Text,
		}, nil

	case model.ToolResultFile:
		return c.toolResultFile(typed)

	case model.ToolResultCustom:
		reference, _, err := model.ProviderValue[ToolReferenceOptions](typed.ProviderOptions, Namespace)
		if err != nil {
			return nil, err
		}

		if reference.Type != "tool-reference" {
			c.warnings.other("unsupported custom tool content part")

			return nil, nil
		}

		return &toolReference{
			Type:     blockToolReference,
			ToolName: reference.ToolName,
		}, nil

	default:
		c.warnings.other(fmt.Sprintf("unsupported tool content part type: %T", item))

		return nil, nil
	}
}

func (c *promptConverter) toolResultFile(file model.ToolResultFile) (requestBlock, error) {
	isImage := mediatype.TopLevel(file.MediaType) == "image"

	switch data := file.Data.(type) {
	case model.FileReference:
		fileID, err := anthropicFileID(data)
		if err != nil {
			return nil, err
		}

		c.result.betas.add("files-api-2025-04-14")

		return nestedMediaBlock(isImage, contentSource{
			Type:   "file",
			FileID: fileID,
		}), nil

	case model.FileURL:
		if data.URL == nil {
			return nil, fmt.Errorf("%w: file URL is missing", model.ErrInvalidPrompt)
		}

		return nestedMediaBlock(isImage, contentSource{
			Type: "url",
			URL:  data.URL.String(),
		}), nil

	case model.FileBytes:
		return c.inlineToolResultFile(file, data, isImage)

	default:
		c.warnings.other(fmt.Sprintf("unsupported tool content part type: file with data type: %s", fileDataType(file.Data)))

		return nil, nil
	}
}

func (c *promptConverter) inlineToolResultFile(file model.ToolResultFile, data model.FileBytes, isImage bool) (requestBlock, error) {
	mediaType, err := mediatype.ResolveFile(file.MediaType, data)
	if err != nil {
		return nil, err
	}

	encoded := base64.StdEncoding.EncodeToString(data.Data)

	if isImage {
		return nestedMediaBlock(true, contentSource{
			Type:      "base64",
			MediaType: mediaType,
			Data:      encoded,
		}), nil
	}

	if mediaType == "application/pdf" {
		c.result.betas.add("pdfs-2024-09-25")

		return nestedMediaBlock(false, contentSource{
			Type:      "base64",
			MediaType: "application/pdf",
			Data:      encoded,
		}), nil
	}

	c.warnings.other(fmt.Sprintf("unsupported tool content part type: file with media type: %s", file.MediaType))

	return nil, nil
}

// nestedMediaBlock is an image or document nested in a tool result, which
// cannot carry its own cache breakpoint.
func nestedMediaBlock(isImage bool, source contentSource) requestBlock {
	if isImage {
		return &imageBlock{
			Type:   blockImage,
			Source: source,
		}
	}

	return &documentBlock{
		Type:   blockDocument,
		Source: source,
	}
}

func fileDataType(data model.FileData) string {
	switch data.(type) {
	case model.FileText:
		return "text"
	case model.FileReference:
		return "reference"
	case model.FileURL:
		return "url"
	default:
		return "data"
	}
}
