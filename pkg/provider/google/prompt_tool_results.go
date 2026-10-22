package google

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/internal/toolresult"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	toolExecutedText   = "Tool executed successfully."
	schemaReferenceKey = "$ref"
)

var base64DataURLPattern = regexp.MustCompile(`(?s)^data:([^;,]+);base64,(.+)$`)

func (c *promptConverter) convertToolMessage(message model.ToolMessage) error {
	parts := make([]part, 0, len(message.Content))

	for _, toolPart := range message.Content {
		result, ok := toolPart.(model.ToolResultPart)
		if !ok {
			continue
		}

		appended, err := c.appendServerToolResponse(result)
		if err != nil {
			return err
		}

		if appended {
			continue
		}

		converted, err := c.convertToolResult(result)
		if err != nil {
			return err
		}

		parts = append(parts, converted...)
	}

	c.contents = append(c.contents, content{
		Role:  roleUser,
		Parts: parts,
	})

	return nil
}

// appendServerToolResponse adds the result of a server-side tool to the
// preceding model content, where Gemini expects it.
func (c *promptConverter) appendServerToolResponse(result model.ToolResultPart) (bool, error) {
	metadata, err := c.partMetadata(result.ProviderOptions)
	if err != nil {
		return false, err
	}

	if !metadata.isServerTool() || len(c.contents) == 0 {
		return false, nil
	}

	last := &c.contents[len(c.contents)-1]
	if last.Role != roleModel {
		return false, nil
	}

	last.Parts = append(last.Parts, serverToolResponsePart(result.Output, metadata))

	return true, nil
}

func (c *promptConverter) convertToolResult(result model.ToolResultPart) ([]part, error) {
	switch output := result.Output.(type) {
	case model.ContentOutput:
		if c.settings.isGemini3 {
			return c.multimodalToolResult(result, output)
		}

		return legacyToolResult(result, output)

	case model.TextOutput:
		return functionResponseParts(result, functionResponseContent{
			Content: encodeJSONText(output.Value),
		}), nil

	case model.JSONOutput:
		return functionResponseParts(result, functionResponseContent{
			Content: serializeResponseValue(output.Value),
		}), nil

	case model.ErrorTextOutput:
		return functionResponseParts(result, functionResponseContent{
			Error: encodeJSONText(output.Value),
		}), nil

	case model.ErrorJSONOutput:
		return functionResponseParts(result, functionResponseContent{
			Error: serializeResponseValue(output.Value),
		}), nil

	case model.ExecutionDeniedOutput:
		reason := output.Reason
		if reason == "" {
			reason = toolresult.DeniedText
		}

		return functionResponseParts(result, functionResponseContent{
			Error: encodeJSONText(reason),
		}), nil

	default:
		return nil, fmt.Errorf("%w: unsupported tool result output %T", model.ErrInvalidPrompt, result.Output)
	}
}

func functionResponseParts(result model.ToolResultPart, response functionResponseContent) []part {
	response.Name = result.ToolName

	return []part{
		{
			FunctionResponse: &functionResponse{
				ID:       result.ToolCallID,
				Name:     result.ToolName,
				Response: response,
			},
		},
	}
}

// multimodalToolResult returns one function response whose files travel as
// function response parts, as Gemini 3 models support. Other items are
// described in the response text.
func (c *promptConverter) multimodalToolResult(result model.ToolResultPart, output model.ContentOutput) ([]part, error) {
	var fileParts []functionResponsePart
	var texts []string

	for _, item := range output.Value {
		if text, ok := item.(model.ToolResultText); ok {
			texts = append(texts, text.Text)

			continue
		}

		filePart, ok, err := c.functionResponseFile(item)
		if err != nil {
			return nil, err
		}

		if ok {
			fileParts = append(fileParts, filePart)

			continue
		}

		description, err := serializeToolResultContent(item)
		if err != nil {
			return nil, err
		}

		texts = append(texts, description)
	}

	text := toolExecutedText
	if len(texts) > 0 {
		text = strings.Join(texts, "\n")
	}

	return []part{
		{
			FunctionResponse: &functionResponse{
				ID:   result.ToolCallID,
				Name: result.ToolName,
				Response: functionResponseContent{
					Name:    result.ToolName,
					Content: encodeJSONText(text),
				},
				Parts: fileParts,
			},
		},
	}, nil
}

// functionResponseFile converts files with inline bytes or a base64 data
// URL. It reports false for every other item.
func (c *promptConverter) functionResponseFile(item model.ToolResultContent) (functionResponsePart, bool, error) {
	file, ok := item.(model.ToolResultFile)
	if !ok {
		return functionResponsePart{}, false, nil
	}

	switch data := file.Data.(type) {
	case model.FileBytes:
		mediaType, err := mediatype.ResolveFile(file.MediaType, file.Data)
		if err != nil {
			return functionResponsePart{}, false, err
		}

		return functionResponsePart{
			InlineData: &InlineData{
				MimeType: mediaType,
				Data:     base64.StdEncoding.EncodeToString(data.Data),
			},
		}, true, nil

	case model.FileURL:
		location := data.URL.String()

		match := base64DataURLPattern.FindStringSubmatch(location)
		if match != nil {
			return functionResponsePart{
				InlineData: &InlineData{
					MimeType: match[1],
					Data:     match[2],
				},
			}, true, nil
		}

		if !mediatype.IsFull(file.MediaType) || !isURLSupported(location, file.MediaType, c.settings.functionResponseURLs) {
			return functionResponsePart{}, false, nil
		}

		return functionResponsePart{
			FileData: &FileData{
				MimeType: file.MediaType,
				FileURI:  location,
			},
		}, true, nil

	default:
		return functionResponsePart{}, false, nil
	}
}

// legacyToolResult sends each content item as its own part, for models
// before Gemini 3 that do not accept files inside function responses.
func legacyToolResult(result model.ToolResultPart, output model.ContentOutput) ([]part, error) {
	var parts []part

	for _, item := range output.Value {
		itemParts, err := legacyItemParts(result, item)
		if err != nil {
			return nil, err
		}

		parts = append(parts, itemParts...)
	}

	return parts, nil
}

func legacyItemParts(result model.ToolResultPart, item model.ToolResultContent) ([]part, error) {
	if text, ok := item.(model.ToolResultText); ok {
		return functionResponseParts(result, functionResponseContent{
			Content: encodeJSONText(text.Text),
		}), nil
	}

	file, isFile := item.(model.ToolResultFile)
	data, hasBytes := file.Data.(model.FileBytes)

	if !isFile || !hasBytes {
		description, err := serializeToolResultContent(item)
		if err != nil {
			return nil, err
		}

		return []part{
			{
				Text: &description,
			},
		}, nil
	}

	mediaType, err := mediatype.ResolveFile(file.MediaType, file.Data)
	if err != nil {
		return nil, err
	}

	kind := "file"
	if mediatype.TopLevel(file.MediaType) == "image" {
		kind = "image"
	}

	return []part{
		{
			InlineData: &InlineData{
				MimeType: mediaType,
				Data:     base64.StdEncoding.EncodeToString(data.Data),
			},
		},
		{
			Text: new("Tool executed successfully and returned this " + kind + " as a response"),
		},
	}, nil
}

// serializeResponseValue returns a JSON tool result. Google reserves
// {"$ref": name} in function responses for multimodal parts, so values that
// contain JSON Schema references are sent as a JSON string instead.
func serializeResponseValue(value json.RawMessage) json.RawMessage {
	if !containsSchemaReference(value) {
		return value
	}

	return encodeJSONText(string(value))
}

func encodeJSONText(text string) json.RawMessage {
	return json.RawMessage(encodeJSONString(text))
}

// containsSchemaReference reports whether any object in value has a "$ref"
// key.
func containsSchemaReference(value json.RawMessage) bool {
	decoder := json.NewDecoder(bytes.NewReader(value))

	var open []tokenContainer

	for {
		token, err := decoder.Token()
		if err != nil {
			return false
		}

		switch token := token.(type) {
		case json.Delim:
			switch token {
			case '{':
				open = append(open, tokenContainer{
					isObject:     true,
					expectingKey: true,
				})
			case '[':
				open = append(open, tokenContainer{})
			default:
				open = open[:len(open)-1]
				finishValue(open)
			}

		case string:
			if !expectsKey(open) {
				finishValue(open)

				continue
			}

			if token == schemaReferenceKey {
				return true
			}

			open[len(open)-1].expectingKey = false

		default:
			finishValue(open)
		}
	}
}

// tokenContainer is an open object or array while scanning JSON tokens.
type tokenContainer struct {
	isObject     bool
	expectingKey bool
}

func expectsKey(open []tokenContainer) bool {
	return len(open) > 0 && open[len(open)-1].expectingKey
}

// finishValue records that a member value ended, so an enclosing object
// expects its next key.
func finishValue(open []tokenContainer) {
	if len(open) == 0 {
		return
	}

	innermost := &open[len(open)-1]
	if innermost.isObject {
		innermost.expectingKey = true
	}
}

// serializedContent is the JSON description of a tool result item that
// Gemini cannot receive as a part.
type serializedContent struct {
	Type            string                `json:"type"`
	MediaType       string                `json:"mediaType,omitempty"`
	Filename        string                `json:"filename,omitempty"`
	Data            *serializedFileData   `json:"data,omitempty"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

type serializedFileData struct {
	Type      string            `json:"type"`
	Data      string            `json:"data,omitempty"`
	URL       string            `json:"url,omitempty"`
	Reference map[string]string `json:"reference,omitempty"`
	Text      *string           `json:"text,omitempty"`
}

func serializeToolResultContent(item model.ToolResultContent) (string, error) {
	var serialized serializedContent

	switch item := item.(type) {
	case model.ToolResultFile:
		serialized = serializedContent{
			Type:            "file",
			MediaType:       item.MediaType,
			Filename:        item.Filename,
			Data:            serializeFileData(item.Data),
			ProviderOptions: item.ProviderOptions,
		}

	case model.ToolResultCustom:
		serialized = serializedContent{
			Type:            "custom",
			ProviderOptions: item.ProviderOptions,
		}

	default:
		return "", fmt.Errorf("%w: unsupported tool result content %T", model.ErrInvalidPrompt, item)
	}

	encoded, err := json.Marshal(serialized)
	if err != nil {
		return "", fmt.Errorf("%w: encode tool result content: %w", model.ErrInvalidPrompt, err)
	}

	return string(encoded), nil
}

func serializeFileData(data model.FileData) *serializedFileData {
	switch data := data.(type) {
	case model.FileBytes:
		return &serializedFileData{
			Type: "data",
			Data: base64.StdEncoding.EncodeToString(data.Data),
		}

	case model.FileURL:
		return &serializedFileData{
			Type: "url",
			URL:  data.URL.String(),
		}

	case model.FileReference:
		return &serializedFileData{
			Type:      "reference",
			Reference: data.IDs,
		}

	case model.FileText:
		return &serializedFileData{
			Type: "text",
			Text: new(data.Text),
		}

	default:
		return nil
	}
}
