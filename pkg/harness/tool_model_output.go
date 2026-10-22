package harness

import (
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/mediatype"
	"github.com/aarock1234/ai/pkg/model"
)

// toolModelOutputLimitation explains which tool model outputs harnesses
// accept.
const toolModelOutputLimitation = "Harnesses support only text and inline images in tool model output."

// NormalizeToolModelOutput checks that a model-facing tool output only holds
// what runtimes accept: text and inline images. Images get a full media
// type, detected from their bytes when the declared one is a wildcard.
// Other content is reported with *CapabilityUnsupportedError.
func NormalizeToolModelOutput(output model.ToolResultOutput) (model.ToolResultOutput, error) {
	content, ok := output.(model.ContentOutput)
	if !ok {
		return output, nil
	}

	normalized := make([]model.ToolResultContent, 0, len(content.Value))

	for _, item := range content.Value {
		normalizedItem, err := normalizeToolModelOutputItem(item)
		if err != nil {
			return nil, &CapabilityUnsupportedError{
				Message: fmt.Sprintf("%s %v", toolModelOutputLimitation, err),
				Cause:   err,
			}
		}

		normalized = append(normalized, normalizedItem)
	}

	return model.ContentOutput{
		Value: normalized,
	}, nil
}

// normalizeToolModelOutputItem normalizes one content item.
func normalizeToolModelOutputItem(item model.ToolResultContent) (model.ToolResultContent, error) {
	switch typed := item.(type) {
	case model.ToolResultText:
		return typed, nil
	case model.ToolResultFile:
		data, inline := typed.Data.(model.FileBytes)
		if !inline || mediatype.TopLevel(typed.MediaType) != "image" {
			return nil, fmt.Errorf("unsupported content 'file' (%s, %T)", typed.MediaType, typed.Data)
		}

		if mediatype.IsFull(typed.MediaType) {
			return typed, nil
		}

		detected, ok := mediatype.Detect(data.Data, "image")
		if !ok {
			return nil, fmt.Errorf("cannot detect the inline image media type")
		}

		typed.MediaType = detected

		return typed, nil
	default:
		return nil, fmt.Errorf("unsupported content %T", item)
	}
}

// ToolModelOutputContentType distinguishes the items of converted tool
// output.
type ToolModelOutputContentType string

// Converted tool output content types.
const (
	ToolModelOutputText  ToolModelOutputContentType = "text"
	ToolModelOutputImage ToolModelOutputContentType = "image"
)

// ToolModelOutputContent is one item of a tool result as runtimes accept it.
type ToolModelOutputContent struct {
	Type ToolModelOutputContentType `json:"type"`
	Text string                     `json:"text,omitempty"`

	// Data is the base64 data of an image.
	Data      string `json:"data,omitempty"`
	MediaType string `json:"mediaType,omitempty"`
}

// ConvertedToolModelOutput is a tool model output converted for a runtime.
type ConvertedToolModelOutput struct {
	Content []ToolModelOutputContent `json:"content"`
	IsError bool                     `json:"isError"`
}

// ConvertToolModelOutput converts a model-facing tool output into text and
// inline images for a runtime's tool response. JSON outputs become their
// JSON text, and an execution denial becomes an error with its reason. Other
// content is reported with *CapabilityUnsupportedError.
func ConvertToolModelOutput(output model.ToolResultOutput) (ConvertedToolModelOutput, error) {
	switch typed := output.(type) {
	case model.TextOutput:
		return textToolOutput(typed.Value, false), nil
	case model.ErrorTextOutput:
		return textToolOutput(typed.Value, true), nil
	case model.JSONOutput:
		return textToolOutput(string(jsonValueOrNull(typed.Value)), false), nil
	case model.ErrorJSONOutput:
		return textToolOutput(string(jsonValueOrNull(typed.Value)), true), nil
	case model.ExecutionDeniedOutput:
		reason := typed.Reason
		if reason == "" {
			reason = "Tool execution denied."
		}

		return textToolOutput(reason, true), nil
	case model.ContentOutput:
		return convertContentToolOutput(typed)
	default:
		return ConvertedToolModelOutput{}, &CapabilityUnsupportedError{
			Message: fmt.Sprintf("%s Unsupported output %T.", toolModelOutputLimitation, output),
		}
	}
}

// convertContentToolOutput converts the items of a content output.
func convertContentToolOutput(output model.ContentOutput) (ConvertedToolModelOutput, error) {
	converted := ConvertedToolModelOutput{
		Content: make([]ToolModelOutputContent, 0, len(output.Value)),
	}

	for _, item := range output.Value {
		switch typed := item.(type) {
		case model.ToolResultText:
			converted.Content = append(converted.Content, ToolModelOutputContent{
				Type: ToolModelOutputText,
				Text: typed.Text,
			})
		case model.ToolResultFile:
			image, ok := inlineImage(typed)
			if !ok {
				return ConvertedToolModelOutput{}, unsupportedToolOutputContent("file")
			}

			converted.Content = append(converted.Content, image)
		default:
			return ConvertedToolModelOutput{}, unsupportedToolOutputContent(fmt.Sprintf("%T", item))
		}
	}

	return converted, nil
}

// inlineImage converts an inline image with a full media type.
func inlineImage(file model.ToolResultFile) (ToolModelOutputContent, bool) {
	data, inline := file.Data.(model.FileBytes)
	isImage := mediatype.TopLevel(file.MediaType) == "image" && mediatype.IsFull(file.MediaType)

	if !inline || !isImage {
		return ToolModelOutputContent{}, false
	}

	return ToolModelOutputContent{
		Type:      ToolModelOutputImage,
		Data:      base64.StdEncoding.EncodeToString(data.Data),
		MediaType: file.MediaType,
	}, true
}

// textToolOutput returns a single text item.
func textToolOutput(text string, isError bool) ConvertedToolModelOutput {
	return ConvertedToolModelOutput{
		Content: []ToolModelOutputContent{
			{
				Type: ToolModelOutputText,
				Text: text,
			},
		},
		IsError: isError,
	}
}

// unsupportedToolOutputContent reports content runtimes do not accept.
func unsupportedToolOutputContent(kind string) error {
	return &CapabilityUnsupportedError{
		Message: fmt.Sprintf("%s Unsupported content '%s'.", toolModelOutputLimitation, kind),
	}
}

// unwrapToolResultOutput returns the raw output and error flag a
// caller-provided tool result reports to consumers.
func unwrapToolResultOutput(part model.ToolResultPart) (any, bool, error) {
	switch typed := part.Output.(type) {
	case model.TextOutput:
		return typed.Value, false, nil
	case model.JSONOutput:
		return typed.Value, false, nil
	case model.ErrorTextOutput:
		return typed.Value, true, nil
	case model.ErrorJSONOutput:
		return typed.Value, true, nil
	case model.ExecutionDeniedOutput:
		return executionDeniedOutput{
			Type:   "execution-denied",
			Reason: typed.Reason,
		}, false, nil
	default:
		encoded, err := MarshalToolResultOutput(part.Output)
		if err != nil {
			return nil, false, err
		}

		return json.RawMessage(encoded), false, nil
	}
}

// executionDeniedOutput is the raw output submitted for a denied call.
type executionDeniedOutput struct {
	Type   string `json:"type"`
	Reason string `json:"reason,omitempty"`
}

// toolFailureOutput is the raw output submitted for a failed call.
type toolFailureOutput struct {
	Error string `json:"error"`
}
