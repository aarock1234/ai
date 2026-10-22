package wirejson

import (
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// ToolOutput is an encoded tool result output. Value holds the text, the
// JSON value or the content items, and Reason the reason of a denial.
type ToolOutput struct {
	Type            string                `json:"type"`
	Value           any                   `json:"value,omitempty"`
	Reason          string                `json:"reason,omitempty"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

// ToolOutputContent is one encoded content item of a multipart tool
// output. Text is set for text items only.
type ToolOutputContent struct {
	Type            string                `json:"type"`
	Text            *string               `json:"text,omitempty"`
	MediaType       string                `json:"mediaType,omitempty"`
	Filename        string                `json:"filename,omitempty"`
	Data            FileData              `json:"data,omitempty"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

// ToolOutput encodes a tool result output.
func (e Encoder) ToolOutput(output model.ToolResultOutput) ToolOutput {
	switch typed := output.(type) {
	case model.TextOutput:
		return ToolOutput{
			Type:            "text",
			Value:           typed.Value,
			ProviderOptions: typed.ProviderOptions,
		}

	case model.JSONOutput:
		return ToolOutput{
			Type:            "json",
			Value:           RawValue(typed.Value),
			ProviderOptions: typed.ProviderOptions,
		}

	case model.ErrorTextOutput:
		return ToolOutput{
			Type:            "error-text",
			Value:           typed.Value,
			ProviderOptions: typed.ProviderOptions,
		}

	case model.ErrorJSONOutput:
		return ToolOutput{
			Type:            "error-json",
			Value:           RawValue(typed.Value),
			ProviderOptions: typed.ProviderOptions,
		}

	case model.ExecutionDeniedOutput:
		return ToolOutput{
			Type:            "execution-denied",
			Reason:          typed.Reason,
			ProviderOptions: typed.ProviderOptions,
		}

	case model.ContentOutput:
		return ToolOutput{
			Type:  "content",
			Value: e.toolOutputContents(typed.Value),
		}

	default:
		return ToolOutput{
			Type: "unknown",
		}
	}
}

// toolOutputContents encodes content items in the encoder's file data
// encoding, skipping unknown items.
func (e Encoder) toolOutputContents(contents []model.ToolResultContent) []ToolOutputContent {
	encoded := make([]ToolOutputContent, 0, len(contents))

	encodeData := func(data model.FileData) (FileData, error) {
		return e.fileData(data), nil
	}

	for _, content := range contents {
		item, err := toolOutputContent(content, encodeData)
		if err != nil {
			continue
		}

		encoded = append(encoded, item)
	}

	return encoded
}

// TaggedToolOutputContents encodes content items with file contents in the
// tagged file data union, as providers that accept tool results only as
// text serialize them. Unknown items and file contents that cannot be
// encoded fail.
func TaggedToolOutputContents(contents []model.ToolResultContent) ([]ToolOutputContent, error) {
	encoded := make([]ToolOutputContent, 0, len(contents))

	for _, content := range contents {
		item, err := toolOutputContent(content, TagFileData)
		if err != nil {
			return nil, err
		}

		encoded = append(encoded, item)
	}

	return encoded, nil
}

// toolOutputContent encodes one content item, with its file contents
// encoded by encodeData.
func toolOutputContent(
	content model.ToolResultContent,
	encodeData func(model.FileData) (FileData, error),
) (ToolOutputContent, error) {
	switch typed := content.(type) {
	case model.ToolResultText:
		return ToolOutputContent{
			Type:            "text",
			Text:            &typed.Text,
			ProviderOptions: typed.ProviderOptions,
		}, nil

	case model.ToolResultFile:
		data, err := encodeData(typed.Data)
		if err != nil {
			return ToolOutputContent{}, err
		}

		return ToolOutputContent{
			Type:            "file",
			MediaType:       typed.MediaType,
			Filename:        typed.Filename,
			Data:            data,
			ProviderOptions: typed.ProviderOptions,
		}, nil

	case model.ToolResultCustom:
		return ToolOutputContent{
			Type:            "custom",
			ProviderOptions: typed.ProviderOptions,
		}, nil

	default:
		return ToolOutputContent{}, fmt.Errorf("%w: tool content part %T", model.ErrUnsupported, content)
	}
}
