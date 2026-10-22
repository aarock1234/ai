package harness

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/url"

	"github.com/aarock1234/ai/pkg/model"
)

// toolResultPartJSON is the upstream JSON shape of a tool result part.
type toolResultPartJSON struct {
	Type            string                `json:"type"`
	ToolCallID      string                `json:"toolCallId"`
	ToolName        string                `json:"toolName"`
	Output          json.RawMessage       `json:"output"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

// toolResultOutputJSON is the upstream JSON shape of a tool result output.
type toolResultOutputJSON struct {
	Type            string                `json:"type"`
	Value           json.RawMessage       `json:"value,omitempty"`
	Reason          string                `json:"reason,omitempty"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

// toolResultContentJSON is the upstream JSON shape of one item of a content
// output.
type toolResultContentJSON struct {
	Type            string                `json:"type"`
	Text            string                `json:"text,omitempty"`
	MediaType       string                `json:"mediaType,omitempty"`
	Filename        string                `json:"filename,omitempty"`
	Data            json.RawMessage       `json:"data,omitempty"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

// fileDataJSON is the upstream JSON shape of file data.
type fileDataJSON struct {
	Type      string            `json:"type"`
	Data      string            `json:"data,omitempty"`
	URL       string            `json:"url,omitempty"`
	Text      string            `json:"text,omitempty"`
	Reference map[string]string `json:"reference,omitempty"`
}

// MarshalToolResultPart encodes a tool result part in the upstream JSON
// shape, as persisted in lifecycle states and sent over bridges.
func MarshalToolResultPart(part model.ToolResultPart) (json.RawMessage, error) {
	output, err := MarshalToolResultOutput(part.Output)
	if err != nil {
		return nil, err
	}

	return json.Marshal(toolResultPartJSON{
		Type:            "tool-result",
		ToolCallID:      part.ToolCallID,
		ToolName:        part.ToolName,
		Output:          output,
		ProviderOptions: part.ProviderOptions,
	})
}

// UnmarshalToolResultPart decodes a tool result part from the upstream JSON
// shape.
func UnmarshalToolResultPart(data json.RawMessage) (model.ToolResultPart, error) {
	var decoded toolResultPartJSON
	if err := json.Unmarshal(data, &decoded); err != nil {
		return model.ToolResultPart{}, fmt.Errorf("decode tool result part: %w", err)
	}

	output, err := UnmarshalToolResultOutput(decoded.Output)
	if err != nil {
		return model.ToolResultPart{}, err
	}

	return model.ToolResultPart{
		ToolCallID:      decoded.ToolCallID,
		ToolName:        decoded.ToolName,
		Output:          output,
		ProviderOptions: decoded.ProviderOptions,
	}, nil
}

// MarshalToolResultOutput encodes a tool result output in the upstream JSON
// shape.
func MarshalToolResultOutput(output model.ToolResultOutput) (json.RawMessage, error) {
	encoded, err := encodeToolResultOutput(output)
	if err != nil {
		return nil, err
	}

	return json.Marshal(encoded)
}

// UnmarshalToolResultOutput decodes a tool result output from the upstream
// JSON shape.
func UnmarshalToolResultOutput(data json.RawMessage) (model.ToolResultOutput, error) {
	var decoded toolResultOutputJSON
	if err := json.Unmarshal(data, &decoded); err != nil {
		return nil, fmt.Errorf("decode tool result output: %w", err)
	}

	switch decoded.Type {
	case "text", "error-text":
		var value string
		if err := json.Unmarshal(decoded.Value, &value); err != nil {
			return nil, fmt.Errorf("decode %s tool result output: %w", decoded.Type, err)
		}

		if decoded.Type == "text" {
			return model.TextOutput{
				Value:           value,
				ProviderOptions: decoded.ProviderOptions,
			}, nil
		}

		return model.ErrorTextOutput{
			Value:           value,
			ProviderOptions: decoded.ProviderOptions,
		}, nil
	case "json":
		return model.JSONOutput{
			Value:           decoded.Value,
			ProviderOptions: decoded.ProviderOptions,
		}, nil
	case "error-json":
		return model.ErrorJSONOutput{
			Value:           decoded.Value,
			ProviderOptions: decoded.ProviderOptions,
		}, nil
	case "execution-denied":
		return model.ExecutionDeniedOutput{
			Reason:          decoded.Reason,
			ProviderOptions: decoded.ProviderOptions,
		}, nil
	case "content":
		return decodeContentOutput(decoded.Value)
	default:
		return nil, fmt.Errorf("decode tool result output: unknown type %q", decoded.Type)
	}
}

// encodeToolResultOutput converts an output into its JSON shape.
func encodeToolResultOutput(output model.ToolResultOutput) (toolResultOutputJSON, error) {
	switch typed := output.(type) {
	case model.TextOutput:
		return stringOutputJSON("text", typed.Value, typed.ProviderOptions)
	case model.ErrorTextOutput:
		return stringOutputJSON("error-text", typed.Value, typed.ProviderOptions)
	case model.JSONOutput:
		return toolResultOutputJSON{
			Type:            "json",
			Value:           jsonValueOrNull(typed.Value),
			ProviderOptions: typed.ProviderOptions,
		}, nil
	case model.ErrorJSONOutput:
		return toolResultOutputJSON{
			Type:            "error-json",
			Value:           jsonValueOrNull(typed.Value),
			ProviderOptions: typed.ProviderOptions,
		}, nil
	case model.ExecutionDeniedOutput:
		return toolResultOutputJSON{
			Type:            "execution-denied",
			Reason:          typed.Reason,
			ProviderOptions: typed.ProviderOptions,
		}, nil
	case model.ContentOutput:
		return encodeContentOutput(typed)
	default:
		return toolResultOutputJSON{}, fmt.Errorf("encode tool result output: unsupported type %T", output)
	}
}

// stringOutputJSON encodes a text output.
func stringOutputJSON(outputType string, value string, providerOptions model.ProviderOptions) (toolResultOutputJSON, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return toolResultOutputJSON{}, fmt.Errorf("encode %s tool result output: %w", outputType, err)
	}

	return toolResultOutputJSON{
		Type:            outputType,
		Value:           encoded,
		ProviderOptions: providerOptions,
	}, nil
}

// encodeContentOutput encodes a multipart output.
func encodeContentOutput(output model.ContentOutput) (toolResultOutputJSON, error) {
	items := make([]toolResultContentJSON, 0, len(output.Value))

	for _, item := range output.Value {
		encoded, err := encodeContentItem(item)
		if err != nil {
			return toolResultOutputJSON{}, err
		}

		items = append(items, encoded)
	}

	value, err := json.Marshal(items)
	if err != nil {
		return toolResultOutputJSON{}, fmt.Errorf("encode content tool result output: %w", err)
	}

	return toolResultOutputJSON{
		Type:  "content",
		Value: value,
	}, nil
}

// encodeContentItem encodes one item of a content output.
func encodeContentItem(item model.ToolResultContent) (toolResultContentJSON, error) {
	switch typed := item.(type) {
	case model.ToolResultText:
		return toolResultContentJSON{
			Type:            "text",
			Text:            typed.Text,
			ProviderOptions: typed.ProviderOptions,
		}, nil
	case model.ToolResultFile:
		data, err := encodeFileData(typed.Data)
		if err != nil {
			return toolResultContentJSON{}, err
		}

		return toolResultContentJSON{
			Type:            "file",
			MediaType:       typed.MediaType,
			Filename:        typed.Filename,
			Data:            data,
			ProviderOptions: typed.ProviderOptions,
		}, nil
	case model.ToolResultCustom:
		return toolResultContentJSON{
			Type:            "custom",
			ProviderOptions: typed.ProviderOptions,
		}, nil
	default:
		return toolResultContentJSON{}, fmt.Errorf("encode tool result content: unsupported type %T", item)
	}
}

// encodeFileData encodes file data, inlining bytes as base64.
func encodeFileData(data model.FileData) (json.RawMessage, error) {
	var encoded fileDataJSON

	switch typed := data.(type) {
	case model.FileBytes:
		encoded = fileDataJSON{
			Type: "data",
			Data: base64.StdEncoding.EncodeToString(typed.Data),
		}
	case model.FileURL:
		encoded = fileDataJSON{
			Type: "url",
			URL:  typed.URL.String(),
		}
	case model.FileText:
		encoded = fileDataJSON{
			Type: "text",
			Text: typed.Text,
		}
	case model.FileReference:
		encoded = fileDataJSON{
			Type:      "reference",
			Reference: typed.IDs,
		}
	default:
		return nil, fmt.Errorf("encode file data: unsupported type %T", data)
	}

	return json.Marshal(encoded)
}

// decodeContentOutput decodes a multipart output, accepting the legacy
// "image-data" and "file-data" items as inline files.
func decodeContentOutput(value json.RawMessage) (model.ToolResultOutput, error) {
	var items []toolResultContentJSON
	if err := json.Unmarshal(value, &items); err != nil {
		return nil, fmt.Errorf("decode content tool result output: %w", err)
	}

	content := make([]model.ToolResultContent, 0, len(items))

	for _, item := range items {
		decoded, err := decodeContentItem(item)
		if err != nil {
			return nil, err
		}

		content = append(content, decoded)
	}

	return model.ContentOutput{
		Value: content,
	}, nil
}

// decodeContentItem decodes one item of a content output.
func decodeContentItem(item toolResultContentJSON) (model.ToolResultContent, error) {
	switch item.Type {
	case "text":
		return model.ToolResultText{
			Text:            item.Text,
			ProviderOptions: item.ProviderOptions,
		}, nil
	case "custom":
		return model.ToolResultCustom{
			ProviderOptions: item.ProviderOptions,
		}, nil
	case "file", "image-data", "file-data":
		data, err := decodeFileData(item)
		if err != nil {
			return nil, err
		}

		return model.ToolResultFile{
			Filename:        item.Filename,
			MediaType:       item.MediaType,
			Data:            data,
			ProviderOptions: item.ProviderOptions,
		}, nil
	default:
		return nil, fmt.Errorf("decode tool result content: unsupported type %q", item.Type)
	}
}

// decodeFileData decodes the data of a file item. Legacy items carry the
// base64 data as a plain string.
func decodeFileData(item toolResultContentJSON) (model.FileData, error) {
	var base64Data string
	if err := json.Unmarshal(item.Data, &base64Data); err == nil {
		return decodeBase64File(base64Data)
	}

	var data fileDataJSON
	if err := json.Unmarshal(item.Data, &data); err != nil {
		return nil, fmt.Errorf("decode file data: %w", err)
	}

	switch data.Type {
	case "data":
		return decodeBase64File(data.Data)
	case "url":
		parsed, err := url.Parse(data.URL)
		if err != nil {
			return nil, fmt.Errorf("decode file url: %w", err)
		}

		return model.FileURL{
			URL: parsed,
		}, nil
	case "text":
		return model.FileText{
			Text: data.Text,
		}, nil
	case "reference":
		return model.FileReference{
			IDs: data.Reference,
		}, nil
	default:
		return nil, fmt.Errorf("decode file data: unknown type %q", data.Type)
	}
}

// decodeBase64File decodes inline base64 file content.
func decodeBase64File(data string) (model.FileData, error) {
	decoded, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		return nil, fmt.Errorf("decode base64 file data: %w", err)
	}

	return model.FileBytes{
		Data: decoded,
	}, nil
}

// jsonValueOrNull returns value, or JSON null when it is empty.
func jsonValueOrNull(value json.RawMessage) json.RawMessage {
	if len(value) == 0 {
		return json.RawMessage("null")
	}

	return value
}
