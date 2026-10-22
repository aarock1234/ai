package pi

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/harness"
)

// convertToolResult ports upstream toPiHostToolResult: the content the
// adapter's extension returns to Pi for an application tool call. A raw
// output is serialized to one text part, a string as itself; a model-facing
// output keeps its text and inline images. The extension truncates the
// content as upstream does.
func convertToolResult(submission harness.ToolResultSubmission) (hostToolResult, error) {
	if submission.ToolResult == nil {
		text, err := serializeToolOutput(submission.Output)
		if err != nil {
			return hostToolResult{}, err
		}

		return hostToolResult{
			Content: []hostToolContent{
				{
					Type: "text",
					Text: text,
				},
			},
			IsError: submission.IsError,
		}, nil
	}

	converted, err := harness.ConvertToolModelOutput(submission.ToolResult.Output)
	if err != nil {
		return hostToolResult{}, err
	}

	content := make([]hostToolContent, 0, len(converted.Content))

	for _, item := range converted.Content {
		if item.Type == harness.ToolModelOutputImage {
			content = append(content, hostToolContent{
				Type:     "image",
				Data:     item.Data,
				MIMEType: item.MediaType,
			})

			continue
		}

		content = append(content, hostToolContent{
			Type: "text",
			Text: item.Text,
		})
	}

	return hostToolResult{
		Content:         content,
		FromModelOutput: true,
		IsError:         submission.IsError || converted.IsError,
	}, nil
}

// serializeToolOutput returns the text Pi feeds back to the model.
func serializeToolOutput(output any) (string, error) {
	switch typed := output.(type) {
	case string:
		return typed, nil
	case nil:
		return "null", nil
	}

	encoded, err := json.Marshal(output)
	if err != nil {
		return "", fmt.Errorf("encode pi tool output: %w", err)
	}

	var text string
	if err := json.Unmarshal(encoded, &text); err == nil {
		return text, nil
	}

	return string(encoded), nil
}

// encodeOutput returns the JSON of the output reported to consumers.
func encodeOutput(output any) json.RawMessage {
	encoded, err := json.Marshal(output)
	if err != nil {
		return json.RawMessage("null")
	}

	return encoded
}
