// Package toolresult encodes tool results for providers that accept them only
// as text. It mirrors how upstream providers serialize tool result outputs
// with JSON.stringify.
package toolresult

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/model"
)

// DeniedText is the result text of a denied tool execution without a reason.
const DeniedText = "Tool call execution denied."

// Text returns a tool result output as text: text outputs unchanged, denied
// executions as their reason, and structured outputs as compact JSON.
func Text(output model.ToolResultOutput) (string, error) {
	switch typed := output.(type) {
	case model.TextOutput:
		return typed.Value, nil

	case model.ErrorTextOutput:
		return typed.Value, nil

	case model.ExecutionDeniedOutput:
		if typed.Reason == "" {
			return DeniedText, nil
		}

		return typed.Reason, nil

	case model.JSONOutput:
		return CompactJSON(typed.Value)

	case model.ErrorJSONOutput:
		return CompactJSON(typed.Value)

	case model.ContentOutput:
		return ContentJSON(typed.Value)

	default:
		return "", fmt.Errorf("%w: tool result output %T", model.ErrUnsupported, output)
	}
}

// CompactJSON returns a JSON value, such as tool input or a JSON tool result,
// without insignificant whitespace. An absent value is null.
func CompactJSON(value json.RawMessage) (string, error) {
	if len(bytes.TrimSpace(value)) == 0 {
		return "null", nil
	}

	var compacted bytes.Buffer
	if err := json.Compact(&compacted, value); err != nil {
		return "", fmt.Errorf("%w: invalid JSON value: %w", model.ErrInvalidPrompt, err)
	}

	return compacted.String(), nil
}

// ContentJSON returns multipart tool result content as JSON text, with file
// contents in the tagged file data union.
func ContentJSON(items []model.ToolResultContent) (string, error) {
	encoded, err := wirejson.TaggedToolOutputContents(items)
	if err != nil {
		return "", err
	}

	text, err := json.Marshal(encoded)
	if err != nil {
		return "", fmt.Errorf("%w: encode tool result content: %w", model.ErrInvalidPrompt, err)
	}

	return string(text), nil
}
