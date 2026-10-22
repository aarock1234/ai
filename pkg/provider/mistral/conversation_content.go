package mistral

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/toolnames"
	"github.com/aarock1234/ai/pkg/model"
)

// ToolExecutionMetadata is the provider metadata of a built-in tool call
// that Mistral ran. Replaying it lets Mistral recognize the call.
type ToolExecutionMetadata struct {
	// Type is "tool.execution".
	Type string `json:"type"`

	// Name is Mistral's tool name, such as "web_search".
	Name string `json:"name"`

	// Function is the function the tool invoked, if reported.
	Function string `json:"function,omitempty"`
}

// executionInput is the input of a built-in tool call.
type executionInput struct {
	Arguments string `json:"arguments"`
}

// executionResult is the result of a built-in tool call.
type executionResult struct {
	Info json.RawMessage `json:"info,omitempty"`
}

// contentConverter converts conversation outputs. It names sources and
// skips repeated source URLs within one response.
type contentConverter struct {
	mapping    toolnames.Mapping
	generateID func() string
	sourceURLs map[string]bool
}

func newContentConverter(tools []model.Tool, generateID func() string) *contentConverter {
	return &contentConverter{
		mapping:    toolnames.New(tools, providerToolNames),
		generateID: generateID,
		sourceURLs: make(map[string]bool),
	}
}

// part converts a content part into text, reasoning or a source. Empty parts
// and repeated sources produce nothing.
func (c *contentConverter) part(part conversationPart) (model.Content, bool) {
	switch part.Type {
	case conversationText:
		if part.Text == "" {
			return nil, false
		}

		return model.Text{
			Text: part.Text,
		}, true

	case conversationThinking:
		text := joinThinking(part.Thinking)
		if text == "" {
			return nil, false
		}

		return model.Reasoning{
			Text: text,
		}, true

	case conversationToolReference:
		return c.source(part)

	default:
		return nil, false
	}
}

func (c *contentConverter) source(part conversationPart) (model.Content, bool) {
	if part.URL == nil || c.sourceURLs[*part.URL] {
		return nil, false
	}

	c.sourceURLs[*part.URL] = true

	title := *part.URL
	if part.Title != nil {
		title = *part.Title
	}

	return model.Source{
		SourceType: model.SourceURL,
		ID:         c.generateID(),
		URL:        *part.URL,
		Title:      title,
	}, true
}

// toolExecution converts a built-in tool run into a provider-executed tool
// call and its result.
func (c *contentConverter) toolExecution(execution toolExecution) (model.ToolCall, model.ToolResult, error) {
	toolName := c.mapping.ToApplication(execution.Name)

	input, err := json.Marshal(executionInput{
		Arguments: execution.Arguments,
	})
	if err != nil {
		return model.ToolCall{}, model.ToolResult{}, fmt.Errorf("encode tool input: %w", err)
	}

	info := execution.Info
	if isJSONNull(info) {
		info = nil
	}

	result, err := json.Marshal(executionResult{
		Info: info,
	})
	if err != nil {
		return model.ToolCall{}, model.ToolResult{}, fmt.Errorf("%w: tool execution info: %w", model.ErrInvalidResponse, err)
	}

	metadata := ToolExecutionMetadata{
		Type: "tool.execution",
		Name: execution.Name,
	}
	if execution.Function != nil {
		metadata.Function = *execution.Function
	}

	call := model.ToolCall{
		ToolCallID:       execution.ID,
		ToolName:         toolName,
		Input:            string(input),
		ProviderExecuted: true,
		ProviderMetadata: model.ProviderMetadata{
			Namespace: metadata,
		},
	}

	toolResult := model.ToolResult{
		ToolCallID: execution.ID,
		ToolName:   toolName,
		Result:     result,
	}

	return call, toolResult, nil
}
