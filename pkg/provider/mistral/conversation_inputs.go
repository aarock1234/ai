package mistral

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aarock1234/ai/internal/toolnames"
	"github.com/aarock1234/ai/internal/toolresult"
	"github.com/aarock1234/ai/pkg/model"
)

// conversationPrompt is a prompt converted for the Conversations API.
type conversationPrompt struct {
	Inputs []conversationInput

	// Instructions joins the system messages.
	Instructions string
}

// inputConverter converts a prompt into conversation inputs. Built-in tool
// runs are replayed as function calls and results, because their entry IDs
// and execution entries cannot be sent as inputs.
type inputConverter struct {
	mapping toolnames.Mapping

	inputs       []conversationInput
	instructions []string

	// executions holds the IDs of replayed built-in tool calls.
	executions map[string]bool

	// pending collects assistant content until the next tool entry.
	pending []contentPart
}

func convertConversationPrompt(prompt []model.Message, mapping toolnames.Mapping) (conversationPrompt, error) {
	converter := &inputConverter{
		mapping:    mapping,
		executions: make(map[string]bool),
	}

	for _, message := range prompt {
		if err := converter.message(message); err != nil {
			return conversationPrompt{}, err
		}
	}

	return conversationPrompt{
		Inputs:       converter.inputs,
		Instructions: strings.Join(converter.instructions, "\n\n"),
	}, nil
}

func (c *inputConverter) message(message model.Message) error {
	switch message := message.(type) {
	case model.SystemMessage:
		c.instructions = append(c.instructions, message.Content)

		return nil

	case model.UserMessage:
		content, err := userContent(message)
		if err != nil {
			return err
		}

		c.inputs = append(c.inputs, messageInput{
			Type:    "message.input",
			Role:    "user",
			Content: content,
		})

		return nil

	case model.AssistantMessage:
		return c.assistant(message)

	case model.ToolMessage:
		for _, part := range message.Content {
			if result, ok := part.(model.ToolResultPart); ok {
				if err := c.toolResult(result); err != nil {
					return err
				}
			}
		}

		return nil

	default:
		return fmt.Errorf("%w: unsupported message role %q", model.ErrInvalidPrompt, message.Role())
	}
}

func (c *inputConverter) assistant(message model.AssistantMessage) error {
	for _, part := range message.Content {
		switch part := part.(type) {
		case model.TextPart:
			c.pending = append(c.pending, textPart(part.Text))

		case model.ReasoningPart:
			c.pending = append(c.pending, thinkingPart(part.Text))

		case model.ToolCallPart:
			c.flushAssistant()

			if err := c.toolCall(part); err != nil {
				return err
			}

		case model.ToolResultPart:
			c.flushAssistant()

			if err := c.toolResult(part); err != nil {
				return err
			}

		default:
			return fmt.Errorf("%w: assistant content type %T", model.ErrUnsupported, part)
		}
	}

	c.flushAssistant()

	return nil
}

// flushAssistant sends the collected assistant content as one message.
func (c *inputConverter) flushAssistant() {
	if len(c.pending) == 0 {
		return
	}

	c.inputs = append(c.inputs, messageInput{
		Type:    "message.input",
		Role:    "assistant",
		Content: c.pending,
	})
	c.pending = nil
}

func (c *inputConverter) toolCall(part model.ToolCallPart) error {
	if !part.ProviderExecuted {
		arguments, err := toolresult.CompactJSON(part.Input)
		if err != nil {
			return err
		}

		c.inputs = append(c.inputs, functionCallInput{
			Type:       "function.call",
			ToolCallID: part.ToolCallID,
			Name:       part.ToolName,
			Arguments:  arguments,
		})

		return nil
	}

	name, err := c.executionName(part)
	if err != nil {
		return err
	}

	arguments, err := executionArguments(part.Input)
	if err != nil {
		return err
	}

	c.inputs = append(c.inputs, functionCallInput{
		Type:       "function.call",
		ToolCallID: part.ToolCallID,
		Name:       name,
		Arguments:  arguments,
	})
	c.executions[part.ToolCallID] = true

	return nil
}

// executionName returns the function or tool name recorded in the call's
// metadata, or the provider name of its tool.
func (c *inputConverter) executionName(part model.ToolCallPart) (string, error) {
	metadata, _, err := model.ProviderValue[ToolExecutionMetadata](part.ProviderOptions, Namespace)
	if err != nil {
		return "", err
	}

	switch {
	case metadata.Function != "":
		return metadata.Function, nil
	case metadata.Name != "":
		return metadata.Name, nil
	default:
		return c.mapping.ToProvider(part.ToolName), nil
	}
}

// executionArguments returns the original arguments of a built-in tool call,
// or the whole input when it has no arguments string.
func executionArguments(input json.RawMessage) (string, error) {
	var decoded struct {
		Arguments *string `json:"arguments"`
	}

	if err := json.Unmarshal(input, &decoded); err == nil && decoded.Arguments != nil {
		return *decoded.Arguments, nil
	}

	return toolresult.CompactJSON(input)
}

func (c *inputConverter) toolResult(part model.ToolResultPart) error {
	result, err := c.resultText(part)
	if err != nil {
		return err
	}

	c.inputs = append(c.inputs, functionResultInput{
		Type:       "function.result",
		ToolCallID: part.ToolCallID,
		Result:     result,
	})

	return nil
}

func (c *inputConverter) resultText(part model.ToolResultPart) (string, error) {
	if !c.executions[part.ToolCallID] {
		return toolresult.Text(part.Output)
	}

	output, ok := part.Output.(model.JSONOutput)
	if !ok {
		return "null", nil
	}

	return executionResultText(output.Value)
}

// executionResultText extracts the result of a replayed built-in tool run:
// info.result when it is set, otherwise info, otherwise null.
func executionResultText(value json.RawMessage) (string, error) {
	var result executionResult
	if err := json.Unmarshal(value, &result); err != nil || !isJSONObject(result.Info) {
		return "null", nil
	}

	var info struct {
		Result json.RawMessage `json:"result"`
	}
	if err := json.Unmarshal(result.Info, &info); err != nil {
		return "", fmt.Errorf("%w: tool execution info: %w", model.ErrInvalidPrompt, err)
	}

	selected := result.Info
	if len(info.Result) > 0 && !isJSONNull(info.Result) {
		selected = info.Result
	}

	var text string
	if err := json.Unmarshal(selected, &text); err == nil {
		return text, nil
	}

	return toolresult.CompactJSON(selected)
}

func isJSONObject(value json.RawMessage) bool {
	return bytes.HasPrefix(bytes.TrimSpace(value), []byte("{"))
}

func isJSONNull(value json.RawMessage) bool {
	return bytes.Equal(bytes.TrimSpace(value), []byte("null"))
}
