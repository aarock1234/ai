package openai

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// EncodeRequest converts a normalized model.Request into the OpenAI
// chat completions wire format.
func EncodeRequest(modelID string, req *model.Request, stream bool) (*ChatRequest, error) {
	msgs := convertMessages(req.Messages)

	cr := &ChatRequest{
		Model:       modelID,
		Messages:    msgs,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		TopP:        req.TopP,
		Stop:        req.StopSequences,
		Stream:      stream,
	}

	metadata, err := convertProviderMetadata(req.ProviderOptions)
	if err != nil {
		return nil, err
	}
	if len(metadata) > 0 {
		cr.Metadata = metadata
	}

	if stream {
		cr.StreamOptions = &streamOptions{IncludeUsage: true}
	}

	if len(req.Tools) > 0 {
		cr.Tools = convertTools(req.Tools)
	}

	if req.ToolChoice != nil {
		cr.ToolChoice = convertToolChoice(req.ToolChoice)
	}

	if req.ResponseFormat != nil {
		cr.ResponseFormat = convertResponseFormat(req.ResponseFormat)
	}

	return cr, nil
}

// DecodeResponse converts an OpenAI chat completions response into
// the normalized model.Response.
func DecodeResponse(resp *ChatResponse) *model.Response {
	if len(resp.Choices) == 0 {
		return &model.Response{
			FinishReason: model.FinishReasonUnknown,
			Usage: model.Usage{
				InputTokens:  resp.Usage.PromptTokens,
				OutputTokens: resp.Usage.CompletionTokens,
			},
			Metadata: responseMetadata(resp),
		}
	}

	choice := resp.Choices[0]
	var content []model.Content

	if choice.Message.Content != nil {
		if text, ok := choice.Message.Content.(string); ok && text != "" {
			content = append(content, model.TextContent{Text: text})
		}
	}

	for _, tc := range choice.Message.ToolCalls {
		content = append(content, model.ToolCallContent{
			ToolCallID: tc.ID,
			ToolName:   tc.Function.Name,
			Args:       tc.Function.Arguments,
		})
	}

	return &model.Response{
		Content:      content,
		FinishReason: mapFinishReason(choice.FinishReason),
		Usage: model.Usage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
		},
		Metadata: responseMetadata(resp),
	}
}

func convertProviderMetadata(opts model.ProviderOptions) (map[string]string, error) {
	raw, ok := opts.Lookup("openai", "metadata")
	if !ok {
		return nil, nil
	}

	var metadata map[string]string
	if err := json.Unmarshal(raw, &metadata); err != nil {
		return nil, fmt.Errorf("openai metadata option must be an object of string values: %w", err)
	}

	return metadata, nil
}

func responseMetadata(resp *ChatResponse) model.ProviderMetadata {
	metadata := model.ProviderMetadata{
		RequestID: resp.ID,
		ModelID:   resp.Model,
	}

	if len(resp.Metadata) > 0 {
		metadata.Provider = make(map[string]string, len(resp.Metadata))
		for key, value := range resp.Metadata {
			metadata.Provider[key] = value
		}
	}

	return metadata
}

// convertMessages translates model messages into the OpenAI wire format.
func convertMessages(msgs []model.Message) []chatMessage {
	result := make([]chatMessage, 0, len(msgs))

	for _, msg := range msgs {
		result = append(result, convertMessage(msg)...)
	}

	return result
}

// convertMessage converts a single model.Message. A message with
// tool call parts becomes an assistant message; tool result parts
// each become separate tool-role messages.
func convertMessage(msg model.Message) []chatMessage {
	// Collect text, tool calls, and tool results from parts.
	var text string
	var toolCalls []wireToolCall
	var toolResults []chatMessage

	for _, part := range msg.Parts {
		switch p := part.(type) {
		case model.TextPart:
			text += p.Text
		case model.ToolCallPart:
			toolCalls = append(toolCalls, wireToolCall{
				ID:   p.ToolCallID,
				Type: "function",
				Function: wireFunction{
					Name:      p.ToolName,
					Arguments: string(p.Args),
				},
			})
		case model.ToolResultPart:
			toolResults = append(toolResults, chatMessage{
				Role:       "tool",
				Content:    p.Content,
				ToolCallID: p.ToolCallID,
				Name:       p.ToolName,
			})
		case model.ImagePart:
			// Image parts use the content array format.
			// For now, only URL-based images are supported.
			// TODO: support base64 inline images.
		}
	}

	var result []chatMessage

	// If the message has tool calls, it's an assistant message.
	if len(toolCalls) > 0 {
		cm := chatMessage{
			Role:      string(msg.Role),
			ToolCalls: toolCalls,
		}
		if text != "" {
			cm.Content = text
		}
		result = append(result, cm)
	} else if text != "" || msg.Role == model.RoleSystem {
		result = append(result, chatMessage{
			Role:    string(msg.Role),
			Content: text,
		})
	}

	// Tool results are separate messages with role "tool".
	result = append(result, toolResults...)

	return result
}

// convertTools translates model tool definitions to the wire format.
func convertTools(tools []model.ToolDefinition) []chatTool {
	result := make([]chatTool, len(tools))
	for i, t := range tools {
		result[i] = chatTool{
			Type: "function",
			Function: toolFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.InputSchema,
			},
		}
	}

	return result
}

// convertToolChoice translates the model ToolChoice to the wire format.
// OpenAI accepts "auto", "none", "required" as strings, or a specific
// tool as an object.
func convertToolChoice(tc *model.ToolChoice) any {
	switch tc.Type {
	case "tool":
		return toolChoiceSpec{
			Type:     "function",
			Function: toolChoiceSpecFunction{Name: tc.ToolName},
		}
	default:
		return tc.Type
	}
}

// convertResponseFormat translates the model ResponseFormat to wire format.
func convertResponseFormat(rf *model.ResponseFormat) *responseFormat {
	r := &responseFormat{Type: rf.Type}

	if rf.JSONSchema != nil {
		r.JSONSchema = &jsonSchemaSpec{
			Name:   rf.JSONSchema.Name,
			Schema: rf.JSONSchema.Schema,
			Strict: rf.JSONSchema.Strict,
		}
	}

	return r
}

// mapFinishReason converts an OpenAI finish reason string to the
// model.FinishReason enum.
func mapFinishReason(reason string) model.FinishReason {
	switch reason {
	case "stop":
		return model.FinishReasonStop
	case "length":
		return model.FinishReasonLength
	case "tool_calls":
		return model.FinishReasonToolCalls
	case "content_filter":
		return model.FinishReasonContentFilter
	default:
		return model.FinishReasonUnknown
	}
}

// MarshalRequest serializes a chatRequest to JSON.
func MarshalRequest(req *ChatRequest) ([]byte, error) {
	return json.Marshal(req)
}
