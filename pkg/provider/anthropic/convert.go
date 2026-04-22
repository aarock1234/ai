package anthropic

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/aarock1234/ai/pkg/model"
)

// EncodeRequest converts a normalized model.Request into the Anthropic
// Messages API wire format.
func EncodeRequest(modelID string, req *model.Request, stream bool) (*MessageRequest, error) {
	if req.ResponseFormat != nil {
		return nil, errors.New("response_format is not supported by the anthropic provider")
	}

	system, messages, err := convertMessages(req.Messages)
	if err != nil {
		return nil, err
	}

	maxTokens := defaultMaxTokens
	if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}

	msgReq := &MessageRequest{
		Model:         modelID,
		System:        system,
		Messages:      messages,
		MaxTokens:     maxTokens,
		Temperature:   req.Temperature,
		TopP:          req.TopP,
		TopK:          req.TopK,
		StopSequences: req.StopSequences,
		Stream:        stream,
	}

	metadata, err := convertProviderMetadata(req.ProviderOptions)
	if err != nil {
		return nil, err
	}
	if len(metadata) > 0 {
		msgReq.Metadata = metadata
	}

	if len(req.Tools) > 0 {
		msgReq.Tools = convertTools(req.Tools)
	}

	if req.ToolChoice != nil {
		toolChoice, err := convertToolChoice(req.ToolChoice)
		if err != nil {
			return nil, err
		}
		msgReq.ToolChoice = toolChoice
	}

	if len(req.Headers) > 0 {
		msgReq.Headers = req.Headers.Clone()
	}

	return msgReq, nil
}

// DecodeResponse converts an Anthropic Messages API response into the
// normalized model.Response.
func DecodeResponse(resp *MessageResponse) *model.Response {
	result := &model.Response{
		FinishReason:    mapFinishReason(resp.StopReason),
		RawFinishReason: resp.StopReason,
		Usage: model.Usage{
			InputTokens:  resp.Usage.InputTokens,
			OutputTokens: resp.Usage.OutputTokens,
		},
		Metadata: responseMetadata(resp),
	}

	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			if block.Text == "" {
				continue
			}
			result.Content = append(result.Content, model.TextContent{Text: block.Text})
		case "thinking":
			result.Content = append(result.Content, model.ReasoningContent{
				Text:      block.Thinking,
				Signature: block.Signature,
				Format:    "anthropic-thinking",
			})
		case "redacted_thinking":
			result.Content = append(result.Content, model.ReasoningContent{
				Text:     block.Data,
				Format:   "anthropic-redacted-thinking",
				Redacted: true,
			})
		case "tool_use":
			result.Content = append(result.Content, model.ToolCallContent{
				ToolCallID: block.ID,
				ToolName:   block.Name,
				Args:       rawJSON(block.Input),
			})
		}
	}

	return result
}

func convertProviderMetadata(opts model.ProviderOptions) (map[string]string, error) {
	raw, ok := opts.Lookup("anthropic", "metadata")
	if !ok {
		return nil, nil
	}

	var metadata map[string]string
	if err := json.Unmarshal(raw, &metadata); err != nil {
		return nil, fmt.Errorf("anthropic metadata option must be an object of string values: %w", err)
	}

	return metadata, nil
}

func responseMetadata(resp *MessageResponse) model.ProviderMetadata {
	metadata := model.ProviderMetadata{
		RequestID: resp.ID,
		ModelID:   resp.Model,
	}

	provider := make(map[string]string)
	if resp.StopReason != "" {
		provider["stop_reason"] = resp.StopReason
	}
	if resp.StopSequence != nil && *resp.StopSequence != "" {
		provider["stop_sequence"] = *resp.StopSequence
	}
	for key, value := range resp.Metadata {
		provider[key] = value
	}
	if len(provider) > 0 {
		metadata.Provider = provider
	}

	return metadata
}

func convertMessages(msgs []model.Message) (string, []message, error) {
	var systemParts []string
	result := make([]message, 0, len(msgs))

	for _, msg := range msgs {
		switch msg.Role {
		case model.RoleSystem:
			text, err := convertSystemMessage(msg)
			if err != nil {
				return "", nil, err
			}
			if text != "" {
				systemParts = append(systemParts, text)
			}
		case model.RoleUser:
			wireMsg, err := convertUserMessage(msg)
			if err != nil {
				return "", nil, err
			}
			if wireMsg != nil {
				result = append(result, *wireMsg)
			}
		case model.RoleAssistant:
			wireMsg, err := convertAssistantMessage(msg)
			if err != nil {
				return "", nil, err
			}
			if wireMsg != nil {
				result = append(result, *wireMsg)
			}
		case model.RoleTool:
			wireMsg, err := convertToolResultMessage(msg)
			if err != nil {
				return "", nil, err
			}
			if wireMsg != nil {
				result = append(result, *wireMsg)
			}
		default:
			return "", nil, fmt.Errorf("unsupported message role: %s", msg.Role)
		}
	}

	return strings.Join(systemParts, "\n\n"), result, nil
}

func convertSystemMessage(msg model.Message) (string, error) {
	var parts []string

	for _, part := range msg.Parts {
		textPart, ok := part.(model.TextPart)
		if !ok {
			return "", errors.New("system messages only support text parts")
		}
		if textPart.Text != "" {
			parts = append(parts, textPart.Text)
		}
	}

	return strings.Join(parts, "\n\n"), nil
}

func convertUserMessage(msg model.Message) (*message, error) {
	blocks := make([]contentBlock, 0, len(msg.Parts))

	for _, part := range msg.Parts {
		block, err := convertUserPart(part)
		if err != nil {
			return nil, err
		}
		if block != nil {
			blocks = append(blocks, *block)
		}
	}

	if len(blocks) == 0 {
		return nil, nil
	}

	return &message{
		Role:    "user",
		Content: blocks,
	}, nil
}

func convertAssistantMessage(msg model.Message) (*message, error) {
	blocks := make([]contentBlock, 0, len(msg.Parts))

	for _, part := range msg.Parts {
		switch p := part.(type) {
		case model.TextPart:
			if p.Text == "" {
				continue
			}
			blocks = append(blocks, contentBlock{
				Type: "text",
				Text: p.Text,
			})
		case model.ReasoningPart:
			if p.Redacted {
				blocks = append(blocks, contentBlock{
					Type: "redacted_thinking",
					Data: p.Text,
				})

				continue
			}
			if p.Text == "" {
				continue
			}
			blocks = append(blocks, contentBlock{
				Type:      "thinking",
				Thinking:  p.Text,
				Signature: p.Signature,
			})
		case model.ToolCallPart:
			blocks = append(blocks, contentBlock{
				Type:  "tool_use",
				ID:    p.ToolCallID,
				Name:  p.ToolName,
				Input: objectJSON(p.Args),
			})
		case model.ImagePart:
			return nil, errors.New("assistant messages do not support image parts")
		case model.FilePart:
			return nil, errors.New("assistant messages do not support file parts")
		case model.ToolResultPart:
			return nil, errors.New("assistant messages do not support tool result parts")
		default:
			return nil, fmt.Errorf("unsupported assistant message part: %T", part)
		}
	}

	if len(blocks) == 0 {
		return nil, nil
	}

	return &message{
		Role:    "assistant",
		Content: blocks,
	}, nil
}

func convertToolResultMessage(msg model.Message) (*message, error) {
	blocks := make([]contentBlock, 0, len(msg.Parts))

	for _, part := range msg.Parts {
		toolResult, ok := part.(model.ToolResultPart)
		if !ok {
			return nil, errors.New("tool messages only support tool result parts")
		}
		blocks = append(blocks, contentBlock{
			Type:      "tool_result",
			ToolUseID: toolResult.ToolCallID,
			Content:   toolResult.Content,
			IsError:   toolResult.IsError,
		})
	}

	if len(blocks) == 0 {
		return nil, nil
	}

	return &message{
		Role:    "user",
		Content: blocks,
	}, nil
}

func convertUserPart(part model.Part) (*contentBlock, error) {
	switch p := part.(type) {
	case model.TextPart:
		if p.Text == "" {
			return nil, nil
		}
		return &contentBlock{
			Type: "text",
			Text: p.Text,
		}, nil
	case model.ImagePart:
		return convertImagePart(p)
	case model.FilePart:
		return convertFilePart(p)
	case model.ToolResultPart:
		return &contentBlock{
			Type:      "tool_result",
			ToolUseID: p.ToolCallID,
			Content:   p.Content,
			IsError:   p.IsError,
		}, nil
	case model.ToolCallPart:
		return nil, errors.New("user messages do not support tool call parts")
	default:
		return nil, fmt.Errorf("unsupported user message part: %T", part)
	}
}

// convertFilePart maps a generic FilePart to an Anthropic content block.
// Images fall back to the image path; PDFs become document blocks.
// Other MIME types are currently unsupported.
func convertFilePart(part model.FilePart) (*contentBlock, error) {
	switch {
	case strings.HasPrefix(part.MIMEType, "image/"):
		return convertImagePart(model.ImagePart{
			URL:      part.URL,
			Data:     part.Data,
			MIMEType: part.MIMEType,
		})
	case part.MIMEType == "application/pdf":
		switch {
		case len(part.Data) > 0:
			return &contentBlock{
				Type: "document",
				Source: &blockSource{
					Type:      "base64",
					MediaType: part.MIMEType,
					Data:      base64.StdEncoding.EncodeToString(part.Data),
				},
			}, nil
		case part.URL != "":
			return &contentBlock{
				Type: "document",
				Source: &blockSource{
					Type: "url",
					URL:  part.URL,
				},
			}, nil
		default:
			return nil, errors.New("document file part requires url or data")
		}
	default:
		return nil, fmt.Errorf("unsupported file mime type: %s", part.MIMEType)
	}
}

func convertImagePart(part model.ImagePart) (*contentBlock, error) {
	switch {
	case len(part.Data) > 0:
		if part.MIMEType == "" {
			return nil, errors.New("image data requires mime type")
		}
		return &contentBlock{
			Type: "image",
			Source: &blockSource{
				Type:      "base64",
				MediaType: part.MIMEType,
				Data:      base64.StdEncoding.EncodeToString(part.Data),
			},
		}, nil
	case part.URL != "":
		return &contentBlock{
			Type: "image",
			Source: &blockSource{
				Type: "url",
				URL:  part.URL,
			},
		}, nil
	default:
		return nil, errors.New("image part requires url or data")
	}
}

func convertTools(tools []model.ToolDefinition) []toolDefinition {
	result := make([]toolDefinition, len(tools))
	for i, tool := range tools {
		result[i] = toolDefinition{
			Name:        tool.Name,
			Description: tool.Description,
			InputSchema: objectJSON(tool.InputSchema),
			Strict:      tool.Strict,
		}
	}

	return result
}

func convertToolChoice(choice *model.ToolChoice) (*toolChoice, error) {
	switch choice.Type {
	case "", "auto":
		return &toolChoice{Type: "auto"}, nil
	case "none":
		return &toolChoice{Type: "none"}, nil
	case "required":
		return &toolChoice{Type: "any"}, nil
	case "tool":
		if choice.ToolName == "" {
			return nil, errors.New("tool choice type tool requires tool name")
		}
		return &toolChoice{
			Type: "tool",
			Name: choice.ToolName,
		}, nil
	default:
		return nil, fmt.Errorf("unsupported tool choice type: %s", choice.Type)
	}
}

func mapFinishReason(reason string) model.FinishReason {
	switch reason {
	case "end_turn", "stop_sequence":
		return model.FinishReasonStop
	case "max_tokens":
		return model.FinishReasonLength
	case "tool_use":
		return model.FinishReasonToolCalls
	default:
		return model.FinishReasonUnknown
	}
}

func objectJSON(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return json.RawMessage(`{}`)
	}

	return raw
}

func rawJSON(raw json.RawMessage) string {
	if len(raw) == 0 {
		return "{}"
	}

	return string(raw)
}

// MarshalRequest serializes a MessageRequest to JSON.
func MarshalRequest(req *MessageRequest) ([]byte, error) {
	return json.Marshal(req)
}
