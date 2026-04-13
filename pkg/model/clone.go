package model

import (
	"encoding/json"
	"net/http"
)

// Clone returns a deep copy of the request.
func (r *Request) Clone() *Request {
	if r == nil {
		return nil
	}

	cloned := &Request{
		Temperature:      r.Temperature,
		MaxTokens:        r.MaxTokens,
		TopP:             r.TopP,
		TopK:             r.TopK,
		PresencePenalty:  r.PresencePenalty,
		FrequencyPenalty: r.FrequencyPenalty,
		Seed:             r.Seed,
		StopSequences:    append([]string(nil), r.StopSequences...),
		ActiveTools:      append([]string(nil), r.ActiveTools...),
		Headers:          cloneHeader(r.Headers),
		ProviderOptions:  r.ProviderOptions.Clone(),
	}

	if len(r.Messages) > 0 {
		cloned.Messages = make([]Message, len(r.Messages))
		for i, msg := range r.Messages {
			cloned.Messages[i] = cloneMessage(msg)
		}
	}

	if len(r.Tools) > 0 {
		cloned.Tools = make([]ToolDefinition, len(r.Tools))
		for i, tool := range r.Tools {
			cloned.Tools[i] = ToolDefinition{
				Name:        tool.Name,
				Description: tool.Description,
				InputSchema: append(json.RawMessage(nil), tool.InputSchema...),
				Strict:      tool.Strict,
			}
		}
	}

	if r.ToolChoice != nil {
		cloned.ToolChoice = &ToolChoice{
			Type:     r.ToolChoice.Type,
			ToolName: r.ToolChoice.ToolName,
		}
	}

	if r.ResponseFormat != nil {
		cloned.ResponseFormat = &ResponseFormat{
			Type: r.ResponseFormat.Type,
		}
		if r.ResponseFormat.JSONSchema != nil {
			cloned.ResponseFormat.JSONSchema = &JSONSchema{
				Name:   r.ResponseFormat.JSONSchema.Name,
				Schema: append(json.RawMessage(nil), r.ResponseFormat.JSONSchema.Schema...),
				Strict: r.ResponseFormat.JSONSchema.Strict,
			}
		}
	}

	return cloned
}

func cloneMessage(msg Message) Message {
	cloned := Message{
		Role: msg.Role,
	}

	if len(msg.Parts) > 0 {
		cloned.Parts = make([]Part, len(msg.Parts))
		for i, part := range msg.Parts {
			cloned.Parts[i] = clonePart(part)
		}
	}

	return cloned
}

func clonePart(part Part) Part {
	switch p := part.(type) {
	case TextPart:
		return TextPart{Text: p.Text}
	case ImagePart:
		return ImagePart{
			URL:      p.URL,
			Data:     append([]byte(nil), p.Data...),
			MIMEType: p.MIMEType,
		}
	case ToolCallPart:
		return ToolCallPart{
			ToolCallID: p.ToolCallID,
			ToolName:   p.ToolName,
			Args:       append(json.RawMessage(nil), p.Args...),
		}
	case ToolResultPart:
		return ToolResultPart{
			ToolCallID: p.ToolCallID,
			ToolName:   p.ToolName,
			Content:    p.Content,
			IsError:    p.IsError,
		}
	default:
		return nil
	}
}

func cloneHeader(headers http.Header) http.Header {
	if len(headers) == 0 {
		return nil
	}

	return headers.Clone()
}
