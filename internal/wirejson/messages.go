package wirejson

import (
	"encoding/json"

	"github.com/aarock1234/ai/pkg/model"
)

// Message is an encoded prompt message: a SystemMessage or a PartsMessage.
type Message interface {
	isMessage()
}

// SystemMessage is an encoded system message.
type SystemMessage struct {
	Role            model.Role            `json:"role"`
	Content         string                `json:"content"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

// PartsMessage is an encoded user, assistant or tool message.
type PartsMessage struct {
	Role            model.Role            `json:"role"`
	Content         []Part                `json:"content"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

func (SystemMessage) isMessage() {}
func (PartsMessage) isMessage()  {}

// Part is one encoded part of a PartsMessage.
type Part interface {
	isPart()
}

// TextPart is an encoded text or reasoning part.
type TextPart struct {
	Type            string                `json:"type"`
	Text            string                `json:"text"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

// FilePart is an encoded file or reasoning file part. Data holds the file
// content in the encoder's file data encoding.
type FilePart struct {
	Type            string                `json:"type"`
	Filename        string                `json:"filename,omitempty"`
	MediaType       string                `json:"mediaType"`
	Data            FileData              `json:"data"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

// CustomPart is an encoded provider-specific part.
type CustomPart struct {
	Type            string                `json:"type"`
	Kind            string                `json:"kind"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

// ToolCallPart is an encoded tool call part.
type ToolCallPart struct {
	Type             string                `json:"type"`
	ToolCallID       string                `json:"toolCallId"`
	ToolName         string                `json:"toolName"`
	Input            json.RawMessage       `json:"input"`
	ProviderExecuted bool                  `json:"providerExecuted,omitempty"`
	ProviderOptions  model.ProviderOptions `json:"providerOptions,omitempty"`
}

// ToolResultPart is an encoded tool result part.
type ToolResultPart struct {
	Type            string                `json:"type"`
	ToolCallID      string                `json:"toolCallId"`
	ToolName        string                `json:"toolName"`
	Output          ToolOutput            `json:"output"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

// ToolApprovalRequestPart is an encoded tool approval request.
type ToolApprovalRequestPart struct {
	Type        string `json:"type"`
	ApprovalID  string `json:"approvalId"`
	ToolCallID  string `json:"toolCallId"`
	Reason      string `json:"reason,omitempty"`
	IsAutomatic bool   `json:"isAutomatic,omitempty"`
}

// ToolApprovalResponsePart is an encoded tool approval response.
type ToolApprovalResponsePart struct {
	Type             string                `json:"type"`
	ApprovalID       string                `json:"approvalId"`
	Approved         bool                  `json:"approved"`
	Reason           string                `json:"reason,omitempty"`
	ProviderExecuted bool                  `json:"providerExecuted,omitempty"`
	ProviderOptions  model.ProviderOptions `json:"providerOptions,omitempty"`
}

func (TextPart) isPart()                 {}
func (FilePart) isPart()                 {}
func (CustomPart) isPart()               {}
func (ToolCallPart) isPart()             {}
func (ToolResultPart) isPart()           {}
func (ToolApprovalRequestPart) isPart()  {}
func (ToolApprovalResponsePart) isPart() {}

// Messages encodes a prompt.
func (e Encoder) Messages(messages []model.Message) []Message {
	encoded := make([]Message, 0, len(messages))

	for _, message := range messages {
		encoded = append(encoded, e.message(message))
	}

	return encoded
}

func (e Encoder) message(message model.Message) Message {
	switch typed := message.(type) {
	case model.SystemMessage:
		return SystemMessage{
			Role:            model.RoleSystem,
			Content:         typed.Content,
			ProviderOptions: typed.ProviderOptions,
		}

	case model.UserMessage:
		return PartsMessage{
			Role:            model.RoleUser,
			Content:         encodeParts(e, typed.Content),
			ProviderOptions: typed.ProviderOptions,
		}

	case model.AssistantMessage:
		return PartsMessage{
			Role:            model.RoleAssistant,
			Content:         encodeParts(e, typed.Content),
			ProviderOptions: typed.ProviderOptions,
		}

	case model.ToolMessage:
		return PartsMessage{
			Role:            model.RoleTool,
			Content:         encodeParts(e, typed.Content),
			ProviderOptions: typed.ProviderOptions,
		}

	case nil:
		return PartsMessage{
			Content: []Part{},
		}

	default:
		return PartsMessage{
			Role:    message.Role(),
			Content: []Part{},
		}
	}
}

// encodeParts encodes the parts of one message, skipping unknown parts.
func encodeParts[MessagePart any](e Encoder, parts []MessagePart) []Part {
	encoded := make([]Part, 0, len(parts))

	for _, part := range parts {
		if wire, ok := e.part(part); ok {
			encoded = append(encoded, wire)
		}
	}

	return encoded
}

func (e Encoder) part(part any) (Part, bool) {
	switch typed := part.(type) {
	case model.TextPart:
		return TextPart{
			Type:            "text",
			Text:            typed.Text,
			ProviderOptions: typed.ProviderOptions,
		}, true

	case model.ReasoningPart:
		return TextPart{
			Type:            "reasoning",
			Text:            typed.Text,
			ProviderOptions: typed.ProviderOptions,
		}, true

	case model.FilePart:
		return FilePart{
			Type:            "file",
			Filename:        typed.Filename,
			MediaType:       typed.MediaType,
			Data:            e.fileData(typed.Data),
			ProviderOptions: typed.ProviderOptions,
		}, true

	case model.ReasoningFilePart:
		return FilePart{
			Type:            "reasoning-file",
			MediaType:       typed.MediaType,
			Data:            e.fileData(typed.Data),
			ProviderOptions: typed.ProviderOptions,
		}, true

	case model.CustomPart:
		return CustomPart{
			Type:            "custom",
			Kind:            typed.Kind,
			ProviderOptions: typed.ProviderOptions,
		}, true

	case model.ToolCallPart:
		return ToolCallPart{
			Type:             "tool-call",
			ToolCallID:       typed.ToolCallID,
			ToolName:         typed.ToolName,
			Input:            RawValue(typed.Input),
			ProviderExecuted: typed.ProviderExecuted,
			ProviderOptions:  typed.ProviderOptions,
		}, true

	case model.ToolResultPart:
		return ToolResultPart{
			Type:            "tool-result",
			ToolCallID:      typed.ToolCallID,
			ToolName:        typed.ToolName,
			Output:          e.ToolOutput(typed.Output),
			ProviderOptions: typed.ProviderOptions,
		}, true

	case model.ToolApprovalRequestPart:
		return ToolApprovalRequestPart{
			Type:        "tool-approval-request",
			ApprovalID:  typed.ApprovalID,
			ToolCallID:  typed.ToolCallID,
			Reason:      typed.Reason,
			IsAutomatic: typed.IsAutomatic,
		}, true

	case model.ToolApprovalResponsePart:
		return ToolApprovalResponsePart{
			Type:             "tool-approval-response",
			ApprovalID:       typed.ApprovalID,
			Approved:         typed.Approved,
			Reason:           typed.Reason,
			ProviderExecuted: typed.ProviderExecuted,
			ProviderOptions:  typed.ProviderOptions,
		}, true

	default:
		return nil, false
	}
}
