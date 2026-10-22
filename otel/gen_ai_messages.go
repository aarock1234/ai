package otel

import (
	"encoding/base64"
	"encoding/json"
	"strings"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// genAIMessage is a message in the gen_ai.input.messages and
// gen_ai.output.messages formats of the semantic conventions. Output
// messages carry a finish reason.
type genAIMessage struct {
	Role         string      `json:"role"`
	Parts        []genAIPart `json:"parts"`
	FinishReason string      `json:"finish_reason,omitempty"`
}

// genAIPart is one part of a genAIMessage.
type genAIPart interface {
	isGenAIPart()
}

// genAITextPart is text or reasoning content.
type genAITextPart struct {
	Type    string `json:"type"`
	Content string `json:"content"`
}

type genAIToolCallPart struct {
	Type      string          `json:"type"`
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
}

type genAIToolCallResponsePart struct {
	Type     string `json:"type"`
	ID       string `json:"id"`
	Response any    `json:"response"`
}

// genAIBlobPart is inline file content. MimeType is null when the media
// type is unknown.
type genAIBlobPart struct {
	Type     string  `json:"type"`
	Modality string  `json:"modality"`
	MimeType *string `json:"mime_type"`
	Content  string  `json:"content"`
}

type genAIURIPart struct {
	Type     string  `json:"type"`
	Modality string  `json:"modality"`
	MimeType *string `json:"mime_type"`
	URI      string  `json:"uri"`
}

type genAIToolApprovalResponsePart struct {
	Type       string `json:"type"`
	ApprovalID string `json:"approval_id"`
	Approved   bool   `json:"approved"`
	Reason     string `json:"reason,omitempty"`
}

type genAICustomPart struct {
	Type string `json:"type"`
	Kind string `json:"kind"`
}

// genAITypePart is a part described only by its type.
type genAITypePart struct {
	Type string `json:"type"`
}

func (genAITextPart) isGenAIPart()                 {}
func (genAIToolCallPart) isGenAIPart()             {}
func (genAIToolCallResponsePart) isGenAIPart()     {}
func (genAIBlobPart) isGenAIPart()                 {}
func (genAIURIPart) isGenAIPart()                  {}
func (genAIToolApprovalResponsePart) isGenAIPart() {}
func (genAICustomPart) isGenAIPart()               {}
func (genAITypePart) isGenAIPart()                 {}

// deniedToolResponse is the response of a tool call the user denied.
type deniedToolResponse struct {
	Denied bool   `json:"denied"`
	Reason string `json:"reason,omitempty"`
}

// systemInstructions returns the gen_ai.system_instructions format of a
// system prompt.
func systemInstructions(system string) []genAITextPart {
	return []genAITextPart{
		textPart(system),
	}
}

// formatMessages converts messages to the gen_ai.input.messages format,
// preserving their order.
func formatMessages(messages []model.Message) []genAIMessage {
	formatted := make([]genAIMessage, 0, len(messages))

	for _, message := range messages {
		formatted = append(formatted, formatMessage(message))
	}

	return formatted
}

func formatMessage(message model.Message) genAIMessage {
	switch typed := message.(type) {
	case model.SystemMessage:
		return genAIMessage{
			Role: string(model.RoleSystem),
			Parts: []genAIPart{
				textPart(typed.Content),
			},
		}

	case model.UserMessage:
		return genAIMessage{
			Role:  string(model.RoleUser),
			Parts: formatParts(typed.Content),
		}

	case model.AssistantMessage:
		return genAIMessage{
			Role:  string(model.RoleAssistant),
			Parts: formatParts(typed.Content),
		}

	case model.ToolMessage:
		return genAIMessage{
			Role:  string(model.RoleTool),
			Parts: formatParts(typed.Content),
		}

	default:
		return genAIMessage{
			Role:  string(message.Role()),
			Parts: []genAIPart{},
		}
	}
}

func formatParts[Part any](parts []Part) []genAIPart {
	formatted := make([]genAIPart, 0, len(parts))

	for _, part := range parts {
		formatted = append(formatted, formatPart(part))
	}

	return formatted
}

func formatPart(part any) genAIPart {
	switch typed := part.(type) {
	case model.TextPart:
		return textPart(typed.Text)

	case model.ReasoningPart:
		return genAITextPart{
			Type:    "reasoning",
			Content: typed.Text,
		}

	case model.FilePart:
		return filePart(typed.MediaType, typed.Data)

	case model.ToolCallPart:
		return genAIToolCallPart{
			Type:      "tool_call",
			ID:        typed.ToolCallID,
			Name:      typed.ToolName,
			Arguments: typed.Input,
		}

	case model.ToolResultPart:
		return genAIToolCallResponsePart{
			Type:     "tool_call_response",
			ID:       typed.ToolCallID,
			Response: toolResultResponse(typed.Output),
		}

	case model.ToolApprovalResponsePart:
		return genAIToolApprovalResponsePart{
			Type:       "tool_approval_response",
			ApprovalID: typed.ApprovalID,
			Approved:   typed.Approved,
			Reason:     typed.Reason,
		}

	case model.CustomPart:
		return genAICustomPart{
			Type: "custom",
			Kind: typed.Kind,
		}

	case model.ReasoningFilePart:
		return genAITypePart{
			Type: "reasoning-file",
		}

	case model.ToolApprovalRequestPart:
		return genAITypePart{
			Type: "tool-approval-request",
		}

	default:
		return genAITypePart{
			Type: "unknown",
		}
	}
}

func textPart(text string) genAITextPart {
	return genAITextPart{
		Type:    "text",
		Content: text,
	}
}

// filePart returns a uri part for linked files and a blob part with base64
// or text content otherwise.
func filePart(mediaType string, data model.FileData) genAIPart {
	switch typed := data.(type) {
	case model.FileURL:
		if typed.URL == nil {
			return blobPart(mediaType, "")
		}

		return uriPart(mediaType, typed.URL.String())

	case model.FileBytes:
		return blobPart(mediaType, base64.StdEncoding.EncodeToString(typed.Data))

	case model.FileText:
		if isHTTPURL(typed.Text) {
			return uriPart(mediaType, typed.Text)
		}

		return blobPart(mediaType, typed.Text)

	default:
		return blobPart(mediaType, "")
	}
}

func uriPart(mediaType, uri string) genAIURIPart {
	return genAIURIPart{
		Type:     "uri",
		Modality: modality(mediaType),
		MimeType: optionalMediaType(mediaType),
		URI:      uri,
	}
}

func blobPart(mediaType, content string) genAIBlobPart {
	return genAIBlobPart{
		Type:     "blob",
		Modality: modality(mediaType),
		MimeType: optionalMediaType(mediaType),
		Content:  content,
	}
}

func isHTTPURL(text string) bool {
	return strings.HasPrefix(text, "http://") || strings.HasPrefix(text, "https://")
}

func optionalMediaType(mediaType string) *string {
	if mediaType == "" {
		return nil
	}

	return &mediaType
}

// modality returns the modality of a media type. Unknown media types are
// reported as images, as upstream does.
func modality(mediaType string) string {
	switch {
	case strings.HasPrefix(mediaType, "video/"):
		return "video"
	case strings.HasPrefix(mediaType, "audio/"):
		return "audio"
	default:
		return "image"
	}
}

// toolResultResponse returns the response of a tool result: the value of
// text, JSON and error outputs, a denial for denied calls, and the encoded
// output otherwise.
func toolResultResponse(output model.ToolResultOutput) any {
	switch typed := output.(type) {
	case model.TextOutput:
		return typed.Value
	case model.ErrorTextOutput:
		return typed.Value
	case model.JSONOutput:
		return typed.Value
	case model.ErrorJSONOutput:
		return typed.Value
	case model.ExecutionDeniedOutput:
		return deniedToolResponse{
			Denied: true,
			Reason: typed.Reason,
		}
	case nil:
		return nil
	default:
		return telemetryJSON.ToolOutput(output)
	}
}

// outputMessageContent is the content of a model response, reported as a
// gen_ai.output.messages assistant message.
type outputMessageContent struct {
	text         string
	reasoning    []string
	toolCalls    []ai.ToolCall
	toolResults  []ai.ToolResult
	files        []ai.GeneratedFile
	finishReason model.UnifiedFinishReason
}

// formatOutputMessages converts a response to the gen_ai.output.messages
// format: reasoning, text, tool calls, tool results and files, in that
// order.
func formatOutputMessages(content outputMessageContent) []genAIMessage {
	parts := make([]genAIPart, 0)

	for _, reasoning := range content.reasoning {
		parts = append(parts, genAITextPart{
			Type:    "reasoning",
			Content: reasoning,
		})
	}

	if content.text != "" {
		parts = append(parts, textPart(content.text))
	}

	for _, call := range content.toolCalls {
		parts = append(parts, genAIToolCallPart{
			Type:      "tool_call",
			ID:        call.ToolCallID,
			Name:      call.ToolName,
			Arguments: call.Input,
		})
	}

	for _, result := range content.toolResults {
		parts = append(parts, genAIToolCallResponsePart{
			Type:     "tool_call_response",
			ID:       result.ToolCallID,
			Response: result.Output,
		})
	}

	for _, file := range content.files {
		parts = append(parts, blobPart(file.MediaType, file.Base64()))
	}

	return []genAIMessage{
		{
			Role:         string(model.RoleAssistant),
			Parts:        parts,
			FinishReason: finishReason(content.finishReason),
		},
	}
}

// responseContent collects the output message content of a model call.
func responseContent(content []ai.ContentPart, reason model.UnifiedFinishReason) outputMessageContent {
	collected := outputMessageContent{
		finishReason: reason,
	}

	var text strings.Builder

	for _, part := range content {
		switch typed := part.(type) {
		case ai.TextContent:
			text.WriteString(typed.Text)
		case ai.ReasoningContent:
			collected.reasoning = appendNonEmpty(collected.reasoning, typed.Text)
		case ai.ToolCall:
			collected.toolCalls = append(collected.toolCalls, typed)
		case ai.ToolResult:
			collected.toolResults = append(collected.toolResults, typed)
		case ai.FileContent:
			collected.files = append(collected.files, typed.File)
		}
	}

	collected.text = text.String()

	return collected
}

// stepContent collects the output message content of a step's text,
// reasoning, tool calls, tool results and files.
func stepContent(step ai.StepResult) outputMessageContent {
	collected := outputMessageContent{
		text:         step.Text(),
		toolCalls:    step.ToolCalls(),
		toolResults:  step.ToolResults(),
		files:        step.Files(),
		finishReason: step.FinishReason,
	}

	for _, part := range step.Reasoning() {
		if reasoning, ok := part.(ai.ReasoningContent); ok {
			collected.reasoning = appendNonEmpty(collected.reasoning, reasoning.Text)
		}
	}

	return collected
}

func appendNonEmpty(values []string, value string) []string {
	if value == "" {
		return values
	}

	return append(values, value)
}

// finishReason maps a finish reason to the semantic conventions.
func finishReason(reason model.UnifiedFinishReason) string {
	switch reason {
	case model.FinishContentFilter:
		return "content_filter"
	case model.FinishToolCalls:
		return "tool_call"
	case model.FinishOther, "unknown":
		return "stop"
	default:
		return string(reason)
	}
}

// providerName maps an AI SDK provider string, such as "openai.chat" or
// "google.vertex.chat", to a gen_ai.provider.name value. Longer prefixes are
// matched first.
func providerName(provider string) string {
	lower := strings.ToLower(provider)

	for _, known := range wellKnownProviders {
		if hasProviderPrefix(lower, known.prefix) {
			return known.name
		}
	}

	return provider
}

// wellKnownProvider maps a provider string prefix to its semantic
// conventions name.
type wellKnownProvider struct {
	prefix string
	name   string
}

var wellKnownProviders = []wellKnownProvider{
	{
		prefix: "google.vertex",
		name:   "gcp.vertex_ai",
	},
	{
		prefix: "google.generative-ai",
		name:   "gcp.gemini",
	},
	{
		prefix: "google-vertex",
		name:   "gcp.vertex_ai",
	},
	{
		prefix: "amazon-bedrock",
		name:   "aws.bedrock",
	},
	{
		prefix: "azure-openai",
		name:   "azure.ai.openai",
	},
	{
		prefix: "anthropic",
		name:   "anthropic",
	},
	{
		prefix: "openai",
		name:   "openai",
	},
	{
		prefix: "azure",
		name:   "azure.ai.inference",
	},
	{
		prefix: "google",
		name:   "gcp.gemini",
	},
	{
		prefix: "mistral",
		name:   "mistral_ai",
	},
	{
		prefix: "cohere",
		name:   "cohere",
	},
	{
		prefix: "bedrock",
		name:   "aws.bedrock",
	},
	{
		prefix: "groq",
		name:   "groq",
	},
	{
		prefix: "deepseek",
		name:   "deepseek",
	},
	{
		prefix: "perplexity",
		name:   "perplexity",
	},
	{
		prefix: "xai",
		name:   "x_ai",
	},
}

func hasProviderPrefix(provider, prefix string) bool {
	return provider == prefix ||
		strings.HasPrefix(provider, prefix+".") ||
		strings.HasPrefix(provider, prefix+"-")
}

// operationName maps an operation ID to a gen_ai.operation.name value.
// Speech and transcription keep their operation IDs.
func operationName(operationID string) string {
	switch operationID {
	case operationGenerateText, operationStreamText:
		return "invoke_agent"
	case operationEmbed, operationEmbedMany:
		return "embeddings"
	case operationRerank:
		return "rerank"
	default:
		return operationID
	}
}
