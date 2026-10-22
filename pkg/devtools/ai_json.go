package devtools

import (
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// The records in this file encode the generated content and usage of
// package ai steps in the JSON shape of upstream step results.

// contentRecord is one recorded item of step content.
type contentRecord interface {
	isContentRecord()
}

// textContentRecord is generated text or reasoning.
type textContentRecord struct {
	Type             string                 `json:"type"`
	Text             string                 `json:"text"`
	Citations        []wirejson.Citation    `json:"citations,omitempty"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

type customContentRecord struct {
	Type             string                 `json:"type"`
	Kind             string                 `json:"kind"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// fileContentRecord is a generated file or reasoning file.
type fileContentRecord struct {
	Type             string                 `json:"type"`
	File             generatedFileRecord    `json:"file"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

// generatedFileRecord is a generated file. Upstream generated files keep
// their bytes in uint8ArrayData, which DevTools records as base64.
type generatedFileRecord struct {
	Data             string                 `json:"uint8ArrayData"`
	MediaType        string                 `json:"mediaType"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
}

type toolCallRecord struct {
	Type             string                 `json:"type"`
	ToolCallID       string                 `json:"toolCallId"`
	ToolName         string                 `json:"toolName"`
	Input            json.RawMessage        `json:"input"`
	ProviderExecuted bool                   `json:"providerExecuted,omitempty"`
	Dynamic          bool                   `json:"dynamic,omitempty"`
	Invalid          bool                   `json:"invalid,omitempty"`
	Error            string                 `json:"error,omitempty"`
	Title            string                 `json:"title,omitempty"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
	ToolMetadata     map[string]any         `json:"toolMetadata,omitempty"`
}

type toolResultRecord struct {
	Type             string                 `json:"type"`
	ToolCallID       string                 `json:"toolCallId"`
	ToolName         string                 `json:"toolName"`
	Input            json.RawMessage        `json:"input"`
	Output           json.RawMessage        `json:"output"`
	ProviderExecuted bool                   `json:"providerExecuted,omitempty"`
	Dynamic          bool                   `json:"dynamic,omitempty"`
	Preliminary      bool                   `json:"preliminary,omitempty"`
	Title            string                 `json:"title,omitempty"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
	ToolMetadata     map[string]any         `json:"toolMetadata,omitempty"`
}

type toolErrorRecord struct {
	Type             string                 `json:"type"`
	ToolCallID       string                 `json:"toolCallId"`
	ToolName         string                 `json:"toolName"`
	Input            json.RawMessage        `json:"input"`
	Error            string                 `json:"error"`
	ProviderExecuted bool                   `json:"providerExecuted,omitempty"`
	Dynamic          bool                   `json:"dynamic,omitempty"`
	Title            string                 `json:"title,omitempty"`
	ProviderMetadata model.ProviderMetadata `json:"providerMetadata,omitempty"`
	ToolMetadata     map[string]any         `json:"toolMetadata,omitempty"`
}

type toolApprovalRequestRecord struct {
	Type        string         `json:"type"`
	ApprovalID  string         `json:"approvalId"`
	ToolCall    toolCallRecord `json:"toolCall"`
	Reason      string         `json:"reason,omitempty"`
	IsAutomatic bool           `json:"isAutomatic,omitempty"`
	Signature   string         `json:"signature,omitempty"`
}

type toolApprovalResponseRecord struct {
	Type             string         `json:"type"`
	ApprovalID       string         `json:"approvalId"`
	ToolCall         toolCallRecord `json:"toolCall"`
	Approved         bool           `json:"approved"`
	Reason           string         `json:"reason,omitempty"`
	ProviderExecuted bool           `json:"providerExecuted,omitempty"`
}

// sourceRecord is a source, encoded with its fields at the top level.
type sourceRecord struct {
	wirejson.Source
}

func (textContentRecord) isContentRecord()          {}
func (customContentRecord) isContentRecord()        {}
func (fileContentRecord) isContentRecord()          {}
func (toolCallRecord) isContentRecord()             {}
func (toolResultRecord) isContentRecord()           {}
func (toolErrorRecord) isContentRecord()            {}
func (toolApprovalRequestRecord) isContentRecord()  {}
func (toolApprovalResponseRecord) isContentRecord() {}
func (sourceRecord) isContentRecord()               {}

// contentRecords encodes the content of a step.
func contentRecords(contents []ai.ContentPart) []contentRecord {
	records := make([]contentRecord, 0, len(contents))

	for _, content := range contents {
		if record, ok := newContentRecord(content); ok {
			records = append(records, record)
		}
	}

	return records
}

func newContentRecord(content ai.ContentPart) (contentRecord, bool) {
	switch typed := content.(type) {
	case ai.TextContent:
		return textContentRecord{
			Type:             "text",
			Text:             typed.Text,
			Citations:        wirejson.EncodeCitations(typed.Citations),
			ProviderMetadata: typed.ProviderMetadata,
		}, true

	case ai.ReasoningContent:
		return textContentRecord{
			Type:             "reasoning",
			Text:             typed.Text,
			ProviderMetadata: typed.ProviderMetadata,
		}, true

	case ai.CustomContent:
		return customContentRecord{
			Type:             "custom",
			Kind:             typed.Kind,
			ProviderMetadata: typed.ProviderMetadata,
		}, true

	case ai.FileContent:
		return fileContentRecord{
			Type:             "file",
			File:             newGeneratedFileRecord(typed.File),
			ProviderMetadata: typed.ProviderMetadata,
		}, true

	case ai.ReasoningFileContent:
		return fileContentRecord{
			Type:             "reasoning-file",
			File:             newGeneratedFileRecord(typed.File),
			ProviderMetadata: typed.ProviderMetadata,
		}, true

	case ai.SourceContent:
		return sourceRecord{
			Source: wirejson.EncodeSource(typed.Source),
		}, true

	default:
		return toolContentRecord(content)
	}
}

// toolContentRecord encodes tool calls, their outcomes and approvals.
func toolContentRecord(content ai.ContentPart) (contentRecord, bool) {
	switch typed := content.(type) {
	case ai.ToolCall:
		return newToolCallRecord(typed), true

	case ai.ToolResult:
		return toolResultRecord{
			Type:             "tool-result",
			ToolCallID:       typed.ToolCallID,
			ToolName:         typed.ToolName,
			Input:            wirejson.RawValue(typed.Input),
			Output:           applicationValue(typed.Output),
			ProviderExecuted: typed.ProviderExecuted,
			Dynamic:          typed.Dynamic,
			Preliminary:      typed.Preliminary,
			Title:            typed.Title,
			ProviderMetadata: typed.ProviderMetadata,
			ToolMetadata:     typed.ToolMetadata,
		}, true

	case ai.ToolError:
		return toolErrorRecord{
			Type:             "tool-error",
			ToolCallID:       typed.ToolCallID,
			ToolName:         typed.ToolName,
			Input:            wirejson.RawValue(typed.Input),
			Error:            errorText(typed.Error),
			ProviderExecuted: typed.ProviderExecuted,
			Dynamic:          typed.Dynamic,
			Title:            typed.Title,
			ProviderMetadata: typed.ProviderMetadata,
			ToolMetadata:     typed.ToolMetadata,
		}, true

	case ai.ToolApprovalRequest:
		return toolApprovalRequestRecord{
			Type:        "tool-approval-request",
			ApprovalID:  typed.ApprovalID,
			ToolCall:    newToolCallRecord(typed.ToolCall),
			Reason:      typed.Reason,
			IsAutomatic: typed.IsAutomatic,
			Signature:   typed.Signature,
		}, true

	case ai.ToolApprovalResponse:
		return toolApprovalResponseRecord{
			Type:             "tool-approval-response",
			ApprovalID:       typed.ApprovalID,
			ToolCall:         newToolCallRecord(typed.ToolCall),
			Approved:         typed.Approved,
			Reason:           typed.Reason,
			ProviderExecuted: typed.ProviderExecuted,
		}, true

	default:
		return nil, false
	}
}

func newToolCallRecord(call ai.ToolCall) toolCallRecord {
	return toolCallRecord{
		Type:             "tool-call",
		ToolCallID:       call.ToolCallID,
		ToolName:         call.ToolName,
		Input:            wirejson.RawValue(call.Input),
		ProviderExecuted: call.ProviderExecuted,
		Dynamic:          call.Dynamic,
		Invalid:          call.Invalid,
		Error:            errorText(call.Error),
		Title:            call.Title,
		ProviderMetadata: call.ProviderMetadata,
		ToolMetadata:     call.ToolMetadata,
	}
}

func newGeneratedFileRecord(file ai.GeneratedFile) generatedFileRecord {
	return generatedFileRecord{
		Data:             base64.StdEncoding.EncodeToString(file.Data),
		MediaType:        file.MediaType,
		ProviderMetadata: file.ProviderMetadata,
	}
}

// applicationValue encodes a value produced by application code, such as a
// tool output, with encoding/json so that its JSON tags apply. A value that
// cannot be encoded is recorded as its text representation.
func applicationValue(value any) json.RawMessage {
	encoded, err := json.Marshal(value)
	if err == nil {
		return encoded
	}

	encoded, err = json.Marshal(fmt.Sprint(value))
	if err != nil {
		return json.RawMessage("null")
	}

	return encoded
}

// errorText returns the message of err, or an empty string when it is nil.
func errorText(err error) string {
	if err == nil {
		return ""
	}

	return err.Error()
}

// usageRecord is the recorded token usage of a step.
type usageRecord struct {
	InputTokens        *int                     `json:"inputTokens,omitempty"`
	InputTokenDetails  inputTokenDetailsRecord  `json:"inputTokenDetails"`
	OutputTokens       *int                     `json:"outputTokens,omitempty"`
	OutputTokenDetails outputTokenDetailsRecord `json:"outputTokenDetails"`
	TotalTokens        *int                     `json:"totalTokens,omitempty"`
	Raw                json.RawMessage          `json:"raw,omitempty"`
}

type inputTokenDetailsRecord struct {
	NoCacheTokens    *int `json:"noCacheTokens,omitempty"`
	CacheReadTokens  *int `json:"cacheReadTokens,omitempty"`
	CacheWriteTokens *int `json:"cacheWriteTokens,omitempty"`
}

type outputTokenDetailsRecord struct {
	TextTokens      *int `json:"textTokens,omitempty"`
	ReasoningTokens *int `json:"reasoningTokens,omitempty"`
}

func newUsageRecord(usage ai.LanguageModelUsage) usageRecord {
	record := usageRecord{
		InputTokens: usage.InputTokens,
		InputTokenDetails: inputTokenDetailsRecord{
			NoCacheTokens:    usage.InputTokenDetails.NoCacheTokens,
			CacheReadTokens:  usage.InputTokenDetails.CacheReadTokens,
			CacheWriteTokens: usage.InputTokenDetails.CacheWriteTokens,
		},
		OutputTokens: usage.OutputTokens,
		OutputTokenDetails: outputTokenDetailsRecord{
			TextTokens:      usage.OutputTokenDetails.TextTokens,
			ReasoningTokens: usage.OutputTokenDetails.ReasoningTokens,
		},
		TotalTokens: usage.TotalTokens,
	}

	if len(usage.Raw) > 0 {
		record.Raw = wirejson.RawValue(usage.Raw)
	}

	return record
}
