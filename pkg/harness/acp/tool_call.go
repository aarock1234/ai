package acp

import (
	"encoding/json"
)

// ToolCall is an ACP tool call as the implementation reported it.
type ToolCall struct {
	ToolCallID string `json:"toolCallId"`

	// Name is the programmatic tool name, when the implementation sends
	// one.
	Name string `json:"name,omitempty"`

	Title     string             `json:"title"`
	Kind      ToolKind           `json:"kind,omitempty"`
	Status    ToolCallStatus     `json:"status,omitempty"`
	Content   []ToolCallContent  `json:"content,omitempty"`
	Locations []ToolCallLocation `json:"locations,omitempty"`

	// RawInput and RawOutput are the implementation's raw tool input and
	// output.
	RawInput  json.RawMessage `json:"rawInput,omitempty"`
	RawOutput json.RawMessage `json:"rawOutput,omitempty"`

	Meta map[string]any `json:"_meta,omitempty"`
}

// ToolKind is the ACP category of a tool.
type ToolKind string

// ACP tool kinds.
const (
	KindRead       ToolKind = "read"
	KindEdit       ToolKind = "edit"
	KindDelete     ToolKind = "delete"
	KindMove       ToolKind = "move"
	KindSearch     ToolKind = "search"
	KindExecute    ToolKind = "execute"
	KindThink      ToolKind = "think"
	KindFetch      ToolKind = "fetch"
	KindSwitchMode ToolKind = "switch_mode"
	KindOther      ToolKind = "other"
)

// ToolCallStatus is the ACP execution status of a tool call.
type ToolCallStatus string

// ACP tool call statuses.
const (
	StatusPending    ToolCallStatus = "pending"
	StatusInProgress ToolCallStatus = "in_progress"
	StatusCompleted  ToolCallStatus = "completed"
	StatusFailed     ToolCallStatus = "failed"
)

// ToolCallContentType distinguishes the content of a tool call.
type ToolCallContentType string

// Tool call content types.
const (
	ToolContentBlock    ToolCallContentType = "content"
	ToolContentDiff     ToolCallContentType = "diff"
	ToolContentTerminal ToolCallContentType = "terminal"
)

// ToolCallContent is one content item of a tool call: a content block, a
// file diff or a terminal reference.
type ToolCallContent struct {
	Type ToolCallContentType `json:"type"`

	// Content is the block of the content type.
	Content *ContentBlock `json:"content,omitempty"`

	// Path, OldText and NewText describe the diff type. OldText is nil for
	// a new file.
	Path    string  `json:"path,omitempty"`
	OldText *string `json:"oldText,omitempty"`
	NewText string  `json:"newText,omitempty"`

	// TerminalID identifies the terminal of the terminal type.
	TerminalID string `json:"terminalId,omitempty"`

	Meta map[string]any `json:"_meta,omitempty"`
}

// ContentBlockType distinguishes ACP content blocks.
type ContentBlockType string

// ACP content block types.
const (
	BlockText         ContentBlockType = "text"
	BlockImage        ContentBlockType = "image"
	BlockAudio        ContentBlockType = "audio"
	BlockResourceLink ContentBlockType = "resource_link"
	BlockResource     ContentBlockType = "resource"
)

// ContentBlock is an ACP content block. Its type selects the fields: text
// carries Text; image and audio carry Data and MimeType; a resource link
// carries URI, Name and its descriptive fields; an embedded resource
// carries Resource.
type ContentBlock struct {
	Type ContentBlockType `json:"type"`

	Text string `json:"text,omitempty"`

	// Data is base64-encoded image or audio data.
	Data     string `json:"data,omitempty"`
	MimeType string `json:"mimeType,omitempty"`

	URI         string `json:"uri,omitempty"`
	Name        string `json:"name,omitempty"`
	Title       string `json:"title,omitempty"`
	Description string `json:"description,omitempty"`
	Size        *int64 `json:"size,omitempty"`

	Resource *EmbeddedResourceContents `json:"resource,omitempty"`

	Annotations *Annotations   `json:"annotations,omitempty"`
	Meta        map[string]any `json:"_meta,omitempty"`
}

// EmbeddedResourceContents are the contents of an embedded resource: Text
// for a text resource, or base64-encoded Blob for a binary one.
type EmbeddedResourceContents struct {
	URI      string         `json:"uri"`
	MimeType string         `json:"mimeType,omitempty"`
	Text     *string        `json:"text,omitempty"`
	Blob     *string        `json:"blob,omitempty"`
	Meta     map[string]any `json:"_meta,omitempty"`
}

// Annotations describe the audience and relevance of content.
type Annotations struct {
	Audience     []Role         `json:"audience,omitempty"`
	LastModified string         `json:"lastModified,omitempty"`
	Priority     *float64       `json:"priority,omitempty"`
	Meta         map[string]any `json:"_meta,omitempty"`
}

// Role is the audience of annotated content.
type Role string

// ACP roles.
const (
	RoleAssistant Role = "assistant"
	RoleUser      Role = "user"
)

// ToolCallLocation is a file location a tool call touches.
type ToolCallLocation struct {
	Path string         `json:"path"`
	Line *int           `json:"line,omitempty"`
	Meta map[string]any `json:"_meta,omitempty"`
}

// decodeToolCall decodes a tool call the bridge forwarded. A call whose
// optional fields do not match the ACP types keeps only its identity and raw
// payloads, as upstream checks only the identity.
func decodeToolCall(data json.RawMessage) (ToolCall, bool) {
	var toolCall ToolCall
	if err := json.Unmarshal(data, &toolCall); err == nil {
		return toolCall, toolCall.ToolCallID != ""
	}

	var identity struct {
		ToolCallID string          `json:"toolCallId"`
		Title      string          `json:"title"`
		RawInput   json.RawMessage `json:"rawInput"`
		RawOutput  json.RawMessage `json:"rawOutput"`
	}

	if err := json.Unmarshal(data, &identity); err != nil || identity.ToolCallID == "" {
		return ToolCall{}, false
	}

	return ToolCall{
		ToolCallID: identity.ToolCallID,
		Title:      identity.Title,
		RawInput:   identity.RawInput,
		RawOutput:  identity.RawOutput,
	}, true
}
