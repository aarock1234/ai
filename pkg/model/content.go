package model

import "encoding/json"

// Content is one item of model output: Text, Reasoning, Custom,
// ReasoningFile, File, ToolApprovalRequest, Source, ToolCall or ToolResult.
type Content interface {
	isContent()
}

// Text is generated text.
type Text struct {
	Text             string
	Citations        []Citation
	ProviderMetadata ProviderMetadata
}

// Reasoning is reasoning text. Signatures and encrypted content travel in
// ProviderMetadata.
type Reasoning struct {
	Text             string
	ProviderMetadata ProviderMetadata
}

// Custom is provider-specific output with no standard meaning. Kind has the
// form "provider.type"; the content lives in ProviderMetadata.
type Custom struct {
	Kind             string
	ProviderMetadata ProviderMetadata
}

// ReasoningFile is a file produced during reasoning. Data is FileBytes or
// FileURL.
type ReasoningFile struct {
	MediaType        string
	Data             FileData
	ProviderMetadata ProviderMetadata
}

// File is a generated file. Data is FileBytes or FileURL.
type File struct {
	MediaType        string
	Data             FileData
	ProviderMetadata ProviderMetadata
}

// ToolApprovalRequest asks the application to approve a provider-executed
// tool call before the provider runs it.
type ToolApprovalRequest struct {
	ApprovalID       string
	ToolCallID       string
	ProviderMetadata ProviderMetadata
}

// ToolCall is a request to call a tool.
type ToolCall struct {
	ToolCallID string
	ToolName   string

	// Input is the JSON arguments as produced by the model. It may be
	// malformed and must be validated before execution.
	Input string

	// ProviderExecuted marks calls that the provider runs itself. The
	// application must not execute them.
	ProviderExecuted bool

	// Dynamic marks tools that were not declared in the call options, such as
	// tools discovered from an MCP server.
	Dynamic bool

	ProviderMetadata ProviderMetadata
}

// ToolResult is the result of a provider-executed tool call.
type ToolResult struct {
	ToolCallID string
	ToolName   string

	// Result is a non-null JSON value.
	Result json.RawMessage

	IsError bool

	// Preliminary marks a partial result that a later result replaces.
	Preliminary bool

	Dynamic          bool
	ProviderMetadata ProviderMetadata
}

// SourceType identifies the kind of source.
type SourceType string

// Source types.
const (
	SourceURL      SourceType = "url"
	SourceDocument SourceType = "document"
)

// Source is a reference used to produce the output.
type Source struct {
	SourceType SourceType
	ID         string

	// URL is set for URL sources.
	URL string

	// Title is optional for URL sources and required for document sources.
	Title string

	// MediaType and Filename describe document sources.
	MediaType string
	Filename  string

	ProviderMetadata ProviderMetadata
}

// Citation links a span of text to a source.
type Citation struct {
	Source Source

	// StartIndex and EndIndex are optional character offsets into the text.
	StartIndex *int
	EndIndex   *int

	// CitedText is the optional quoted passage.
	CitedText string
}

func (Text) isContent()                {}
func (Reasoning) isContent()           {}
func (Custom) isContent()              {}
func (ReasoningFile) isContent()       {}
func (File) isContent()                {}
func (ToolApprovalRequest) isContent() {}
func (ToolCall) isContent()            {}
func (ToolResult) isContent()          {}
func (Source) isContent()              {}
