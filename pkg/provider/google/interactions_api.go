package google

import (
	"encoding/json"
	"slices"
)

// Interaction statuses.
const (
	interactionInProgress     = "in_progress"
	interactionRequiresAction = "requires_action"
	interactionCompleted      = "completed"
	interactionFailed         = "failed"
	interactionCancelled      = "cancelled"
	interactionIncomplete     = "incomplete"
)

var interactionStatuses = []string{
	interactionInProgress,
	interactionRequiresAction,
	interactionCompleted,
	interactionFailed,
	interactionCancelled,
	interactionIncomplete,
}

// interactionTerminalStatuses end an interaction.
var interactionTerminalStatuses = []string{
	interactionCompleted,
	interactionFailed,
	interactionCancelled,
	interactionIncomplete,
}

func isTerminalInteraction(status string) bool {
	return slices.Contains(interactionTerminalStatuses, status)
}

// Built-in tool step types, whose names drop the _call or _result suffix.
var (
	builtinToolCallSteps = []string{
		"google_search_call",
		"code_execution_call",
		"url_context_call",
		"file_search_call",
		"google_maps_call",
		"mcp_server_tool_call",
	}

	builtinToolResultSteps = []string{
		"google_search_result",
		"code_execution_result",
		"url_context_result",
		"file_search_result",
		"google_maps_result",
		"mcp_server_tool_result",
	}
)

// Steps of MCP server tools carry the tool name.
const (
	mcpServerToolCallStep   = "mcp_server_tool_call"
	mcpServerToolResultStep = "mcp_server_tool_result"
	mcpServerToolName       = "mcp_server_tool"
)

// interactionsRequest is the body of POST /interactions. It names a model
// or an agent.
type interactionsRequest struct {
	Model                 string                         `json:"model,omitempty"`
	Agent                 string                         `json:"agent,omitempty"`
	Input                 []interactionsStep             `json:"input"`
	SystemInstruction     string                         `json:"system_instruction,omitempty"`
	Tools                 []interactionsTool             `json:"tools,omitempty"`
	ResponseFormat        []interactionsResponseFormat   `json:"response_format,omitempty"`
	ResponseModalities    []InteractionsResponseModality `json:"response_modalities,omitempty"`
	PreviousInteractionID string                         `json:"previous_interaction_id,omitempty"`
	ServiceTier           InteractionsServiceTier        `json:"service_tier,omitempty"`
	Store                 *bool                          `json:"store,omitempty"`
	GenerationConfig      *interactionsGenerationConfig  `json:"generation_config,omitempty"`
	AgentConfig           *interactionsAgentConfig       `json:"agent_config,omitempty"`
	Environment           *InteractionsEnvironment       `json:"environment,omitempty"`
	Background            *bool                          `json:"background,omitempty"`
	Stream                bool                           `json:"stream,omitempty"`
}

// interactionsStep is one step of the input or of a response. Function
// calls, thoughts and built-in tool calls and results carry their payload
// on the step; user input and model output carry content blocks.
type interactionsStep struct {
	Type string `json:"type"`

	Content []interactionsContent `json:"content,omitempty"`

	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
	Signature *string         `json:"signature,omitempty"`

	Summary []interactionsSummaryItem `json:"summary,omitempty"`

	CallID  string          `json:"call_id,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	IsError *bool           `json:"is_error,omitempty"`

	ServerName string `json:"server_name,omitempty"`
	SearchType string `json:"search_type,omitempty"`
}

// interactionsContent is a content block: text, image, audio, document or
// video, or the function result of a tool turn.
type interactionsContent struct {
	Type string `json:"type"`

	Text        *string                  `json:"text,omitempty"`
	Annotations []interactionsAnnotation `json:"annotations,omitempty"`

	Data       string                      `json:"data,omitempty"`
	MimeType   string                      `json:"mime_type,omitempty"`
	URI        string                      `json:"uri,omitempty"`
	Resolution InteractionsMediaResolution `json:"resolution,omitempty"`
	Processing json.RawMessage             `json:"processing,omitempty"`

	CallID    string          `json:"call_id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Signature string          `json:"signature,omitempty"`
	Result    json.RawMessage `json:"result,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`
}

// interactionsSummaryItem is one item of a thought summary.
type interactionsSummaryItem struct {
	Type     string  `json:"type"`
	Text     *string `json:"text,omitempty"`
	Data     string  `json:"data,omitempty"`
	MimeType string  `json:"mime_type,omitempty"`
}

// interactionsAnnotation is a url_citation, file_citation or
// place_citation of a text block.
type interactionsAnnotation struct {
	Type        string  `json:"type"`
	URL         string  `json:"url"`
	Title       *string `json:"title"`
	FileName    *string `json:"file_name"`
	DocumentURI string  `json:"document_uri"`
	Name        *string `json:"name"`
}

// interactionsVideoStaticProcessing is static video processing on the
// wire, with offsets as durations such as "5s".
type interactionsVideoStaticProcessing struct {
	Type        string   `json:"type"`
	StartOffset string   `json:"start_offset,omitempty"`
	EndOffset   string   `json:"end_offset,omitempty"`
	FPS         *float64 `json:"fps,omitempty"`
}

// interactionsResponseFormat is one entry of response_format.
type interactionsResponseFormat struct {
	Type        InteractionsResponseFormatType `json:"type"`
	MimeType    string                         `json:"mime_type,omitempty"`
	Schema      json.RawMessage                `json:"schema,omitempty"`
	AspectRatio string                         `json:"aspect_ratio,omitempty"`
	ImageSize   string                         `json:"image_size,omitempty"`
	Resolution  string                         `json:"resolution,omitempty"`
	Duration    string                         `json:"duration,omitempty"`
	Delivery    string                         `json:"delivery,omitempty"`
	GCSURI      string                         `json:"gcs_uri,omitempty"`
}

type interactionsGenerationConfig struct {
	Temperature       *float64                      `json:"temperature,omitempty"`
	TopP              *float64                      `json:"top_p,omitempty"`
	TopK              *int                          `json:"top_k,omitempty"`
	Seed              *int                          `json:"seed,omitempty"`
	StopSequences     []string                      `json:"stop_sequences,omitempty"`
	MaxOutputTokens   *int                          `json:"max_output_tokens,omitempty"`
	ThinkingLevel     InteractionsThinkingLevel     `json:"thinking_level,omitempty"`
	ThinkingSummaries InteractionsThinkingSummaries `json:"thinking_summaries,omitempty"`
	ToolChoice        *interactionsToolChoice       `json:"tool_choice,omitempty"`
}

// isEmpty reports whether no field is set.
func (c interactionsGenerationConfig) isEmpty() bool {
	return c.Temperature == nil &&
		c.TopP == nil &&
		c.TopK == nil &&
		c.Seed == nil &&
		c.StopSequences == nil &&
		c.MaxOutputTokens == nil &&
		c.ThinkingLevel == "" &&
		c.ThinkingSummaries == "" &&
		c.ToolChoice == nil
}

type interactionsAgentConfig struct {
	Type                  InteractionsAgentConfigType   `json:"type"`
	ThinkingSummaries     InteractionsThinkingSummaries `json:"thinking_summaries,omitempty"`
	Visualization         string                        `json:"visualization,omitempty"`
	CollaborativePlanning *bool                         `json:"collaborative_planning,omitempty"`
}

// interactionsToolChoice is a mode, such as "auto", or an allowed tools
// configuration.
type interactionsToolChoice struct {
	Mode         string
	AllowedTools *interactionsAllowedTools
}

type interactionsAllowedToolsConfig struct {
	AllowedTools interactionsAllowedTools `json:"allowed_tools"`
}

type interactionsAllowedTools struct {
	Mode  string   `json:"mode"`
	Tools []string `json:"tools"`
}

// MarshalJSON encodes the mode as a string, or the allowed tools as an
// object.
func (c interactionsToolChoice) MarshalJSON() ([]byte, error) {
	if c.AllowedTools == nil {
		return json.Marshal(c.Mode)
	}

	return json.Marshal(interactionsAllowedToolsConfig{
		AllowedTools: *c.AllowedTools,
	})
}

// interactionsResponse is an interaction. ID is absent when the
// interaction is not stored.
type interactionsResponse struct {
	ID          *string            `json:"id"`
	Created     *string            `json:"created"`
	Status      string             `json:"status"`
	Model       *string            `json:"model"`
	Steps       []interactionsStep `json:"steps"`
	Usage       json.RawMessage    `json:"usage"`
	ServiceTier *string            `json:"service_tier"`
}

// interactionsUsage is the token usage of an interaction.
type interactionsUsage struct {
	TotalInputTokens       *int                         `json:"total_input_tokens"`
	TotalOutputTokens      *int                         `json:"total_output_tokens"`
	TotalThoughtTokens     *int                         `json:"total_thought_tokens"`
	TotalCachedTokens      *int                         `json:"total_cached_tokens"`
	OutputTokensByModality []interactionsModalityTokens `json:"output_tokens_by_modality"`
}

type interactionsModalityTokens struct {
	Modality *string `json:"modality"`
	Tokens   *int    `json:"tokens"`
}

// interactionsEvent is one server-sent event of an interaction stream.
// The fields beyond the type depend on it.
type interactionsEvent struct {
	EventType string  `json:"event_type"`
	EventID   *string `json:"event_id"`

	Interaction *interactionsEventInteraction `json:"interaction"`

	Index *int               `json:"index"`
	Step  *interactionsStep  `json:"step"`
	Delta *interactionsDelta `json:"delta"`

	Status *string `json:"status"`

	Error *interactionsEventError `json:"error"`
}

type interactionsEventInteraction struct {
	ID          *string         `json:"id"`
	Created     *string         `json:"created"`
	Model       *string         `json:"model"`
	Status      *string         `json:"status"`
	Usage       json.RawMessage `json:"usage"`
	ServiceTier *string         `json:"service_tier"`
}

// interactionsDelta is the payload of a step.delta event. Arguments is a
// partial JSON string for arguments_delta and an object for built-in tool
// calls.
type interactionsDelta struct {
	Type string `json:"type"`

	Text        *string                  `json:"text"`
	Annotations []interactionsAnnotation `json:"annotations"`

	Content   *interactionsSummaryItem `json:"content"`
	Signature *string                  `json:"signature"`

	ID        *string         `json:"id"`
	Arguments json.RawMessage `json:"arguments"`
	Name      *string         `json:"name"`

	Data     *string `json:"data"`
	MimeType *string `json:"mime_type"`
	URI      *string `json:"uri"`

	CallID  *string         `json:"call_id"`
	Result  json.RawMessage `json:"result"`
	IsError *bool           `json:"is_error"`
}

type interactionsEventError struct {
	Code    *string `json:"code"`
	Message *string `json:"message"`
}
