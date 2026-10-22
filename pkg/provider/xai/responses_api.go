package xai

import (
	"bytes"
	"encoding/json"
	"fmt"
)

type responsesRequest struct {
	Model              string           `json:"model"`
	Input              []inputItem      `json:"input"`
	Logprobs           *bool            `json:"logprobs,omitempty"`
	TopLogprobs        *int             `json:"top_logprobs,omitempty"`
	MaxOutputTokens    *int             `json:"max_output_tokens,omitempty"`
	Temperature        *float64         `json:"temperature,omitempty"`
	TopP               *float64         `json:"top_p,omitempty"`
	TopK               *int             `json:"top_k,omitempty"`
	MinP               *float64         `json:"min_p,omitempty"`
	Seed               *int             `json:"seed,omitempty"`
	Text               *textConfig      `json:"text,omitempty"`
	Reasoning          *reasoningConfig `json:"reasoning,omitempty"`
	Store              *bool            `json:"store,omitempty"`
	Include            []IncludeValue   `json:"include,omitempty"`
	PreviousResponseID *string          `json:"previous_response_id,omitempty"`
	MaxTurns           *int             `json:"max_turns,omitempty"`
	ParallelToolCalls  *bool            `json:"parallel_tool_calls,omitempty"`
	PromptCacheKey     *string          `json:"prompt_cache_key,omitempty"`
	SafetyIdentifier   *string          `json:"safety_identifier,omitempty"`
	ServiceTier        string           `json:"service_tier,omitempty"`
	User               *string          `json:"user,omitempty"`
	Tools              []responsesTool  `json:"tools,omitempty"`
	ToolChoice         *toolChoice      `json:"tool_choice,omitempty"`
	Stream             bool             `json:"stream,omitempty"`
}

type textConfig struct {
	Format textFormat `json:"format"`
}

type textFormat struct {
	Type        string          `json:"type"`
	Strict      *bool           `json:"strict,omitempty"`
	Name        string          `json:"name,omitempty"`
	Description string          `json:"description,omitempty"`
	Schema      json.RawMessage `json:"schema,omitempty"`
}

type reasoningConfig struct {
	Effort ReasoningEffort `json:"effort"`
}

// responsesTool is a function tool or a server-side tool. Each tool type
// sets its own fields.
type responsesTool struct {
	Type string `json:"type"`

	// Function tools.
	Name        string          `json:"name,omitempty"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
	Strict      *bool           `json:"strict,omitempty"`

	// web_search and x_search.
	AllowedDomains           []string `json:"allowed_domains,omitempty"`
	ExcludedDomains          []string `json:"excluded_domains,omitempty"`
	EnableImageSearch        *bool    `json:"enable_image_search,omitempty"`
	EnableImageUnderstanding *bool    `json:"enable_image_understanding,omitempty"`
	AllowedXHandles          []string `json:"allowed_x_handles,omitempty"`
	ExcludedXHandles         []string `json:"excluded_x_handles,omitempty"`
	FromDate                 string   `json:"from_date,omitempty"`
	ToDate                   string   `json:"to_date,omitempty"`
	EnableVideoUnderstanding *bool    `json:"enable_video_understanding,omitempty"`

	// image_generation.
	Action ImageGenerationAction `json:"action,omitempty"`

	// file_search.
	VectorStoreIDs []string `json:"vector_store_ids,omitempty"`
	MaxNumResults  *float64 `json:"max_num_results,omitempty"`

	// mcp.
	ServerURL         string            `json:"server_url,omitempty"`
	ServerLabel       string            `json:"server_label,omitempty"`
	ServerDescription string            `json:"server_description,omitempty"`
	AllowedTools      []string          `json:"allowed_tools,omitempty"`
	Headers           map[string]string `json:"headers,omitempty"`
	Authorization     string            `json:"authorization,omitempty"`
}

// toolChoice is "auto", "none", "required" or a named function.
type toolChoice struct {
	mode     string
	function string
}

type namedToolChoice struct {
	Type string `json:"type"`
	Name string `json:"name"`
}

// MarshalJSON encodes a mode as a string and a function as an object.
func (c toolChoice) MarshalJSON() ([]byte, error) {
	if c.function == "" {
		return json.Marshal(c.mode)
	}

	return json.Marshal(namedToolChoice{
		Type: "function",
		Name: c.function,
	})
}

// inputItem is a message, a function call or output, or a reasoning item.
type inputItem interface {
	isInputItem()
}

type systemInput struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type userInput struct {
	Role    string      `json:"role"`
	Content []inputPart `json:"content"`
}

// inputPart is a text, image or file part. Each type sets its own fields.
type inputPart struct {
	Type     string  `json:"type"`
	Text     *string `json:"text,omitempty"`
	ImageURL string  `json:"image_url,omitempty"`
	Detail   string  `json:"detail,omitempty"`
	FileID   string  `json:"file_id,omitempty"`
	FileURL  string  `json:"file_url,omitempty"`
}

type assistantInput struct {
	Role    string  `json:"role"`
	Content string  `json:"content"`
	ID      *string `json:"id,omitempty"`
}

type functionCallInput struct {
	Type      string `json:"type"`
	ID        string `json:"id"`
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
	Status    string `json:"status"`
}

type functionOutputInput struct {
	Type   string         `json:"type"`
	CallID string         `json:"call_id"`
	Output functionOutput `json:"output"`
}

// functionOutput is text, or content parts for multipart results.
type functionOutput struct {
	text  string
	parts []inputPart
}

// MarshalJSON encodes the parts, or the text when there are no parts.
func (o functionOutput) MarshalJSON() ([]byte, error) {
	if o.parts != nil {
		return json.Marshal(o.parts)
	}

	return json.Marshal(o.text)
}

type reasoningInput struct {
	Type             string        `json:"type"`
	ID               string        `json:"id"`
	Summary          []summaryText `json:"summary"`
	Status           string        `json:"status"`
	EncryptedContent *string       `json:"encrypted_content,omitempty"`
}

type summaryText struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

func (systemInput) isInputItem()         {}
func (userInput) isInputItem()           {}
func (assistantInput) isInputItem()      {}
func (functionCallInput) isInputItem()   {}
func (functionOutputInput) isInputItem() {}
func (reasoningInput) isInputItem()      {}

type responsesResponse struct {
	ID                *string            `json:"id"`
	CreatedAt         *float64           `json:"created_at"`
	Model             *string            `json:"model"`
	Output            []outputItem       `json:"output"`
	Usage             *responsesUsage    `json:"usage"`
	Status            *string            `json:"status"`
	ServiceTier       *string            `json:"service_tier"`
	PromptCacheKey    *string            `json:"prompt_cache_key"`
	SafetyIdentifier  *string            `json:"safety_identifier"`
	IncompleteDetails *incompleteDetails `json:"incomplete_details"`
	Error             *responseError     `json:"error"`
}

type incompleteDetails struct {
	Reason string `json:"reason"`
}

type responseError struct {
	Code    *string `json:"code"`
	Message string  `json:"message"`
}

// responsesUsage is the token usage of a response. The raw object is kept as
// the usage Raw value.
type responsesUsage struct {
	InputTokens         int                  `json:"input_tokens"`
	OutputTokens        int                  `json:"output_tokens"`
	InputTokensDetails  *inputTokensDetails  `json:"input_tokens_details"`
	OutputTokensDetails *outputTokensDetails `json:"output_tokens_details"`
	CostInUSDTicks      *float64             `json:"cost_in_usd_ticks"`

	raw json.RawMessage
}

type inputTokensDetails struct {
	CachedTokens *int `json:"cached_tokens"`
}

type outputTokensDetails struct {
	ReasoningTokens *int `json:"reasoning_tokens"`
}

// UnmarshalJSON decodes the usage and keeps the raw object.
func (u *responsesUsage) UnmarshalJSON(data []byte) error {
	type plain responsesUsage

	var decoded plain
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}

	*u = responsesUsage(decoded)
	u.raw = bytes.Clone(data)

	return nil
}

// Output item types.
const (
	itemMessage             = "message"
	itemFunctionCall        = "function_call"
	itemReasoning           = "reasoning"
	itemFileSearchCall      = "file_search_call"
	itemImageGenerationCall = "image_generation_call"
	itemWebSearchCall       = "web_search_call"
	itemXSearchCall         = "x_search_call"
	itemCodeInterpreterCall = "code_interpreter_call"
	itemCodeExecutionCall   = "code_execution_call"
	itemViewImageCall       = "view_image_call"
	itemViewXVideoCall      = "view_x_video_call"
	itemCustomToolCall      = "custom_tool_call"
	itemMCPCall             = "mcp_call"
)

// serverToolCallTypes are the output items of server-side tool calls that
// share the tool call shape.
var serverToolCallTypes = map[string]bool{
	itemWebSearchCall:       true,
	itemXSearchCall:         true,
	itemCodeInterpreterCall: true,
	itemCodeExecutionCall:   true,
	itemViewImageCall:       true,
	itemViewXVideoCall:      true,
	itemCustomToolCall:      true,
	itemMCPCall:             true,
}

// outputItem is one response output item. Its fields are those of every
// item type; each type sets its own.
type outputItem struct {
	Type   string `json:"type"`
	ID     string `json:"id"`
	Status string `json:"status"`

	// message.
	Role    string           `json:"role"`
	Content []messageContent `json:"content"`

	// function_call and server-side tool calls.
	Name      *string         `json:"name"`
	Arguments *string         `json:"arguments"`
	Input     *string         `json:"input"`
	CallID    *string         `json:"call_id"`
	Action    json.RawMessage `json:"action"`

	// reasoning.
	Summary          []reasoningText `json:"summary"`
	EncryptedContent *string         `json:"encrypted_content"`

	// file_search_call.
	Queries []string           `json:"queries"`
	Results *[]fileSearchMatch `json:"results"`

	// image_generation_call.
	Prompt *string `json:"prompt"`
	Result *string `json:"result"`
}

// messageContent is a message content part, or a reasoning content part.
type messageContent struct {
	Type        string            `json:"type"`
	Text        string            `json:"text"`
	Annotations []annotation      `json:"annotations"`
	Logprobs    []json.RawMessage `json:"logprobs"`
}

type annotation struct {
	Type  string  `json:"type"`
	URL   *string `json:"url"`
	Title *string `json:"title"`
}

type reasoningText struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type fileSearchMatch struct {
	FileID   string  `json:"file_id"`
	Filename string  `json:"filename"`
	Score    float64 `json:"score"`
	Text     string  `json:"text"`
}

// streamChunk is one streamed event. The fields of each event type share
// their JSON names.
type streamChunk struct {
	Type        string             `json:"type"`
	Response    *responsesResponse `json:"response"`
	Item        *outputItem        `json:"item"`
	OutputIndex int                `json:"output_index"`
	ItemID      string             `json:"item_id"`
	Delta       string             `json:"delta"`
	Annotations []annotation       `json:"annotations"`
	Annotation  *annotation        `json:"annotation"`
	Code        *string            `json:"code"`
	Message     string             `json:"message"`
}

// webSearchWireAction is the action of a web search call.
type webSearchWireAction struct {
	Type    string            `json:"type"`
	Query   *string           `json:"query"`
	Queries []string          `json:"queries"`
	URL     *string           `json:"url"`
	Pattern *string           `json:"pattern"`
	Sources []json.RawMessage `json:"sources"`
}

// decodeJSONValue decodes raw into target and reports success.
func decodeJSONValue[T any](raw json.RawMessage, target *T) bool {
	if len(bytes.TrimSpace(raw)) == 0 {
		return false
	}

	return json.Unmarshal(raw, target) == nil
}

// encodeResult encodes a tool result value.
func encodeResult(value any) (json.RawMessage, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("encode tool result: %w", err)
	}

	return encoded, nil
}
