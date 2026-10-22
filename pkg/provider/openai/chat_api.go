package openai

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// chatRequest is the body of a Chat Completions request.
type chatRequest struct {
	Model string `json:"model"`

	LogitBias         map[int]float64 `json:"logit_bias,omitempty"`
	Logprobs          *bool           `json:"logprobs,omitempty"`
	TopLogprobs       *int            `json:"top_logprobs,omitempty"`
	User              string          `json:"user,omitempty"`
	ParallelToolCalls *bool           `json:"parallel_tool_calls,omitempty"`

	MaxTokens        *int                `json:"max_tokens,omitempty"`
	Temperature      *float64            `json:"temperature,omitempty"`
	TopP             *float64            `json:"top_p,omitempty"`
	FrequencyPenalty *float64            `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64            `json:"presence_penalty,omitempty"`
	ResponseFormat   *chatResponseFormat `json:"response_format,omitempty"`
	Stop             []string            `json:"stop,omitempty"`
	Seed             *int                `json:"seed,omitempty"`
	Verbosity        ChatTextVerbosity   `json:"verbosity,omitempty"`

	MaxCompletionTokens  *int                     `json:"max_completion_tokens,omitempty"`
	Store                *bool                    `json:"store,omitempty"`
	Metadata             map[string]string        `json:"metadata,omitempty"`
	Prediction           json.RawMessage          `json:"prediction,omitempty"`
	ReasoningEffort      model.ReasoningEffort    `json:"reasoning_effort,omitempty"`
	ServiceTier          ChatServiceTier          `json:"service_tier,omitempty"`
	PromptCacheKey       string                   `json:"prompt_cache_key,omitempty"`
	PromptCacheOptions   *ChatPromptCacheOptions  `json:"prompt_cache_options,omitempty"`
	PromptCacheRetention ChatPromptCacheRetention `json:"prompt_cache_retention,omitempty"`
	SafetyIdentifier     string                   `json:"safety_identifier,omitempty"`

	Messages   []chatMessage   `json:"messages"`
	Tools      []chatTool      `json:"tools,omitempty"`
	ToolChoice *chatToolChoice `json:"tool_choice,omitempty"`

	Stream        bool               `json:"stream,omitempty"`
	StreamOptions *chatStreamOptions `json:"stream_options,omitempty"`
}

// chatStreamOptions configures a streaming request.
type chatStreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// chatResponseFormatType is the type of a response_format value.
type chatResponseFormatType string

const (
	chatResponseFormatJSONObject chatResponseFormatType = "json_object"
	chatResponseFormatJSONSchema chatResponseFormatType = "json_schema"
)

// chatResponseFormat requests JSON output.
type chatResponseFormat struct {
	Type       chatResponseFormatType `json:"type"`
	JSONSchema *chatJSONSchemaFormat  `json:"json_schema,omitempty"`
}

// chatJSONSchemaFormat requests output matching a JSON Schema.
type chatJSONSchemaFormat struct {
	Schema      json.RawMessage `json:"schema"`
	Strict      bool            `json:"strict"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
}

// chatRole is the role of a Chat Completions message.
type chatRole string

const (
	chatRoleSystem    chatRole = "system"
	chatRoleDeveloper chatRole = "developer"
	chatRoleUser      chatRole = "user"
	chatRoleAssistant chatRole = "assistant"
	chatRoleTool      chatRole = "tool"
)

// chatMessage is one message of a Chat Completions prompt.
type chatMessage struct {
	Role       chatRole           `json:"role"`
	Content    chatMessageContent `json:"content"`
	ToolCalls  []chatToolCall     `json:"tool_calls,omitempty"`
	ToolCallID string             `json:"tool_call_id,omitempty"`
}

// chatMessageContent is a message's content: a string, an array of parts,
// or null when neither is set.
type chatMessageContent struct {
	text  *string
	parts []chatContentPart
}

// chatTextContent returns string content.
func chatTextContent(text string) chatMessageContent {
	return chatMessageContent{
		text: &text,
	}
}

// chatPartsContent returns array content.
func chatPartsContent(parts []chatContentPart) chatMessageContent {
	return chatMessageContent{
		parts: parts,
	}
}

// MarshalJSON encodes the parts, the text, or null.
func (c chatMessageContent) MarshalJSON() ([]byte, error) {
	switch {
	case c.parts != nil:
		return json.Marshal(c.parts)
	case c.text != nil:
		return json.Marshal(*c.text)
	default:
		return []byte("null"), nil
	}
}

// chatPartType is the type of a content part.
type chatPartType string

const (
	chatPartText       chatPartType = "text"
	chatPartImageURL   chatPartType = "image_url"
	chatPartInputAudio chatPartType = "input_audio"
	chatPartFile       chatPartType = "file"
)

// chatContentPart is one part of array content. Exactly one of the payload
// fields matching Type is set.
type chatContentPart struct {
	Type chatPartType `json:"type"`

	Text       *string         `json:"text,omitempty"`
	ImageURL   *chatImageURL   `json:"image_url,omitempty"`
	InputAudio *chatInputAudio `json:"input_audio,omitempty"`
	File       *chatFile       `json:"file,omitempty"`

	PromptCacheBreakpoint *ChatPromptCacheBreakpoint `json:"prompt_cache_breakpoint,omitempty"`
}

// chatImageURL is an image given by URL or data URL.
type chatImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// chatAudioFormat is the format of inline input audio.
type chatAudioFormat string

const (
	chatAudioWAV chatAudioFormat = "wav"
	chatAudioMP3 chatAudioFormat = "mp3"
)

// chatInputAudio is base64-encoded input audio.
type chatInputAudio struct {
	Data   string          `json:"data"`
	Format chatAudioFormat `json:"format"`
}

// chatFile is an inline file or an uploaded file ID.
type chatFile struct {
	Filename string `json:"filename,omitempty"`
	FileData string `json:"file_data,omitempty"`
	FileID   string `json:"file_id,omitempty"`
}

// chatToolCall is a function call in an assistant message.
type chatToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function chatFunctionCall `json:"function"`
}

// chatFunctionCall is a function name with JSON-encoded arguments.
type chatFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// chatTool is a function tool definition.
type chatTool struct {
	Type     string           `json:"type"`
	Function chatToolFunction `json:"function"`
}

// chatToolFunction describes a callable function.
type chatToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
	Strict      *bool           `json:"strict,omitempty"`
}

// chatToolChoice is "auto", "none", "required", or a specific function.
type chatToolChoice struct {
	mode         model.ToolChoiceType
	functionName string
}

// chatNamedToolChoice is the object form of a tool choice.
type chatNamedToolChoice struct {
	Type     string                  `json:"type"`
	Function chatNamedToolChoiceName `json:"function"`
}

// chatNamedToolChoiceName names the required function.
type chatNamedToolChoiceName struct {
	Name string `json:"name"`
}

// MarshalJSON encodes a mode string or a named function object.
func (c chatToolChoice) MarshalJSON() ([]byte, error) {
	if c.functionName == "" {
		return json.Marshal(c.mode)
	}

	return json.Marshal(chatNamedToolChoice{
		Type: "function",
		Function: chatNamedToolChoiceName{
			Name: c.functionName,
		},
	})
}

// chatResponse is a Chat Completions response.
type chatResponse struct {
	ID      string       `json:"id"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []chatChoice `json:"choices"`
	Usage   *chatUsage   `json:"usage"`
}

// chatChoice is one generated choice.
type chatChoice struct {
	Message      chatResponseMessage `json:"message"`
	Index        int                 `json:"index"`
	Logprobs     *chatLogprobs       `json:"logprobs"`
	FinishReason string              `json:"finish_reason"`
}

// chatResponseMessage is the assistant message of a choice.
type chatResponseMessage struct {
	Role        string                 `json:"role"`
	Content     string                 `json:"content"`
	Audio       *chatResponseAudio     `json:"audio"`
	ToolCalls   []chatResponseToolCall `json:"tool_calls"`
	Annotations []chatAnnotation       `json:"annotations"`
}

// chatResponseAudio is generated audio; only its transcript is used.
type chatResponseAudio struct {
	Transcript string `json:"transcript"`
}

// chatResponseToolCall is a generated function call.
type chatResponseToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function chatFunctionCall `json:"function"`
}

// chatAnnotation is a citation attached to generated text.
type chatAnnotation struct {
	Type        string          `json:"type"`
	URLCitation chatURLCitation `json:"url_citation"`
}

// chatURLCitation is a cited web page.
type chatURLCitation struct {
	StartIndex int    `json:"start_index"`
	EndIndex   int    `json:"end_index"`
	URL        string `json:"url"`
	Title      string `json:"title"`
}

// chatLogprobs holds the log probabilities of a choice.
type chatLogprobs struct {
	Content []ChatLogprob `json:"content"`
}

// chatUsage is the token usage of a Chat Completions call. It keeps the
// original JSON for model.Usage.Raw.
type chatUsage struct {
	PromptTokens            *int                     `json:"prompt_tokens"`
	CompletionTokens        *int                     `json:"completion_tokens"`
	TotalTokens             *int                     `json:"total_tokens"`
	PromptTokensDetails     *chatPromptTokensDetails `json:"prompt_tokens_details"`
	CompletionTokensDetails *chatOutputTokensDetails `json:"completion_tokens_details"`

	raw json.RawMessage
}

// chatPromptTokensDetails breaks down prompt tokens.
type chatPromptTokensDetails struct {
	CachedTokens     *int `json:"cached_tokens"`
	CacheWriteTokens *int `json:"cache_write_tokens"`
}

// chatOutputTokensDetails breaks down completion tokens.
type chatOutputTokensDetails struct {
	ReasoningTokens          *int `json:"reasoning_tokens"`
	AcceptedPredictionTokens *int `json:"accepted_prediction_tokens"`
	RejectedPredictionTokens *int `json:"rejected_prediction_tokens"`
}

// UnmarshalJSON decodes the usage and keeps a copy of the original JSON.
func (u *chatUsage) UnmarshalJSON(data []byte) error {
	type plain chatUsage

	var decoded plain
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}

	*u = chatUsage(decoded)
	u.raw = bytes.Clone(data)

	return nil
}

// chatChunk is one streamed Chat Completions event: a completion chunk or an
// error frame.
type chatChunk struct {
	ID      string            `json:"id"`
	Created int64             `json:"created"`
	Model   string            `json:"model"`
	Choices []chatChunkChoice `json:"choices"`
	Usage   *chatUsage        `json:"usage"`

	Error *chatErrorFrame `json:"error"`
}

// chatChunkChoice is the streamed part of one choice.
type chatChunkChoice struct {
	Delta        *chatDelta    `json:"delta"`
	Logprobs     *chatLogprobs `json:"logprobs"`
	FinishReason string        `json:"finish_reason"`
	Index        int           `json:"index"`
}

// chatDelta is the new content of a streamed choice.
type chatDelta struct {
	Role        string              `json:"role"`
	Content     *string             `json:"content"`
	ToolCalls   []chatToolCallDelta `json:"tool_calls"`
	Annotations []chatAnnotation    `json:"annotations"`
}

// chatToolCallDelta is a streamed fragment of a function call.
type chatToolCallDelta struct {
	Index    *int                   `json:"index"`
	ID       *string                `json:"id"`
	Type     *string                `json:"type"`
	Function *chatFunctionCallDelta `json:"function"`
}

// chatFunctionCallDelta is a streamed fragment of a function name and
// arguments.
type chatFunctionCallDelta struct {
	Name      *string `json:"name"`
	Arguments *string `json:"arguments"`
}

// isErrorFrame reports whether the chunk is an error frame rather than a
// completion chunk.
func (c chatChunk) isErrorFrame() bool {
	return c.Choices == nil && c.Error != nil
}

// hasOutput reports whether the chunk carries generated content.
func (c chatChunk) hasOutput() bool {
	for _, choice := range c.Choices {
		if choice.Delta.hasOutput() {
			return true
		}
	}

	return false
}

func (d *chatDelta) hasOutput() bool {
	if d == nil {
		return false
	}

	hasText := d.Content != nil && *d.Content != ""

	return hasText || len(d.ToolCalls) > 0 || len(d.Annotations) > 0
}

// validate checks the required fields that decoding cannot enforce.
func (r chatResponse) validate() error {
	for _, choice := range r.Choices {
		for _, toolCall := range choice.Message.ToolCalls {
			if toolCall.Type != "function" {
				return fmt.Errorf("%w: tool call type %q is not \"function\"", model.ErrInvalidResponse, toolCall.Type)
			}
		}

		if err := validateChatAnnotations(choice.Message.Annotations); err != nil {
			return err
		}
	}

	return nil
}

// validate checks the required fields of a completion chunk.
func (c chatChunk) validate() error {
	if c.Choices == nil {
		return fmt.Errorf("%w: chunk has no choices", model.ErrInvalidResponse)
	}

	for _, choice := range c.Choices {
		if choice.Delta == nil {
			continue
		}

		if err := validateChatToolCallDeltas(choice.Delta.ToolCalls); err != nil {
			return err
		}

		if err := validateChatAnnotations(choice.Delta.Annotations); err != nil {
			return err
		}
	}

	return nil
}

func validateChatToolCallDeltas(deltas []chatToolCallDelta) error {
	for _, delta := range deltas {
		if delta.Index == nil {
			return fmt.Errorf("%w: tool call delta has no index", model.ErrInvalidResponse)
		}

		if delta.Function == nil {
			return fmt.Errorf("%w: tool call delta has no function", model.ErrInvalidResponse)
		}

		if delta.Type != nil && *delta.Type != "function" {
			return fmt.Errorf("%w: tool call type %q is not \"function\"", model.ErrInvalidResponse, *delta.Type)
		}
	}

	return nil
}

func validateChatAnnotations(annotations []chatAnnotation) error {
	for _, annotation := range annotations {
		if annotation.Type != "url_citation" {
			return fmt.Errorf("%w: annotation type %q is not \"url_citation\"", model.ErrInvalidResponse, annotation.Type)
		}
	}

	return nil
}
