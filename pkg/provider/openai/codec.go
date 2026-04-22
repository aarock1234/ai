// Package openai provides an OpenAI language model provider and shared
// codec for OpenAI-compatible chat completion APIs. Providers that use
// the OpenAI wire format (OpenAI, OpenRouter, Groq, Together, DeepSeek,
// etc.) import this package for request encoding, response decoding,
// and streaming.
package openai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// ChatRequest is the OpenAI chat completions request body. Fields are
// shared across the OpenAI-compatible ecosystem; provider-specific
// additions flow through ProviderOptions on model.Request.
type ChatRequest struct {
	Model            string            `json:"model"`
	Messages         []chatMessage     `json:"messages"`
	Temperature      *float64          `json:"temperature,omitempty"`
	MaxTokens        *int              `json:"max_completion_tokens,omitempty"`
	TopP             *float64          `json:"top_p,omitempty"`
	PresencePenalty  *float64          `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64          `json:"frequency_penalty,omitempty"`
	Seed             *int64            `json:"seed,omitempty"`
	Stop             []string          `json:"stop,omitempty"`
	Tools            []chatTool        `json:"tools,omitempty"`
	ToolChoice       *toolChoiceValue  `json:"tool_choice,omitempty"`
	ResponseFormat   *responseFormat   `json:"response_format,omitempty"`
	Metadata         map[string]string `json:"metadata,omitempty"`
	Modalities       []string          `json:"modalities,omitempty"`
	Audio            *audioConfig      `json:"audio,omitempty"`
	Reasoning        *reasoningConfig  `json:"reasoning,omitempty"`
	Prediction       *predictionConfig `json:"prediction,omitempty"`
	Stream           bool              `json:"stream,omitempty"`
	StreamOptions    *streamOptions    `json:"stream_options,omitempty"`
	Headers          http.Header       `json:"-"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type audioConfig struct {
	Voice  string `json:"voice,omitempty"`
	Format string `json:"format,omitempty"`
}

type reasoningConfig struct {
	Effort    string `json:"effort,omitempty"`
	MaxTokens int    `json:"max_tokens,omitempty"`
	Exclude   bool   `json:"exclude,omitempty"`
}

type predictionConfig struct {
	Type    string `json:"type"`
	Content string `json:"content"`
}

// chatMessage is a single message in the OpenAI wire format. The shape
// covers both request and response flows; unused fields are omitted via
// struct tags.
type chatMessage struct {
	Role             string              `json:"role"`
	Content          messageContent      `json:"content,omitempty"`
	ToolCalls        []wireToolCall      `json:"tool_calls,omitempty"`
	ToolCallID       string              `json:"tool_call_id,omitempty"`
	Name             string              `json:"name,omitempty"`
	Reasoning        string              `json:"reasoning,omitempty"`
	ReasoningContent string              `json:"reasoning_content,omitempty"`
	ReasoningDetails []wireReasoningPart `json:"reasoning_details,omitempty"`
	Annotations      []wireAnnotation    `json:"annotations,omitempty"`
	Images           []wireMessageImage  `json:"images,omitempty"`
	Audio            *wireMessageAudio   `json:"audio,omitempty"`
	Refusal          string              `json:"refusal,omitempty"`
}

// messageContent is the OpenAI chat message content union. On the wire
// it is either a plain string, an array of typed content parts, or
// absent. Callers populate exactly one of Text or Parts; MarshalJSON
// picks the right shape.
type messageContent struct {
	Text  string
	Parts []contentPart
}

// MarshalJSON emits the string form when only Text is set, the parts
// array when Parts is set, and null otherwise.
func (c messageContent) MarshalJSON() ([]byte, error) {
	switch {
	case len(c.Parts) > 0:
		return json.Marshal(c.Parts)
	case c.Text != "":
		return json.Marshal(c.Text)
	default:
		return []byte("null"), nil
	}
}

// UnmarshalJSON accepts both wire shapes: a JSON string or an array
// of typed parts. Other shapes (including null) leave the zero value.
func (c *messageContent) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		*c = messageContent{}

		return nil
	}

	if trimmed[0] == '"' {
		var s string
		if err := json.Unmarshal(trimmed, &s); err != nil {
			return fmt.Errorf("decoding message content string: %w", err)
		}
		*c = messageContent{Text: s}

		return nil
	}

	if trimmed[0] == '[' {
		var parts []contentPart
		if err := json.Unmarshal(trimmed, &parts); err != nil {
			return fmt.Errorf("decoding message content parts: %w", err)
		}
		*c = messageContent{Parts: parts}

		return nil
	}

	return fmt.Errorf("unexpected message content shape: %s", trimmed)
}

// TextValue returns the concatenated text across all plain-text parts,
// or the Text field when the content is a plain string.
func (c messageContent) TextValue() string {
	if len(c.Parts) == 0 {
		return c.Text
	}

	var b strings.Builder
	for _, part := range c.Parts {
		if part.Type == "text" && part.Text != "" {
			b.WriteString(part.Text)
		}
	}

	return b.String()
}

// contentPart is a multi-part content segment for a user message. The
// OpenAI API accepts typed parts: text, image_url, input_audio, file.
type contentPart struct {
	Type       string            `json:"type"`
	Text       string            `json:"text,omitempty"`
	ImageURL   *contentPartURL   `json:"image_url,omitempty"`
	InputAudio *contentPartAudio `json:"input_audio,omitempty"`
	File       *contentPartFile  `json:"file,omitempty"`
}

type contentPartURL struct {
	URL string `json:"url"`
}

type contentPartAudio struct {
	Data   string `json:"data"`
	Format string `json:"format,omitempty"`
}

type contentPartFile struct {
	FileData string `json:"file_data,omitempty"`
	FileID   string `json:"file_id,omitempty"`
	Filename string `json:"filename,omitempty"`
}

// wireMessageImage mirrors OpenRouter's `message.images[]` output: the
// generated image is returned as a base64 data URL.
type wireMessageImage struct {
	Type     string          `json:"type"`
	ImageURL *contentPartURL `json:"image_url,omitempty"`
}

// wireMessageAudio mirrors OpenAI's `message.audio` output for gpt-audio
// models: base64-encoded audio bytes plus a transcript.
type wireMessageAudio struct {
	ID         string `json:"id,omitempty"`
	Data       string `json:"data,omitempty"`
	Transcript string `json:"transcript,omitempty"`
	ExpiresAt  int64  `json:"expires_at,omitempty"`
	Format     string `json:"format,omitempty"`
}

// wireReasoningPart mirrors OpenRouter's `reasoning_details[]` array
// with the three reasoning block variants: text, summary, encrypted.
type wireReasoningPart struct {
	Type      string `json:"type,omitempty"`
	Text      string `json:"text,omitempty"`
	Summary   string `json:"summary,omitempty"`
	Data      string `json:"data,omitempty"`
	Signature string `json:"signature,omitempty"`
	ID        string `json:"id,omitempty"`
	Format    string `json:"format,omitempty"`
	Index     int    `json:"index,omitempty"`
}

// wireAnnotation is an OpenAI message annotation. Currently only
// url_citation is emitted (web search citations).
type wireAnnotation struct {
	Type         string            `json:"type"`
	URLCitation  *wireURLCitation  `json:"url_citation,omitempty"`
	FileCitation *wireFileCitation `json:"file_citation,omitempty"`
}

type wireURLCitation struct {
	URL        string `json:"url"`
	Title      string `json:"title,omitempty"`
	StartIndex int    `json:"start_index,omitempty"`
	EndIndex   int    `json:"end_index,omitempty"`
}

type wireFileCitation struct {
	FileID   string `json:"file_id,omitempty"`
	Filename string `json:"filename,omitempty"`
	Index    int    `json:"index,omitempty"`
	Title    string `json:"title,omitempty"`
}

// wireToolCall is a tool call in the OpenAI wire format. In streaming
// responses, Index identifies which tool call a delta belongs to; ID and
// Function.Name appear only on the first delta for each tool call.
type wireToolCall struct {
	Index    int          `json:"index"`
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function wireFunction `json:"function"`
}

// wireFunction describes the function invocation in a tool call.
type wireFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// chatTool describes a tool available to the model.
type chatTool struct {
	Type     string       `json:"type"`
	Function toolFunction `json:"function"`
}

// toolFunction describes a function tool's schema.
type toolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
	Strict      bool            `json:"strict,omitempty"`
}

// responseFormat requests a specific output structure.
type responseFormat struct {
	Type       string          `json:"type"`
	JSONSchema *jsonSchemaSpec `json:"json_schema,omitempty"`
}

// jsonSchemaSpec is the schema specification for structured output.
type jsonSchemaSpec struct {
	Name   string          `json:"name"`
	Schema json.RawMessage `json:"schema"`
	Strict bool            `json:"strict"`
}

// ChatResponse is the OpenAI chat completions response.
type ChatResponse struct {
	ID       string            `json:"id"`
	Choices  []chatChoice      `json:"choices"`
	Usage    wireUsage         `json:"usage"`
	Model    string            `json:"model"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

// chatChoice is a single completion choice.
type chatChoice struct {
	Index        int         `json:"index"`
	Message      chatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

// wireUsage tracks token consumption, including the detail breakdowns
// that OpenAI and OpenRouter emit for reasoning/audio/image and cached
// tokens.
type wireUsage struct {
	PromptTokens            int                     `json:"prompt_tokens"`
	CompletionTokens        int                     `json:"completion_tokens"`
	TotalTokens             int                     `json:"total_tokens,omitempty"`
	PromptTokensDetails     *wirePromptTokenDetails `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *wireCompletionDetails  `json:"completion_tokens_details,omitempty"`
}

type wirePromptTokenDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
	AudioTokens  int `json:"audio_tokens,omitempty"`
	ImageTokens  int `json:"image_tokens,omitempty"`
}

type wireCompletionDetails struct {
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
	AudioTokens     int `json:"audio_tokens,omitempty"`
	ImageTokens     int `json:"image_tokens,omitempty"`
}

// chatStreamChunk is a single chunk in a streaming response.
type chatStreamChunk struct {
	ID       string              `json:"id"`
	Choices  []streamChunkChoice `json:"choices"`
	Usage    *wireUsage          `json:"usage,omitempty"`
	Model    string              `json:"model,omitempty"`
	Metadata map[string]string   `json:"metadata,omitempty"`
}

// streamChunkChoice is a single choice delta in a streaming chunk.
type streamChunkChoice struct {
	Index        int       `json:"index"`
	Delta        chatDelta `json:"delta"`
	FinishReason *string   `json:"finish_reason"`
}

// chatDelta is the incremental content in a streaming chunk.
type chatDelta struct {
	Role             string              `json:"role,omitempty"`
	Content          *string             `json:"content,omitempty"`
	ToolCalls        []wireToolCall      `json:"tool_calls,omitempty"`
	Reasoning        string              `json:"reasoning,omitempty"`
	ReasoningContent string              `json:"reasoning_content,omitempty"`
	ReasoningDetails []wireReasoningPart `json:"reasoning_details,omitempty"`
	Annotations      []wireAnnotation    `json:"annotations,omitempty"`
	Images           []wireMessageImage  `json:"images,omitempty"`
	Audio            *wireMessageAudio   `json:"audio,omitempty"`
	Refusal          string              `json:"refusal,omitempty"`
}

// toolChoiceValue is the OpenAI `tool_choice` union. On the wire it is
// either a string ("auto" / "none" / "required") or an object naming a
// specific tool. Callers populate exactly one of Mode or Named.
type toolChoiceValue struct {
	Mode  string
	Named *toolChoiceSpec
}

// MarshalJSON emits the string form when Mode is set, the object form
// when Named is set. An empty value marshals to null.
func (v toolChoiceValue) MarshalJSON() ([]byte, error) {
	switch {
	case v.Named != nil:
		return json.Marshal(v.Named)
	case v.Mode != "":
		return json.Marshal(v.Mode)
	default:
		return []byte("null"), nil
	}
}

// toolChoiceSpec is the object form of the tool_choice union, naming
// the specific function to invoke.
type toolChoiceSpec struct {
	Type     string                 `json:"type"`
	Function toolChoiceSpecFunction `json:"function"`
}

// toolChoiceSpecFunction names the specific tool to call.
type toolChoiceSpecFunction struct {
	Name string `json:"name"`
}
