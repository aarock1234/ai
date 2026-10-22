package openaicompatible

import (
	"encoding/json"
)

// chatRequest holds the request fields that provider passthrough options may
// replace.
type chatRequest struct {
	Model string  `json:"model"`
	User  *string `json:"user,omitempty"`

	MaxTokens        *int                `json:"max_tokens,omitempty"`
	Temperature      *float64            `json:"temperature,omitempty"`
	TopP             *float64            `json:"top_p,omitempty"`
	FrequencyPenalty *float64            `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64            `json:"presence_penalty,omitempty"`
	ResponseFormat   *chatResponseFormat `json:"response_format,omitempty"`
	Stop             []string            `json:"stop,omitempty"`
	Seed             *int                `json:"seed,omitempty"`
}

// chatRequestTail holds the request fields that always take precedence over
// provider passthrough options, even when they are absent.
type chatRequestTail struct {
	ReasoningEffort *string         `json:"reasoning_effort,omitempty"`
	Verbosity       *string         `json:"verbosity,omitempty"`
	Messages        []chatMessage   `json:"messages"`
	Tools           []chatTool      `json:"tools,omitempty"`
	ToolChoice      *chatToolChoice `json:"tool_choice,omitempty"`
}

// chatRequestTailKeys are the JSON names of chatRequestTail.
var chatRequestTailKeys = []string{
	"reasoning_effort",
	"verbosity",
	"messages",
	"tools",
	"tool_choice",
}

// chatResponseFormatType is the wire response format type.
type chatResponseFormatType string

const (
	chatResponseFormatJSONSchema chatResponseFormatType = "json_schema"
	chatResponseFormatJSONObject chatResponseFormatType = "json_object"
)

type chatResponseFormat struct {
	Type       chatResponseFormatType `json:"type"`
	JSONSchema *chatJSONSchema        `json:"json_schema,omitempty"`
}

type chatJSONSchema struct {
	Schema      json.RawMessage `json:"schema"`
	Strict      bool            `json:"strict"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
}

// chatMessage is a systemMessage, userMessage, assistantMessage or
// toolMessage.
type chatMessage interface {
	isChatMessage()
}

// messageExtra holds caller-supplied members that are added to a message or
// part, from its "openaiCompatible" provider options.
type messageExtra = map[string]json.RawMessage

type systemMessage struct {
	Role    string       `json:"role"`
	Content string       `json:"content"`
	Extra   messageExtra `json:"-"`
}

type userMessage struct {
	Role    string         `json:"role"`
	Content messageContent `json:"content"`
	Extra   messageExtra   `json:"-"`
}

type assistantMessage struct {
	Role string `json:"role"`

	// Content is null when the message has tool calls and no text.
	Content *string `json:"content"`

	ReasoningContent string            `json:"reasoning_content,omitempty"`
	ToolCalls        []messageToolCall `json:"tool_calls,omitempty"`
	Extra            messageExtra      `json:"-"`
}

type toolMessage struct {
	Role       string         `json:"role"`
	ToolCallID string         `json:"tool_call_id"`
	Content    messageContent `json:"content"`
	Extra      messageExtra   `json:"-"`
}

func (systemMessage) isChatMessage()    {}
func (userMessage) isChatMessage()      {}
func (assistantMessage) isChatMessage() {}
func (toolMessage) isChatMessage()      {}

// MarshalJSON encodes the message with its extra members.
func (m systemMessage) MarshalJSON() ([]byte, error) {
	type plain systemMessage

	return encodeExtended(plain(m), m.Extra)
}

// MarshalJSON encodes the message with its extra members.
func (m userMessage) MarshalJSON() ([]byte, error) {
	type plain userMessage

	return encodeExtended(plain(m), m.Extra)
}

// MarshalJSON encodes the message with its extra members.
func (m assistantMessage) MarshalJSON() ([]byte, error) {
	type plain assistantMessage

	return encodeExtended(plain(m), m.Extra)
}

// MarshalJSON encodes the message with its extra members.
func (m toolMessage) MarshalJSON() ([]byte, error) {
	type plain toolMessage

	return encodeExtended(plain(m), m.Extra)
}

// messageContent is either plain text or a list of content parts.
type messageContent struct {
	Text  string
	Parts []contentPart
}

// MarshalJSON encodes the parts when there are any and the text otherwise.
func (c messageContent) MarshalJSON() ([]byte, error) {
	if c.Parts != nil {
		return json.Marshal(c.Parts)
	}

	return json.Marshal(c.Text)
}

// contentPartType is the wire type of a content part.
type contentPartType string

const (
	contentPartText       contentPartType = "text"
	contentPartImageURL   contentPartType = "image_url"
	contentPartVideoURL   contentPartType = "video_url"
	contentPartInputAudio contentPartType = "input_audio"
	contentPartFile       contentPartType = "file"
)

// contentPart is one part of user or tool content. The field matching Type is
// set.
type contentPart struct {
	Type contentPartType `json:"type"`

	Text       *string         `json:"text,omitempty"`
	ImageURL   *mediaURL       `json:"image_url,omitempty"`
	VideoURL   *mediaURL       `json:"video_url,omitempty"`
	InputAudio *inputAudio     `json:"input_audio,omitempty"`
	File       *fileAttachment `json:"file,omitempty"`

	Extra messageExtra `json:"-"`
}

// MarshalJSON encodes the part with its extra members.
func (p contentPart) MarshalJSON() ([]byte, error) {
	type plain contentPart

	return encodeExtended(plain(p), p.Extra)
}

type mediaURL struct {
	URL string `json:"url"`
}

// audioFormat is an input audio format.
type audioFormat string

const (
	audioFormatWAV audioFormat = "wav"
	audioFormatMP3 audioFormat = "mp3"
)

type inputAudio struct {
	Data   string      `json:"data"`
	Format audioFormat `json:"format"`
}

// fileAttachment is either inline file data or an uploaded file ID.
type fileAttachment struct {
	Filename string `json:"filename,omitempty"`
	FileData string `json:"file_data,omitempty"`
	FileID   string `json:"file_id,omitempty"`
}

type messageToolCall struct {
	ID           string        `json:"id"`
	Type         string        `json:"type"`
	Function     functionCall  `json:"function"`
	ExtraContent *extraContent `json:"extra_content,omitempty"`
	Extra        messageExtra  `json:"-"`
}

// MarshalJSON encodes the tool call with its extra members.
func (c messageToolCall) MarshalJSON() ([]byte, error) {
	type plain messageToolCall

	return encodeExtended(plain(c), c.Extra)
}

type functionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// extraContent carries provider-specific tool call data. Google Gemini uses
// it for thought signatures through its OpenAI compatibility layer.
type extraContent struct {
	Google *googleExtraContent `json:"google,omitempty"`
}

type googleExtraContent struct {
	ThoughtSignature *string `json:"thought_signature,omitempty"`
}

// thoughtSignature returns the Google thought signature, if any.
func (c *extraContent) thoughtSignature() string {
	if c == nil || c.Google == nil || c.Google.ThoughtSignature == nil {
		return ""
	}

	return *c.Google.ThoughtSignature
}

type chatTool struct {
	Type     string             `json:"type"`
	Function chatToolDefinition `json:"function"`
}

type chatToolDefinition struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
	Strict      *bool           `json:"strict,omitempty"`
}

// chatToolChoice is a mode such as "auto" or a specific function.
type chatToolChoice struct {
	Mode         string
	FunctionName string
}

// MarshalJSON encodes a mode as a string and a function as an object.
func (c chatToolChoice) MarshalJSON() ([]byte, error) {
	if c.FunctionName == "" {
		return json.Marshal(c.Mode)
	}

	return json.Marshal(namedToolChoice{
		Type: "function",
		Function: namedFunction{
			Name: c.FunctionName,
		},
	})
}

type namedToolChoice struct {
	Type     string        `json:"type"`
	Function namedFunction `json:"function"`
}

type namedFunction struct {
	Name string `json:"name"`
}
