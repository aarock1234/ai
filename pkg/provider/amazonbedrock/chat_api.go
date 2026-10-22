package amazonbedrock

import (
	"encoding/json"

	"github.com/aarock1234/ai/internal/jsonobject"
)

// converseRequest is the body of a Converse call, without the provider
// options passed through and the tool configuration, which follow it.
type converseRequest struct {
	System   []contentBlock    `json:"system"`
	Messages []converseMessage `json:"messages"`

	AdditionalModelRequestFields      jsonobject.Object `json:"additionalModelRequestFields,omitempty"`
	AdditionalModelResponseFieldPaths []string          `json:"additionalModelResponseFieldPaths,omitempty"`

	InferenceConfig *inferenceConfig   `json:"inferenceConfig,omitempty"`
	ServiceTier     *serviceTierConfig `json:"serviceTier,omitempty"`
	RequestMetadata map[string]string  `json:"requestMetadata,omitempty"`
}

type inferenceConfig struct {
	MaxTokens     *int     `json:"maxTokens,omitempty"`
	Temperature   *float64 `json:"temperature,omitempty"`
	TopP          *float64 `json:"topP,omitempty"`
	TopK          *int     `json:"topK,omitempty"`
	StopSequences []string `json:"stopSequences,omitempty"`
}

func (c inferenceConfig) isEmpty() bool {
	return c.MaxTokens == nil && c.Temperature == nil && c.TopP == nil && c.TopK == nil && c.StopSequences == nil
}

type serviceTierConfig struct {
	Type ServiceTier `json:"type"`
}

type conversationRole string

const (
	roleUser      conversationRole = "user"
	roleAssistant conversationRole = "assistant"
)

type converseMessage struct {
	Role    conversationRole `json:"role"`
	Content []contentBlock   `json:"content"`
}

// contentBlock is one member of the Converse content block union; exactly
// one field is set.
type contentBlock struct {
	Text             *string                `json:"text,omitempty"`
	Image            *mediaBlock            `json:"image,omitempty"`
	Video            *mediaBlock            `json:"video,omitempty"`
	Document         *documentBlock         `json:"document,omitempty"`
	ToolUse          *toolUseBlock          `json:"toolUse,omitempty"`
	ToolResult       *toolResultBlock       `json:"toolResult,omitempty"`
	ReasoningContent *reasoningContentBlock `json:"reasoningContent,omitempty"`
	GuardContent     *guardContentBlock     `json:"guardContent,omitempty"`
	CachePoint       *CachePoint            `json:"cachePoint,omitempty"`
}

func textBlock(text string) contentBlock {
	return contentBlock{
		Text: &text,
	}
}

func cachePointBlock(point *CachePoint) contentBlock {
	return contentBlock{
		CachePoint: point,
	}
}

// mediaBlock is an image or video with inline bytes or an S3 location.
type mediaBlock struct {
	Format string      `json:"format"`
	Source mediaSource `json:"source"`
}

type mediaSource struct {
	// Bytes is base64.
	Bytes      string      `json:"bytes,omitempty"`
	S3Location *s3Location `json:"s3Location,omitempty"`
}

type s3Location struct {
	URI string `json:"uri"`
}

type documentBlock struct {
	Format    string             `json:"format"`
	Name      string             `json:"name"`
	Source    documentSource     `json:"source"`
	Citations *DocumentCitations `json:"citations,omitempty"`
}

type documentSource struct {
	// Bytes is base64.
	Bytes string `json:"bytes"`
}

type toolUseBlock struct {
	ToolUseID string          `json:"toolUseId"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
}

type toolResultBlock struct {
	ToolUseID string         `json:"toolUseId"`
	Content   []contentBlock `json:"content"`
}

type reasoningContentBlock struct {
	ReasoningText     *reasoningText     `json:"reasoningText,omitempty"`
	RedactedReasoning *redactedReasoning `json:"redactedReasoning,omitempty"`
	RedactedContent   *string            `json:"redactedContent,omitempty"`
}

type reasoningText struct {
	Text      string `json:"text"`
	Signature string `json:"signature,omitempty"`
}

type redactedReasoning struct {
	Data string `json:"data"`
}

type guardContentBlock struct {
	Text  *guardText  `json:"text,omitempty"`
	Image *mediaBlock `json:"image,omitempty"`
}

type guardText struct {
	Text       string                  `json:"text"`
	Qualifiers []GuardContentQualifier `json:"qualifiers,omitempty"`
}

// toolConfig is the Converse tool configuration.
type toolConfig struct {
	Tools      []toolEntry         `json:"tools,omitempty"`
	ToolChoice *converseToolChoice `json:"toolChoice,omitempty"`
}

// toolEntry is a tool specification or a cache point.
type toolEntry struct {
	ToolSpec   *toolSpec   `json:"toolSpec,omitempty"`
	CachePoint *CachePoint `json:"cachePoint,omitempty"`
}

type toolSpec struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Strict      *bool           `json:"strict,omitempty"`
	InputSchema toolInputSchema `json:"inputSchema"`
}

type toolInputSchema struct {
	JSON json.RawMessage `json:"json"`
}

// converseToolChoice is one of auto, any or a named tool.
type converseToolChoice struct {
	Auto *struct{}      `json:"auto,omitempty"`
	Any  *struct{}      `json:"any,omitempty"`
	Tool *namedToolName `json:"tool,omitempty"`
}

type namedToolName struct {
	Name string `json:"name"`
}

// converseResponse is the body of a Converse response, limited to the
// fields the model reads.
type converseResponse struct {
	Output                        *converseOutput           `json:"output"`
	StopReason                    *string                   `json:"stopReason"`
	AdditionalModelResponseFields *additionalResponseFields `json:"additionalModelResponseFields"`
	Trace                         json.RawMessage           `json:"trace"`
	PerformanceConfig             *PerformanceConfig        `json:"performanceConfig"`
	ServiceTier                   *ServiceTierInfo          `json:"serviceTier"`
	Usage                         json.RawMessage           `json:"usage"`
}

type converseOutput struct {
	Message *converseOutputMessage `json:"message"`
}

type converseOutputMessage struct {
	Content []responseBlock `json:"content"`
	Role    string          `json:"role"`
}

type responseBlock struct {
	Text             *string                   `json:"text"`
	CitationsContent *citationsContent         `json:"citationsContent"`
	ToolUse          *responseToolUse          `json:"toolUse"`
	ReasoningContent *responseReasoningContent `json:"reasoningContent"`
}

type citationsContent struct {
	Content []citationText `json:"content"`
}

type citationText struct {
	Text *string `json:"text"`
}

type responseToolUse struct {
	ToolUseID string          `json:"toolUseId"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
}

type responseReasoningContent struct {
	ReasoningText     *responseReasoningText `json:"reasoningText"`
	RedactedReasoning *redactedReasoning     `json:"redactedReasoning"`
	RedactedContent   *string                `json:"redactedContent"`
}

type responseReasoningText struct {
	Signature *string `json:"signature"`
	Text      string  `json:"text"`
}

// additionalResponseFields are the requested additional response fields.
type additionalResponseFields struct {
	Delta *responseFieldDelta `json:"delta"`
}

type responseFieldDelta struct {
	StopSequence *string `json:"stop_sequence"`
}

func (f *additionalResponseFields) stopSequence() *string {
	if f == nil || f.Delta == nil {
		return nil
	}

	return f.Delta.StopSequence
}

// converseUsage is the token usage of a call.
type converseUsage struct {
	InputTokens           *int          `json:"inputTokens"`
	OutputTokens          *int          `json:"outputTokens"`
	TotalTokens           *int          `json:"totalTokens"`
	CacheReadInputTokens  *int          `json:"cacheReadInputTokens"`
	CacheWriteInputTokens *int          `json:"cacheWriteInputTokens"`
	CacheDetails          []CacheDetail `json:"cacheDetails"`
}

// Stream event types.
const (
	eventContentBlockStart = "contentBlockStart"
	eventContentBlockDelta = "contentBlockDelta"
	eventContentBlockStop  = "contentBlockStop"
	eventMessageStop       = "messageStop"
	eventMetadata          = "metadata"
)

// streamExceptionTypes are the in-band exception event types.
var streamExceptionTypes = map[string]bool{
	"internalServerException":     true,
	"modelStreamErrorException":   true,
	"serviceUnavailableException": true,
	"throttlingException":         true,
	"validationException":         true,
}

type contentBlockStartEvent struct {
	ContentBlockIndex *int               `json:"contentBlockIndex"`
	Start             *contentBlockStart `json:"start"`
}

type contentBlockStart struct {
	ToolUse *responseToolUse `json:"toolUse"`
}

type contentBlockDeltaEvent struct {
	ContentBlockIndex *int         `json:"contentBlockIndex"`
	Delta             *streamDelta `json:"delta"`
}

// streamDelta is one member of the delta union; the first present field
// selects it.
type streamDelta struct {
	Text             *string                `json:"text"`
	Citation         json.RawMessage        `json:"citation"`
	ToolUse          *toolInputDelta        `json:"toolUse"`
	ReasoningContent *reasoningContentDelta `json:"reasoningContent"`
}

type toolInputDelta struct {
	Input *string `json:"input"`
}

// reasoningContentDelta is one member of the reasoning delta union; the
// first present field selects it.
type reasoningContentDelta struct {
	Text            *string `json:"text"`
	Signature       *string `json:"signature"`
	Data            *string `json:"data"`
	RedactedContent *string `json:"redactedContent"`
}

type contentBlockStopEvent struct {
	ContentBlockIndex *int `json:"contentBlockIndex"`
}

type messageStopEvent struct {
	AdditionalModelResponseFields *additionalResponseFields `json:"additionalModelResponseFields"`
	StopReason                    *string                   `json:"stopReason"`
}

type metadataEvent struct {
	Trace             json.RawMessage    `json:"trace"`
	PerformanceConfig *PerformanceConfig `json:"performanceConfig"`
	ServiceTier       *ServiceTierInfo   `json:"serviceTier"`
	Usage             json.RawMessage    `json:"usage"`
}
