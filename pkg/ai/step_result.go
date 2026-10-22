package ai

import (
	"encoding/json"
	"strings"
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

// StepResult is the outcome of one step of a generation: one language model
// call and the execution of the tools it called.
type StepResult struct {
	// CallID identifies the generation the step belongs to.
	CallID string

	// StepNumber is the zero-based index of the step.
	StepNumber int

	// Model identifies the model that produced the step.
	Model StepModel

	// Content is the generated content in order, followed by tool outputs
	// and approvals.
	Content []ContentPart

	FinishReason    model.UnifiedFinishReason
	RawFinishReason string

	Usage       LanguageModelUsage
	Performance StepPerformance

	// Warnings are the provider's warnings for the call.
	Warnings []model.Warning

	Request  StepRequest
	Response StepResponse

	ProviderMetadata model.ProviderMetadata
}

// StepModel identifies a language model.
type StepModel struct {
	Provider string
	ModelID  string
}

// StepRequest describes the request of a step. Body and Messages are only
// retained when the generation's Include settings ask for them.
type StepRequest struct {
	Body     json.RawMessage
	Messages []model.Message
}

// retainedStepRequest describes a step request and the payloads to retain.
type retainedStepRequest struct {
	body     json.RawMessage
	messages []model.Message

	includeBody     bool
	includeMessages bool
}

// newStepRequest retains the request payloads that the Include settings
// select.
func newStepRequest(request retainedStepRequest) StepRequest {
	var stepRequest StepRequest

	if request.includeBody {
		stepRequest.Body = request.body
	}

	if request.includeMessages {
		stepRequest.Messages = request.messages
	}

	return stepRequest
}

// StepResponse describes the response of a step.
type StepResponse struct {
	model.ResponseInfo

	// Messages are the assistant and tool messages generated in the step.
	// Append them to the conversation to continue it.
	Messages []model.Message
}

// StepPerformance measures the timing of a step: the performance of its
// model call, the total step time and the time of each tool execution.
// Durations replace the upstream millisecond values.
type StepPerformance struct {
	LanguageModelCallPerformance

	// StepTime is the total time of the step, including tool execution.
	StepTime time.Duration

	// ToolExecution is the execution time of each client-side tool call,
	// keyed by tool call ID.
	ToolExecution map[string]time.Duration
}

// OutputChunkTimingStats summarizes the gaps between streamed output chunks.
type OutputChunkTimingStats struct {
	Min    time.Duration
	P10    time.Duration
	Median time.Duration
	Avg    time.Duration
	P90    time.Duration
	Max    time.Duration
}

// Text returns the concatenated text of the step, or the empty string when
// it has no text.
func (s StepResult) Text() string {
	var text strings.Builder

	for _, part := range s.Content {
		if typed, ok := part.(TextContent); ok {
			text.WriteString(typed.Text)
		}
	}

	return text.String()
}

// Reasoning returns the ReasoningContent and ReasoningFileContent parts of
// the step.
func (s StepResult) Reasoning() []ContentPart {
	var reasoning []ContentPart

	for _, part := range s.Content {
		switch part.(type) {
		case ReasoningContent, ReasoningFileContent:
			reasoning = append(reasoning, part)
		}
	}

	return reasoning
}

// ReasoningText returns the concatenated reasoning text, excluding reasoning
// files. It reports false when the step has no reasoning text.
func (s StepResult) ReasoningText() (string, bool) {
	var text strings.Builder

	for _, part := range s.Content {
		if typed, ok := part.(ReasoningContent); ok {
			text.WriteString(typed.Text)
		}
	}

	return text.String(), text.Len() > 0
}

// Files returns the generated files of the step.
func (s StepResult) Files() []GeneratedFile {
	return collectContent(s.Content, func(part FileContent) (GeneratedFile, bool) {
		return part.File, true
	})
}

// Sources returns the sources of the step.
func (s StepResult) Sources() []SourceContent {
	return collectContent(s.Content, func(part SourceContent) (SourceContent, bool) {
		return part, true
	})
}

// ToolCalls returns the tool calls of the step.
func (s StepResult) ToolCalls() []ToolCall {
	return collectContent(s.Content, func(part ToolCall) (ToolCall, bool) {
		return part, true
	})
}

// StaticToolCalls returns the tool calls of tools declared with known types.
func (s StepResult) StaticToolCalls() []ToolCall {
	return collectContent(s.Content, func(part ToolCall) (ToolCall, bool) {
		return part, !part.Dynamic
	})
}

// DynamicToolCalls returns the calls of dynamic tools and invalid calls.
func (s StepResult) DynamicToolCalls() []ToolCall {
	return collectContent(s.Content, func(part ToolCall) (ToolCall, bool) {
		return part, part.Dynamic
	})
}

// ToolResults returns the successful tool results of the step.
func (s StepResult) ToolResults() []ToolResult {
	return collectContent(s.Content, func(part ToolResult) (ToolResult, bool) {
		return part, true
	})
}

// StaticToolResults returns the results of tools declared with known types.
func (s StepResult) StaticToolResults() []ToolResult {
	return collectContent(s.Content, func(part ToolResult) (ToolResult, bool) {
		return part, !part.Dynamic
	})
}

// DynamicToolResults returns the results of dynamic tools.
func (s StepResult) DynamicToolResults() []ToolResult {
	return collectContent(s.Content, func(part ToolResult) (ToolResult, bool) {
		return part, part.Dynamic
	})
}

// collectContent returns the parts of type Part that select accepts, in
// order.
func collectContent[Part ContentPart, Value any](content []ContentPart, selectPart func(part Part) (Value, bool)) []Value {
	var values []Value

	for _, part := range content {
		typed, ok := part.(Part)
		if !ok {
			continue
		}

		if value, selected := selectPart(typed); selected {
			values = append(values, value)
		}
	}

	return values
}
