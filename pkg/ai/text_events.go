package ai

import (
	"github.com/aarock1234/ai/pkg/model"
)

// GenerateTextStartEvent describes a GenerateText or StreamText call before
// any model call.
type GenerateTextStartEvent struct {
	CallID string

	// OperationID is "ai.generateText" or "ai.streamText".
	OperationID string

	Provider string
	ModelID  string

	System   string
	Messages []model.Message

	Tools       []Tool
	ToolChoice  *model.ToolChoice
	ActiveTools []string
	ToolOrder   []string

	CallSettings    CallSettings
	ProviderOptions model.ProviderOptions
	Output          Output
}

// GenerateTextStepStartEvent describes a step before its model call.
type GenerateTextStepStartEvent struct {
	CallID     string
	Provider   string
	ModelID    string
	StepNumber int

	System   string
	Messages []model.Message

	Tools       []Tool
	ToolChoice  *model.ToolChoice
	ActiveTools []string
	ToolOrder   []string

	// Steps are the completed steps.
	Steps []StepResult

	ProviderOptions model.ProviderOptions
	Output          Output

	// PromptMessages is the converted prompt sent to the model.
	PromptMessages []model.Message

	// StepTools and StepToolChoice are the tool declarations and the tool
	// choice sent to the model.
	StepTools      []model.Tool
	StepToolChoice *model.ToolChoice
}

// GenerateTextEndEvent describes a completed generation. Output is nil
// when it could not be parsed.
type GenerateTextEndEvent struct {
	CallID string

	GenerateTextResult
}
