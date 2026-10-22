package anthropic

import (
	"encoding/json"
	"fmt"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// ModelCapabilities are the features of a Claude model that platforms
// serving Claude through their own APIs, such as the Amazon Bedrock Converse
// API, adapt requests to.
type ModelCapabilities struct {
	// MaxOutputTokens is the model's output token limit.
	MaxOutputTokens int

	// SupportsStructuredOutput means the model accepts native structured
	// output and strict tools.
	SupportsStructuredOutput bool

	// SupportsAdaptiveThinking means the model sizes its thinking by
	// effort rather than by a token budget.
	SupportsAdaptiveThinking bool

	// RejectsSamplingParameters means temperature, topP and topK are
	// rejected.
	RejectsSamplingParameters bool

	// RejectsForcedToolUse means a required or named tool choice is
	// rejected.
	RejectsForcedToolUse bool

	// IsKnownModel means the model ID matched a known Claude model.
	IsKnownModel bool
}

// CapabilitiesOf returns the capabilities of a Claude model ID. IDs with a
// platform prefix, such as "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
// are matched by the Claude model name they contain.
func CapabilitiesOf(modelID string) ModelCapabilities {
	capabilities := capabilitiesOf(modelID)

	return ModelCapabilities{
		MaxOutputTokens:           capabilities.maxOutputTokens,
		SupportsStructuredOutput:  capabilities.supportsStructuredOutput,
		SupportsAdaptiveThinking:  capabilities.supportsAdaptiveThinking,
		RejectsSamplingParameters: capabilities.rejectsSamplingParameters,
		RejectsForcedToolUse:      capabilities.rejectsForcedToolUse,
		IsKnownModel:              capabilities.isKnownModel,
	}
}

// SanitizeJSONSchema removes the JSON Schema keywords that Claude's
// constrained decoder rejects, moving value constraints into descriptions.
func SanitizeJSONSchema(schema json.RawMessage) json.RawMessage {
	return sanitizeJSONSchema(schema)
}

// ProviderToolsInput are the Anthropic provider-defined tools of a call and
// its tool choice.
type ProviderToolsInput struct {
	Tools      []model.ProviderTool
	ToolChoice *model.ToolChoice

	// DisableParallelToolUse limits the model to one tool call per turn.
	DisableParallelToolUse *bool

	// RejectsForcedToolUse downgrades required and named tool choices to
	// auto with a warning.
	RejectsForcedToolUse bool
}

// PreparedProviderTools are the Messages API settings that provider-defined
// tools need.
type PreparedProviderTools struct {
	// ToolChoice is the Messages API tool_choice object, or nil when the
	// call has no tool choice.
	ToolChoice json.RawMessage

	// Betas are the beta features the tools require.
	Betas []string

	Warnings []model.Warning
}

// PrepareProviderTools converts the tool choice of Anthropic
// provider-defined tools and collects the betas they require, for
// platforms that accept Claude's tool_choice but declare tools themselves.
func PrepareProviderTools(input ProviderToolsInput) (PreparedProviderTools, error) {
	tools := make([]model.Tool, 0, len(input.Tools))
	for _, tool := range input.Tools {
		tools = append(tools, tool)
	}

	prepared, err := prepareTools(toolPreparation{
		tools:                  tools,
		toolChoice:             input.ToolChoice,
		disableParallelToolUse: input.DisableParallelToolUse,
		rejectsForcedToolUse:   input.RejectsForcedToolUse,
	})
	if err != nil {
		return PreparedProviderTools{}, err
	}

	result := PreparedProviderTools{
		Betas:    slices.Clone(prepared.betas.names),
		Warnings: prepared.warnings.list,
	}

	if prepared.toolChoice == nil {
		return result, nil
	}

	choice, err := json.Marshal(prepared.toolChoice)
	if err != nil {
		return PreparedProviderTools{}, fmt.Errorf("encode tool choice: %w", err)
	}

	result.ToolChoice = choice

	return result, nil
}
