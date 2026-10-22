package anthropic

import (
	"encoding/json"
	"fmt"
	"slices"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// bedrockAnthropicVersion is the API version Bedrock expects in the
	// body.
	bedrockAnthropicVersion = "bedrock-2023-05-31"

	// fineGrainedToolStreamingBeta replaces the per-tool
	// eager_input_streaming field, which Bedrock rejects.
	fineGrainedToolStreamingBeta = "fine-grained-tool-streaming-2025-05-14"
)

// toolVersionUpgrades map tool versions Bedrock does not accept to newer
// ones.
var toolVersionUpgrades = map[string]string{
	"bash_20241022":        "bash_20250124",
	"text_editor_20241022": "text_editor_20250728",
	"computer_20241022":    "computer_20250124",
}

// toolNames are the names that tool versions require.
var toolNames = map[string]string{
	"text_editor_20250728": "str_replace_based_edit_tool",
}

// toolBetas are the beta features Bedrock requires for tool types.
var toolBetas = map[string]string{
	"bash_20250124":        "computer-use-2025-01-24",
	"bash_20241022":        "computer-use-2024-10-22",
	"text_editor_20250124": "computer-use-2025-01-24",
	"text_editor_20241022": "computer-use-2024-10-22",
	"text_editor_20250429": "computer-use-2025-01-24",
	"text_editor_20250728": "computer-use-2025-01-24",
	"computer_20250124":    "computer-use-2025-01-24",
	"computer_20241022":    "computer-use-2024-10-22",

	"tool_search_tool_regex_20251119": "tool-search-tool-2025-10-19",

	// BM25 tool search is not offered on Bedrock; the beta makes Bedrock
	// return a more useful error.
	"tool_search_tool_bm25_20251119": "tool-search-tool-2025-10-19",
}

// toolChoiceFields are the tool choice members Bedrock accepts.
type toolChoiceFields struct {
	Type json.RawMessage `json:"type"`
	Name json.RawMessage `json:"name"`
}

// thinkingBinding is the block binding of a thinking configuration.
type thinkingBinding struct {
	PrefixMismatchBehavior json.RawMessage `json:"prefix_mismatch_behavior"`
}

// toolType is the type member of a tool definition.
type toolType struct {
	Type *string `json:"type"`
}

// transformRequestBody adapts a Messages API body to Bedrock InvokeModel:
// it removes the model and stream fields, upgrades tool versions and
// declares their betas in the body, keeps only the type and name of the
// tool choice, renames the thinking block binding field and adds the
// Bedrock API version.
func transformRequestBody(body json.RawMessage, betas []string) (json.RawMessage, error) {
	object, err := jsonobject.Parse(body)
	if err != nil {
		return nil, fmt.Errorf("%w: request body: %w", model.ErrInvalidArgument, err)
	}

	transformed := object
	for _, key := range []string{"model", "stream", "tool_choice", "thinking", "tools"} {
		transformed = transformed.Without(key)
	}

	required := slices.Clone(betas)

	if thinking, found := presentMember(object, "thinking"); found {
		value, err := transformThinking(thinking)
		if err != nil {
			return nil, err
		}

		transformed = transformed.With("thinking", value)
	}

	if tools, found := presentMember(object, "tools"); found {
		value, err := transformTools(tools, &required)
		if err != nil {
			return nil, err
		}

		transformed = transformed.With("tools", value)
	}

	if choice, found := presentMember(object, "tool_choice"); found {
		value, err := transformToolChoice(choice)
		if err != nil {
			return nil, err
		}

		transformed = transformed.With("tool_choice", value)
	}

	if len(required) > 0 {
		encoded, err := json.Marshal(required)
		if err != nil {
			return nil, fmt.Errorf("encode anthropic_beta: %w", err)
		}

		transformed = transformed.With("anthropic_beta", encoded)
	}

	transformed = transformed.With("anthropic_version", jsonobject.EncodeString(bedrockAnthropicVersion))

	return transformed.Encode()
}

// transformThinking renames prefix_mismatch_behavior to the
// mismatch_behavior field Bedrock expects; only some regions alias it.
func transformThinking(thinking json.RawMessage) (json.RawMessage, error) {
	object, err := jsonobject.Parse(thinking)
	if err != nil {
		return nil, fmt.Errorf("%w: thinking: %w", model.ErrInvalidArgument, err)
	}

	bindingValue, found := presentMember(object, "block_binding")
	if !found {
		return thinking, nil
	}

	var binding thinkingBinding
	if err := json.Unmarshal(bindingValue, &binding); err != nil {
		return nil, fmt.Errorf("%w: thinking block binding: %w", model.ErrInvalidArgument, err)
	}

	renamed := jsonobject.Object{}
	if binding.PrefixMismatchBehavior != nil {
		renamed = renamed.With("mismatch_behavior", binding.PrefixMismatchBehavior)
	}

	encodedBinding, err := renamed.Encode()
	if err != nil {
		return nil, fmt.Errorf("%w: thinking block binding: %w", model.ErrInvalidArgument, err)
	}

	return object.With("block_binding", encodedBinding).Encode()
}

// transformTools upgrades tool versions and adds the betas tools require
// to required, removing eager_input_streaming in favor of the
// fine-grained tool streaming beta.
func transformTools(tools json.RawMessage, required *[]string) (json.RawMessage, error) {
	var definitions []jsonobject.Object
	if err := json.Unmarshal(tools, &definitions); err != nil {
		return nil, fmt.Errorf("%w: tools: %w", model.ErrInvalidArgument, err)
	}

	transformed := make([]jsonobject.Object, 0, len(definitions))
	for _, definition := range definitions {
		tool, err := transformTool(definition, required)
		if err != nil {
			return nil, err
		}

		transformed = append(transformed, tool)
	}

	encoded, err := json.Marshal(transformed)
	if err != nil {
		return nil, fmt.Errorf("encode tools: %w", err)
	}

	return encoded, nil
}

func transformTool(tool jsonobject.Object, required *[]string) (jsonobject.Object, error) {
	if eager, found := tool.Get("eager_input_streaming"); found && string(eager) == "true" {
		addBeta(required, fineGrainedToolStreamingBeta)
	}

	tool = tool.Without("eager_input_streaming")

	encoded, err := tool.Encode()
	if err != nil {
		return nil, fmt.Errorf("%w: tool: %w", model.ErrInvalidArgument, err)
	}

	var declared toolType
	if err := json.Unmarshal(encoded, &declared); err != nil {
		return nil, fmt.Errorf("%w: tool type: %w", model.ErrInvalidArgument, err)
	}

	if declared.Type == nil {
		return tool, nil
	}

	currentType := *declared.Type

	if upgraded, found := toolVersionUpgrades[currentType]; found {
		addBeta(required, toolBetas[upgraded])

		tool = tool.With("type", jsonobject.EncodeString(upgraded))
		if name, renamed := toolNames[upgraded]; renamed {
			tool = tool.With("name", jsonobject.EncodeString(name))
		}

		return tool, nil
	}

	addBeta(required, toolBetas[currentType])

	if name, renamed := toolNames[currentType]; renamed {
		tool = tool.With("name", jsonobject.EncodeString(name))
	}

	return tool, nil
}

// transformToolChoice keeps the type and name of a tool choice; Bedrock
// rejects disable_parallel_tool_use.
func transformToolChoice(choice json.RawMessage) (json.RawMessage, error) {
	var fields toolChoiceFields
	if err := json.Unmarshal(choice, &fields); err != nil {
		return nil, fmt.Errorf("%w: tool choice: %w", model.ErrInvalidArgument, err)
	}

	transformed := jsonobject.Object{}
	if fields.Type != nil {
		transformed = transformed.With("type", fields.Type)
	}

	if fields.Name != nil && string(fields.Name) != "null" {
		transformed = transformed.With("name", fields.Name)
	}

	return transformed.Encode()
}

// presentMember returns a member that is present and not null.
func presentMember(object jsonobject.Object, key string) (json.RawMessage, bool) {
	value, found := object.Get(key)
	if !found || string(value) == "null" {
		return nil, false
	}

	return value, true
}

// addBeta adds a beta feature once, keeping order. An empty name is
// ignored.
func addBeta(betas *[]string, beta string) {
	if beta == "" || slices.Contains(*betas, beta) {
		return
	}

	*betas = append(*betas, beta)
}
