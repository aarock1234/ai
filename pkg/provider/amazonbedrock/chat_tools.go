package amazonbedrock

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/internal/optional"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/amazonbedrock/internal/bedrock"
	"github.com/aarock1234/ai/pkg/provider/anthropic"
)

// anthropicToolPrefix starts the IDs of Anthropic provider tools.
const anthropicToolPrefix = "anthropic."

// unsupportedProviderTools are the Anthropic tools Bedrock does not offer.
var unsupportedProviderTools = map[string]bool{
	"anthropic.web_search_20250305": true,
	"anthropic.web_search_20260318": true,
	"anthropic.web_fetch_20260318":  true,
}

// providerToolSchemasJSON holds the input schemas of the Anthropic provider
// tools by tool ID. Converse requires a schema for every tool, including
// provider tools whose definitions Claude knows.
//
//go:embed provider_tool_schemas.json
var providerToolSchemasJSON []byte

var providerToolSchemas = sync.OnceValues(func() (map[string]json.RawMessage, error) {
	var schemas map[string]json.RawMessage
	if err := json.Unmarshal(providerToolSchemasJSON, &schemas); err != nil {
		return nil, fmt.Errorf("decode provider tool schemas: %w", err)
	}

	return schemas, nil
})

// toolPreparation are the inputs of tool preparation.
type toolPreparation struct {
	tools      []model.Tool
	toolChoice *model.ToolChoice

	modelID               string
	family                ModelFamily
	reasoningBudgetTokens *int

	disableParallelToolUse *bool
	rejectsForcedToolUse   bool
}

// preparedTools are the tool configuration of a request and the Anthropic
// tool choice that goes into the additional model request fields.
type preparedTools struct {
	config *toolConfig

	// anthropicToolChoice, when set, is sent as the tool_choice additional
	// model request field.
	anthropicToolChoice json.RawMessage

	betas    []string
	warnings []model.Warning
}

// hasTools reports whether the request declares tools or an Anthropic
// tool choice.
func (p preparedTools) hasTools() bool {
	return p.config != nil || p.anthropicToolChoice != nil
}

// prepareTools converts the call's tools into the Converse tool
// configuration. Anthropic provider tools of Claude models are declared as
// tool specifications, with Claude's tool choice in the additional model
// request fields.
func prepareTools(preparation toolPreparation) (preparedTools, error) {
	prepared := preparedTools{}

	functionTools, providerTools := prepared.splitTools(preparation.tools)
	if len(functionTools) == 0 && len(providerTools) == 0 {
		return prepared, nil
	}

	anthropicModel := isAnthropicModel(preparation.modelID, preparation.family, preparation.reasoningBudgetTokens)
	choice := preparation.toolChoice

	rejectsForcedChoice := anthropicModel && preparation.rejectsForcedToolUse && isForcedChoice(choice)

	requestProviderTools := providerTools
	if rejectsForcedChoice && choice.Type == model.ToolChoiceTool {
		requestProviderTools = toolsNamed(providerTools, choice.ToolName)
	}

	usingAnthropicTools := anthropicModel && len(requestProviderTools) > 0

	var specs []toolEntry

	if usingAnthropicTools {
		providerSpecs, err := prepared.anthropicTools(preparation, providerTools, requestProviderTools)
		if err != nil {
			return preparedTools{}, err
		}

		specs = providerSpecs
	} else {
		for _, tool := range requestProviderTools {
			prepared.warnings = append(prepared.warnings, model.UnsupportedWarning("tool "+tool.ID, ""))
		}
	}

	preparedChoice := choice
	if !usingAnthropicTools && rejectsForcedChoice {
		prepared.warnForcedChoice(choice)
		preparedChoice = &model.ToolChoice{
			Type: model.ToolChoiceAuto,
		}
	}

	if choice != nil && choice.Type == model.ToolChoiceTool {
		functionTools = functionToolsNamed(functionTools, choice.ToolName)
	}

	functionSpecs, err := prepared.functionTools(functionTools, preparation.modelID)
	if err != nil {
		return preparedTools{}, err
	}

	specs = append(specs, functionSpecs...)

	disableParallel := optional.Or(preparation.disableParallelToolUse, false)
	if anthropicModel && !usingAnthropicTools && disableParallel && len(specs) > 0 && !isChoice(preparedChoice, model.ToolChoiceNone) {
		prepared.anthropicToolChoice, err = parallelDisabledChoice(preparedChoice)
		if err != nil {
			return preparedTools{}, err
		}
	}

	var toolChoice *converseToolChoice
	if !usingAnthropicTools && prepared.anthropicToolChoice == nil && len(specs) > 0 && preparedChoice != nil {
		toolChoice, specs, err = converseChoiceOf(*preparedChoice, specs)
		if err != nil {
			return preparedTools{}, err
		}
	}

	if len(specs) > 0 {
		prepared.config = &toolConfig{
			Tools:      specs,
			ToolChoice: toolChoice,
		}
	}

	return prepared, nil
}

// splitTools separates function tools from provider tools, dropping the
// Anthropic tools that Bedrock does not offer with a warning.
func (p *preparedTools) splitTools(tools []model.Tool) ([]model.FunctionTool, []model.ProviderTool) {
	var (
		functionTools []model.FunctionTool
		providerTools []model.ProviderTool
	)

	for _, tool := range tools {
		switch tool := tool.(type) {
		case model.FunctionTool:
			functionTools = append(functionTools, tool)

		case *model.FunctionTool:
			functionTools = append(functionTools, *tool)

		case model.ProviderTool:
			providerTools = p.keepSupported(providerTools, tool)

		case *model.ProviderTool:
			providerTools = p.keepSupported(providerTools, *tool)
		}
	}

	return functionTools, providerTools
}

func (p *preparedTools) keepSupported(tools []model.ProviderTool, tool model.ProviderTool) []model.ProviderTool {
	if !unsupportedProviderTools[tool.ID] {
		return append(tools, tool)
	}

	name := strings.TrimPrefix(tool.ID, anthropicToolPrefix)
	p.warnings = append(p.warnings, model.UnsupportedWarning(
		name+" tool",
		fmt.Sprintf("The %s tool is not supported on Amazon Bedrock.", name),
	))

	return tools
}

// anthropicTools prepares Claude's tool choice for the provider tools and
// declares the requested ones as tool specifications.
func (p *preparedTools) anthropicTools(preparation toolPreparation, providerTools []model.ProviderTool, requestTools []model.ProviderTool) ([]toolEntry, error) {
	anthropicTools, err := anthropic.PrepareProviderTools(anthropic.ProviderToolsInput{
		Tools:                  providerTools,
		ToolChoice:             preparation.toolChoice,
		DisableParallelToolUse: preparation.disableParallelToolUse,
		RejectsForcedToolUse:   preparation.rejectsForcedToolUse,
	})
	if err != nil {
		return nil, err
	}

	p.warnings = append(p.warnings, anthropicTools.Warnings...)
	p.betas = append(p.betas, anthropicTools.Betas...)
	p.anthropicToolChoice = anthropicTools.ToolChoice

	schemas, err := providerToolSchemas()
	if err != nil {
		return nil, err
	}

	var specs []toolEntry
	for _, tool := range requestTools {
		schema, known := schemas[tool.ID]
		if !known {
			p.warnings = append(p.warnings, model.UnsupportedWarning("tool "+tool.ID, ""))

			continue
		}

		specs = append(specs, toolEntry{
			ToolSpec: &toolSpec{
				Name: tool.Name,
				InputSchema: toolInputSchema{
					JSON: schema,
				},
			},
		})
	}

	return specs, nil
}

func (p *preparedTools) warnForcedChoice(choice *model.ToolChoice) {
	if choice.Type == model.ToolChoiceTool {
		p.warnings = append(p.warnings, model.UnsupportedWarning(
			"toolChoice",
			fmt.Sprintf(
				"toolChoice 'tool' is not supported by this model because it rejects forced tool use. Only the '%s' tool is sent with 'auto' tool choice. Instruct the model to use the tool in the prompt and verify that a tool call was made.",
				choice.ToolName,
			),
		))

		return
	}

	p.warnings = append(p.warnings, model.UnsupportedWarning(
		"toolChoice",
		"toolChoice 'required' is not supported by this model because it rejects forced tool use. Using 'auto' instead. Instruct the model to use a tool in the prompt and verify that a tool call was made.",
	))
}

// functionTools declares function tools, keeping strict only where Bedrock
// accepts it.
func (p *preparedTools) functionTools(tools []model.FunctionTool, modelID string) ([]toolEntry, error) {
	strictSupported := bedrock.SupportsStrictTools(modelID)

	specs := make([]toolEntry, 0, len(tools))
	for _, tool := range tools {
		if len(tool.InputSchema) == 0 {
			return nil, fmt.Errorf("%w: function tool %q has no input schema", model.ErrInvalidArgument, tool.Name)
		}

		strictRequested := optional.Or(tool.Strict, false)
		strictForTool := strictSupported && (!strictRequested || isStrictSchemaCompatible(tool.InputSchema))

		switch {
		case !strictSupported && strictRequested:
			p.warnings = append(p.warnings, model.UnsupportedWarning(
				"strict",
				fmt.Sprintf(
					"Tool '%s' has strict: true, but strict mode is not supported by this model on Amazon Bedrock. The strict property will be ignored.",
					tool.Name,
				),
			))

		case strictRequested && !strictForTool:
			p.warnings = append(p.warnings, model.UnsupportedWarning(
				"strict",
				fmt.Sprintf(
					"Tool '%s' has strict: true, but Amazon Bedrock requires every object in a strict tool schema to set additionalProperties: false. The strict property will be ignored.",
					tool.Name,
				),
			))
		}

		spec := &toolSpec{
			Name: tool.Name,
			InputSchema: toolInputSchema{
				JSON: tool.InputSchema,
			},
		}

		if strings.TrimSpace(tool.Description) != "" {
			spec.Description = tool.Description
		}

		if tool.Strict != nil && strictForTool {
			spec.Strict = tool.Strict
		}

		specs = append(specs, toolEntry{
			ToolSpec: spec,
		})
	}

	return specs, nil
}

// parallelDisabledChoice is Claude's tool choice with parallel tool use
// disabled.
func parallelDisabledChoice(choice *model.ToolChoice) (json.RawMessage, error) {
	choiceType := "auto"
	if choice != nil && choice.Type == model.ToolChoiceRequired {
		choiceType = "any"
	}

	object := jsonobject.Object{
		{
			Key:   "type",
			Value: jsonobject.EncodeString(choiceType),
		},
	}

	if choice != nil && choice.Type == model.ToolChoiceTool {
		object = jsonobject.Object{
			{
				Key:   "type",
				Value: jsonobject.EncodeString("tool"),
			},
			{
				Key:   "name",
				Value: jsonobject.EncodeString(choice.ToolName),
			},
		}
	}

	return object.With("disable_parallel_tool_use", json.RawMessage(`true`)).Encode()
}

// converseChoiceOf converts a tool choice. A "none" choice removes the
// tools instead.
func converseChoiceOf(choice model.ToolChoice, specs []toolEntry) (*converseToolChoice, []toolEntry, error) {
	switch choice.Type {
	case model.ToolChoiceAuto:
		return &converseToolChoice{
			Auto: &struct{}{},
		}, specs, nil

	case model.ToolChoiceRequired:
		return &converseToolChoice{
			Any: &struct{}{},
		}, specs, nil

	case model.ToolChoiceNone:
		return nil, nil, nil

	case model.ToolChoiceTool:
		return &converseToolChoice{
			Tool: &namedToolName{
				Name: choice.ToolName,
			},
		}, specs, nil

	default:
		return nil, nil, fmt.Errorf("%w: tool choice type: %s", model.ErrUnsupported, choice.Type)
	}
}

func isForcedChoice(choice *model.ToolChoice) bool {
	return isChoice(choice, model.ToolChoiceRequired) || isChoice(choice, model.ToolChoiceTool)
}

func isChoice(choice *model.ToolChoice, choiceType model.ToolChoiceType) bool {
	return choice != nil && choice.Type == choiceType
}

func toolsNamed(tools []model.ProviderTool, name string) []model.ProviderTool {
	var named []model.ProviderTool
	for _, tool := range tools {
		if tool.Name == name {
			named = append(named, tool)
		}
	}

	return named
}

func functionToolsNamed(tools []model.FunctionTool, name string) []model.FunctionTool {
	var named []model.FunctionTool
	for _, tool := range tools {
		if tool.Name == name {
			named = append(named, tool)
		}
	}

	return named
}
