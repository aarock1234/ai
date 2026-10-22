package google

import (
	"fmt"

	"github.com/aarock1234/ai/internal/optional"

	"github.com/aarock1234/ai/pkg/model"
)

// preparedTools are the tool declarations and tool configuration of a
// request.
type preparedTools struct {
	tools    []toolDeclaration
	config   *toolConfig
	warnings []model.Warning
}

// prepareTools converts the call's tools and tool choice. Provider tools
// become built-in tool declarations and function tools become function
// declarations. Gemini 3 models accept both kinds in one request.
func prepareTools(tools []model.Tool, toolChoice *model.ToolChoice, modelID string, isVertex bool) (preparedTools, error) {
	var prepared preparedTools
	if len(tools) == 0 {
		return prepared, nil
	}

	capabilities := capabilitiesOf(modelID)

	functionTools, providerTools, err := splitTools(tools)
	if err != nil {
		return preparedTools{}, err
	}

	if len(providerTools) == 0 {
		return prepareFunctionTools(functionTools, toolChoice)
	}

	if len(functionTools) > 0 && !capabilities.usesGemini3Features {
		combination := model.UnsupportedWarning("combination of function and provider-defined tools", "")
		prepared.warnings = append(prepared.warnings, combination)
	}

	for _, tool := range providerTools {
		declaration, warning, err := prepareProviderTool(tool, capabilities)
		if err != nil {
			return preparedTools{}, err
		}

		if warning != nil {
			prepared.warnings = append(prepared.warnings, *warning)

			continue
		}

		prepared.tools = append(prepared.tools, declaration)
	}

	canCombine := len(functionTools) > 0 && capabilities.usesGemini3Features && len(prepared.tools) > 0
	if canCombine {
		prepared.tools = append(prepared.tools, toolDeclaration{
			FunctionDeclarations: functionDeclarations(functionTools),
		})
		prepared.config = combinedToolConfig(toolChoice, isVertex)
	}

	return prepared, nil
}

func splitTools(tools []model.Tool) ([]model.FunctionTool, []model.ProviderTool, error) {
	var functionTools []model.FunctionTool
	var providerTools []model.ProviderTool

	for _, tool := range tools {
		switch tool := tool.(type) {
		case model.FunctionTool:
			functionTools = append(functionTools, tool)
		case model.ProviderTool:
			providerTools = append(providerTools, tool)
		default:
			return nil, nil, fmt.Errorf("%w: tool %q of type %T", model.ErrUnsupported, tool.ToolName(), tool)
		}
	}

	return functionTools, providerTools, nil
}

// prepareProviderTool returns the declaration of a provider tool, or a
// warning when the model does not support it.
func prepareProviderTool(tool model.ProviderTool, capabilities modelCapabilities) (toolDeclaration, *model.Warning, error) {
	unsupported := func(details string) (toolDeclaration, *model.Warning, error) {
		warning := model.UnsupportedWarning("provider-defined tool "+tool.ID, details)

		return toolDeclaration{}, &warning, nil
	}

	switch tool.ID {
	case googleSearchToolID:
		if !capabilities.supportsGemini2Tools {
			return unsupported("Google Search requires Gemini 2.0 or newer.")
		}

		args, err := decodeToolArgs[SearchArgs](tool)
		if err != nil {
			return toolDeclaration{}, nil, err
		}

		return toolDeclaration{
			GoogleSearch: &args,
		}, nil, nil

	case enterpriseWebSearchToolID:
		if !capabilities.supportsGemini2Tools {
			return unsupported("Enterprise Web Search requires Gemini 2.0 or newer.")
		}

		return toolDeclaration{
			EnterpriseWebSearch: &emptyObject{},
		}, nil, nil

	case urlContextToolID:
		if !capabilities.supportsGemini2Tools {
			return unsupported("The URL context tool is not supported with other Gemini models than Gemini 2.")
		}

		return toolDeclaration{
			URLContext: &emptyObject{},
		}, nil, nil

	case codeExecutionToolID:
		if !capabilities.supportsGemini2Tools {
			return unsupported("The code execution tool is not supported with other Gemini models than Gemini 2.")
		}

		return toolDeclaration{
			CodeExecution: &emptyObject{},
		}, nil, nil

	case fileSearchToolID:
		if !capabilities.supportsFileSearch {
			return unsupported("The file search tool is only supported with Gemini 2.5 models and Gemini 3 models.")
		}

		args, err := decodeToolArgs[FileSearchArgs](tool)
		if err != nil {
			return toolDeclaration{}, nil, err
		}

		if err := args.validate(); err != nil {
			return toolDeclaration{}, nil, err
		}

		return toolDeclaration{
			FileSearch: &args,
		}, nil, nil

	case vertexRAGStoreToolID:
		if !capabilities.supportsGemini2Tools {
			return unsupported("The RAG store tool is not supported with other Gemini models than Gemini 2.")
		}

		args, err := decodeToolArgs[VertexRAGStoreArgs](tool)
		if err != nil {
			return toolDeclaration{}, nil, err
		}

		return toolDeclaration{
			Retrieval: &retrieval{
				VertexRAGStore: vertexRAGStore{
					RAGResources: ragResources{
						RAGCorpus: args.RAGCorpus,
					},
					SimilarityTopK: args.TopK,
				},
			},
		}, nil, nil

	case googleMapsToolID:
		if !capabilities.supportsGemini2Tools {
			return unsupported("The Google Maps grounding tool is not supported with Gemini models other than Gemini 2 or newer.")
		}

		return toolDeclaration{
			GoogleMaps: &emptyObject{},
		}, nil, nil

	default:
		return unsupported("")
	}
}

// combinedToolConfig configures function calling when function tools are
// combined with server-side tools on Gemini 3 models. Only the Gemini API
// accepts the request to include server-side tool invocations.
func combinedToolConfig(toolChoice *model.ToolChoice, isVertex bool) *toolConfig {
	config := &toolConfig{
		FunctionCallingConfig: &functionCallingConfig{
			Mode: functionCallingValidated,
		},
		IncludeServerSideToolInvocations: !isVertex,
	}

	if toolChoice == nil {
		return config
	}

	switch toolChoice.Type {
	case model.ToolChoiceNone:
		config.FunctionCallingConfig = &functionCallingConfig{
			Mode: functionCallingNone,
		}

	case model.ToolChoiceRequired:
		config.FunctionCallingConfig = &functionCallingConfig{
			Mode: functionCallingAny,
		}

	case model.ToolChoiceTool:
		config.FunctionCallingConfig = &functionCallingConfig{
			Mode: functionCallingAny,
			AllowedFunctionNames: []string{
				toolChoice.ToolName,
			},
		}
	}

	return config
}

// prepareFunctionTools declares function tools without provider tools. Strict
// tools switch automatic function calling to validated mode.
func prepareFunctionTools(tools []model.FunctionTool, toolChoice *model.ToolChoice) (preparedTools, error) {
	prepared := preparedTools{
		tools: []toolDeclaration{
			{
				FunctionDeclarations: functionDeclarations(tools),
			},
		},
	}

	automaticMode := functionCallingAuto
	if hasStrictTool(tools) {
		automaticMode = functionCallingValidated
	}

	if toolChoice == nil {
		if automaticMode == functionCallingValidated {
			prepared.config = &toolConfig{
				FunctionCallingConfig: &functionCallingConfig{
					Mode: functionCallingValidated,
				},
			}
		}

		return prepared, nil
	}

	callingConfig := &functionCallingConfig{}

	switch toolChoice.Type {
	case model.ToolChoiceAuto:
		callingConfig.Mode = automaticMode

	case model.ToolChoiceNone:
		callingConfig.Mode = functionCallingNone

	case model.ToolChoiceRequired:
		callingConfig.Mode = functionCallingAny

	case model.ToolChoiceTool:
		callingConfig.Mode = functionCallingAny
		callingConfig.AllowedFunctionNames = []string{
			toolChoice.ToolName,
		}

	default:
		return preparedTools{}, fmt.Errorf("%w: tool choice type %q", model.ErrUnsupported, toolChoice.Type)
	}

	prepared.config = &toolConfig{
		FunctionCallingConfig: callingConfig,
	}

	return prepared, nil
}

func functionDeclarations(tools []model.FunctionTool) []functionDeclaration {
	declarations := make([]functionDeclaration, 0, len(tools))
	for _, tool := range tools {
		declarations = append(declarations, functionDeclaration{
			Name:                 tool.Name,
			Description:          tool.Description,
			ParametersJSONSchema: tool.InputSchema,
		})
	}

	return declarations
}

func hasStrictTool(tools []model.FunctionTool) bool {
	for _, tool := range tools {
		if optional.Or(tool.Strict, false) {
			return true
		}
	}

	return false
}
