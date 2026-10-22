package google

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// Provider tool IDs that only the Interactions API accepts. Create them as
// model.ProviderTool values with the matching arguments.
const (
	computerUseToolID = "google.computer_use"
	mcpServerToolID   = "google.mcp_server"
	retrievalToolID   = "google.retrieval"
)

// interactionsTool is one entry of the tools of an interaction.
type interactionsTool interface {
	isInteractionsTool()
}

type interactionsFunctionTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// interactionsBuiltinTool is a built-in tool without settings, such as
// code execution.
type interactionsBuiltinTool struct {
	Type string `json:"type"`
}

type interactionsSearchTool struct {
	Type        string   `json:"type"`
	SearchTypes []string `json:"search_types,omitempty"`
}

type interactionsFileSearchTool struct {
	Type                 string   `json:"type"`
	FileSearchStoreNames []string `json:"file_search_store_names,omitempty"`
	TopK                 *int     `json:"top_k,omitempty"`
	MetadataFilter       string   `json:"metadata_filter,omitempty"`
}

type interactionsMapsTool struct {
	Type         string   `json:"type"`
	Latitude     *float64 `json:"latitude,omitempty"`
	Longitude    *float64 `json:"longitude,omitempty"`
	EnableWidget *bool    `json:"enable_widget,omitempty"`
}

// interactionsComputerUseTool keeps the camelCase member the API expects
// for excluded functions.
type interactionsComputerUseTool struct {
	Type                        string   `json:"type"`
	Environment                 string   `json:"environment"`
	ExcludedPredefinedFunctions []string `json:"excludedPredefinedFunctions,omitempty"`
}

type interactionsMCPServerTool struct {
	Type         string            `json:"type"`
	Name         string            `json:"name,omitempty"`
	URL          string            `json:"url,omitempty"`
	Headers      map[string]string `json:"headers,omitempty"`
	AllowedTools json.RawMessage   `json:"allowed_tools,omitempty"`
}

type interactionsRetrievalTool struct {
	Type                 string                      `json:"type"`
	RetrievalTypes       []string                    `json:"retrieval_types"`
	VertexAISearchConfig *InteractionsVertexAISearch `json:"vertex_ai_search_config,omitempty"`
}

func (interactionsFunctionTool) isInteractionsTool()    {}
func (interactionsBuiltinTool) isInteractionsTool()     {}
func (interactionsSearchTool) isInteractionsTool()      {}
func (interactionsFileSearchTool) isInteractionsTool()  {}
func (interactionsMapsTool) isInteractionsTool()        {}
func (interactionsComputerUseTool) isInteractionsTool() {}
func (interactionsMCPServerTool) isInteractionsTool()   {}
func (interactionsRetrievalTool) isInteractionsTool()   {}

// interactionsMapsArgs are the Google Maps settings of the Interactions
// API.
type interactionsMapsArgs struct {
	Latitude     *float64 `json:"latitude,omitempty"`
	Longitude    *float64 `json:"longitude,omitempty"`
	EnableWidget *bool    `json:"enableWidget,omitempty"`
}

// InteractionsComputerUseArgs are the arguments of the "google.computer_use"
// provider tool.
type InteractionsComputerUseArgs struct {
	// Environment is "browser", the default.
	Environment                 string   `json:"environment,omitempty"`
	ExcludedPredefinedFunctions []string `json:"excludedPredefinedFunctions,omitempty"`
}

// InteractionsMCPServerArgs are the arguments of the "google.mcp_server"
// provider tool.
type InteractionsMCPServerArgs struct {
	Name         string            `json:"name,omitempty"`
	URL          string            `json:"url,omitempty"`
	Headers      map[string]string `json:"headers,omitempty"`
	AllowedTools json.RawMessage   `json:"allowedTools,omitempty"`
}

// InteractionsRetrievalArgs are the arguments of the "google.retrieval"
// provider tool. RetrievalTypes defaults to vertex_ai_search.
type InteractionsRetrievalArgs struct {
	RetrievalTypes       []string                    `json:"retrievalTypes,omitempty"`
	VertexAISearchConfig *InteractionsVertexAISearch `json:"vertexAiSearchConfig,omitempty"`
}

// InteractionsVertexAISearch selects the Vertex AI Search data stores or
// engine.
type InteractionsVertexAISearch struct {
	Datastores []string `json:"datastores,omitempty"`
	Engine     string   `json:"engine,omitempty"`
}

// preparedInteractionsTools are the converted tools and tool choice.
type preparedInteractionsTools struct {
	tools      []interactionsTool
	toolChoice *interactionsToolChoice
	warnings   []model.Warning
}

// prepareInteractionsTools converts function tools and the google.*
// provider tools. The tool choice governs function calling only, so it is
// dropped without function tools.
func prepareInteractionsTools(tools []model.Tool, toolChoice *model.ToolChoice) (preparedInteractionsTools, error) {
	var prepared preparedInteractionsTools

	hasFunction := false

	for _, tool := range tools {
		switch typed := tool.(type) {
		case model.FunctionTool:
			hasFunction = true

			prepared.tools = append(prepared.tools, interactionsFunctionTool{
				Type:        "function",
				Name:        typed.Name,
				Description: typed.Description,
				Parameters:  typed.InputSchema,
			})

		case model.ProviderTool:
			converted, ok, err := interactionsProviderTool(typed)
			if err != nil {
				return preparedInteractionsTools{}, err
			}

			if !ok {
				prepared.warnings = append(prepared.warnings, model.UnsupportedWarning(
					"provider-defined tool "+typed.ID,
					fmt.Sprintf("provider-defined tool %s is not supported by google.interactions; tool dropped.", typed.ID),
				))

				continue
			}

			prepared.tools = append(prepared.tools, converted)

		default:
			prepared.warnings = append(prepared.warnings, model.UnsupportedWarning(
				fmt.Sprintf("tool of type %T", tool),
				"Only function tools and google.* provider-defined tools are supported by google.interactions; tool dropped.",
			))
		}
	}

	if toolChoice != nil && hasFunction {
		prepared.toolChoice = interactionsToolChoiceOf(*toolChoice)
	}

	return prepared, nil
}

// interactionsProviderTool converts a provider tool. It reports false for
// tools the Interactions API does not support.
func interactionsProviderTool(tool model.ProviderTool) (interactionsTool, bool, error) {
	switch tool.ID {
	case googleSearchToolID:
		args, err := decodeToolArgs[SearchArgs](tool)
		if err != nil {
			return nil, false, err
		}

		return interactionsSearchTool{
			Type:        "google_search",
			SearchTypes: interactionsSearchTypes(args.SearchTypes),
		}, true, nil

	case codeExecutionToolID:
		return interactionsBuiltinTool{
			Type: "code_execution",
		}, true, nil

	case urlContextToolID:
		return interactionsBuiltinTool{
			Type: "url_context",
		}, true, nil

	case fileSearchToolID:
		args, err := decodeToolArgs[FileSearchArgs](tool)
		if err != nil {
			return nil, false, err
		}

		return interactionsFileSearchTool{
			Type:                 "file_search",
			FileSearchStoreNames: args.FileSearchStoreNames,
			TopK:                 args.TopK,
			MetadataFilter:       args.MetadataFilter,
		}, true, nil

	case googleMapsToolID:
		args, err := decodeToolArgs[interactionsMapsArgs](tool)
		if err != nil {
			return nil, false, err
		}

		return interactionsMapsTool{
			Type:         "google_maps",
			Latitude:     args.Latitude,
			Longitude:    args.Longitude,
			EnableWidget: args.EnableWidget,
		}, true, nil

	default:
		return interactionsAgentTool(tool)
	}
}

// interactionsAgentTool converts the tools that only the Interactions API
// accepts.
func interactionsAgentTool(tool model.ProviderTool) (interactionsTool, bool, error) {
	switch tool.ID {
	case computerUseToolID:
		args, err := decodeToolArgs[InteractionsComputerUseArgs](tool)
		if err != nil {
			return nil, false, err
		}

		environment := args.Environment
		if environment == "" {
			environment = "browser"
		}

		return interactionsComputerUseTool{
			Type:                        "computer_use",
			Environment:                 environment,
			ExcludedPredefinedFunctions: args.ExcludedPredefinedFunctions,
		}, true, nil

	case mcpServerToolID:
		args, err := decodeToolArgs[InteractionsMCPServerArgs](tool)
		if err != nil {
			return nil, false, err
		}

		return interactionsMCPServerTool{
			Type:         "mcp_server",
			Name:         args.Name,
			URL:          args.URL,
			Headers:      args.Headers,
			AllowedTools: args.AllowedTools,
		}, true, nil

	case retrievalToolID:
		args, err := decodeToolArgs[InteractionsRetrievalArgs](tool)
		if err != nil {
			return nil, false, err
		}

		retrievalTypes := args.RetrievalTypes
		if retrievalTypes == nil {
			retrievalTypes = []string{
				"vertex_ai_search",
			}
		}

		return interactionsRetrievalTool{
			Type:                 "retrieval",
			RetrievalTypes:       retrievalTypes,
			VertexAISearchConfig: args.VertexAISearchConfig,
		}, true, nil

	default:
		return nil, false, nil
	}
}

func interactionsSearchTypes(types *SearchTypes) []string {
	if types == nil {
		return nil
	}

	var searchTypes []string

	if types.WebSearch != nil {
		searchTypes = append(searchTypes, "web_search")
	}

	if types.ImageSearch != nil {
		searchTypes = append(searchTypes, "image_search")
	}

	return searchTypes
}

// interactionsToolChoiceOf maps the tool choice. A specific tool becomes a
// validated allowed tools configuration.
func interactionsToolChoiceOf(choice model.ToolChoice) *interactionsToolChoice {
	switch choice.Type {
	case model.ToolChoiceAuto:
		return &interactionsToolChoice{
			Mode: "auto",
		}

	case model.ToolChoiceRequired:
		return &interactionsToolChoice{
			Mode: "any",
		}

	case model.ToolChoiceNone:
		return &interactionsToolChoice{
			Mode: "none",
		}

	case model.ToolChoiceTool:
		return &interactionsToolChoice{
			AllowedTools: &interactionsAllowedTools{
				Mode: "validated",
				Tools: []string{
					choice.ToolName,
				},
			},
		}

	default:
		return nil
	}
}
