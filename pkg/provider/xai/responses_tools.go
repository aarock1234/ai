package xai

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

// preparedTools are the tools and tool choice of a request.
type preparedTools struct {
	tools      []responsesTool
	toolChoice *toolChoice
	warnings   []model.Warning
}

// prepareTools converts function tools and xAI's server-side tools. Only
// function tools can be forced with a tool choice.
func prepareTools(tools []model.Tool, choice *model.ToolChoice) (preparedTools, error) {
	if len(tools) == 0 {
		return preparedTools{}, nil
	}

	var prepared preparedTools
	byName := make(map[string]model.Tool, len(tools))

	for _, tool := range tools {
		byName[tool.ToolName()] = tool

		switch tool := tool.(type) {
		case model.FunctionTool:
			prepared.tools = append(prepared.tools, responsesTool{
				Type:        "function",
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.InputSchema,
				Strict:      tool.Strict,
			})

		case model.ProviderTool:
			converted, ok, err := serverTool(tool)
			if err != nil {
				return preparedTools{}, err
			}

			if !ok {
				prepared.warnings = append(prepared.warnings, model.UnsupportedWarning("provider-defined tool "+tool.Name, ""))

				continue
			}

			prepared.tools = append(prepared.tools, converted)
		}
	}

	if choice == nil {
		return prepared, nil
	}

	switch choice.Type {
	case model.ToolChoiceAuto, model.ToolChoiceNone, model.ToolChoiceRequired:
		prepared.toolChoice = &toolChoice{
			mode: string(choice.Type),
		}

	case model.ToolChoiceTool:
		selected, ok := byName[choice.ToolName]
		if !ok {
			return prepared, nil
		}

		if _, isProviderTool := selected.(model.ProviderTool); isProviderTool {
			prepared.warnings = append(prepared.warnings, model.UnsupportedWarning(
				fmt.Sprintf("toolChoice for server-side tool \"%s\"", selected.ToolName()),
				"",
			))

			return prepared, nil
		}

		prepared.toolChoice = &toolChoice{
			function: selected.ToolName(),
		}

	default:
		return preparedTools{}, fmt.Errorf("%w: tool choice type %q", model.ErrUnsupported, choice.Type)
	}

	return prepared, nil
}

// serverTool converts an xAI provider tool. It reports false for tools of
// other providers.
func serverTool(tool model.ProviderTool) (responsesTool, bool, error) {
	switch tool.ID {
	case toolWebSearch:
		converted, err := webSearchTool(tool)

		return converted, true, err

	case toolXSearch:
		converted, err := xSearchTool(tool)

		return converted, true, err

	case toolCodeExecution:
		return responsesTool{
			Type: "code_interpreter",
		}, true, nil

	case toolViewImage:
		return responsesTool{
			Type: "view_image",
		}, true, nil

	case toolViewXVideo:
		return responsesTool{
			Type: "view_x_video",
		}, true, nil

	case toolImageGeneration:
		converted, err := imageGenerationTool(tool)

		return converted, true, err

	case toolFileSearch:
		converted, err := fileSearchTool(tool)

		return converted, true, err

	case toolMCP:
		converted, err := mcpTool(tool)

		return converted, true, err

	default:
		return responsesTool{}, false, nil
	}
}

func webSearchTool(tool model.ProviderTool) (responsesTool, error) {
	args, err := decodeToolArgs[WebSearchArgs](tool)
	if err != nil {
		return responsesTool{}, err
	}

	if len(args.AllowedDomains) > maxSearchDomains || len(args.ExcludedDomains) > maxSearchDomains {
		return responsesTool{}, fmt.Errorf("%w: %s allows at most %d allowed and excluded domains", model.ErrInvalidArgument, tool.ID, maxSearchDomains)
	}

	return responsesTool{
		Type:                     "web_search",
		AllowedDomains:           args.AllowedDomains,
		ExcludedDomains:          args.ExcludedDomains,
		EnableImageSearch:        args.EnableImageSearch,
		EnableImageUnderstanding: args.EnableImageUnderstanding,
	}, nil
}

func xSearchTool(tool model.ProviderTool) (responsesTool, error) {
	args, err := decodeToolArgs[XSearchArgs](tool)
	if err != nil {
		return responsesTool{}, err
	}

	if len(args.AllowedXHandles) > maxXHandles || len(args.ExcludedXHandles) > maxXHandles {
		return responsesTool{}, fmt.Errorf("%w: %s allows at most %d allowed and excluded handles", model.ErrInvalidArgument, tool.ID, maxXHandles)
	}

	return responsesTool{
		Type:                     "x_search",
		AllowedXHandles:          args.AllowedXHandles,
		ExcludedXHandles:         args.ExcludedXHandles,
		FromDate:                 args.FromDate,
		ToDate:                   args.ToDate,
		EnableImageUnderstanding: args.EnableImageUnderstanding,
		EnableVideoUnderstanding: args.EnableVideoUnderstanding,
	}, nil
}

func imageGenerationTool(tool model.ProviderTool) (responsesTool, error) {
	args, err := decodeToolArgs[ImageGenerationArgs](tool)
	if err != nil {
		return responsesTool{}, err
	}

	switch args.Action {
	case "", ImageGenerationAuto, ImageGenerationGenerate, ImageGenerationEdit:
	default:
		return responsesTool{}, fmt.Errorf("%w: %s action %q", model.ErrInvalidArgument, tool.ID, args.Action)
	}

	return responsesTool{
		Type:   "image_generation",
		Action: args.Action,
	}, nil
}

// errMissingVectorStores reports a file search without collections.
var errMissingVectorStores = errors.New("vectorStoreIds is required")

func fileSearchTool(tool model.ProviderTool) (responsesTool, error) {
	args, err := decodeToolArgs[FileSearchArgs](tool)
	if err != nil {
		return responsesTool{}, err
	}

	if args.VectorStoreIDs == nil {
		return responsesTool{}, fmt.Errorf("%w: %s: %w", model.ErrInvalidArgument, tool.ID, errMissingVectorStores)
	}

	return responsesTool{
		Type:           "file_search",
		VectorStoreIDs: args.VectorStoreIDs,
		MaxNumResults:  args.MaxNumResults,
	}, nil
}

func mcpTool(tool model.ProviderTool) (responsesTool, error) {
	args, err := decodeToolArgs[MCPServerArgs](tool)
	if err != nil {
		return responsesTool{}, err
	}

	if args.ServerURL == "" {
		return responsesTool{}, fmt.Errorf("%w: %s needs serverUrl", model.ErrInvalidArgument, tool.ID)
	}

	return responsesTool{
		Type:              "mcp",
		ServerURL:         args.ServerURL,
		ServerLabel:       args.ServerLabel,
		ServerDescription: args.ServerDescription,
		AllowedTools:      args.AllowedTools,
		Headers:           args.Headers,
		Authorization:     args.Authorization,
	}, nil
}

// toolNames are the call's names of the server-side tools whose results the
// response reports.
type toolNames struct {
	webSearch       string
	xSearch         string
	codeExecution   string
	mcp             string
	fileSearch      string
	imageGeneration string
}

// serverToolNames returns the name of the first declared tool of each kind,
// or its default name.
func serverToolNames(tools []model.Tool) toolNames {
	return toolNames{
		webSearch:       declaredName(tools, toolWebSearch, nameWebSearch),
		xSearch:         declaredName(tools, toolXSearch, nameXSearch),
		codeExecution:   declaredName(tools, toolCodeExecution, nameCodeExecution),
		mcp:             declaredName(tools, toolMCP, ""),
		fileSearch:      declaredName(tools, toolFileSearch, nameFileSearch),
		imageGeneration: declaredName(tools, toolImageGeneration, nameImageGeneration),
	}
}

func declaredName(tools []model.Tool, toolID string, fallback string) string {
	for _, tool := range tools {
		if providerTool, ok := tool.(model.ProviderTool); ok && providerTool.ID == toolID {
			return providerTool.Name
		}
	}

	return fallback
}

// Sub-tool names that xAI reports for its search tools.
var (
	webSearchSubTools = []string{
		"web_search",
		"web_search_with_snippets",
		"browse_page",
	}

	xSearchSubTools = []string{
		"x_user_search",
		"x_keyword_search",
		"x_semantic_search",
		"x_thread_fetch",
	}
)

// toolName returns the application name of a server-side tool call.
func (n toolNames) toolName(item outputItem) string {
	name := ""
	if item.Name != nil {
		name = *item.Name
	}

	switch {
	case item.Type == itemWebSearchCall || slices.Contains(webSearchSubTools, name):
		return n.webSearch

	case item.Type == itemXSearchCall || slices.Contains(xSearchSubTools, name):
		return n.xSearch

	case name == nameCodeExecution || item.Type == itemCodeInterpreterCall || item.Type == itemCodeExecutionCall:
		return n.codeExecution

	case item.Type == itemMCPCall:
		switch {
		case n.mcp != "":
			return n.mcp
		case name != "":
			return name
		default:
			return nameMCP
		}

	default:
		return name
	}
}

// toolInput returns the input of a server-side tool call.
func toolInput(item outputItem) string {
	if item.Type == itemCustomToolCall {
		if item.Input == nil {
			return ""
		}

		return *item.Input
	}

	if item.Arguments == nil {
		return ""
	}

	return *item.Arguments
}

// Web search action types on the wire and in results.
var webSearchActionTypes = map[string]string{
	"search":       "search",
	"open_page":    "openPage",
	"find_in_page": "findInPage",
}

// mapWebSearchAction converts the action of a web search call. An action of
// an unknown shape gives an empty result.
func mapWebSearchAction(raw json.RawMessage) WebSearchResult {
	var action webSearchWireAction
	if !decodeJSONValue(raw, &action) {
		return WebSearchResult{}
	}

	resultType, known := webSearchActionTypes[action.Type]
	if !known {
		return WebSearchResult{}
	}

	result := WebSearchResult{
		Action: &WebSearchAction{
			Type: resultType,
		},
		Sources: webSources(action.Sources),
	}

	switch action.Type {
	case "search":
		result.Action.Query = action.Query
		result.Action.Queries = action.Queries

	case "open_page":
		result.Action.URL = &nullableString{
			value: action.URL,
		}

	case "find_in_page":
		result.Action.URL = &nullableString{
			value: action.URL,
		}
		result.Action.Pattern = &nullableString{
			value: action.Pattern,
		}
	}

	return result
}

// webSources keeps the sources that are URL sources.
func webSources(raw []json.RawMessage) []WebSource {
	var sources []WebSource

	for _, item := range raw {
		var source WebSource
		if !decodeJSONValue(item, &source) || source.Type != "url" || source.URL == "" {
			continue
		}

		sources = append(sources, source)
	}

	return sources
}
