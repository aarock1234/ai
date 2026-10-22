package anthropic

import (
	"fmt"
	"slices"

	"github.com/aarock1234/ai/internal/optional"

	"github.com/aarock1234/ai/pkg/model"
)

// toolPreparation are the inputs of tool preparation.
type toolPreparation struct {
	tools                  []model.Tool
	toolChoice             *model.ToolChoice
	disableParallelToolUse *bool
	cache                  *cacheControlValidator

	supportsStructuredOutput bool
	supportsStrictTools      bool

	// defaultEagerInputStreaming applies to function tools that do not set
	// eager input streaming themselves.
	defaultEagerInputStreaming bool

	// rejectsForcedToolUse downgrades required and named tool choices to
	// auto.
	rejectsForcedToolUse bool
}

// preparedTools are the tool definitions and tool choice of a request.
type preparedTools struct {
	tools      []requestTool
	toolChoice *toolChoice
	warnings   *callWarnings
	betas      *betaSet
}

// prepareTools converts the call's tools and tool choice.
func prepareTools(preparation toolPreparation) (preparedTools, error) {
	prepared := preparedTools{
		warnings: &callWarnings{},
		betas:    newBetaSet(),
	}

	if len(preparation.tools) == 0 {
		return prepared, nil
	}

	for _, tool := range preparation.tools {
		converted, err := prepared.convertTool(tool, preparation)
		if err != nil {
			return preparedTools{}, err
		}

		if converted != nil {
			prepared.tools = append(prepared.tools, converted)
		}
	}

	if prepared.tools == nil {
		prepared.tools = []requestTool{}
	}

	if err := prepared.chooseTool(preparation); err != nil {
		return preparedTools{}, err
	}

	return prepared, nil
}

func (p *preparedTools) convertTool(tool model.Tool, preparation toolPreparation) (requestTool, error) {
	switch typed := tool.(type) {
	case model.FunctionTool:
		return p.functionTool(typed, preparation)
	case *model.FunctionTool:
		return p.functionTool(*typed, preparation)
	case model.ProviderTool:
		return p.providerTool(typed)
	case *model.ProviderTool:
		return p.providerTool(*typed)
	default:
		p.warnings.unsupported(fmt.Sprintf("tool %T", tool), "")

		return nil, nil
	}
}

func (p *preparedTools) functionTool(tool model.FunctionTool, preparation toolPreparation) (requestTool, error) {
	if len(tool.InputSchema) == 0 {
		return nil, fmt.Errorf("%w: function tool %q has no input schema", model.ErrInvalidArgument, tool.Name)
	}

	cacheControl := preparation.cache.cacheControl(tool.ProviderOptions, cacheToolDefinition)

	options, _, err := model.ProviderValue[ToolOptions](tool.ProviderOptions, Namespace)
	if err != nil {
		return nil, err
	}

	strictRequested := optional.Or(tool.Strict, false)
	if !preparation.supportsStrictTools && strictRequested {
		p.warnings.unsupported(
			"strict",
			fmt.Sprintf(
				"Tool '%s' has strict: true, but strict mode is not supported by this provider. The strict property will be ignored.",
				tool.Name,
			),
		)
	}

	converted := &functionTool{
		Name:                tool.Name,
		Description:         tool.Description,
		InputSchema:         tool.InputSchema,
		CacheControl:        cacheControl,
		EagerInputStreaming: optional.Or(options.EagerInputStreaming, preparation.defaultEagerInputStreaming),
		DeferLoading:        options.DeferLoading,
		AllowedCallers:      options.AllowedCallers,
		InputExamples:       tool.InputExamples,
	}

	if preparation.supportsStrictTools {
		converted.Strict = tool.Strict
	}

	if preparation.supportsStructuredOutput {
		p.betas.add("structured-outputs-2025-11-13")
	}

	if tool.InputExamples != nil || options.AllowedCallers != nil {
		p.betas.add("advanced-tool-use-2025-11-20")
	}

	return converted, nil
}

// providerTool converts a provider-defined tool. Provider tools have no
// provider options, so they never carry cache breakpoints.
func (p *preparedTools) providerTool(tool model.ProviderTool) (requestTool, error) {
	switch tool.ID {
	case toolCodeExecution20250522:
		p.betas.add("code-execution-2025-05-22")

		return builtin("code_execution_20250522", nameCodeExecution), nil

	case toolCodeExecution20250825:
		p.betas.add("code-execution-2025-08-25")

		return builtin("code_execution_20250825", nameCodeExecution), nil

	case toolCodeExecution20260120:
		return builtin("code_execution_20260120", nameCodeExecution), nil

	case toolComputer20241022:
		p.betas.add("computer-use-2024-10-22")

		return computerToolOf(tool, "computer_20241022")

	case toolComputer20250124:
		p.betas.add("computer-use-2025-01-24")

		return computerToolOf(tool, "computer_20250124")

	case toolComputer20251124:
		p.betas.add("computer-use-2025-11-24")

		return zoomComputerToolOf(tool)

	case toolComputerToolset20260801:
		return computerToolsetOf(tool)

	case toolTextEditor20241022:
		p.betas.add("computer-use-2024-10-22")

		return builtin("text_editor_20241022", nameStrReplaceEditor), nil

	case toolTextEditor20250124:
		p.betas.add("computer-use-2025-01-24")

		return builtin("text_editor_20250124", nameStrReplaceEditor), nil

	case toolTextEditor20250429:
		p.betas.add("computer-use-2025-01-24")

		return builtin("text_editor_20250429", nameStrReplaceBased), nil

	case toolTextEditor20250728:
		return textEditorToolOf(tool)

	case toolBash20241022:
		p.betas.add("computer-use-2024-10-22")

		return builtin("bash_20241022", nameBash), nil

	case toolBash20250124:
		p.betas.add("computer-use-2025-01-24")

		return builtin("bash_20250124", nameBash), nil

	case toolMemory20250818:
		p.betas.add("context-management-2025-06-27")

		return builtin("memory_20250818", nameMemory), nil

	case toolWebFetch20250910:
		p.betas.add("web-fetch-2025-09-10")

		return webFetchToolOf(tool, "web_fetch_20250910")

	case toolWebFetch20260209:
		p.betas.add("code-execution-web-tools-2026-02-09")

		return webFetchToolOf(tool, "web_fetch_20260209")

	case toolWebFetch20260318:
		return latestWebFetchToolOf(tool)

	case toolWebSearch20250305:
		return webSearchToolOf(tool, "web_search_20250305")

	case toolWebSearch20260209:
		p.betas.add("code-execution-web-tools-2026-02-09")

		return webSearchToolOf(tool, "web_search_20260209")

	case toolWebSearch20260318:
		return latestWebSearchToolOf(tool)

	case toolSearchRegex20251119:
		return builtin("tool_search_tool_regex_20251119", nameToolSearchRegex), nil

	case toolSearchBM2520251119:
		return builtin("tool_search_tool_bm25_20251119", nameToolSearchBM25), nil

	case toolAdvisor20260301:
		p.betas.add("advisor-tool-2026-03-01")

		return advisorToolOf(tool)

	default:
		p.warnings.unsupported("provider-defined tool "+tool.ID, "")

		return nil, nil
	}
}

func builtin(toolType string, name string) requestTool {
	return &builtinTool{
		Type: toolType,
		Name: name,
	}
}

func computerToolOf(tool model.ProviderTool, toolType string) (requestTool, error) {
	args, err := decodeToolArgs[ComputerArgs](tool)
	if err != nil {
		return nil, err
	}

	return &computerTool{
		Name:            nameComputer,
		Type:            toolType,
		DisplayWidthPx:  args.DisplayWidthPx,
		DisplayHeightPx: args.DisplayHeightPx,
		DisplayNumber:   args.DisplayNumber,
	}, nil
}

func zoomComputerToolOf(tool model.ProviderTool) (requestTool, error) {
	args, err := decodeToolArgs[Computer20251124Args](tool)
	if err != nil {
		return nil, err
	}

	return &computerTool{
		Name:            nameComputer,
		Type:            "computer_20251124",
		DisplayWidthPx:  args.DisplayWidthPx,
		DisplayHeightPx: args.DisplayHeightPx,
		DisplayNumber:   args.DisplayNumber,
		EnableZoom:      args.EnableZoom,
	}, nil
}

func computerToolsetOf(tool model.ProviderTool) (requestTool, error) {
	args, err := decodeToolArgs[ComputerToolsetArgs](tool)
	if err != nil {
		return nil, err
	}

	converted := &computerToolsetTool{
		Type: "computer_toolset_20260801",
	}

	if args.Configs == nil {
		return converted, nil
	}

	converted.Configs = make(map[ComputerAction]memberConfig, len(args.Configs))
	for member, config := range args.Configs {
		if !slices.Contains(computerActions, member) {
			return nil, fmt.Errorf("%w: computer toolset has no member %q", model.ErrInvalidArgument, member)
		}

		converted.Configs[member] = memberConfig(config)
	}

	return converted, nil
}

func textEditorToolOf(tool model.ProviderTool) (requestTool, error) {
	args, err := decodeToolArgs[TextEditor20250728Args](tool)
	if err != nil {
		return nil, err
	}

	return &textEditorTool{
		Name:          nameStrReplaceBased,
		Type:          "text_editor_20250728",
		MaxCharacters: args.MaxCharacters,
	}, nil
}

func webFetchToolOf(tool model.ProviderTool, toolType string) (requestTool, error) {
	args, err := decodeToolArgs[WebFetchArgs](tool)
	if err != nil {
		return nil, err
	}

	return &webFetchTool{
		Type:             toolType,
		Name:             nameWebFetch,
		MaxUses:          args.MaxUses,
		AllowedDomains:   args.AllowedDomains,
		BlockedDomains:   args.BlockedDomains,
		Citations:        args.Citations,
		MaxContentTokens: args.MaxContentTokens,
	}, nil
}

func latestWebFetchToolOf(tool model.ProviderTool) (requestTool, error) {
	args, err := decodeToolArgs[WebFetch20260318Args](tool)
	if err != nil {
		return nil, err
	}

	err = checkEnum("responseInclusion", args.ResponseInclusion, ResponseInclusionFull, ResponseInclusionExcluded)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", model.ErrInvalidArgument, err)
	}

	return &webFetchTool{
		Type:              "web_fetch_20260318",
		Name:              nameWebFetch,
		MaxUses:           args.MaxUses,
		AllowedDomains:    args.AllowedDomains,
		BlockedDomains:    args.BlockedDomains,
		Citations:         args.Citations,
		MaxContentTokens:  args.MaxContentTokens,
		UseCache:          args.UseCache,
		ResponseInclusion: args.ResponseInclusion,
	}, nil
}

func webSearchToolOf(tool model.ProviderTool, toolType string) (requestTool, error) {
	args, err := decodeToolArgs[WebSearchArgs](tool)
	if err != nil {
		return nil, err
	}

	if err := validateUserLocation(args.UserLocation); err != nil {
		return nil, err
	}

	return &webSearchTool{
		Type:           toolType,
		Name:           nameWebSearch,
		MaxUses:        args.MaxUses,
		AllowedDomains: args.AllowedDomains,
		BlockedDomains: args.BlockedDomains,
		UserLocation:   args.UserLocation,
	}, nil
}

func latestWebSearchToolOf(tool model.ProviderTool) (requestTool, error) {
	args, err := decodeToolArgs[WebSearch20260318Args](tool)
	if err != nil {
		return nil, err
	}

	if err := validateUserLocation(args.UserLocation); err != nil {
		return nil, err
	}

	err = checkEnum("responseInclusion", args.ResponseInclusion, ResponseInclusionFull, ResponseInclusionExcluded)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", model.ErrInvalidArgument, err)
	}

	return &webSearchTool{
		Type:              "web_search_20260318",
		Name:              nameWebSearch,
		MaxUses:           args.MaxUses,
		AllowedDomains:    args.AllowedDomains,
		BlockedDomains:    args.BlockedDomains,
		UserLocation:      args.UserLocation,
		ResponseInclusion: args.ResponseInclusion,
	}, nil
}

func validateUserLocation(location *UserLocation) error {
	if location == nil || location.Type == "approximate" {
		return nil
	}

	return fmt.Errorf("%w: web search user location type must be \"approximate\", got %q", model.ErrInvalidArgument, location.Type)
}

func advisorToolOf(tool model.ProviderTool) (requestTool, error) {
	args, err := decodeToolArgs[AdvisorArgs](tool)
	if err != nil {
		return nil, err
	}

	if args.MaxTokens != nil && *args.MaxTokens < 1024 {
		return nil, fmt.Errorf("%w: advisor maxTokens must be at least 1024, got %d", model.ErrInvalidArgument, *args.MaxTokens)
	}

	if err := validateAdvisorCaching(args.Caching); err != nil {
		return nil, fmt.Errorf("%w: %w", model.ErrInvalidArgument, err)
	}

	return &advisorTool{
		Type:      "advisor_20260301",
		Name:      nameAdvisor,
		Model:     args.Model,
		MaxUses:   args.MaxUses,
		MaxTokens: args.MaxTokens,
		Caching:   args.Caching,
	}, nil
}

func validateAdvisorCaching(caching *AdvisorCaching) error {
	if caching == nil {
		return nil
	}

	if err := checkRequiredEnum("advisor caching type", caching.Type, CacheEphemeral); err != nil {
		return err
	}

	return checkRequiredEnum("advisor caching ttl", caching.TTL, CacheTTL5Minutes, CacheTTL1Hour)
}

// chooseTool converts the tool choice. Models or thinking modes that reject
// forced tool use fall back to auto with a warning; a named tool is then the
// only tool sent.
func (p *preparedTools) chooseTool(preparation toolPreparation) error {
	disableParallel := preparation.disableParallelToolUse
	choice := preparation.toolChoice

	if choice == nil {
		if optional.Or(disableParallel, false) {
			p.toolChoice = &toolChoice{
				Type:                   "auto",
				DisableParallelToolUse: disableParallel,
			}
		}

		return nil
	}

	switch choice.Type {
	case model.ToolChoiceAuto:
		p.toolChoice = autoChoice(disableParallel)

	case model.ToolChoiceRequired:
		if preparation.rejectsForcedToolUse {
			p.warnings.unsupported(
				"toolChoice",
				"toolChoice 'required' is not supported with this model or thinking configuration. Using 'auto' instead. Instruct the model to use a tool in the prompt and verify that a tool call was made.",
			)
			p.toolChoice = autoChoice(disableParallel)

			return nil
		}

		p.toolChoice = &toolChoice{
			Type:                   "any",
			DisableParallelToolUse: disableParallel,
		}

	case model.ToolChoiceNone:
		p.toolChoice = &toolChoice{
			Type: "none",
		}

	case model.ToolChoiceTool:
		if preparation.rejectsForcedToolUse {
			p.warnings.unsupported(
				"toolChoice",
				fmt.Sprintf(
					"toolChoice 'tool' is not supported with this model or thinking configuration. Only the '%s' tool is sent with 'auto' tool choice. Instruct the model to use the tool in the prompt and verify that a tool call was made.",
					choice.ToolName,
				),
			)
			p.tools = onlyTool(p.tools, choice.ToolName)
			p.toolChoice = autoChoice(disableParallel)

			return nil
		}

		p.toolChoice = &toolChoice{
			Type:                   "tool",
			Name:                   choice.ToolName,
			DisableParallelToolUse: disableParallel,
		}

	default:
		return fmt.Errorf("%w: tool choice type %q", model.ErrUnsupported, choice.Type)
	}

	return nil
}

func autoChoice(disableParallel *bool) *toolChoice {
	return &toolChoice{
		Type:                   "auto",
		DisableParallelToolUse: disableParallel,
	}
}

func onlyTool(tools []requestTool, name string) []requestTool {
	kept := []requestTool{}
	for _, tool := range tools {
		if tool.toolName() == name {
			kept = append(kept, tool)
		}
	}

	return kept
}
