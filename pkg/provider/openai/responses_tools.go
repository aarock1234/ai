package openai

import (
	"fmt"
	"strings"

	"github.com/aarock1234/ai/internal/optional"

	"github.com/aarock1234/ai/internal/toolnames"
	"github.com/aarock1234/ai/pkg/model"
)

// responsesAllowedToolResolution is a tool that tool_choice.allowed_tools
// can name, or the reason it cannot.
type responsesAllowedToolResolution struct {
	entry             responsesAllowedTool
	unsupportedReason string
}

func (r responsesAllowedToolResolution) supported() bool {
	return r.unsupportedReason == ""
}

func (r responsesAllowedToolResolution) same(other responsesAllowedToolResolution) bool {
	if r.supported() && other.supported() {
		return r.entry.key() == other.entry.key()
	}

	return r.unsupportedReason == other.unsupportedReason
}

// responsesAllowedToolAlias is the resolution reached through a provider
// tool name, which several tools may share.
type responsesAllowedToolAlias struct {
	resolution responsesAllowedToolResolution
	ambiguous  bool
}

// responsesToolsRequest is the input of tool preparation.
type responsesToolsRequest struct {
	tools                    []model.Tool
	toolChoice               *model.ToolChoice
	allowedTools             *ResponsesAllowedTools
	toolNames                toolnames.Mapping
	supportsAsyncToolCalling bool
}

// responsesPreparedTools is the result of tool preparation.
type responsesPreparedTools struct {
	tools      []responsesTool
	toolChoice *responsesToolChoice
	warnings   []model.Warning

	// customToolNames holds the custom tools, whose calls and results use
	// custom tool items.
	customToolNames map[string]bool

	// outputSchemaToolNames holds the functions with an output schema,
	// whose results must be valid JSON.
	outputSchemaToolNames map[string]bool

	// shellProviderExecuted reports that the shell tool runs in a container.
	shellProviderExecuted bool
}

// responsesToolPreparer converts call tools into API tools.
type responsesToolPreparer struct {
	request  responsesToolsRequest
	prepared responsesPreparedTools

	namespaces  map[string]*responsesNamespaceTool
	resolutions map[string]responsesAllowedToolResolution
	aliases     map[string]*responsesAllowedToolAlias
	shellSeen   bool
}

// prepareResponsesTools converts the call's tools and tool choice.
func prepareResponsesTools(request responsesToolsRequest) (responsesPreparedTools, error) {
	p := &responsesToolPreparer{
		request: request,
		prepared: responsesPreparedTools{
			customToolNames:       make(map[string]bool),
			outputSchemaToolNames: make(map[string]bool),
		},
		namespaces:  make(map[string]*responsesNamespaceTool),
		resolutions: make(map[string]responsesAllowedToolResolution),
		aliases:     make(map[string]*responsesAllowedToolAlias),
	}

	if len(request.tools) == 0 {
		return p.prepared, nil
	}

	for _, tool := range request.tools {
		if err := p.add(tool); err != nil {
			return responsesPreparedTools{}, err
		}
	}

	if request.allowedTools != nil {
		choice, err := p.allowedToolsChoice(*request.allowedTools)
		if err != nil {
			return responsesPreparedTools{}, err
		}

		p.prepared.toolChoice = choice

		return p.prepared, nil
	}

	p.prepared.toolChoice = p.toolChoice()

	return p.prepared, nil
}

func (p *responsesToolPreparer) add(tool model.Tool) error {
	switch tool := tool.(type) {
	case model.FunctionTool:
		return p.addFunction(tool)
	case model.ProviderTool:
		return p.addProviderTool(tool)
	default:
		p.prepared.warnings = append(p.prepared.warnings, model.UnsupportedWarning(
			fmt.Sprintf("tool %s", tool.ToolName()),
			"",
		))

		return nil
	}
}

// addFunction adds a function tool, possibly inside a namespace.
func (p *responsesToolPreparer) addFunction(tool model.FunctionTool) error {
	options, _, err := model.ProviderValue[ResponsesFunctionToolOptions](tool.ProviderOptions, Namespace)
	if err != nil {
		return err
	}

	if len(options.OutputSchema) > 0 {
		p.prepared.outputSchemaToolNames[tool.Name] = true
	}

	if options.OmitFromRequestTools {
		return p.omitFunction(tool)
	}

	function, err := p.functionTool(tool, options)
	if err != nil {
		return err
	}

	resolution := responsesAllowedToolResolution{
		entry: responsesAllowedTool{
			Type: "function",
			Name: tool.Name,
		},
	}

	switch {
	case options.Namespace != nil:
		resolution.unsupportedReason = "tools inside an OpenAI tool namespace are not visible to tool_choice.allowed_tools"
	case optional.Or(options.DeferLoading, false):
		resolution.unsupportedReason = "deferred tools are not visible to tool_choice.allowed_tools"
	}

	if err := p.placeFunction(function, options.Namespace); err != nil {
		return err
	}

	p.record(tool.Name, resolution, "")

	return nil
}

// omitFunction records a function that is left out of the request tools.
func (p *responsesToolPreparer) omitFunction(tool model.FunctionTool) error {
	p.record(tool.Name, responsesAllowedToolResolution{
		unsupportedReason: "tools omitted from the request tools array cannot be allow-listed",
	}, "")

	choice := p.request.toolChoice
	forcesTool := choice != nil && choice.Type == model.ToolChoiceTool && choice.ToolName == tool.Name
	if p.request.allowedTools == nil && forcesTool {
		return fmt.Errorf(
			"%w: forcing tool %q omitted from the request tools array; use automatic tool choice for history-loaded tools",
			model.ErrUnsupported,
			tool.Name,
		)
	}

	return nil
}

// placeFunction adds a function at the top level or to its namespace.
func (p *responsesToolPreparer) placeFunction(function *responsesFunctionTool, namespace *ResponsesToolNamespace) error {
	if namespace == nil {
		p.prepared.tools = append(p.prepared.tools, function)

		return nil
	}

	group, exists := p.namespaces[namespace.Name]
	if !exists {
		group = &responsesNamespaceTool{
			Type:        "namespace",
			Name:        namespace.Name,
			Description: namespace.Description,
		}
		p.namespaces[namespace.Name] = group
		p.prepared.tools = append(p.prepared.tools, group)
	}

	if group.Description != namespace.Description {
		return fmt.Errorf("%w: conflicting descriptions for OpenAI tool namespace %q", model.ErrUnsupported, namespace.Name)
	}

	group.Tools = append(group.Tools, function)

	return nil
}

// functionTool converts a function tool with normalized schemas.
func (p *responsesToolPreparer) functionTool(tool model.FunctionTool, options ResponsesFunctionToolOptions) (*responsesFunctionTool, error) {
	parameters, warnings, err := normalizeJSONSchema(tool.InputSchema)
	if err != nil {
		return nil, err
	}

	p.prepared.warnings = append(p.prepared.warnings, warnings...)

	function := &responsesFunctionTool{
		Type:           "function",
		Name:           tool.Name,
		Description:    tool.Description,
		Parameters:     parameters,
		Async:          p.asyncOption(options.Async, tool.Name),
		Strict:         optional.Or(tool.Strict, false),
		DeferLoading:   options.DeferLoading,
		AllowedCallers: options.AllowedCallers,
	}

	if len(options.OutputSchema) > 0 {
		outputSchema, warnings, err := normalizeJSONSchema(options.OutputSchema)
		if err != nil {
			return nil, err
		}

		p.prepared.warnings = append(p.prepared.warnings, warnings...)
		function.OutputSchema = outputSchema
	}

	return function, nil
}

// asyncOption drops async on models that do not support it, with a
// warning.
func (p *responsesToolPreparer) asyncOption(async *bool, toolName string) *bool {
	if async == nil || !*async || p.request.supportsAsyncToolCalling {
		return async
	}

	p.prepared.warnings = append(p.prepared.warnings, model.UnsupportedWarning(
		fmt.Sprintf("async tool calling for %q", toolName),
		"Async tool calling is only supported by GPT-6 and later models.",
	))

	return nil
}

// record remembers how tool_choice.allowed_tools can name a tool, by its
// application name and by its provider name.
func (p *responsesToolPreparer) record(toolName string, resolution responsesAllowedToolResolution, providerName string) {
	p.resolutions[toolName] = resolution

	if providerName == "" || providerName == toolName {
		return
	}

	alias, exists := p.aliases[providerName]
	if !exists {
		p.aliases[providerName] = &responsesAllowedToolAlias{
			resolution: resolution,
		}

		return
	}

	if !alias.ambiguous && !alias.resolution.same(resolution) {
		alias.ambiguous = true
	}
}

// allowedToolsChoice builds tool_choice.allowed_tools from tool names.
func (p *responsesToolPreparer) allowedToolsChoice(allowed ResponsesAllowedTools) (*responsesToolChoice, error) {
	var entries []responsesAllowedTool
	var dropped []string

	for _, name := range allowed.ToolNames {
		entry, ok := p.allowedToolEntry(name)
		if !ok {
			dropped = append(dropped, name)

			continue
		}

		entries = append(entries, entry)
	}

	if len(entries) == 0 {
		return nil, fmt.Errorf(
			"%w: allowedTools with only tools that cannot be allow-listed (%s)",
			model.ErrUnsupported,
			strings.Join(dropped, ", "),
		)
	}

	mode := allowed.Mode
	if mode == "" {
		mode = ResponsesAllowedToolsAuto
	}

	return &responsesToolChoice{
		Type:  "allowed_tools",
		Mode:  mode,
		Tools: entries,
	}, nil
}

// allowedToolEntry resolves one allowed tool name. It reports false when
// the tool cannot be allow-listed.
func (p *responsesToolPreparer) allowedToolEntry(name string) (responsesAllowedTool, bool) {
	feature := fmt.Sprintf("allowedTools entry %q", name)

	direct, isDirect := p.resolutions[name]
	alias, isAlias := p.aliases[name]

	if isDirect && isAlias {
		p.warn(feature, "this name is both a tool name and the provider tool name of another tool in this request; the tool with this name is allowed")
	}

	resolution := direct
	if !isDirect && isAlias {
		if alias.ambiguous {
			p.warn(feature, "several tools in this request share this provider tool name; use the tool name from the tools for this request instead")

			return responsesAllowedTool{}, false
		}

		resolution = alias.resolution
	}

	if !isDirect && !isAlias {
		p.warn(feature, "the tool is not part of the tools for this request and is sent as a function tool")

		return responsesAllowedTool{
			Type: "function",
			Name: p.request.toolNames.ToProvider(name),
		}, true
	}

	if !resolution.supported() {
		p.warn(feature, resolution.unsupportedReason+"; the tool is removed from the allowed tools")

		return responsesAllowedTool{}, false
	}

	return resolution.entry, true
}

// toolChoice converts the call's tool choice.
func (p *responsesToolPreparer) toolChoice() *responsesToolChoice {
	choice := p.request.toolChoice
	if choice == nil {
		return nil
	}

	if choice.Type != model.ToolChoiceTool {
		return &responsesToolChoice{
			Keyword: string(choice.Type),
		}
	}

	name := p.request.toolNames.ToProvider(choice.ToolName)

	switch {
	case responsesIsBuiltinChoice(name):
		return &responsesToolChoice{
			Type: name,
		}
	case p.prepared.customToolNames[name]:
		return &responsesToolChoice{
			Type: "custom",
			Name: name,
		}
	default:
		return &responsesToolChoice{
			Type: "function",
			Name: name,
		}
	}
}

func (p *responsesToolPreparer) warn(feature, details string) {
	p.prepared.warnings = append(p.prepared.warnings, model.UnsupportedWarning(feature, details))
}

// responsesIsBuiltinChoice reports whether a forced tool is a built-in tool
// that tool_choice names by type.
func responsesIsBuiltinChoice(name string) bool {
	switch name {
	case "code_interpreter", "file_search", "image_generation", "web_search_preview", "web_search",
		"mcp", "apply_patch", "computer", "programmatic_tool_calling":
		return true
	default:
		return false
	}
}

// responsesAllowedResolution returns how tool_choice.allowed_tools names a
// converted provider tool.
func responsesAllowedResolution(tool responsesTool) responsesAllowedToolResolution {
	switch tool := tool.(type) {
	case responsesCustomTool:
		return responsesAllowedToolResolution{
			entry: responsesAllowedTool{
				Type: "custom",
				Name: tool.Name,
			},
		}
	case responsesMCPTool:
		return responsesAllowedToolResolution{
			entry: responsesAllowedTool{
				Type:        "mcp",
				ServerLabel: tool.ServerLabel,
			},
		}
	}

	switch tool.responsesToolType() {
	case "file_search", "web_search", "web_search_preview", "image_generation", "code_interpreter",
		"computer", "apply_patch", "shell", "local_shell", "programmatic_tool_calling":
		return responsesAllowedToolResolution{
			entry: responsesAllowedTool{
				Type: tool.responsesToolType(),
			},
		}
	default:
		return responsesAllowedToolResolution{
			unsupportedReason: fmt.Sprintf("OpenAI does not support %s tools in tool_choice.allowed_tools", tool.responsesToolType()),
		}
	}
}
