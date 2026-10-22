package anthropic

import (
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/internal/optional"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/internal/toolnames"
	"github.com/aarock1234/ai/pkg/model"
)

const maxTemperature = 1.0

// preparedRequest is a Messages API request ready to send, with everything
// needed to interpret its response.
type preparedRequest struct {
	body     messagesRequest
	warnings []model.Warning
	betas    *betaSet

	// usesJSONResponseTool reports that JSON output arrives as a call of
	// the synthetic "json" tool.
	usesJSONResponseTool bool

	toolNames toolnames.Mapping

	// customNamespace is the provider's own options namespace when the
	// call used it; metadata is then duplicated under it.
	customNamespace string
}

// prepareRequest converts call options into a request body, collecting
// warnings and the beta features the request needs. headerBetas are the
// betas already present in configured or call headers.
func (m *MessagesModel) prepareRequest(options model.CallOptions, stream bool, headerBetas []string) (*preparedRequest, error) {
	settings, err := m.resolveRequestSettings(options)
	if err != nil {
		return nil, err
	}

	warnings := settings.warnings
	providerOptions := settings.providerOptions

	cacheWarnings := &callWarnings{}
	cache := newCacheControlValidator(settings.namespace, cacheWarnings)
	toolNames := toolnames.New(options.Tools, providerToolNames)

	prompt, err := convertPrompt(promptConversion{
		prompt:        options.Prompt,
		sendReasoning: optional.Or(providerOptions.SendReasoning, true),
		warnings:      warnings,
		cache:         cache,
		toolNames:     toolNames,
		toolsets:      toolsetNames(options.Tools),

		convertMediaPart: m.provider.host.convertMediaPart,
	})
	if err != nil {
		return nil, err
	}

	policy := thinkingPolicy{
		modelID:      m.modelID,
		capabilities: settings.capabilities,
		warnings:     warnings,
	}
	policy.applyReasoning(&providerOptions, options.Reasoning)
	policy.normalize(&providerOptions)

	thinking := resolveThinking(providerOptions.Thinking, warnings)

	inputs := requestInputs{
		options:         options,
		providerOptions: providerOptions,
		capabilities:    settings.capabilities,
		sampling:        settings.sampling,
		thinking:        thinking,
		output:          settings.output,
		prompt:          prompt,
		warnings:        warnings,
	}

	body, err := m.requestBody(inputs)
	if err != nil {
		return nil, err
	}

	betas := prompt.betas
	addOptionBetas(betas, providerOptions, thinking, options.Tools, warnings)

	tools, err := prepareTools(toolPreparationOf(inputs, cache, stream))
	if err != nil {
		return nil, err
	}

	if cache.err != nil {
		return nil, fmt.Errorf("%w: cache control: %w", model.ErrInvalidArgument, cache.err)
	}

	body.Tools = tools.tools
	body.ToolChoice = tools.toolChoice
	body.Stream = stream

	betas.add(tools.betas.names...)
	betas.add(headerBetas...)
	betas.add(providerOptions.AnthropicBeta...)

	prepared := &preparedRequest{
		body:                 body,
		warnings:             slices.Concat(warnings.list, tools.warnings.list, cacheWarnings.list),
		betas:                betas,
		usesJSONResponseTool: settings.output.jsonResponseTool != nil,
		toolNames:            toolNames,
	}

	if settings.usedCustomNamespace && settings.namespace != Namespace {
		prepared.customNamespace = settings.namespace
	}

	return prepared, nil
}

// requestSettings are the validated provider options of a call and the
// sampling and output settings that the model accepts.
type requestSettings struct {
	warnings *callWarnings

	// namespace is the provider's own options namespace;
	// usedCustomNamespace reports that the call used it.
	namespace           string
	usedCustomNamespace bool

	providerOptions MessagesOptions
	capabilities    modelCapabilities
	sampling        samplingSettings
	output          structuredOutput
}

// resolveRequestSettings reads the provider options and restricts the
// sampling and structured output settings to what the model supports.
func (m *MessagesModel) resolveRequestSettings(options model.CallOptions) (requestSettings, error) {
	warnings := &callWarnings{}
	warnUnsupportedSettings(options, warnings)

	requested := samplingSettings{
		temperature: m.clampTemperature(options.Temperature, warnings),
		topK:        options.TopK,
		topP:        options.TopP,
	}

	if isJSONWithoutSchema(options.ResponseFormat) {
		warnings.unsupported("responseFormat", "JSON response format requires a schema. The response format is ignored.")
	}

	namespace := m.provider.optionsNamespace()

	providerOptions, usedCustomNamespace, err := readMessagesOptions(options.ProviderOptions, namespace)
	if err != nil {
		return requestSettings{}, err
	}

	capabilities := m.provider.host.restrict(capabilitiesOf(m.modelID))
	if !capabilities.isKnownModel && options.MaxOutputTokens == nil {
		warnings.compatibility(
			"maxOutputTokens",
			fmt.Sprintf(
				"The model %q is unknown. The max output tokens have been limited to %d. Set maxOutputTokens explicitly to override this limit.",
				m.modelID,
				capabilities.maxOutputTokens,
			),
		)
	}

	return requestSettings{
		warnings:            warnings,
		namespace:           namespace,
		usedCustomNamespace: usedCustomNamespace,
		providerOptions:     providerOptions,
		capabilities:        capabilities,
		sampling:            m.restrictSampling(requested, capabilities, warnings),
		output:              m.resolveStructuredOutput(options.ResponseFormat, providerOptions, capabilities, warnings),
	}, nil
}

// readMessagesOptions reads the options under "anthropic" and, for a
// renamed provider, under its own namespace, which overrides individual
// keys. It reports whether the provider's own namespace was present.
func readMessagesOptions(values model.ProviderOptions, namespace string) (MessagesOptions, bool, error) {
	merged, _, err := providerObject(values, Namespace)
	if err != nil {
		return MessagesOptions{}, false, err
	}

	usedCustomNamespace := false
	if namespace != Namespace {
		custom, found, err := providerObject(values, namespace)
		if err != nil {
			return MessagesOptions{}, false, err
		}

		usedCustomNamespace = found
		merged = merged.Merge(custom)
	}

	encoded, err := merged.Encode()
	if err != nil {
		return MessagesOptions{}, false, fmt.Errorf("%w: anthropic provider options: %w", model.ErrInvalidArgument, err)
	}

	var options MessagesOptions
	if err := json.Unmarshal(encoded, &options); err != nil {
		return MessagesOptions{}, false, fmt.Errorf("%w: anthropic provider options: %w", model.ErrInvalidArgument, err)
	}

	if err := options.validate(); err != nil {
		return MessagesOptions{}, false, err
	}

	return options, usedCustomNamespace, nil
}

// providerObject returns the options stored under namespace as a JSON
// object.
func providerObject(values model.ProviderOptions, namespace string) (jsonobject.Object, bool, error) {
	raw, found, err := model.ProviderValue[json.RawMessage](values, namespace)
	if err != nil || !found {
		return nil, found, err
	}

	object, err := jsonobject.Parse(raw)
	if err != nil {
		return nil, true, fmt.Errorf("%w: %s provider options: %w", model.ErrInvalidArgument, namespace, err)
	}

	return object, true, nil
}

func warnUnsupportedSettings(options model.CallOptions, warnings *callWarnings) {
	if options.FrequencyPenalty != nil {
		warnings.unsupported("frequencyPenalty", "")
	}

	if options.PresencePenalty != nil {
		warnings.unsupported("presencePenalty", "")
	}

	if options.Seed != nil {
		warnings.unsupported("seed", "")
	}
}

// samplingSettings holds the sampling settings that survive model restrictions.
type samplingSettings struct {
	temperature *float64
	topK        *int
	topP        *float64
}

// restrictSampling drops the sampling settings of models that reject them.
func (m *MessagesModel) restrictSampling(resolved samplingSettings, capabilities modelCapabilities, warnings *callWarnings) samplingSettings {
	if !capabilities.rejectsSamplingParameters {
		return resolved
	}

	if resolved.temperature != nil {
		warnings.unsupported("temperature", fmt.Sprintf("temperature is not supported by %s and will be ignored", m.modelID))
		resolved.temperature = nil
	}

	if resolved.topK != nil {
		warnings.unsupported("topK", fmt.Sprintf("topK is not supported by %s and will be ignored", m.modelID))
		resolved.topK = nil
	}

	if resolved.topP != nil {
		warnings.unsupported("topP", fmt.Sprintf("topP is not supported by %s and will be ignored", m.modelID))
		resolved.topP = nil
	}

	return resolved
}

func isJSONWithoutSchema(format *model.ResponseFormat) bool {
	return isJSONFormat(format) && len(format.Schema) == 0
}

func hasJSONSchema(format *model.ResponseFormat) bool {
	return isJSONFormat(format) && len(format.Schema) > 0
}

func isJSONFormat(format *model.ResponseFormat) bool {
	return format != nil && format.Type == model.ResponseFormatJSON
}

// clampTemperature keeps the temperature between 0 and the maximum of the
// API. A host's maximum is reported under the provider's options namespace.
func (m *MessagesModel) clampTemperature(temperature *float64, warnings *callWarnings) *float64 {
	if temperature == nil {
		return nil
	}

	maximum := maxTemperature
	provider := Namespace
	if m.provider.host.maxTemperature != nil {
		maximum = *m.provider.host.maxTemperature
		provider = m.provider.optionsNamespace()
	}

	if *temperature > maximum {
		formattedMaximum := strconv.FormatFloat(maximum, 'f', 1, 64)

		warnings.unsupported(
			"temperature",
			fmt.Sprintf(
				"%s exceeds %s maximum of %s. clamped to %s",
				formatNumber(*temperature),
				provider,
				formattedMaximum,
				formattedMaximum,
			),
		)

		return new(maximum)
	}

	if *temperature < 0 {
		warnings.unsupported(
			"temperature",
			fmt.Sprintf("%s is below %s minimum of 0. clamped to 0", formatNumber(*temperature), provider),
		)

		return new(0.0)
	}

	return temperature
}

// structuredOutput is the resolved strategy for JSON output.
type structuredOutput struct {
	// nativeSchema is the schema for output_config.format, when native
	// structured output is used.
	nativeSchema json.RawMessage

	// jsonResponseTool is the synthetic tool that carries JSON output when
	// native structured output is not used.
	jsonResponseTool *model.FunctionTool
}

func (m *MessagesModel) resolveStructuredOutput(
	format *model.ResponseFormat,
	providerOptions MessagesOptions,
	capabilities modelCapabilities,
	warnings *callWarnings,
) structuredOutput {
	resolved := structuredOutput{}

	if !hasJSONSchema(format) {
		return resolved
	}

	mode := providerOptions.StructuredOutputMode
	if mode == "" {
		mode = StructuredOutputAuto
	}

	useNative := mode == StructuredOutputFormat || (mode == StructuredOutputAuto && capabilities.supportsStructuredOutput)

	// The JSON tool relies on forced tool use, which some models reject.
	canFallBackToNative := capabilities.rejectsForcedToolUse && capabilities.supportsStructuredOutput
	if !useNative && canFallBackToNative {
		warnings.unsupported(
			"providerOptions.anthropic.structuredOutputMode",
			fmt.Sprintf(
				"structuredOutputMode 'jsonTool' is not supported by %s because it rejects forced tool use. Using 'outputFormat' instead.",
				m.modelID,
			),
		)

		useNative = true
	}

	if useNative {
		resolved.nativeSchema = sanitizeJSONSchema(format.Schema)

		return resolved
	}

	resolved.jsonResponseTool = &model.FunctionTool{
		Name:        jsonToolName,
		Description: "Respond with a JSON object.",
		InputSchema: format.Schema,
	}

	disableParallel := providerOptions.DisableParallelToolUse
	if disableParallel != nil && !*disableParallel {
		warnings.unsupported(
			"providerOptions.anthropic.disableParallelToolUse",
			"`disableParallelToolUse: false` is ignored when using the JSON response tool. Parallel tool use is disabled to ensure a single coherent JSON tool call.",
		)
	}

	return resolved
}

// requestInputs are the resolved settings a request body is built from.
type requestInputs struct {
	options         model.CallOptions
	providerOptions MessagesOptions
	capabilities    modelCapabilities
	sampling        samplingSettings
	thinking        thinkingRequest
	output          structuredOutput
	prompt          *convertedPrompt
	warnings        *callWarnings
}

func (m *MessagesModel) requestBody(inputs requestInputs) (messagesRequest, error) {
	providerOptions := inputs.providerOptions

	container, err := requestContainerOf(providerOptions.Container)
	if err != nil {
		return messagesRequest{}, err
	}

	maxTokens := optional.Or(inputs.options.MaxOutputTokens, inputs.capabilities.maxOutputTokens)

	body := messagesRequest{
		Model:             m.modelID,
		MaxTokens:         maxTokens,
		Temperature:       inputs.sampling.temperature,
		TopK:              inputs.sampling.topK,
		TopP:              inputs.sampling.topP,
		StopSequences:     inputs.options.StopSequences,
		Thinking:          inputs.thinking.config,
		OutputConfig:      requestOutputConfig(providerOptions, inputs.output.nativeSchema),
		Speed:             providerOptions.Speed,
		ServiceTier:       providerOptions.ServiceTier,
		InferenceGeo:      providerOptions.InferenceGeo,
		CacheControl:      providerOptions.CacheControl,
		Metadata:          requestMetadataOf(providerOptions.Metadata),
		MCPServers:        mcpServersOf(providerOptions.MCPServers),
		Container:         container,
		System:            inputs.prompt.system,
		Messages:          inputs.prompt.messages,
		Safeguards:        safeguardsOf(providerOptions.Safeguards),
		Compaction:        compactionOf(providerOptions.Compaction),
		ContextManagement: contextManagementOf(providerOptions.ContextManagement),
	}

	if providerOptions.Fallbacks != nil && providerOptions.Fallbacks.isSet() {
		body.Fallbacks = providerOptions.Fallbacks
	}

	if inputs.thinking.isThinking {
		m.removeSamplingWithThinking(&body, inputs.warnings)
		body.MaxTokens += inputs.thinking.budget
	} else {
		m.removeTopPWithTemperature(&body, inputs)
	}

	m.limitMaxTokens(&body, inputs)

	return body, nil
}

// removeSamplingWithThinking removes the sampling settings that thinking
// does not accept. Hosts that accept temperature and topP with thinking
// keep them.
func (m *MessagesModel) removeSamplingWithThinking(body *messagesRequest, warnings *callWarnings) {
	keepTemperatureAndTopP := m.provider.host.keepSamplingWithThinking

	if body.Temperature != nil && !keepTemperatureAndTopP {
		body.Temperature = nil
		warnings.unsupported("temperature", "temperature is not supported when thinking is enabled")
	}

	if body.TopK != nil {
		body.TopK = nil
		warnings.unsupported("topK", "topK is not supported when thinking is enabled")
	}

	if body.TopP != nil && !keepTemperatureAndTopP {
		body.TopP = nil
		warnings.unsupported("topP", "topP is not supported when thinking is enabled")
	}
}

// removeTopPWithTemperature keeps only the temperature of Claude models,
// which reject both. Other models behind Anthropic-compatible APIs may
// require both, so their settings are kept.
func (m *MessagesModel) removeTopPWithTemperature(body *messagesRequest, inputs requestInputs) {
	if !m.isAnthropicModel(inputs.capabilities) {
		return
	}

	if body.TopP == nil || body.Temperature == nil {
		return
	}

	inputs.warnings.unsupported("topP", "topP is not supported when temperature is set. topP is ignored.")
	body.TopP = nil
}

// limitMaxTokens keeps max tokens within the limit of known models, so that
// switching models does not break calls.
func (m *MessagesModel) limitMaxTokens(body *messagesRequest, inputs requestInputs) {
	limit := inputs.capabilities.maxOutputTokens
	if !inputs.capabilities.isKnownModel || body.MaxTokens <= limit {
		return
	}

	if inputs.options.MaxOutputTokens != nil {
		warnings := inputs.warnings
		warnings.unsupported(
			"maxOutputTokens",
			fmt.Sprintf(
				"%d (maxOutputTokens + thinkingBudget) is greater than %s %d max output tokens. The max output tokens have been limited to %d.",
				body.MaxTokens,
				m.modelID,
				limit,
				limit,
			),
		)
	}

	body.MaxTokens = limit
}

func (m *MessagesModel) isAnthropicModel(capabilities modelCapabilities) bool {
	return capabilities.isKnownModel || strings.Contains(m.modelID, "claude-")
}

func requestOutputConfig(options MessagesOptions, nativeSchema json.RawMessage) *outputConfig {
	if options.Effort == "" && options.TaskBudget == nil && nativeSchema == nil {
		return nil
	}

	config := &outputConfig{
		Effort:     options.Effort,
		TaskBudget: options.TaskBudget,
	}

	if nativeSchema != nil {
		config.Format = &outputFormat{
			Type:   "json_schema",
			Schema: nativeSchema,
		}
	}

	return config
}

func requestMetadataOf(metadata *RequestMetadata) *requestMetadata {
	if metadata == nil || metadata.UserID == nil {
		return nil
	}

	return &requestMetadata{
		UserID: *metadata.UserID,
	}
}

func mcpServersOf(servers []MCPServer) []mcpServer {
	if len(servers) == 0 {
		return nil
	}

	converted := make([]mcpServer, 0, len(servers))
	for _, server := range servers {
		wire := mcpServer{
			Type:               server.Type,
			Name:               server.Name,
			URL:                server.URL,
			AuthorizationToken: server.AuthorizationToken,
		}

		if server.ToolConfiguration != nil {
			wire.ToolConfiguration = &mcpToolConfiguration{
				AllowedTools: server.ToolConfiguration.AllowedTools,
				Enabled:      server.ToolConfiguration.Enabled,
			}
		}

		converted = append(converted, wire)
	}

	return converted
}

// requestContainerOf returns the container object when skills are loaded,
// and otherwise the container ID alone.
func requestContainerOf(container *ContainerOptions) (*requestContainer, error) {
	if container == nil {
		return nil, nil
	}

	if len(container.Skills) == 0 {
		if container.ID == "" {
			return nil, nil
		}

		return &requestContainer{
			ID: container.ID,
		}, nil
	}

	skills := make([]containerSkill, 0, len(container.Skills))
	for _, skill := range container.Skills {
		skillID, err := resolveSkillID(skill)
		if err != nil {
			return nil, err
		}

		skills = append(skills, containerSkill{
			Type:    skill.Type,
			SkillID: skillID,
			Version: skill.Version,
		})
	}

	return &requestContainer{
		ID:     container.ID,
		Skills: skills,
	}, nil
}

func resolveSkillID(skill Skill) (string, error) {
	if skill.Type != SkillCustom {
		return skill.SkillID, nil
	}

	skillID, found := skill.ProviderReference[Namespace]
	if !found {
		return "", fmt.Errorf("%w: custom skill has no anthropic provider reference", model.ErrInvalidArgument)
	}

	return skillID, nil
}

func safeguardsOf(safeguards []Safeguard) []safeguard {
	if len(safeguards) == 0 {
		return nil
	}

	converted := make([]safeguard, 0, len(safeguards))
	for _, entry := range safeguards {
		converted = append(converted, safeguard(entry))
	}

	return converted
}

func compactionOf(compaction *Compaction) *compactionRequest {
	if compaction == nil {
		return nil
	}

	return &compactionRequest{
		Type:         compaction.Type,
		Instructions: compaction.Instructions,
	}
}

// contextManagementOf converts context edits, keeping only the fields that
// belong to each strategy.
func contextManagementOf(management *ContextManagement) *contextManagementWire {
	if management == nil {
		return nil
	}

	edits := make([]contextEditWire, 0, len(management.Edits))
	for _, edit := range management.Edits {
		wire := contextEditWire{
			Type: edit.Type,
		}

		switch edit.Type {
		case ContextEditClearToolUses:
			wire.Trigger = edit.Trigger
			wire.Keep = edit.Keep
			wire.ClearAtLeast = edit.ClearAtLeast
			wire.ClearToolInputs = edit.ClearToolInputs
			wire.ExcludeTools = edit.ExcludeTools

		case ContextEditClearThinking:
			wire.Keep = edit.Keep

		case ContextEditCompact:
			wire.Trigger = edit.Trigger
			wire.PauseAfterCompaction = edit.PauseAfterCompaction
			wire.Instructions = edit.Instructions
		}

		edits = append(edits, wire)
	}

	return &contextManagementWire{
		Edits: edits,
	}
}

// addOptionBetas adds the beta features that provider options require.
func addOptionBetas(
	betas *betaSet,
	options MessagesOptions,
	thinking thinkingRequest,
	tools []model.Tool,
	warnings *callWarnings,
) {
	if len(options.MCPServers) > 0 {
		betas.add("mcp-client-2025-04-04")
	}

	if len(options.Safeguards) > 0 {
		betas.add("dangerous-tool-use-2026-09-03")
	}

	if options.Compaction != nil {
		betas.add("compact-2026-09-04")
	}

	if options.ContextManagement != nil {
		betas.add("context-management-2025-06-27")

		if hasCompactEdit(options.ContextManagement) {
			betas.add("compact-2026-01-12")
		}
	}

	if options.Container != nil && len(options.Container.Skills) > 0 {
		betas.add("code-execution-2025-08-25", "skills-2025-10-02", "files-api-2025-04-14")

		if !hasSkillsCodeExecution(tools) {
			warnings.other("code execution tool is required when using skills")
		}
	}

	if options.TaskBudget != nil {
		betas.add("task-budgets-2026-03-13")
	}

	if options.Speed == SpeedFast {
		betas.add("fast-mode-2026-02-01")
	}

	if thinking.display == ThinkingDisplayUpdates {
		betas.add("thinking-display-updates-2026-08-18")
	}

	if thinking.hasBinding {
		betas.add("thinking-binding-controls-2026-08-01")
	}

	if options.Fallbacks != nil && options.Fallbacks.Default {
		betas.add("server-side-fallback-2026-07-01")
	} else if options.Fallbacks != nil && len(options.Fallbacks.Chain) > 0 {
		betas.add("server-side-fallback-2026-06-01")
	}
}

func hasCompactEdit(management *ContextManagement) bool {
	for _, edit := range management.Edits {
		if edit.Type == ContextEditCompact {
			return true
		}
	}

	return false
}

// hasSkillsCodeExecution reports whether the tools include a code execution
// version that can run agent skills.
func hasSkillsCodeExecution(tools []model.Tool) bool {
	for _, tool := range tools {
		providerTool, ok := tool.(model.ProviderTool)
		if !ok {
			continue
		}

		if providerTool.ID == toolCodeExecution20250825 || providerTool.ID == toolCodeExecution20260120 {
			return true
		}
	}

	return false
}

// toolsetNames maps the application names of toolset tools to their API
// toolset names.
func toolsetNames(tools []model.Tool) map[string]string {
	names := make(map[string]string)

	for _, tool := range tools {
		providerTool, ok := tool.(model.ProviderTool)
		if ok && providerTool.ID == toolComputerToolset20260801 {
			names[providerTool.Name] = nameComputer
		}
	}

	return names
}

// toolPreparationOf returns the tool preparation of a request. The JSON
// response tool is added as the only required tool choice, without parallel
// calls.
func toolPreparationOf(inputs requestInputs, cache *cacheControlValidator, stream bool) toolPreparation {
	options := inputs.options
	providerOptions := inputs.providerOptions
	capabilities := inputs.capabilities
	output := inputs.output

	preparation := toolPreparation{
		tools:                      options.Tools,
		toolChoice:                 options.ToolChoice,
		disableParallelToolUse:     providerOptions.DisableParallelToolUse,
		cache:                      cache,
		supportsStructuredOutput:   capabilities.supportsStructuredOutput,
		supportsStrictTools:        capabilities.supportsStrictTools,
		defaultEagerInputStreaming: stream && optional.Or(providerOptions.ToolStreaming, true),
		rejectsForcedToolUse:       capabilities.rejectsForcedToolUse || thinkingTypeOf(providerOptions.Thinking) == ThinkingEnabled,
	}

	if output.jsonResponseTool == nil {
		return preparation
	}

	preparation.tools = append(append([]model.Tool{}, options.Tools...), *output.jsonResponseTool)
	preparation.toolChoice = &model.ToolChoice{
		Type: model.ToolChoiceRequired,
	}
	preparation.disableParallelToolUse = new(true)
	preparation.supportsStructuredOutput = false

	return preparation
}

// callWarnings collects the warnings of one call in order.
type callWarnings struct {
	list []model.Warning
}

func (w *callWarnings) add(warnings ...model.Warning) {
	w.list = append(w.list, warnings...)
}

func (w *callWarnings) unsupported(feature string, details string) {
	w.list = append(w.list, model.UnsupportedWarning(feature, details))
}

func (w *callWarnings) compatibility(feature string, details string) {
	w.list = append(w.list, model.CompatibilityWarning(feature, details))
}

func (w *callWarnings) other(message string) {
	w.list = append(w.list, model.Warning{
		Type:    model.WarningOther,
		Message: message,
	})
}

// betaSet is an ordered set of beta feature names.
type betaSet struct {
	names []string
	seen  map[string]bool
}

func newBetaSet() *betaSet {
	return &betaSet{
		seen: make(map[string]bool),
	}
}

func (s *betaSet) add(names ...string) {
	for _, name := range names {
		if s.seen[name] {
			continue
		}

		s.seen[name] = true
		s.names = append(s.names, name)
	}
}

func (s *betaSet) header() string {
	return strings.Join(s.names, ",")
}

// formatNumber formats a number the way JavaScript prints it.
func formatNumber(value float64) string {
	return strconv.FormatFloat(value, 'f', -1, 64)
}
