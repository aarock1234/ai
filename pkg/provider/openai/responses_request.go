package openai

import (
	"fmt"
	"slices"
	"strings"

	"github.com/aarock1234/ai/internal/optional"

	"github.com/aarock1234/ai/internal/reasoning"
	"github.com/aarock1234/ai/internal/toolnames"
	"github.com/aarock1234/ai/pkg/model"
)

// responsesProviderToolNames maps provider tool IDs to their wire names.
var responsesProviderToolNames = map[string]string{
	codeInterpreterToolID:         "code_interpreter",
	computerToolID:                "computer",
	fileSearchToolID:              "file_search",
	imageGenerationToolID:         "image_generation",
	localShellToolID:              "local_shell",
	shellToolID:                   "shell",
	webSearchToolID:               "web_search",
	webSearchPreviewToolID:        "web_search_preview",
	mcpToolID:                     "mcp",
	applyPatchToolID:              "apply_patch",
	toolSearchToolID:              "tool_search",
	programmaticToolCallingToolID: "programmatic_tool_calling",
}

// responsesPreparation holds the intermediate decisions of request
// preparation.
type responsesPreparation struct {
	modelID string

	// requestModelID is the model sent in the request body.
	requestModelID string

	// explicitMessageItems adds "type": "message" to message items.
	explicitMessageItems bool

	// omitWebSearchSourcesInclude stops web search sources from being
	// requested.
	omitWebSearchSourcesInclude bool

	options      model.CallOptions
	openai       ResponsesOptions
	namespace    string
	capabilities modelCapabilities
	warnings     []model.Warning

	reasoningEffort  string
	reasoningSummary *string
	isReasoningModel bool

	generateID func() string
}

// prepare converts call options into a request body.
func (m *ResponsesModel) prepare(options model.CallOptions) (*responsesCall, error) {
	namespace := m.config.optionsNamespace

	openaiOptions, err := readResponsesOptions(options.ProviderOptions, namespace)
	if err != nil {
		return nil, err
	}

	p := &responsesPreparation{
		modelID:                     m.modelID,
		requestModelID:              m.config.requestModelID,
		explicitMessageItems:        m.config.explicitMessageItems,
		omitWebSearchSourcesInclude: m.config.omitWebSearchSourcesInclude,
		generateID:                  m.config.generateID,
		options:                     options,
		openai:                      openaiOptions,
		namespace:                   namespace,
		capabilities:                capabilitiesFor(m.modelID),
	}

	p.warnUnsupportedSettings()
	p.resolveReasoning()

	return p.build()
}

// warnUnsupportedSettings reports standard settings the API lacks.
func (p *responsesPreparation) warnUnsupportedSettings() {
	if p.options.TopK != nil {
		p.warn("topK", "")
	}

	if p.options.Seed != nil {
		p.warn("seed", "")
	}

	if p.options.PresencePenalty != nil {
		p.warn("presencePenalty", "")
	}

	if p.options.FrequencyPenalty != nil {
		p.warn("frequencyPenalty", "")
	}

	if p.options.StopSequences != nil {
		p.warn("stopSequences", "")
	}
}

// resolveReasoning picks the reasoning effort and summary.
func (p *responsesPreparation) resolveReasoning() {
	effort := p.openai.ReasoningEffort
	if effort == "" && reasoning.IsCustom(p.options.Reasoning) {
		effort = string(p.options.Reasoning)
	}

	if effort != "" && !p.capabilities.supportsEffort(model.ReasoningEffort(effort)) {
		p.warn("reasoningEffort", p.unsupportedEffortDetails())
		effort = ""
	}

	p.reasoningEffort = effort

	switch {
	case p.openai.ReasoningSummary != nil:
		p.reasoningSummary = p.openai.ReasoningSummary
	case effort != "" && effort != "none":
		p.reasoningSummary = new("detailed")
	}

	p.isReasoningModel = p.capabilities.isReasoningModel
	if p.openai.ForceReasoning != nil {
		p.isReasoningModel = *p.openai.ForceReasoning
	}
}

// build assembles the request body.
func (p *responsesPreparation) build() (*responsesCall, error) {
	if p.openai.Conversation != "" && p.openai.PreviousResponseID != "" {
		p.warn("conversation", "conversation and previousResponseId cannot be used together")
	}

	toolNames := toolnames.New(p.options.Tools, responsesProviderToolNames)

	tools, err := prepareResponsesTools(responsesToolsRequest{
		tools:                    p.options.Tools,
		toolChoice:               p.options.ToolChoice,
		allowedTools:             p.openai.AllowedTools,
		toolNames:                toolNames,
		supportsAsyncToolCalling: p.capabilities.supportsAsyncToolCalling,
	})
	if err != nil {
		return nil, err
	}

	input, err := p.convertInput(toolNames, tools)
	if err != nil {
		return nil, err
	}

	textFormat, err := p.textFormat()
	if err != nil {
		return nil, err
	}

	body := responsesRequest{
		Model:                p.requestModelID,
		Input:                input,
		Temperature:          p.options.Temperature,
		TopP:                 p.options.TopP,
		MaxOutputTokens:      p.options.MaxOutputTokens,
		Text:                 textFormat,
		Conversation:         p.openai.Conversation,
		MaxToolCalls:         p.openai.MaxToolCalls,
		Metadata:             p.openai.Metadata,
		ParallelToolCalls:    p.openai.ParallelToolCalls,
		PreviousResponseID:   p.openai.PreviousResponseID,
		Store:                p.openai.Store,
		User:                 p.openai.User,
		Instructions:         p.openai.Instructions,
		ServiceTier:          p.openai.ServiceTier,
		Include:              p.include(),
		PromptCacheKey:       p.openai.PromptCacheKey,
		PromptCacheOptions:   p.openai.PromptCacheOptions,
		PromptCacheRetention: p.openai.PromptCacheRetention,
		SafetyIdentifier:     p.openai.SafetyIdentifier,
		TopLogprobs:          p.topLogprobs(),
		Truncation:           p.openai.Truncation,
		ContextManagement:    p.contextManagement(),
		Reasoning:            p.reasoning(),
		Tools:                tools.tools,
		ToolChoice:           tools.toolChoice,
	}

	p.applyModelRestrictions(&body)

	callContext, err := p.callContext(toolNames, tools)
	if err != nil {
		return nil, err
	}

	return &responsesCall{
		body:     body,
		warnings: append(p.warnings, tools.warnings...),
		context:  callContext,
	}, nil
}

// convertInput converts the prompt and applies reasoning effort updates and
// the compaction trigger.
func (p *responsesPreparation) convertInput(toolNames toolnames.Mapping, tools responsesPreparedTools) ([]responsesInputItem, error) {
	updateUnsupportedReason := p.configurationUpdateUnsupportedReason()

	converter := newResponsesInputConverter(responsesInputOptions{
		namespace:                   p.namespace,
		toolNames:                   toolNames,
		systemMessageMode:           p.systemMessageMode(),
		updateUnsupportedReason:     updateUnsupportedReason,
		passThroughUnsupportedFiles: p.openai.PassThroughUnsupportedFiles,
		store:                       optional.Or(p.openai.Store, true),
		isReasoningModel:            p.isReasoningModel,
		hasConversation:             p.openai.Conversation != "",
		hasPreviousResponseID:       p.openai.PreviousResponseID != "",
		hasLocalShellTool:           p.providerToolName(localShellToolID) != "",
		hasShellTool:                p.providerToolName(shellToolID) != "",
		hasApplyPatchTool:           p.providerToolName(applyPatchToolID) != "",
		hasComputerTool:             p.providerToolName(computerToolID) != "",
		toolSearchToolName:          p.providerToolName(toolSearchToolID),
		customToolNames:             tools.customToolNames,
		outputSchemaToolNames:       tools.outputSchemaToolNames,
		explicitMessageItems:        p.explicitMessageItems,
	})

	input, warnings, err := converter.convert(p.options.Prompt)
	if err != nil {
		return nil, err
	}

	p.warnings = append(p.warnings, warnings...)

	if err := p.checkConfigurationUpdates(input); err != nil {
		return nil, err
	}

	input = p.prependReasoningEffortUpdate(input, updateUnsupportedReason)

	if err := responsesCheckAdjacentUpdates(input); err != nil {
		return nil, err
	}

	// A compaction trigger is a request control, not history. OpenAI requires
	// it to be the final input item.
	if p.openai.CompactionTrigger {
		input = append(input, responsesCompactionTriggerInput{
			Type: responsesCompactionTrigger,
		})
	}

	return input, nil
}

// systemMessageMode returns the configured or model default system message
// mode.
func (p *responsesPreparation) systemMessageMode() ResponsesSystemMessageMode {
	if p.openai.SystemMessageMode != "" {
		return p.openai.SystemMessageMode
	}

	if p.isReasoningModel {
		return ResponsesSystemMessageDeveloper
	}

	return ResponsesSystemMessageMode(p.capabilities.systemMessageMode)
}

// configurationUpdateUnsupportedReason explains why reasoning effort
// updates cannot be sent, or returns an empty string when they can.
func (p *responsesPreparation) configurationUpdateUnsupportedReason() string {
	if !p.capabilities.supportsConfigurationUpdate {
		return "reasoningEffortUpdate is only supported by GPT-6 and later models"
	}

	usesAdvancedContext := p.openai.ReasoningMode == ResponsesReasoningPro ||
		p.openai.ContextManagement != nil ||
		p.openai.Truncation == ResponsesTruncationAuto
	if usesAdvancedContext {
		return "reasoningEffortUpdate requires standard reasoning mode without automatic compaction or automatic truncation"
	}

	return ""
}

// unsupportedEffort explains why the model rejects effort, or returns an
// empty string when it accepts it.
func (p *responsesPreparation) unsupportedEffort(effort ResponsesReasoningEffort) string {
	if effort == "" || p.capabilities.supportsEffort(model.ReasoningEffort(effort)) {
		return ""
	}

	return p.unsupportedEffortDetails()
}

func (p *responsesPreparation) unsupportedEffortDetails() string {
	efforts := make([]string, 0, len(p.capabilities.supportedReasoningEfforts))
	for _, effort := range p.capabilities.supportedReasoningEfforts {
		efforts = append(efforts, string(effort))
	}

	return fmt.Sprintf(
		"%s only supports the following reasoning efforts: %s",
		p.modelID,
		strings.Join(efforts, ", "),
	)
}

// checkConfigurationUpdates rejects message-level updates whose effort the
// model does not support.
func (p *responsesPreparation) checkConfigurationUpdates(input []responsesInputItem) error {
	for _, item := range input {
		update, ok := item.(responsesConfigurationUpdateInput)
		if !ok {
			continue
		}

		if reason := p.unsupportedEffort(update.Reasoning.Effort); reason != "" {
			return fmt.Errorf("%w: message-level reasoningEffortUpdate: %s", model.ErrUnsupported, reason)
		}
	}

	return nil
}

// prependReasoningEffortUpdate adds the request-level reasoning effort
// update before the input, or warns when it cannot be sent.
func (p *responsesPreparation) prependReasoningEffortUpdate(input []responsesInputItem, updateUnsupportedReason string) []responsesInputItem {
	effort := p.openai.ReasoningEffortUpdate
	if effort == "" {
		return input
	}

	reason := updateUnsupportedReason
	if reason == "" {
		reason = p.unsupportedEffort(effort)
	}

	if reason != "" {
		p.warn("reasoningEffortUpdate", reason)

		return input
	}

	// An identical update right after this one would form an adjacent pair,
	// which OpenAI rejects.
	if len(input) > 0 {
		first, ok := input[0].(responsesConfigurationUpdateInput)
		if ok && first.Reasoning.Effort == effort {
			return input
		}
	}

	update := responsesConfigurationUpdateInput{
		Type: responsesConfigurationUpdate,
		Reasoning: responsesConfigurationReasoning{
			Effort: effort,
		},
	}

	return append([]responsesInputItem{update}, input...)
}

// responsesCheckAdjacentUpdates rejects consecutive configuration updates,
// which OpenAI does not accept.
func responsesCheckAdjacentUpdates(input []responsesInputItem) error {
	for index := 1; index < len(input); index++ {
		previous := input[index-1].responsesInputType()
		current := input[index].responsesInputType()

		if previous == responsesConfigurationUpdate && current == responsesConfigurationUpdate {
			return fmt.Errorf("%w: adjacent reasoning effort configuration updates", model.ErrUnsupported)
		}
	}

	return nil
}

// textFormat returns the text options for structured output and verbosity.
func (p *responsesPreparation) textFormat() (*responsesText, error) {
	responseFormat := p.options.ResponseFormat
	wantsJSON := responseFormat != nil && responseFormat.Type == model.ResponseFormatJSON

	if !wantsJSON && p.openai.TextVerbosity == "" {
		return nil, nil
	}

	text := &responsesText{
		Verbosity: p.openai.TextVerbosity,
	}

	if !wantsJSON {
		return text, nil
	}

	if len(responseFormat.Schema) == 0 {
		text.Format = &responsesTextFormat{
			Type: "json_object",
		}

		return text, nil
	}

	schema, warnings, err := normalizeJSONSchema(responseFormat.Schema)
	if err != nil {
		return nil, err
	}

	p.warnings = append(p.warnings, warnings...)

	strict := true
	if p.openai.StrictJSONSchema != nil {
		strict = *p.openai.StrictJSONSchema
	}

	name := responseFormat.Name
	if name == "" {
		name = "response"
	}

	text.Format = &responsesTextFormat{
		Type:        "json_schema",
		Strict:      &strict,
		Name:        name,
		Description: responseFormat.Description,
		Schema:      schema,
	}

	return text, nil
}

// include returns the include list with the values the call needs added.
func (p *responsesPreparation) include() []ResponsesInclude {
	include := slices.Clone(p.openai.Include)

	add := func(value ResponsesInclude) {
		if !slices.Contains(include, value) {
			include = append(include, value)
		}
	}

	if p.topLogprobs() != nil {
		add(ResponsesIncludeOutputTextLogprobs)
	}

	includeSources := optional.Or(p.openai.IncludeWebSearchSources, true)
	if p.webSearchToolName() != "" && includeSources && !p.omitWebSearchSourcesInclude {
		add(ResponsesIncludeWebSearchSources)
	}

	if p.providerToolName(codeInterpreterToolID) != "" {
		add(ResponsesIncludeCodeInterpreterOutputs)
	}

	// Store defaults to true, so only an explicit false needs encrypted
	// reasoning to continue the conversation.
	storeDisabled := p.openai.Store != nil && !*p.openai.Store
	if storeDisabled && p.isReasoningModel {
		add(ResponsesIncludeReasoningEncryptedContent)
	}

	return include
}

// topLogprobs returns the requested number of log probability alternatives.
func (p *responsesPreparation) topLogprobs() *int {
	if p.openai.Logprobs == 0 {
		return nil
	}

	return new(int(p.openai.Logprobs))
}

func (p *responsesPreparation) contextManagement() []responsesContextManagement {
	var management []responsesContextManagement

	for _, entry := range p.openai.ContextManagement {
		management = append(management, responsesContextManagement(entry))
	}

	return management
}

// reasoning returns the reasoning options of reasoning models.
func (p *responsesPreparation) reasoning() *responsesReasoning {
	if !p.isReasoningModel {
		return nil
	}

	settings := responsesReasoning{
		Effort:  p.reasoningEffort,
		Summary: p.reasoningSummary,
		Mode:    p.openai.ReasoningMode,
		Context: p.openai.ReasoningContext,
	}

	if settings == (responsesReasoning{}) {
		return nil
	}

	return &settings
}

// applyModelRestrictions removes settings the model does not accept and
// warns about them.
func (p *responsesPreparation) applyModelRestrictions(body *responsesRequest) {
	if p.capabilities.supportsConfigurationUpdate && body.PromptCacheRetention != "" {
		body.PromptCacheRetention = ""
		p.warn("promptCacheRetention", "promptCacheRetention is not supported by GPT-6 and later models; use promptCacheOptions instead")
	}

	if p.isReasoningModel {
		p.restrictReasoningParameters(body)
	} else {
		p.warnReasoningOptions()
	}

	p.restrictServiceTier(body)
}

// restrictReasoningParameters removes sampling settings that reasoning
// models reject, unless the effective effort is "none" on a model that
// accepts them then.
func (p *responsesPreparation) restrictReasoningParameters(body *responsesRequest) {
	effectiveEffort := p.reasoningEffort
	for _, item := range body.Input {
		if update, ok := item.(responsesConfigurationUpdateInput); ok {
			effectiveEffort = string(update.Reasoning.Effort)
		}
	}

	if effectiveEffort == "none" && p.capabilities.supportsNonReasoningParameters {
		return
	}

	if body.Temperature != nil {
		body.Temperature = nil
		p.warn("temperature", "temperature is not supported for reasoning models")
	}

	if body.TopP != nil {
		body.TopP = nil
		p.warn("topP", "topP is not supported for reasoning models")
	}

	requestsLogprobs := body.TopLogprobs != nil || slices.Contains(body.Include, ResponsesIncludeOutputTextLogprobs)
	if p.capabilities.supportedReasoningEfforts == nil || !requestsLogprobs {
		return
	}

	body.TopLogprobs = nil
	body.Include = slices.DeleteFunc(body.Include, func(value ResponsesInclude) bool {
		return value == ResponsesIncludeOutputTextLogprobs
	})

	if len(body.Include) == 0 {
		body.Include = nil
	}

	p.warn("logprobs", "logprobs is not supported for reasoning models")
}

// warnReasoningOptions reports reasoning options sent to a non-reasoning
// model.
func (p *responsesPreparation) warnReasoningOptions() {
	if p.openai.ReasoningEffort != "" {
		p.warn("reasoningEffort", "reasoningEffort is not supported for non-reasoning models")
	}

	if p.openai.ReasoningSummary != nil {
		p.warn("reasoningSummary", "reasoningSummary is not supported for non-reasoning models")
	}

	if p.openai.ReasoningMode != "" {
		p.warn("reasoningMode", "reasoningMode is not supported for non-reasoning models")
	}

	if p.openai.ReasoningContext != "" {
		p.warn("reasoningContext", "reasoningContext is not supported for non-reasoning models")
	}
}

// restrictServiceTier removes service tiers the model does not offer.
func (p *responsesPreparation) restrictServiceTier(body *responsesRequest) {
	tier := p.openai.ServiceTier

	if tier == ResponsesServiceTierFlex && !p.capabilities.supportsFlexProcessing {
		body.ServiceTier = ""
		p.warn("serviceTier", "flex processing is only available for o3, o4-mini, and gpt-5 models")
	}

	isPriority := tier == ResponsesServiceTierPriority || tier == ResponsesServiceTierFast
	if isPriority && !p.capabilities.supportsPriorityProcessing {
		body.ServiceTier = ""
		p.warn("serviceTier", "priority processing is only available for supported models (gpt-4, gpt-5, gpt-5-mini, o3, o4-mini) and requires Enterprise access. gpt-5-nano is not supported")
	}
}

// callContext returns what response mapping needs from the request.
func (p *responsesPreparation) callContext(toolNames toolnames.Mapping, tools responsesPreparedTools) (responsesCallContext, error) {
	approvalToolCallIDs, err := responsesApprovalToolCallIDs(p.options.Prompt)
	if err != nil {
		return responsesCallContext{}, err
	}

	webSearchToolName := p.webSearchToolName()
	if webSearchToolName == "" {
		webSearchToolName = "web_search"
	}

	return responsesCallContext{
		namespace:               p.namespace,
		toolNames:               toolNames,
		webSearchToolName:       webSearchToolName,
		isShellProviderExecuted: tools.shellProviderExecuted,
		store:                   optional.Or(p.openai.Store, false),
		collectLogprobs:         p.openai.Logprobs > 0,
		functionToolNames:       responsesFunctionToolNames(p.options.Tools),
		approvalToolCallIDs:     approvalToolCallIDs,
		generateID:              p.generateID,
	}, nil
}

// providerToolName returns the application name of the first provider tool
// with id, or an empty string.
func (p *responsesPreparation) providerToolName(id string) string {
	for _, tool := range p.options.Tools {
		if providerTool, ok := tool.(model.ProviderTool); ok && providerTool.ID == id {
			return providerTool.Name
		}
	}

	return ""
}

// webSearchToolName returns the application name of the web search tool.
func (p *responsesPreparation) webSearchToolName() string {
	for _, tool := range p.options.Tools {
		providerTool, ok := tool.(model.ProviderTool)
		if !ok {
			continue
		}

		if providerTool.ID == webSearchToolID || providerTool.ID == webSearchPreviewToolID {
			return providerTool.Name
		}
	}

	return ""
}

func (p *responsesPreparation) warn(feature, details string) {
	p.warnings = append(p.warnings, model.UnsupportedWarning(feature, details))
}

// responsesFunctionToolNames returns the names of the declared function
// tools.
func responsesFunctionToolNames(tools []model.Tool) map[string]bool {
	names := make(map[string]bool)

	for _, tool := range tools {
		if function, ok := tool.(model.FunctionTool); ok {
			names[function.Name] = true
		}
	}

	return names
}

// responsesApprovalToolCallIDs maps MCP approval request IDs to the tool
// call IDs that were created for them, so that results of approved calls
// refer to the same tool call.
func responsesApprovalToolCallIDs(prompt []model.Message) (map[string]string, error) {
	mapping := make(map[string]string)

	for _, message := range prompt {
		assistant, ok := message.(model.AssistantMessage)
		if !ok {
			continue
		}

		for _, part := range assistant.Content {
			call, ok := part.(model.ToolCallPart)
			if !ok {
				continue
			}

			metadata, _, err := model.ProviderValue[ResponsesToolCallMetadata](call.ProviderOptions, Namespace)
			if err != nil {
				return nil, err
			}

			if metadata.ApprovalRequestID != "" {
				mapping[metadata.ApprovalRequestID] = call.ToolCallID
			}
		}
	}

	return mapping, nil
}
