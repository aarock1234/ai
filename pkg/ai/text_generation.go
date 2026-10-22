package ai

import (
	"context"
	"slices"
	"time"

	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/internal/merge"
	"github.com/aarock1234/ai/pkg/model"
)

// textGenerationConfig holds the options shared by GenerateText and
// StreamText.
type textGenerationConfig struct {
	model        model.LanguageModel
	callSettings CallSettings
	prompt       Prompt

	tools               []Tool
	toolChoice          *model.ToolChoice
	activeTools         []string
	toolOrder           []string
	toolCallConcurrency int
	stopWhen            []StopCondition
	output              Output

	toolApproval       ToolApprovalConfiguration
	toolApprovalSecret []byte
	toolCallers        map[string][]string

	providerOptions model.ProviderOptions
	prepareStep     PrepareStepFunc
	repairToolCall  ToolCallRepairFunc
	refineToolInput map[string]ToolInputRefinement
	download        DownloadFunction

	timeout     TimeoutConfiguration
	telemetry   TelemetryOptions
	logWarnings LogWarningsFunc

	onLanguageModelCallStart func(ctx context.Context, event LanguageModelCallStartEvent)
	onLanguageModelCallEnd   func(ctx context.Context, event LanguageModelCallEndEvent)
	onToolExecutionStart     func(ctx context.Context, event ToolExecutionStartEvent)
	onToolExecutionEnd       func(ctx context.Context, event ToolExecutionEndEvent)
}

// newTextGenerationConfig returns the settings of a GenerateText call.
// StreamText converts its options with StreamTextOptions.generateTextOptions
// first.
func newTextGenerationConfig(options GenerateTextOptions) textGenerationConfig {
	return textGenerationConfig{
		model:                    options.Model,
		callSettings:             options.CallSettings,
		prompt:                   options.Prompt,
		tools:                    options.Tools,
		toolChoice:               options.ToolChoice,
		activeTools:              options.ActiveTools,
		toolOrder:                options.ToolOrder,
		toolCallConcurrency:      options.ToolCallConcurrency,
		stopWhen:                 options.StopWhen,
		output:                   options.Output,
		toolApproval:             options.ToolApproval,
		toolApprovalSecret:       options.ToolApprovalSecret,
		toolCallers:              options.ToolCallers,
		providerOptions:          options.ProviderOptions,
		prepareStep:              options.PrepareStep,
		repairToolCall:           options.RepairToolCall,
		refineToolInput:          options.RefineToolInput,
		download:                 options.Download,
		timeout:                  options.Timeout,
		telemetry:                options.Telemetry,
		logWarnings:              options.LogWarnings,
		onLanguageModelCallStart: options.OnLanguageModelCallStart,
		onLanguageModelCallEnd:   options.OnLanguageModelCallEnd,
		onToolExecutionStart:     options.OnToolExecutionStart,
		onToolExecutionEnd:       options.OnToolExecutionEnd,
	}
}

// textGeneration is a validated text generation shared by GenerateText and
// StreamText.
type textGeneration struct {
	textGenerationConfig

	callID         string
	initialPrompt  standardizedPrompt
	retry          retryPolicy
	responseFormat *model.ResponseFormat

	resolvedToolCallers resolvedToolCallers
	toolSearch          *toolSearchState
	telemetryDispatcher telemetryDispatcher
}

// newTextGeneration validates the configuration before any model call.
func newTextGeneration(config textGenerationConfig) (*textGeneration, error) {
	if err := validateTextGenerationConfig(config); err != nil {
		return nil, err
	}

	retry, err := prepareRetries(config.callSettings.MaxRetries)
	if err != nil {
		return nil, err
	}

	toolCallers, err := resolveToolCallers(config.tools, config.toolCallers)
	if err != nil {
		return nil, err
	}

	toolSearch, err := newToolSearchState(config.tools, toolCallers)
	if err != nil {
		return nil, err
	}

	if config.output == nil {
		config.output = TextOutput()
	}

	responseFormat, err := config.output.ResponseFormat()
	if err != nil {
		return nil, err
	}

	initialPrompt, err := standardizePrompt(config.prompt)
	if err != nil {
		return nil, err
	}

	config.stopWhen = stopConditionsOrDefault(config.stopWhen)

	return &textGeneration{
		textGenerationConfig: config,
		callID:               id.New("call"),
		initialPrompt:        initialPrompt,
		retry:                retry,
		responseFormat:       responseFormat,
		resolvedToolCallers:  toolCallers,
		toolSearch:           toolSearch,
		telemetryDispatcher:  newTelemetryDispatcher(config.telemetry),
	}, nil
}

// validateTextGenerationConfig rejects options that cannot produce a valid
// model call.
func validateTextGenerationConfig(config textGenerationConfig) error {
	if config.model == nil {
		return &InvalidArgumentError{
			Parameter: "model",
			Message:   "model must not be nil",
		}
	}

	if err := validateTools(config.tools); err != nil {
		return err
	}

	if config.toolCallConcurrency < 0 {
		return &InvalidArgumentError{
			Parameter: "toolCallConcurrency",
			Value:     config.toolCallConcurrency,
			Message:   "toolCallConcurrency must be >= 0",
		}
	}

	if err := config.timeout.validate(); err != nil {
		return err
	}

	_, err := prepareCallSettings(config.callSettings)

	return err
}

// startEvent describes the generation for the OnStart callback.
func (g *textGeneration) startEvent(operationID string) GenerateTextStartEvent {
	return GenerateTextStartEvent{
		CallID:          g.callID,
		OperationID:     operationID,
		Provider:        g.model.Provider(),
		ModelID:         g.model.ModelID(),
		System:          g.prompt.System,
		Messages:        g.initialPrompt.messages,
		Tools:           g.tools,
		ToolChoice:      g.toolChoice,
		ActiveTools:     g.activeTools,
		ToolOrder:       g.toolOrder,
		CallSettings:    g.callSettings,
		ProviderOptions: g.providerOptions,
		Output:          g.output,
	}
}

// stepInput is the state a step starts from.
type stepInput struct {
	number           int
	steps            []StepResult
	system           string
	messages         []model.Message
	responseMessages []model.Message
}

// preparedStep is a step ready for its model call.
type preparedStep struct {
	number int
	model  model.LanguageModel
	system string

	// messages are the step's messages before conversion, including the
	// messages of local tool callers; they carry forward to the next step.
	messages []model.Message

	// modelTools are the tools declared to the model. Their calls are
	// parsed against them.
	modelTools []Tool

	// executionTools are the tools that run the step's calls, with tool
	// searches and local callers bound.
	executionTools []Tool

	toolChoice      *model.ToolChoice
	activeTools     []string
	toolOrder       []string
	providerOptions model.ProviderOptions
	callSettings    CallSettings

	// callOptions are the complete options of the model call.
	callOptions model.CallOptions
}

// prepareStepCall applies PrepareStep and converts the step's prompt.
func (g *textGeneration) prepareStepCall(ctx context.Context, input stepInput) (preparedStep, error) {
	overrides, err := g.stepOverrides(ctx, input)
	if err != nil {
		return preparedStep{}, err
	}

	step := preparedStep{
		number:          input.number,
		model:           g.model,
		system:          input.system,
		messages:        input.messages,
		toolChoice:      g.toolChoice,
		activeTools:     g.activeTools,
		toolOrder:       g.toolOrder,
		providerOptions: merge.Deep(g.providerOptions, overrides.ProviderOptions),
		callSettings:    stepCallSettings(g.callSettings, overrides),
	}

	if overrides.Model != nil {
		step.model = overrides.Model
	}

	if overrides.System != nil {
		step.system = *overrides.System
	}

	if overrides.Messages != nil {
		step.messages = overrides.Messages
	}

	if overrides.ToolChoice != nil {
		step.toolChoice = overrides.ToolChoice
	}

	if overrides.ActiveTools != nil {
		step.activeTools = overrides.ActiveTools
	}

	if overrides.ToolOrder != nil {
		step.toolOrder = overrides.ToolOrder
	}

	g.prepareStepTools(&step)

	callOptions, err := g.stepCallOptions(ctx, step)
	if err != nil {
		return preparedStep{}, err
	}

	step.callOptions = callOptions

	return step, nil
}

// prepareStepTools selects the active tools, hides the deferred tools that
// no search has discovered, routes the tools through their callers and
// appends the callers' messages to the step's messages.
func (g *textGeneration) prepareStepTools(step *preparedStep) {
	active := FilterActiveTools(g.tools, step.activeTools)
	searchable := g.toolSearch.prepare(active)
	routed := prepareToolsForToolCallers(searchable, g.resolvedToolCallers)

	step.modelTools = routed.modelTools
	step.executionTools = routed.executionTools
	step.messages = appendToolCallerMessages(step.messages, routed.messages)
}

func (g *textGeneration) stepOverrides(ctx context.Context, input stepInput) (PrepareStepResult, error) {
	if g.prepareStep == nil {
		return PrepareStepResult{}, nil
	}

	overrides, err := g.prepareStep(ctx, PrepareStepOptions{
		Model:            g.model,
		Steps:            input.steps,
		StepNumber:       input.number,
		System:           input.system,
		InitialSystem:    g.prompt.System,
		Messages:         input.messages,
		InitialMessages:  g.initialPrompt.messages,
		ResponseMessages: input.responseMessages,
	})
	if err != nil {
		return PrepareStepResult{}, err
	}

	if overrides == nil {
		return PrepareStepResult{}, nil
	}

	return *overrides, nil
}

// stepCallOptions builds the model call options of a step.
func (g *textGeneration) stepCallOptions(ctx context.Context, step preparedStep) (model.CallOptions, error) {
	if err := validateMessages(step.messages, g.prompt.AllowSystemInMessages); err != nil {
		return model.CallOptions{}, err
	}

	prompt, err := convertToModelPrompt(ctx, convertPromptOptions{
		prompt: standardizedPrompt{
			instructions: systemInstructions(step.system),
			messages:     step.messages,
		},
		supportedURLs: step.model.SupportedURLs(),
		download:      g.download,
	})
	if err != nil {
		return model.CallOptions{}, err
	}

	callOptions, err := prepareCallSettings(step.callSettings)
	if err != nil {
		return model.CallOptions{}, err
	}

	callOptions.Prompt = prompt
	callOptions.Tools = prepareTools(step.modelTools, step.toolOrder)
	callOptions.ToolChoice = prepareToolChoice(step.toolChoice)
	callOptions.ResponseFormat = g.responseFormat
	callOptions.ProviderOptions = step.providerOptions

	return callOptions, nil
}

// stepStartEvent describes a prepared step for the OnStepStart callback.
func (g *textGeneration) stepStartEvent(step preparedStep, steps []StepResult) GenerateTextStepStartEvent {
	return GenerateTextStepStartEvent{
		CallID:          g.callID,
		Provider:        step.model.Provider(),
		ModelID:         step.model.ModelID(),
		StepNumber:      step.number,
		System:          step.system,
		Messages:        step.messages,
		Tools:           g.tools,
		ToolChoice:      step.toolChoice,
		ActiveTools:     step.activeTools,
		ToolOrder:       step.toolOrder,
		Steps:           steps,
		ProviderOptions: step.providerOptions,
		Output:          g.output,
		PromptMessages:  step.callOptions.Prompt,
		StepTools:       step.callOptions.Tools,
		StepToolChoice:  step.callOptions.ToolChoice,
	}
}

// notifyStart calls OnStart and the telemetry integrations when the
// generation begins.
func (g *textGeneration) notifyStart(ctx context.Context, event GenerateTextStartEvent, onStart func(ctx context.Context, event GenerateTextStartEvent)) {
	if onStart != nil {
		onStart(ctx, event)
	}

	g.telemetryDispatcher.onGenerateTextStart(ctx, event)
}

// notifyStepStart calls OnStepStart and the telemetry integrations before
// a step's model call.
func (g *textGeneration) notifyStepStart(ctx context.Context, event GenerateTextStepStartEvent, onStepStart func(ctx context.Context, event GenerateTextStepStartEvent)) {
	if onStepStart != nil {
		onStepStart(ctx, event)
	}

	g.telemetryDispatcher.onGenerateTextStepStart(ctx, event)
}

// notifyStepEnd calls OnStepEnd and the telemetry integrations after
// a step, and logs the step's warnings.
func (g *textGeneration) notifyStepEnd(ctx context.Context, step StepResult, onStepEnd func(ctx context.Context, step StepResult)) {
	if onStepEnd != nil {
		onStepEnd(ctx, step)
	}

	g.telemetryDispatcher.onGenerateTextStepEnd(ctx, step)

	g.logWarnings.log(ctx, WarningsEvent{
		Warnings: step.Warnings,
		Provider: step.Model.Provider,
		ModelID:  step.Model.ModelID,
	})
}

// notifyEnd calls OnEnd and the telemetry integrations when the
// generation completes.
func (g *textGeneration) notifyEnd(ctx context.Context, event GenerateTextEndEvent, onEnd func(ctx context.Context, event GenerateTextEndEvent)) {
	if onEnd != nil {
		onEnd(ctx, event)
	}

	g.telemetryDispatcher.onGenerateTextEnd(ctx, event)
}

// languageModelCall describes the streamed model call of a prepared step.
func (g *textGeneration) languageModelCall(step preparedStep, collectContent bool) languageModelCall {
	return languageModelCall{
		start:          g.languageModelCallStartEvent(step),
		model:          step.model,
		callOptions:    step.callOptions,
		tools:          step.modelTools,
		repair:         g.repairToolCall,
		refine:         g.refineToolInput,
		collectContent: collectContent,
		telemetry:      g.telemetryDispatcher,
		onStart:        g.onLanguageModelCallStart,
		onEnd:          g.onLanguageModelCallEnd,
	}
}

// languageModelCallStartEvent describes the model call of a prepared step.
func (g *textGeneration) languageModelCallStartEvent(step preparedStep) LanguageModelCallStartEvent {
	return LanguageModelCallStartEvent{
		CallID:       g.callID,
		Provider:     step.model.Provider(),
		ModelID:      step.model.ModelID(),
		System:       step.system,
		Messages:     step.messages,
		Tools:        step.callOptions.Tools,
		CallSettings: step.callSettings,
	}
}

// observesLanguageModelCallEnd reports whether anything receives the end
// of model calls, which requires converting the response content.
func (g *textGeneration) observesLanguageModelCallEnd() bool {
	return g.onLanguageModelCallEnd != nil || g.telemetryDispatcher.enabled()
}

// toolExecution describes how the tool calls of a step run.
func (g *textGeneration) toolExecution(tools []Tool, messages []model.Message) toolExecution {
	return toolExecution{
		callID:      g.callID,
		tools:       tools,
		messages:    messages,
		concurrency: g.toolCallConcurrency,
		timeout:     g.timeout,
		telemetry:   g.telemetryDispatcher,
		onStart:     g.onToolExecutionStart,
		onEnd:       g.onToolExecutionEnd,
	}
}

// toolCallApproval is the outcome of resolving the approval of one call.
type toolCallApproval struct {
	// request is nil when the call needs no approval.
	request *ToolApprovalRequest

	// response is set when the approval policy decided automatically.
	response *ToolApprovalResponse

	// blocked reports that the call must not run.
	blocked bool
}

// resolveCallApproval resolves the approval of a valid call and creates the
// approval request and automatic response. Calls that need no approval do
// not consume an approval ID.
func (g *textGeneration) resolveCallApproval(ctx context.Context, call ToolCall, tools []Tool, messages []model.Message) (toolCallApproval, error) {
	status, err := resolveToolApproval(ctx, toolApprovalRequest{
		tools:         tools,
		configuration: g.toolApproval,
		call:          call,
		messages:      messages,
	})
	if err != nil {
		return toolCallApproval{}, err
	}

	if status.Type == ToolApprovalNotApplicable {
		return toolCallApproval{}, nil
	}

	approvalID := id.New("aitxt")

	signature, err := signToolApproval(toolApprovalSignature{
		secret:     g.toolApprovalSecret,
		approvalID: approvalID,
		toolCallID: call.ToolCallID,
		toolName:   call.ToolName,
		input:      call.Input,
	})
	if err != nil {
		return toolCallApproval{}, err
	}

	request := &ToolApprovalRequest{
		ApprovalID: approvalID,
		ToolCall:   call,
		Signature:  signature,
	}

	if status.Type == ToolApprovalUserApproval {
		request.Reason = status.Reason

		return toolCallApproval{
			request: request,
			blocked: true,
		}, nil
	}

	request.IsAutomatic = true
	approved := status.Type == ToolApprovalApproved

	return toolCallApproval{
		request: request,
		response: &ToolApprovalResponse{
			ApprovalID:       approvalID,
			ToolCall:         call,
			Approved:         approved,
			Reason:           status.Reason,
			ProviderExecuted: call.ProviderExecuted,
		},
		blocked: !approved,
	}, nil
}

// initialApprovals are the approval responses of the prompt, resolved before
// the first step.
type initialApprovals struct {
	// approved are approved local calls that passed revalidation.
	approved []collectedToolApproval

	// deniedLocal are denied local calls, including those the policy denies
	// on revalidation.
	deniedLocal []collectedToolApproval

	deniedProviderExecuted []collectedToolApproval

	invalid []invalidToolApproval
}

func (a initialApprovals) isEmpty() bool {
	return len(a.approved) == 0 &&
		len(a.deniedLocal) == 0 &&
		len(a.deniedProviderExecuted) == 0 &&
		len(a.invalid) == 0
}

func (a initialApprovals) approvedCalls() []ToolCall {
	calls := make([]ToolCall, 0, len(a.approved))
	for _, approval := range a.approved {
		calls = append(calls, approval.call)
	}

	return calls
}

// resolveInitialApprovals collects the approval responses of the prompt and
// revalidates the approved local calls.
func (g *textGeneration) resolveInitialApprovals(ctx context.Context) (initialApprovals, error) {
	collected, err := collectToolApprovals(g.initialPrompt.messages)
	if err != nil {
		return initialApprovals{}, err
	}

	var approvals initialApprovals
	var approvedLocal []collectedToolApproval

	for _, approval := range collected.approved {
		if !approval.call.ProviderExecuted {
			approvedLocal = append(approvedLocal, approval)
		}
	}

	for _, approval := range collected.denied {
		if approval.call.ProviderExecuted {
			approvals.deniedProviderExecuted = append(approvals.deniedProviderExecuted, approval)
		} else {
			approvals.deniedLocal = append(approvals.deniedLocal, approval)
		}
	}

	validated, err := validateApprovedToolApprovals(ctx, approvalValidation{
		approvals:     approvedLocal,
		tools:         g.tools,
		configuration: g.toolApproval,
		messages:      g.initialPrompt.messages,
		secret:        g.toolApprovalSecret,
		refine:        g.refineToolInput,
	})
	if err != nil {
		return initialApprovals{}, err
	}

	approvals.approved = validated.approved
	approvals.deniedLocal = append(approvals.deniedLocal, validated.denied...)
	approvals.invalid = validated.invalid

	return approvals, nil
}

// initialToolMessageInput holds the outcomes reported to the model before
// the first step.
type initialToolMessageInput struct {
	outputs []ToolOutput
	invalid []invalidToolApproval

	// denied are denied calls; those that already have a result are skipped.
	denied []collectedToolApproval
}

// initialToolMessage builds the tool message with the results of approved
// calls, the errors of invalid approved calls and the denials. It returns
// false when there is nothing to report.
func (g *textGeneration) initialToolMessage(ctx context.Context, input initialToolMessageInput) (model.ToolMessage, bool, error) {
	var content []model.ToolPart

	for _, output := range input.outputs {
		part, err := toolOutputPart(ctx, output, g.tools)
		if err != nil {
			return model.ToolMessage{}, false, err
		}

		content = append(content, part)
	}

	for _, approval := range input.invalid {
		modelOutput, err := createToolModelOutput(ctx, toolModelOutputRequest{
			toolCallID: approval.call.ToolCallID,
			input:      approval.call.Input,
			output:     approval.err,
			tool:       lookupTool(g.tools, approval.call.ToolName),
			errorMode:  toolOutputErrorText,
		})
		if err != nil {
			return model.ToolMessage{}, false, err
		}

		content = append(content, model.ToolResultPart{
			ToolCallID: approval.call.ToolCallID,
			ToolName:   approval.call.ToolName,
			Output:     modelOutput,
		})
	}

	for _, approval := range input.denied {
		if approval.existingResult != nil {
			continue
		}

		content = append(content, model.ToolResultPart{
			ToolCallID: approval.call.ToolCallID,
			ToolName:   approval.call.ToolName,
			Output:     executionDeniedOutput(approval),
		})
	}

	if len(content) == 0 {
		return model.ToolMessage{}, false, nil
	}

	return model.ToolMessage{
		Content: content,
	}, true, nil
}

// deniedApprovalOptions carries the approval ID of a denied
// provider-executed call. Upstream writes it under the "openai" namespace,
// the only provider that correlates denials with approvals.
type deniedApprovalOptions struct {
	ApprovalID string `json:"approvalId"`
}

// executionDeniedOutput reports a denial to the model. Provider-executed
// calls carry the approval ID so that the provider can correlate it.
func executionDeniedOutput(approval collectedToolApproval) model.ExecutionDeniedOutput {
	output := model.ExecutionDeniedOutput{
		Reason: approval.response.Reason,
	}

	if approval.call.ProviderExecuted {
		output.ProviderOptions = model.ProviderOptions{
			"openai": deniedApprovalOptions{
				ApprovalID: approval.response.ApprovalID,
			},
		}
	}

	return output
}

// deferredToolCalls tracks provider-executed calls whose results may arrive
// in a later step.
type deferredToolCalls map[string]string

// track records the step's provider-executed calls of tools that support
// deferred results and have no result yet, then removes the calls whose
// results arrived.
func (d deferredToolCalls) track(calls []ToolCall, tools []Tool, resultCallIDs []string) {
	for _, call := range calls {
		if !call.ProviderExecuted {
			continue
		}

		tool, found := findTool(tools, call.ToolName)
		supportsDeferred := found && tool.kind() == ToolKindProvider && tool.SupportsDeferredResults
		if !supportsDeferred {
			continue
		}

		d[call.ToolCallID] = call.ToolName
	}

	for _, toolCallID := range resultCallIDs {
		delete(d, toolCallID)
	}
}

// loopProgress summarizes a step for the continuation decision.
type loopProgress struct {
	clientToolCalls   int
	clientToolOutputs int
	deniedResponses   int
}

// shouldContinue reports whether another step runs: every client tool call
// has an output or a denial, there are client calls or pending deferred
// results, and no stop condition is met.
func (g *textGeneration) shouldContinue(ctx context.Context, progress loopProgress, deferred deferredToolCalls, steps []StepResult) bool {
	allCallsResolved := progress.clientToolOutputs+progress.deniedResponses == progress.clientToolCalls
	hasWork := progress.clientToolCalls > 0 || len(deferred) > 0

	if !allCallsResolved || !hasWork {
		return false
	}

	return !isStopConditionMet(ctx, g.stopWhen, steps)
}

// isToolExecutionAllowed reports whether a finish reason permits running the
// step's tool calls.
func isToolExecutionAllowed(reason model.UnifiedFinishReason) bool {
	return reason == model.FinishStop || reason == model.FinishToolCalls
}

// enforcedToolChoiceViolation reports whether the step's tool calls violate
// a required or tool-specific tool choice.
func enforcedToolChoiceViolation(choice *model.ToolChoice, calls []ToolCall) bool {
	if choice == nil {
		return false
	}

	switch choice.Type {
	case model.ToolChoiceRequired:
		return len(calls) == 0

	case model.ToolChoiceTool:
		calledChosenTool := slices.ContainsFunc(calls, func(call ToolCall) bool {
			return call.ToolName == choice.ToolName
		})

		return !calledChosenTool

	default:
		return false
	}
}

// responseMetadata fills the response fields the provider did not report.
func responseMetadata(metadata model.ResponseMetadata, stepModel model.LanguageModel) model.ResponseMetadata {
	if metadata.ID == "" {
		metadata.ID = id.New("aitxt")
	}

	if metadata.Timestamp.IsZero() {
		metadata.Timestamp = time.Now()
	}

	if metadata.ModelID == "" {
		metadata.ModelID = stepModel.ModelID()
	}

	return metadata
}
