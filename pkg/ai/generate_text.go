package ai

import (
	"context"
	"encoding/json"
	"slices"
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

// GenerateTextOptions configures GenerateText.
type GenerateTextOptions struct {
	Model model.LanguageModel

	CallSettings
	Prompt

	// Tools are the tools the model may call, in declaration order.
	Tools []Tool

	// ToolChoice controls tool selection. Nil selects automatic choice.
	ToolChoice *model.ToolChoice

	// ActiveTools restricts the tools available to the model without
	// changing the declared tools. Nil applies no restriction.
	ActiveTools []string

	// ToolOrder lists tool names to send first; the other tools follow in
	// name order. Nil keeps declaration order. A stable order can improve
	// provider-side caching.
	ToolOrder []string

	// ToolCallConcurrency limits the tool calls that run at once within a
	// step. Zero means no limit; 1 runs them sequentially in call order.
	ToolCallConcurrency int

	// StopWhen ends the tool-calling loop when any condition is met. Nil
	// stops after one step.
	StopWhen []StopCondition

	// Output specifies the shape of the output. Nil generates text.
	Output Output

	// ToolApproval decides whether tool calls need approval: a
	// ToolApprovalFunc for every call, or ToolApprovals per tool. Its
	// decisions replace the NeedsApproval functions of the tools.
	ToolApproval ToolApprovalConfiguration

	// ToolApprovalSecret signs tool approval requests with HMAC-SHA256 and
	// verifies the signatures of approvals in the prompt, so that clients
	// cannot forge approvals.
	ToolApprovalSecret []byte

	// ToolCallers maps tool names to the callers that may call them:
	// DirectToolCall for the model itself and the names of tools with a
	// Caller. Tools without an entry may only be called directly; an empty
	// entry permits no caller. Nil routes nothing. It is upstream's
	// experimental_toolCallers.
	ToolCallers map[string][]string

	ProviderOptions model.ProviderOptions

	// PrepareStep adjusts the settings of each step.
	PrepareStep PrepareStepFunc

	// RepairToolCall attempts to repair tool calls that fail to parse.
	RepairToolCall ToolCallRepairFunc

	// RefineToolInput maps tool names to refinements of their validated
	// inputs.
	RefineToolInput map[string]ToolInputRefinement

	// Download resolves file URLs in the prompt. Nil downloads the URLs the
	// model does not support.
	Download DownloadFunction

	// Include selects the large payloads retained in step results.
	Include GenerateTextInclude

	// Timeout limits each step and tool execution. FirstChunk and Chunk only
	// apply to streaming; GenerateText reports them as unsupported warnings.
	// The total timeout comes from ctx.
	Timeout TimeoutConfiguration

	// Telemetry sends the lifecycle events of the generation to telemetry
	// integrations.
	Telemetry TelemetryOptions

	// LogWarnings receives the warnings of each step. Nil discards them; they
	// are always available on the step results.
	LogWarnings LogWarningsFunc

	// OnStart is called before the first model call.
	OnStart func(ctx context.Context, event GenerateTextStartEvent)

	// OnStepStart is called before each step.
	OnStepStart func(ctx context.Context, event GenerateTextStepStartEvent)

	// OnLanguageModelCallStart is called immediately before each model call
	// and OnLanguageModelCallEnd after its response was parsed, before any
	// client-side tool runs.
	OnLanguageModelCallStart func(ctx context.Context, event LanguageModelCallStartEvent)
	OnLanguageModelCallEnd   func(ctx context.Context, event LanguageModelCallEndEvent)

	// OnToolExecutionStart and OnToolExecutionEnd are called around each
	// tool execution. They may be called concurrently.
	OnToolExecutionStart func(ctx context.Context, event ToolExecutionStartEvent)
	OnToolExecutionEnd   func(ctx context.Context, event ToolExecutionEndEvent)

	// OnStepEnd is called after each step.
	OnStepEnd func(ctx context.Context, step StepResult)

	// OnEnd is called after the final step.
	OnEnd func(ctx context.Context, event GenerateTextEndEvent)
}

// GenerateTextInclude selects large payloads to retain in step results. They
// are dropped by default to reduce memory use.
type GenerateTextInclude struct {
	RequestBody     bool
	RequestMessages bool
	ResponseBody    bool
}

// GenerateText generates text and calls tools with a language model, without
// streaming. With tools and StopWhen it runs a multi-step loop: tool calls
// that the model makes are executed and their results sent back to the
// model until a stop condition is met.
//
// Cancellation and deadlines come from ctx.
func GenerateText(ctx context.Context, options GenerateTextOptions) (*GenerateTextResult, error) {
	generation, err := newTextGeneration(newTextGenerationConfig(options))
	if err != nil {
		return nil, err
	}

	run := &generateTextRun{
		textGeneration: generation,
		options:        options,
	}

	return run.generate(ctx)
}

// generateTextRun is one GenerateText call.
type generateTextRun struct {
	*textGeneration

	options GenerateTextOptions
}

func (r *generateTextRun) generate(ctx context.Context) (*GenerateTextResult, error) {
	r.logUnsupportedTimeouts(ctx)
	r.notifyStart(ctx, r.startEvent("ai.generateText"), r.options.OnStart)

	result, err := r.generateSteps(ctx)
	if err != nil {
		r.telemetryDispatcher.onError(ctx, TelemetryErrorEvent{
			CallID: r.callID,
			Error:  err,
		})

		return nil, err
	}

	return result, nil
}

// logUnsupportedTimeouts warns about the timeouts that only streaming
// enforces.
func (r *generateTextRun) logUnsupportedTimeouts(ctx context.Context) {
	var warnings []model.Warning

	if r.timeout.FirstChunk > 0 {
		warning := model.UnsupportedWarning(
			"timeout.firstChunk",
			"The firstChunk timeout is only supported by streaming functions.",
		)

		warnings = append(warnings, warning)
	}

	if r.timeout.Chunk > 0 {
		warning := model.UnsupportedWarning(
			"timeout.chunk",
			"The chunk timeout is only supported by streaming functions.",
		)

		warnings = append(warnings, warning)
	}

	r.logWarnings.log(ctx, WarningsEvent{
		Warnings: warnings,
		Provider: r.model.Provider(),
		ModelID:  r.model.ModelID(),
	})
}

// generateSteps runs the approved calls of the prompt and the steps, and
// aggregates the result.
func (r *generateTextRun) generateSteps(ctx context.Context) (*GenerateTextResult, error) {
	initialResponseMessages, err := r.runInitialApprovals(ctx)
	if err != nil {
		return nil, err
	}

	steps, err := r.runSteps(ctx, initialResponseMessages)
	if err != nil {
		return nil, err
	}

	finalStep := steps[len(steps)-1]

	output, outputErr := r.parseOutput(ctx, finalStep)

	result := newGenerateTextResult(generateTextResultInput{
		steps:                   steps,
		initialResponseMessages: initialResponseMessages,
		totalUsage:              sumStepUsage(steps),
		output:                  output,
	})

	r.notifyEnd(ctx, GenerateTextEndEvent{
		CallID:             r.callID,
		GenerateTextResult: *result,
	}, r.options.OnEnd)

	if outputErr != nil {
		return nil, outputErr
	}

	return result, nil
}

// runInitialApprovals executes the tool calls approved in the prompt and
// returns the tool message that reports them, denials and invalid approvals.
func (r *generateTextRun) runInitialApprovals(ctx context.Context) ([]model.Message, error) {
	approvals, err := r.resolveInitialApprovals(ctx)
	if err != nil {
		return nil, err
	}

	execution := r.toolExecution(r.tools, r.initialPrompt.messages)

	executed, err := executeToolCalls(ctx, execution, approvals.approvedCalls())
	if err != nil {
		return nil, err
	}

	outputs := make([]ToolOutput, 0, len(executed))
	for _, executedCall := range executed {
		outputs = append(outputs, executedCall.output)
	}

	message, hasContent, err := r.initialToolMessage(ctx, initialToolMessageInput{
		outputs: outputs,
		invalid: approvals.invalid,
		denied:  slices.Concat(approvals.deniedLocal, approvals.deniedProviderExecuted),
	})
	if err != nil {
		return nil, err
	}

	if !hasContent {
		return nil, nil
	}

	return []model.Message{
		message,
	}, nil
}

// runSteps runs the tool-calling loop and returns its steps.
func (r *generateTextRun) runSteps(ctx context.Context, initialResponseMessages []model.Message) ([]StepResult, error) {
	var steps []StepResult

	deferred := make(deferredToolCalls)
	responseMessages := initialResponseMessages
	input := stepInput{
		system:           r.prompt.System,
		messages:         slices.Concat(r.initialPrompt.messages, initialResponseMessages),
		responseMessages: responseMessages,
	}

	for {
		if len(steps) > 0 && ctx.Err() != nil {
			return nil, context.Cause(ctx)
		}

		input.number = len(steps)
		input.steps = steps

		step, err := r.runTimedStep(ctx, input, deferred)
		if err != nil {
			return nil, err
		}

		steps = append(steps, step.result)
		responseMessages = slices.Concat(responseMessages, step.result.Response.Messages)

		r.notifyStepEnd(ctx, step.result, r.options.OnStepEnd)

		if !r.shouldContinue(ctx, step.progress, deferred, steps) {
			return steps, nil
		}

		if step.timeoutErr != nil {
			return nil, step.timeoutErr
		}

		input = stepInput{
			system:           step.prepared.system,
			messages:         slices.Concat(step.prepared.messages, step.result.Response.Messages),
			responseMessages: responseMessages,
		}
	}
}

// generatedStep is a completed step with the state the loop needs.
type generatedStep struct {
	result   StepResult
	prepared preparedStep
	progress loopProgress

	// timeoutErr reports a step timeout that elapsed while the step's tools
	// ran. It ends the generation if the loop would continue.
	timeoutErr error
}

// runTimedStep runs a step limited by the step timeout. The timeout covers
// the model call, its retries and the step's tool executions.
func (r *generateTextRun) runTimedStep(ctx context.Context, input stepInput, deferred deferredToolCalls) (generatedStep, error) {
	stepCtx, cancel := withTimeout(ctx, TimeoutStep, r.timeout.Step)
	defer cancel()

	step, err := r.runStep(stepCtx, input, deferred)

	timedOut := stepCtx.Err() != nil && ctx.Err() == nil
	if !timedOut {
		return step, err
	}

	if err != nil {
		return generatedStep{}, context.Cause(stepCtx)
	}

	step.timeoutErr = context.Cause(stepCtx)

	return step, nil
}

// runStep performs one model call, resolves tool approvals, executes the
// tool calls and assembles the step result.
func (r *generateTextRun) runStep(ctx context.Context, input stepInput, deferred deferredToolCalls) (generatedStep, error) {
	prepared, err := r.prepareStepCall(ctx, input)
	if err != nil {
		return generatedStep{}, err
	}

	r.notifyStepStart(ctx, r.stepStartEvent(prepared, input.steps), r.options.OnStepStart)

	started := time.Now()

	response, err := r.requestModelResponse(ctx, prepared)
	if err != nil {
		return generatedStep{}, err
	}

	responseTime := time.Since(started)
	response.Response.ResponseMetadata = responseMetadata(response.Response.ResponseMetadata, prepared.model)
	usage := newLanguageModelUsage(response.Usage)
	files := newGeneratedFileResolver()

	calls, err := r.parseToolCalls(ctx, prepared, response.Content)
	if err != nil {
		return generatedStep{}, err
	}

	err = r.notifyLanguageModelCallEnd(ctx, languageModelCallOutcome{
		prepared:     prepared,
		response:     response,
		calls:        calls,
		usage:        usage,
		responseTime: responseTime,
		files:        files,
	})
	if err != nil {
		return generatedStep{}, err
	}

	if enforcedToolChoiceViolation(prepared.callOptions.ToolChoice, calls) {
		return generatedStep{}, &ToolChoiceViolationError{
			ToolChoice:   *prepared.callOptions.ToolChoice,
			FinishReason: response.FinishReason,
			Provider:     prepared.model.Provider(),
			ModelID:      prepared.model.ModelID(),
			Content:      response.Content,
		}
	}

	handling, err := r.handleToolCalls(ctx, prepared, calls, response.FinishReason.Unified)
	if err != nil {
		return generatedStep{}, err
	}

	deferred.track(calls, prepared.executionTools, providerResultCallIDs(response.Content))

	content, err := convertModelContent(ctx, modelContentConversion{
		content:           response.Content,
		toolCalls:         calls,
		toolOutputs:       handling.outputs,
		approvalRequests:  handling.approvalRequests,
		approvalResponses: handling.approvalResponses,
		tools:             r.tools,
		files:             files,
	})
	if err != nil {
		return generatedStep{}, err
	}

	responseMessages, err := toResponseMessages(ctx, content, r.tools)
	if err != nil {
		return generatedStep{}, err
	}

	result := r.newStepResult(generatedStepOutcome{
		prepared:         prepared,
		response:         response,
		content:          content,
		responseMessages: responseMessages,
		usage:            usage,
		responseTime:     responseTime,
		stepTime:         time.Since(started),
		executionTimes:   handling.executionTimes,
	})

	return generatedStep{
		result:   result,
		prepared: prepared,
		progress: handling.progress,
	}, nil
}

// generatedStepOutcome is everything a completed step reports in its
// result.
type generatedStepOutcome struct {
	prepared         preparedStep
	response         *model.GenerateResult
	content          []ContentPart
	responseMessages []model.Message
	usage            LanguageModelUsage
	responseTime     time.Duration
	stepTime         time.Duration
	executionTimes   map[string]time.Duration
}

// newStepResult assembles the result of a completed step.
func (r *generateTextRun) newStepResult(outcome generatedStepOutcome) StepResult {
	prepared := outcome.prepared
	response := outcome.response

	return StepResult{
		CallID:     r.callID,
		StepNumber: prepared.number,
		Model: StepModel{
			Provider: prepared.model.Provider(),
			ModelID:  prepared.model.ModelID(),
		},
		Content:         outcome.content,
		FinishReason:    response.FinishReason.Unified,
		RawFinishReason: response.FinishReason.Raw,
		Usage:           outcome.usage,
		Performance: StepPerformance{
			LanguageModelCallPerformance: generatedCallPerformance(outcome.usage, outcome.responseTime),
			StepTime:                     outcome.stepTime,
			ToolExecution:                outcome.executionTimes,
		},
		Warnings:         response.Warnings,
		Request:          r.stepRequest(prepared, response.Request),
		Response:         r.stepResponse(response.Response, outcome.responseMessages),
		ProviderMetadata: response.ProviderMetadata,
	}
}

// requestModelResponse notifies the start of a step's model call and
// performs it, retrying failed attempts.
func (r *generateTextRun) requestModelResponse(ctx context.Context, prepared preparedStep) (*model.GenerateResult, error) {
	callStart := r.languageModelCallStartEvent(prepared)

	if r.onLanguageModelCallStart != nil {
		r.onLanguageModelCallStart(ctx, callStart)
	}

	r.telemetryDispatcher.onLanguageModelCallStart(ctx, callStart)

	return retryCall(ctx, r.retry, func(ctx context.Context) (*model.GenerateResult, error) {
		return r.callModel(ctx, prepared, callStart)
	})
}

// callModel performs one attempt of a step's model call inside the
// telemetry wrappers.
func (r *generateTextRun) callModel(ctx context.Context, prepared preparedStep, callStart LanguageModelCallStartEvent) (*model.GenerateResult, error) {
	var response *model.GenerateResult

	err := r.telemetryDispatcher.executeLanguageModelCall(ctx, callStart, func(ctx context.Context) error {
		generated, err := prepared.model.Generate(ctx, prepared.callOptions)
		response = generated

		return err
	})
	if err != nil {
		return nil, err
	}

	return response, nil
}

// languageModelCallOutcome is a parsed model response, before any tool
// runs.
type languageModelCallOutcome struct {
	prepared     preparedStep
	response     *model.GenerateResult
	calls        []ToolCall
	usage        LanguageModelUsage
	responseTime time.Duration
	files        *generatedFileResolver
}

// notifyLanguageModelCallEnd converts the response content and passes the
// end of the model call to OnLanguageModelCallEnd and the telemetry
// integrations. The content is only converted when something observes it.
func (r *generateTextRun) notifyLanguageModelCallEnd(ctx context.Context, outcome languageModelCallOutcome) error {
	if !r.observesLanguageModelCallEnd() {
		return nil
	}

	content, err := convertModelContent(ctx, modelContentConversion{
		content:   outcome.response.Content,
		toolCalls: outcome.calls,
		tools:     r.tools,
		files:     outcome.files,
	})
	if err != nil {
		return err
	}

	event := LanguageModelCallEndEvent{
		CallID:           r.callID,
		Provider:         outcome.prepared.model.Provider(),
		ModelID:          outcome.response.Response.ModelID,
		FinishReason:     outcome.response.FinishReason.Unified,
		Usage:            outcome.usage,
		Content:          content,
		ResponseID:       outcome.response.Response.ID,
		ProviderMetadata: outcome.response.ProviderMetadata,
		Performance:      generatedCallPerformance(outcome.usage, outcome.responseTime),
	}

	if r.onLanguageModelCallEnd != nil {
		r.onLanguageModelCallEnd(ctx, event)
	}

	r.telemetryDispatcher.onLanguageModelCallEnd(ctx, event)

	return nil
}

// parseToolCalls parses the tool calls of a response against the tools
// declared to the model in the step.
func (r *generateTextRun) parseToolCalls(ctx context.Context, prepared preparedStep, content []model.Content) ([]ToolCall, error) {
	var calls []ToolCall

	for _, part := range content {
		call, ok := part.(model.ToolCall)
		if !ok {
			continue
		}

		parsed, err := parseToolCall(ctx, toolCallParseRequest{
			call:     call,
			tools:    prepared.modelTools,
			repair:   r.repairToolCall,
			refine:   r.refineToolInput,
			system:   prepared.system,
			messages: prepared.messages,
		})
		if err != nil {
			return nil, err
		}

		calls = append(calls, parsed)
	}

	return calls, nil
}

// toolCallHandling is the outcome of processing a step's tool calls.
type toolCallHandling struct {
	outputs           []ToolOutput
	approvalRequests  []ToolApprovalRequest
	approvalResponses []ToolApprovalResponse
	executionTimes    map[string]time.Duration
	progress          loopProgress
}

// handleToolCalls notifies the tools of their inputs, resolves approvals,
// reports invalid calls and executes the allowed calls.
func (r *generateTextRun) handleToolCalls(
	ctx context.Context,
	prepared preparedStep,
	calls []ToolCall,
	finishReason model.UnifiedFinishReason,
) (toolCallHandling, error) {
	approvals, err := r.resolveApprovals(ctx, prepared, calls)
	if err != nil {
		return toolCallHandling{}, err
	}

	partition := partitionClientCalls(calls, approvals.blocked)

	handling := toolCallHandling{
		outputs:           partition.invalidOutputs,
		approvalRequests:  approvals.requests,
		approvalResponses: approvals.responses,
		executionTimes:    make(map[string]time.Duration),
	}

	if isToolExecutionAllowed(finishReason) {
		execution := r.toolExecution(prepared.executionTools, prepared.messages)

		executed, executionErr := executeToolCalls(ctx, execution, partition.executable)
		if executionErr != nil {
			return toolCallHandling{}, executionErr
		}

		for _, executedCall := range executed {
			callID := toolOutputCallID(executedCall.output)

			handling.outputs = append(handling.outputs, executedCall.output)
			handling.executionTimes[callID] = executedCall.executionTime
		}
	}

	handling.progress = loopProgress{
		clientToolCalls:   partition.clientCalls,
		clientToolOutputs: len(handling.outputs),
		deniedResponses:   countDenied(handling.approvalResponses),
	}

	return handling, nil
}

// stepApprovals are the approval outcomes of a step's tool calls.
type stepApprovals struct {
	requests  []ToolApprovalRequest
	responses []ToolApprovalResponse

	// blocked holds the IDs of calls that must not run yet.
	blocked map[string]bool
}

// resolveApprovals notifies the tools of their complete inputs and resolves
// the approval of each valid call of a known tool.
func (r *generateTextRun) resolveApprovals(ctx context.Context, prepared preparedStep, calls []ToolCall) (stepApprovals, error) {
	approvals := stepApprovals{
		blocked: make(map[string]bool),
	}

	for _, call := range calls {
		if call.Invalid {
			continue
		}

		tool, found := findTool(prepared.executionTools, call.ToolName)
		if !found {
			continue
		}

		notifyToolInput(ctx, tool, call, prepared.messages)

		approval, err := r.resolveCallApproval(ctx, call, prepared.executionTools, prepared.messages)
		if err != nil {
			return stepApprovals{}, err
		}

		if approval.request != nil {
			approvals.requests = append(approvals.requests, *approval.request)
		}

		if approval.response != nil {
			approvals.responses = append(approvals.responses, *approval.response)
		}

		if approval.blocked {
			approvals.blocked[call.ToolCallID] = true
		}
	}

	return approvals, nil
}

// clientCallPartition sorts the client tool calls of a step.
type clientCallPartition struct {
	// clientCalls counts the calls that are not provider-executed.
	clientCalls int

	// invalidOutputs report the invalid calls to the model.
	invalidOutputs []ToolOutput

	// executable are the valid calls that are not blocked by an approval.
	executable []ToolCall
}

// partitionClientCalls separates the invalid client calls from those that
// may run. Provider-executed calls are left out.
func partitionClientCalls(calls []ToolCall, blocked map[string]bool) clientCallPartition {
	var partition clientCallPartition

	for _, call := range calls {
		if call.ProviderExecuted {
			continue
		}

		partition.clientCalls++

		switch {
		case call.Invalid:
			partition.invalidOutputs = append(partition.invalidOutputs, invalidToolCallError(call))

		case !blocked[call.ToolCallID]:
			partition.executable = append(partition.executable, call)
		}
	}

	return partition
}

// notifyToolInput calls the tool's input callbacks for a complete call.
func notifyToolInput(ctx context.Context, tool Tool, call ToolCall, messages []model.Message) {
	options := ToolExecutionOptions{
		ToolCallID: call.ToolCallID,
		Messages:   messages,
	}

	if tool.OnInputStart != nil {
		tool.OnInputStart(ctx, options)
	}

	if tool.OnInputAvailable != nil {
		tool.OnInputAvailable(ctx, ToolInputEvent[json.RawMessage]{
			ToolExecutionOptions: options,
			Input:                call.Input,
		})
	}
}

// invalidToolCallError reports an invalid client tool call to the model.
func invalidToolCallError(call ToolCall) ToolError {
	return ToolError{
		ToolCallID:   call.ToolCallID,
		ToolName:     call.ToolName,
		Input:        call.Input,
		Error:        call.Error,
		Dynamic:      true,
		Title:        call.Title,
		ToolMetadata: call.ToolMetadata,
	}
}

func countDenied(responses []ToolApprovalResponse) int {
	denied := 0
	for _, response := range responses {
		if !response.Approved {
			denied++
		}
	}

	return denied
}

// providerResultCallIDs returns the tool call IDs of the provider-executed
// results in a response.
func providerResultCallIDs(content []model.Content) []string {
	var ids []string

	for _, part := range content {
		if result, ok := part.(model.ToolResult); ok {
			ids = append(ids, result.ToolCallID)
		}
	}

	return ids
}

// stepRequest retains the request payloads that Include selects.
func (r *generateTextRun) stepRequest(prepared preparedStep, request model.RequestInfo) StepRequest {
	return newStepRequest(retainedStepRequest{
		body:            request.Body,
		messages:        prepared.messages,
		includeBody:     r.options.Include.RequestBody,
		includeMessages: r.options.Include.RequestMessages,
	})
}

// stepResponse retains the response body when Include selects it.
func (r *generateTextRun) stepResponse(response model.ResponseInfo, messages []model.Message) StepResponse {
	if !r.options.Include.ResponseBody {
		response.Body = nil
	}

	return StepResponse{
		ResponseInfo: response,
		Messages:     messages,
	}
}

// parseOutput parses the final step's text when the step stopped normally or
// produced text without calling tools. It returns nil otherwise.
func (r *generateTextRun) parseOutput(ctx context.Context, finalStep StepResult) (any, error) {
	text := finalStep.Text()

	stoppedNormally := finalStep.FinishReason == model.FinishStop
	producedText := finalStep.FinishReason != model.FinishToolCalls && text != ""

	if !stoppedNormally && !producedText {
		return nil, nil
	}

	return r.output.ParseCompleteOutput(ctx, text, OutputContext{
		Response:     finalStep.Response.ResponseInfo,
		Usage:        finalStep.Usage,
		FinishReason: finalStep.FinishReason,
	})
}
