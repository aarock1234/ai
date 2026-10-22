package harness

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"slices"

	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/sandbox"
)

// managedHeaders are the request headers the harness sets itself.
var managedHeaders = []string{
	"authorization",
	"x-api-key",
	"user-agent",
	"x-client-app",
}

// Agent drives a third-party agent runtime through a harness adapter, the Go
// form of upstream HarnessAgent.
//
// An Agent is a stateless definition: construct it once and pass an
// AgentSession, created with CreateSession, to every Generate and Stream
// call. Application tools run on the host when the runtime calls them, with
// the session's restricted sandbox attached to their context (see
// sandbox.SessionFromContext); built-in tools run inside the runtime.
//
// Go adaptation: Agent does not implement ai.Agent, because a harness turn
// needs a session and its stream is a StreamResult rather than an
// ai.StreamTextResult. The stream parts and results use the ai types.
type Agent struct {
	settings         AgentSettings
	tools            []ai.Tool
	builtinFiltering *BuiltinToolFiltering
	permissionMode   PermissionMode
	headers          http.Header
}

// New validates the settings and returns an agent.
func New(settings AgentSettings) (*Agent, error) {
	if settings.Harness == nil {
		return nil, errors.New("harness agent: a harness is required")
	}

	if err := validateSandboxConfig(settings.SandboxConfig); err != nil {
		return nil, err
	}

	headers, err := normalizeHeaders(settings.Headers)
	if err != nil {
		return nil, err
	}

	agent := &Agent{
		settings:       settings,
		permissionMode: resolvePermissionMode(settings.PermissionMode),
		headers:        headers,
	}

	prepared, err := agent.prepareTurnSettings(PreparedCall{
		Model:        settings.Model,
		Skills:       settings.Skills,
		Instructions: settings.Instructions,
		Tools:        settings.Tools,
	})
	if err != nil {
		return nil, err
	}

	if err := agent.checkPermissionModeSupport(); err != nil {
		return nil, err
	}

	agent.tools = prepared.tools
	agent.builtinFiltering = prepared.builtinToolFiltering

	return agent, nil
}

// ID returns the agent's identifier, which may be empty.
func (a *Agent) ID() string {
	return a.settings.ID
}

// HarnessID returns the identifier of the agent's harness.
func (a *Agent) HarnessID() string {
	return a.settings.Harness.HarnessID()
}

// HasOutput reports whether the agent parses turns with a configured output.
func (a *Agent) HasOutput() bool {
	return a.settings.Output != nil
}

// Tools returns the merged tool set: the harness's built-in tools and the
// application tools, application tools overriding built-in tools of the same
// name.
func (a *Agent) Tools() []ai.Tool {
	return slices.Clone(a.tools)
}

// SandboxTemplate returns the template that prepares sandboxes for this
// agent's harness and bootstrap hook, or nil when nothing needs preparing.
// Pass it to a sandbox session creator to reuse snapshots.
func (a *Agent) SandboxTemplate(ctx context.Context) (*SandboxTemplate, error) {
	return CreateSandboxTemplate(ctx, CreateSandboxTemplateOptions{
		Harnesses: []Harness{
			a.settings.Harness,
		},
		SandboxConfig: a.settings.SandboxConfig,
	})
}

// CreateSession starts a fresh session, or resumes one from ResumeFrom or
// ContinueFrom. It applies the harness's bootstrap recipe and the sandbox
// hooks, creates the session's working directory and starts the adapter.
func (a *Agent) CreateSession(ctx context.Context, options CreateSessionOptions) (*AgentSession, error) {
	if options.SandboxSession == nil {
		return nil, errors.New("harness agent: create session requires a sandbox session")
	}

	if options.ResumeFrom != nil && options.ContinueFrom != nil {
		return nil, errors.New("harness agent: pass either ResumeFrom or ContinueFrom, not both")
	}

	resumeFrom, continueFrom, err := a.validateLifecycleStates(options)
	if err != nil {
		return nil, err
	}

	sessionID := options.SessionID
	if sessionID == "" {
		sessionID = id.New("")
	}

	sessionWorkDir, err := a.prepareSandbox(ctx, options.SandboxSession, sessionID)
	if err != nil {
		return nil, err
	}

	underlying, err := a.settings.Harness.Start(ctx, StartOptions{
		SessionID:            sessionID,
		Headers:              a.headers,
		ResumeFrom:           resumeFrom,
		ContinueFrom:         continueFrom,
		PermissionMode:       a.permissionMode,
		BuiltinToolFiltering: a.builtinFiltering,
		Observability: buildObservability(observabilitySetup{
			debug:     a.settings.Debug,
			onLog:     a.settings.OnLog,
			telemetry: a.settings.Telemetry,
		}),
		SandboxSession: options.SandboxSession,
		SessionWorkDir: sessionWorkDir,
	})
	if err != nil {
		return nil, fmt.Errorf("start harness %q: %w", a.HarnessID(), err)
	}

	return newAgentSession(agentSessionSetup{
		sessionID:      sessionID,
		harness:        a.settings.Harness,
		underlying:     underlying,
		sandbox:        options.SandboxSession,
		sessionWorkDir: sessionWorkDir,
		toolApproval:   a.settings.ToolApproval,
		continueFrom:   continueFrom,
	}), nil
}

// validateLifecycleStates validates the resume or continuation state and
// returns the continuation to resume, if any.
func (a *Agent) validateLifecycleStates(options CreateSessionOptions) (*ResumeSessionState, *ContinueTurnState, error) {
	var resumeFrom *ResumeSessionState
	var continueFrom *ContinueTurnState

	if options.ResumeFrom != nil {
		validated, err := validateResumeSessionState(a.settings.Harness, *options.ResumeFrom)
		if err != nil {
			return nil, nil, err
		}

		resumeFrom = &validated
		continueFrom = validated.ContinueFrom
	}

	if options.ContinueFrom != nil {
		validated, err := validateContinueTurnState(a.settings.Harness, *options.ContinueFrom)
		if err != nil {
			return nil, nil, err
		}

		continueFrom = &validated
	}

	return resumeFrom, continueFrom, nil
}

// prepareSandbox applies the bootstrap recipe and hooks and returns the
// session's working directory.
func (a *Agent) prepareSandbox(ctx context.Context, session sandbox.Session, sessionID string) (string, error) {
	restricted := RestrictedSession(session)
	config := a.settings.SandboxConfig

	defaultWorkingDirectory, err := ResolveSandboxDefaultWorkingDirectory(ctx, session)
	if err != nil {
		return "", err
	}

	workDir, err := optionalWorkDir(config.WorkDir)
	if err != nil {
		return "", err
	}

	sessionWorkDir := resolveSessionWorkDir(sessionWorkDirInput{
		defaultWorkingDirectory: defaultWorkingDirectory,
		harnessID:               a.HarnessID(),
		sessionID:               sessionID,
		workDir:                 workDir,
	})

	recipe, err := harnessBootstrap(ctx, a.settings.Harness)
	if err != nil {
		return "", err
	}

	if recipe != nil {
		if err := applySessionBootstrap(ctx, restricted, *recipe); err != nil {
			return "", err
		}
	}

	err = runSandboxBootstrap(ctx, sandboxBootstrapInput{
		session:                 restricted,
		workDir:                 workDir,
		onBootstrap:             config.OnBootstrap,
		bootstrapHash:           config.BootstrapHash,
		skipOnBootstrapIfMarked: true,
	})
	if err != nil {
		return "", err
	}

	if err := ensureSandboxDirectory(ctx, session, sessionWorkDir); err != nil {
		return "", err
	}

	if config.OnSession != nil {
		err := config.OnSession(ctx, SandboxSessionEvent{
			Session:        restricted,
			SessionWorkDir: sessionWorkDir,
		})
		if err != nil {
			return "", fmt.Errorf("run sandbox session hook: %w", err)
		}
	}

	return sessionWorkDir, nil
}

// Generate runs one turn to completion and returns its aggregated result. A
// trailing tool message with approval responses or tool results continues
// the session's unfinished turn instead. Ctx governs the turn.
func (a *Agent) Generate(ctx context.Context, options CallOptions) (*ai.GenerateTextResult, error) {
	stream, err := a.Stream(ctx, options)
	if err != nil {
		return nil, err
	}

	return a.generateResult(stream)
}

// Stream starts one turn and returns once the adapter accepted it. A
// trailing tool message with approval responses or tool results continues
// the session's unfinished turn instead. Ctx governs the turn: cancelling it
// ends the stream with an ai.AbortPart.
func (a *Agent) Stream(ctx context.Context, options CallOptions) (*StreamResult, error) {
	if options.Session == nil {
		return nil, errSessionRequired
	}

	continuation, err := continuationOf(options.Prompt, options.Messages)
	if err != nil {
		return nil, err
	}

	if continuation != nil {
		return a.startContinuation(ctx, options.Session, *continuation)
	}

	request, err := a.preparePromptTurn(ctx, options)
	if err != nil {
		return nil, err
	}

	stream, turnID, err := options.Session.startPromptTurn(ctx, request)
	if err != nil {
		return nil, err
	}

	return a.awaitReady(ctx, options.Session, stream, turnID)
}

// ContinueGenerate continues the session's unfinished turn without a new
// prompt and waits for it, after CreateSession with ContinueFrom or after a
// turn paused for host input.
func (a *Agent) ContinueGenerate(ctx context.Context, options ContinueOptions) (*ai.GenerateTextResult, error) {
	stream, err := a.ContinueStream(ctx, options)
	if err != nil {
		return nil, err
	}

	return a.generateResult(stream)
}

// ContinueStream continues the session's unfinished turn without a new
// prompt and streams it. What the adapter can guarantee, a lossless attach
// or a lossy rerun, follows from how it resumed the session.
func (a *Agent) ContinueStream(ctx context.Context, options ContinueOptions) (*StreamResult, error) {
	if options.Session == nil {
		return nil, errSessionRequired
	}

	return a.startContinuation(ctx, options.Session, turnContinuation{
		approvals: options.ToolApprovals,
		results:   options.ToolResults,
	})
}

// SteerOptions configure Agent.Steer.
type SteerOptions struct {
	Session *AgentSession
	Text    string
}

// Steer submits another user message to the session's running turn; see
// AgentSession.SteerTurn. This API is experimental.
func (a *Agent) Steer(ctx context.Context, options SteerOptions) error {
	if options.Session == nil {
		return errSessionRequired
	}

	return options.Session.SteerTurn(ctx, options.Text)
}

// turnContinuation is the host input that continues an unfinished turn.
type turnContinuation struct {
	approvals []model.ToolApprovalResponsePart
	results   []model.ToolResultPart
}

// startContinuation continues the session's unfinished turn with the
// agent's own settings, which the session replaces with the settings frozen
// when the turn began.
func (a *Agent) startContinuation(ctx context.Context, session *AgentSession, continuation turnContinuation) (*StreamResult, error) {
	request, err := a.baseTurnRequest(turnContinue, PreparedCall{
		Model:        a.settings.Model,
		Skills:       a.settings.Skills,
		Instructions: a.settings.Instructions,
		Tools:        a.settings.Tools,
	}, lifecycleCallbacksOf(a.settings, CallOptions{}))
	if err != nil {
		return nil, err
	}

	request.approvalContinuations = continuation.approvals
	request.resultContinuations = continuation.results

	stream, turnID, err := session.startContinueTurn(ctx, request)
	if err != nil {
		return nil, err
	}

	return a.awaitReady(ctx, session, stream, turnID)
}

// awaitReady waits until the adapter started the turn or the turn ended.
func (a *Agent) awaitReady(ctx context.Context, session *AgentSession, stream *StreamResult, turnID int) (*StreamResult, error) {
	if err := session.waitForPromptControl(ctx, turnID); err != nil {
		_ = stream.Close()

		return nil, err
	}

	return stream, nil
}

// generateResult waits for a turn and returns its result, failing when the
// configured output could not be parsed.
func (a *Agent) generateResult(stream *StreamResult) (*ai.GenerateTextResult, error) {
	result, err := stream.Result()
	if err != nil {
		return nil, err
	}

	if stream.result.outcome.outputErr != nil && a.settings.Output != nil {
		return nil, stream.result.outcome.outputErr
	}

	return result, nil
}

// preparePromptTurn applies PrepareCall and prepares a fresh turn.
func (a *Agent) preparePromptTurn(ctx context.Context, options CallOptions) (turnRequest, error) {
	call := PreparedCall{
		Options:      options.Options,
		Prompt:       options.Prompt,
		Messages:     options.Messages,
		Model:        a.settings.Model,
		Skills:       a.settings.Skills,
		Instructions: a.settings.Instructions,
		Tools:        a.settings.Tools,
	}

	if a.settings.PrepareCall != nil {
		prepared, err := a.settings.PrepareCall(ctx, call)
		if err != nil {
			return turnRequest{}, fmt.Errorf("prepare harness call: %w", err)
		}

		call = prepared
	}

	continuation, err := continuationOf(call.Prompt, call.Messages)
	if err != nil {
		return turnRequest{}, err
	}

	if continuation != nil {
		return turnRequest{}, errors.New("harness agent: PrepareCall must return a fresh user prompt, not a tool continuation")
	}

	prompt, err := promptOf(call.Prompt, call.Messages)
	if err != nil {
		return turnRequest{}, err
	}

	request, err := a.baseTurnRequest(turnPrompt, call, lifecycleCallbacksOf(a.settings, options))
	if err != nil {
		return turnRequest{}, err
	}

	request.prompt = prompt

	return request, nil
}

// baseTurnRequest prepares the settings shared by fresh and continued
// turns.
func (a *Agent) baseTurnRequest(mode turnMode, call PreparedCall, callbacks lifecycleCallbacks) (turnRequest, error) {
	prepared, err := a.prepareTurnSettings(call)
	if err != nil {
		return turnRequest{}, err
	}

	responseFormat, err := a.responseFormat()
	if err != nil {
		return turnRequest{}, err
	}

	return turnRequest{
		mode:                 mode,
		settings:             prepared.settings,
		tools:                prepared.tools,
		activeTools:          prepared.activeTools,
		builtinToolFiltering: prepared.builtinToolFiltering,
		responseFormat:       responseFormat,
		output:               a.settings.Output,
		telemetry:            a.settings.Telemetry,
		callbacks:            callbacks,
		stopConditions:       a.settings.StopWhen,
	}, nil
}

// preparedTurnSettings are the resolved settings of a turn.
type preparedTurnSettings struct {
	settings             TurnSettings
	tools                []ai.Tool
	activeTools          []ai.Tool
	builtinToolFiltering *BuiltinToolFiltering
}

// prepareTurnSettings merges the built-in and application tools and
// resolves the tool filtering of a turn.
func (a *Agent) prepareTurnSettings(call PreparedCall) (preparedTurnSettings, error) {
	harness := a.settings.Harness

	if isBuiltinTool(harness, string(ToolAskUserQuestions)) && hasTool(call.Tools, string(ToolAskUserQuestions)) {
		return preparedTurnSettings{}, errors.New("harness agent: tool name askUserQuestions is reserved for harness question requests")
	}

	tools := mergeTools(harness.BuiltinTools(), call.Tools)

	filtering, err := resolveToolFiltering(toolFilteringInput{
		harness:       harness,
		userTools:     call.Tools,
		allTools:      tools,
		activeTools:   a.settings.ActiveTools,
		inactiveTools: a.settings.InactiveTools,
	})
	if err != nil {
		return preparedTurnSettings{}, err
	}

	return preparedTurnSettings{
		settings: TurnSettings{
			Model:        call.Model,
			Skills:       call.Skills,
			Instructions: call.Instructions,
			Tools:        toolSpecsOf(filtering.activeUserTools),
		},
		tools:                tools,
		activeTools:          filtering.activeUserTools,
		builtinToolFiltering: filtering.builtinToolFiltering,
	}, nil
}

// checkPermissionModeSupport fails when a restrictive permission mode needs
// built-in tool approvals the harness cannot request.
func (a *Agent) checkPermissionModeSupport() error {
	if a.permissionMode == PermissionAllowAll || len(a.settings.Harness.BuiltinTools()) == 0 {
		return nil
	}

	if supportsBuiltinToolApprovals(a.settings.Harness) {
		return nil
	}

	return &CapabilityUnsupportedError{
		Message: fmt.Sprintf(
			"Harness '%s' does not support built-in tool approval requests; use permissionMode: 'allow-all'.",
			a.HarnessID(),
		),
		HarnessID: a.HarnessID(),
	}
}

// responseFormat returns the response format of the configured output.
func (a *Agent) responseFormat() (*model.ResponseFormat, error) {
	if a.settings.Output == nil {
		return nil, nil
	}

	format, err := a.settings.Output.ResponseFormat()
	if err != nil {
		return nil, fmt.Errorf("resolve harness response format: %w", err)
	}

	if format == nil || format.Type == model.ResponseFormatText {
		return format, nil
	}

	resolved := *format

	return &resolved, nil
}

// lifecycleCallbacksOf merges the agent's callbacks with a call's, the
// agent's first.
func lifecycleCallbacksOf(settings AgentSettings, call CallOptions) lifecycleCallbacks {
	return lifecycleCallbacks{
		onStart:                  callbacks(settings.OnStart, call.OnStart),
		onStepStart:              callbacks(settings.OnStepStart, call.OnStepStart),
		onLanguageModelCallStart: callbacks(settings.OnLanguageModelCallStart),
		onLanguageModelCallEnd:   callbacks(settings.OnLanguageModelCallEnd),
		onToolExecutionStart:     callbacks(settings.OnToolExecutionStart, call.OnToolExecutionStart),
		onToolExecutionEnd:       callbacks(settings.OnToolExecutionEnd, call.OnToolExecutionEnd),
		onStepEnd:                callbacks(settings.OnStepEnd, call.OnStepEnd),
		onEnd:                    callbacks(settings.OnEnd, call.OnEnd),
	}
}

// callbacks returns the non-nil callbacks in order.
func callbacks[Event any](candidates ...func(ctx context.Context, event Event)) []func(ctx context.Context, event Event) {
	var present []func(ctx context.Context, event Event)

	for _, candidate := range candidates {
		if candidate != nil {
			present = append(present, candidate)
		}
	}

	return present
}
