package ai

import (
	"context"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// defaultAgentStepCount is the number of steps after which a ToolLoopAgent
// without StopWhen conditions stops.
const defaultAgentStepCount = 20

// ToolLoopAgentSettings configure a ToolLoopAgent.
type ToolLoopAgentSettings struct {
	// ID identifies the agent. It may be empty.
	ID string

	// GenerateTextOptions are the settings of every call. System holds the
	// agent's instructions. Prompt and Messages are replaced by those of each
	// call. A nil StopWhen stops after 20 steps. The callbacks run before
	// the callbacks of each call.
	GenerateTextOptions

	// IncludeRawChunks emits RawPart parts with the provider's native events
	// in streaming calls.
	IncludeRawChunks bool

	// PrepareCall adjusts the settings of each call, for example from its
	// Options. Nil uses the settings as they are.
	PrepareCall AgentPrepareCallFunc
}

// AgentPrepareCallFunc returns the generation settings of one agent call.
// Callbacks set in the returned options are ignored; the agent attaches the
// callbacks of its settings and of the call.
type AgentPrepareCallFunc func(ctx context.Context, event AgentPrepareCallEvent) (GenerateTextOptions, error)

// AgentPrepareCallEvent describes the agent call being prepared.
type AgentPrepareCallEvent struct {
	// GenerateTextOptions are the agent's settings with the call's prompt or
	// messages applied and without callbacks.
	GenerateTextOptions

	// Options are the call options passed to Generate or Stream.
	Options any
}

// ToolLoopAgent runs tools in a loop. In each step it calls the model, and
// when the model calls tools it executes them and calls the model again with
// their results. The loop ends when the model finishes without tool calls,
// when a called tool has no Execute function or needs approval, or when a
// stop condition is met. When the default step limit ends the loop, the
// agent reports a warning to LogWarnings.
//
// Go adaptation: upstream tags requests with a user agent suffix; this SDK
// has no user agent, so it is omitted.
type ToolLoopAgent struct {
	settings ToolLoopAgentSettings
}

// NewToolLoopAgent returns an agent with the given settings.
func NewToolLoopAgent(settings ToolLoopAgentSettings) *ToolLoopAgent {
	return &ToolLoopAgent{
		settings: settings,
	}
}

// ID returns the ID of the agent settings.
func (a *ToolLoopAgent) ID() string {
	return a.settings.ID
}

// Tools returns the tools of the agent settings.
func (a *ToolLoopAgent) Tools() []Tool {
	return a.settings.Tools
}

// Generate runs the agent with GenerateText.
func (a *ToolLoopAgent) Generate(ctx context.Context, options AgentCallOptions) (*GenerateTextResult, error) {
	prepared, err := a.prepareCall(ctx, options)
	if err != nil {
		return nil, err
	}

	return GenerateText(ctx, a.withCallOptions(prepared, options))
}

// Stream runs the agent with StreamText.
func (a *ToolLoopAgent) Stream(ctx context.Context, options AgentStreamOptions) (*StreamTextResult, error) {
	prepared, err := a.prepareCall(ctx, options.AgentCallOptions)
	if err != nil {
		return nil, err
	}

	generation := a.withCallOptions(prepared, options.AgentCallOptions)

	return StreamText(ctx, a.streamTextOptions(generation, options.Transforms))
}

// prepareCall applies the call's prompt and the default stop condition to
// the settings and passes them through PrepareCall.
func (a *ToolLoopAgent) prepareCall(ctx context.Context, options AgentCallOptions) (GenerateTextOptions, error) {
	base := withoutCallbacks(a.settings.GenerateTextOptions)
	base.Prompt.Prompt = options.Prompt
	base.Messages = options.Messages

	if base.StopWhen == nil {
		base.StopWhen = []StopCondition{
			defaultAgentStopCondition(defaultAgentStepCount, base.LogWarnings),
		}
	}

	if a.settings.PrepareCall == nil {
		return base, nil
	}

	prepared, err := a.settings.PrepareCall(ctx, AgentPrepareCallEvent{
		GenerateTextOptions: base,
		Options:             options.Options,
	})
	if err != nil {
		return GenerateTextOptions{}, fmt.Errorf("prepare agent call: %w", err)
	}

	return prepared, nil
}

// withCallOptions applies the call's timeouts and attaches the callbacks of
// the settings followed by those of the call.
func (a *ToolLoopAgent) withCallOptions(prepared GenerateTextOptions, call AgentCallOptions) GenerateTextOptions {
	settings := a.settings.GenerateTextOptions

	if call.Timeout != nil {
		prepared.Timeout = *call.Timeout
	}

	prepared.OnStart = chainCallbacks(settings.OnStart, call.OnStart)
	prepared.OnStepStart = chainCallbacks(settings.OnStepStart, call.OnStepStart)
	prepared.OnLanguageModelCallStart = settings.OnLanguageModelCallStart
	prepared.OnLanguageModelCallEnd = settings.OnLanguageModelCallEnd
	prepared.OnToolExecutionStart = chainCallbacks(settings.OnToolExecutionStart, call.OnToolExecutionStart)
	prepared.OnToolExecutionEnd = chainCallbacks(settings.OnToolExecutionEnd, call.OnToolExecutionEnd)
	prepared.OnStepEnd = chainCallbacks(settings.OnStepEnd, call.OnStepEnd)
	prepared.OnEnd = chainCallbacks(settings.OnEnd, call.OnEnd)

	return prepared
}

// streamTextOptions converts the prepared generation settings into the
// options of a StreamText call.
func (a *ToolLoopAgent) streamTextOptions(generation GenerateTextOptions, transforms []StreamTransform) StreamTextOptions {
	return StreamTextOptions{
		Model:                    generation.Model,
		CallSettings:             generation.CallSettings,
		Prompt:                   generation.Prompt,
		Tools:                    generation.Tools,
		ToolChoice:               generation.ToolChoice,
		ActiveTools:              generation.ActiveTools,
		ToolOrder:                generation.ToolOrder,
		ToolCallConcurrency:      generation.ToolCallConcurrency,
		StopWhen:                 generation.StopWhen,
		Output:                   generation.Output,
		ToolApproval:             generation.ToolApproval,
		ToolApprovalSecret:       generation.ToolApprovalSecret,
		ToolCallers:              generation.ToolCallers,
		ProviderOptions:          generation.ProviderOptions,
		PrepareStep:              generation.PrepareStep,
		RepairToolCall:           generation.RepairToolCall,
		RefineToolInput:          generation.RefineToolInput,
		Transforms:               transforms,
		Download:                 generation.Download,
		Timeout:                  generation.Timeout,
		Telemetry:                generation.Telemetry,
		LogWarnings:              generation.LogWarnings,
		OnStart:                  generation.OnStart,
		OnStepStart:              generation.OnStepStart,
		OnLanguageModelCallStart: generation.OnLanguageModelCallStart,
		OnLanguageModelCallEnd:   generation.OnLanguageModelCallEnd,
		OnToolExecutionStart:     generation.OnToolExecutionStart,
		OnToolExecutionEnd:       generation.OnToolExecutionEnd,
		OnStepEnd:                generation.OnStepEnd,
		OnEnd:                    generation.OnEnd,
		Include: StreamTextInclude{
			RequestBody:     generation.Include.RequestBody,
			RequestMessages: generation.Include.RequestMessages,
			RawChunks:       a.settings.IncludeRawChunks,
		},
	}
}

// withoutCallbacks returns options without their callbacks, as PrepareCall
// receives them.
func withoutCallbacks(options GenerateTextOptions) GenerateTextOptions {
	options.OnStart = nil
	options.OnStepStart = nil
	options.OnLanguageModelCallStart = nil
	options.OnLanguageModelCallEnd = nil
	options.OnToolExecutionStart = nil
	options.OnToolExecutionEnd = nil
	options.OnStepEnd = nil
	options.OnEnd = nil

	return options
}

// defaultAgentStopCondition stops the loop after stepCount steps and warns
// that the default limit ended it. Stop conditions are only evaluated when
// the loop could otherwise continue, so a call that finishes naturally at
// the limit does not warn.
func defaultAgentStopCondition(stepCount int, logWarnings LogWarningsFunc) StopCondition {
	return func(ctx context.Context, event StopConditionEvent) bool {
		if len(event.Steps) != stepCount {
			return false
		}

		latest := event.Steps[len(event.Steps)-1]

		logWarnings.log(ctx, WarningsEvent{
			Warnings: []model.Warning{
				{
					Type: model.WarningOther,
					Message: fmt.Sprintf(
						"The tool loop stopped because it reached the default stop condition, IsStepCount(%d). "+
							"To allow more steps, set StopWhen to IsStepCount with a higher limit or provide a custom stop condition.",
						stepCount,
					),
				},
			},
			Provider: latest.Model.Provider,
			ModelID:  latest.Model.ModelID,
		})

		return true
	}
}

// chainCallbacks returns a callback that calls first and then second. Nil
// callbacks are skipped.
func chainCallbacks[Event any](first, second func(ctx context.Context, event Event)) func(ctx context.Context, event Event) {
	if first == nil {
		return second
	}

	if second == nil {
		return first
	}

	return func(ctx context.Context, event Event) {
		first(ctx, event)
		second(ctx, event)
	}
}
