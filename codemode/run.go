package codemode

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"sync/atomic"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// invocationCounter numbers invocations without an outer tool call ID.
var invocationCounter atomic.Int64

// RunOptions configure Run.
type RunOptions struct {
	// JS is the program, JavaScript or TypeScript whose types are stripped
	// before it runs. It runs as the body of an async function, so
	// top-level await and return work.
	JS string

	// Tools are the host tools available to the program as tools.name(input).
	Tools []ai.Tool

	// ToolCallID is the ID of the code mode tool call that runs the
	// program. Host tool calls get IDs derived from it. It defaults to the
	// continuation's ID, or a generated one.
	ToolCallID string

	// Messages are forwarded to the host tools.
	Messages []model.Message

	Options Options

	// Continuation and InterruptResolution resume an interrupted
	// invocation. ContinueInterrupt and ContinueApproval set them.
	Continuation        *Continuation
	InterruptResolution *InterruptResolution
}

// InterruptResolution resolves the next pending interrupt of a
// continuation.
type InterruptResolution struct {
	InterruptID string

	// Resolution must encode to JSON within
	// ExecutionPolicy.MaxToolOutputBytes. Approval interrupts take an
	// ApprovalDecision.
	Resolution any
}

// ResultStatus reports whether an invocation completed or was interrupted.
type ResultStatus string

// Result statuses.
const (
	ResultStatusCompleted   ResultStatus = "completed"
	ResultStatusInterrupted ResultStatus = "interrupted"
)

// Result is the outcome of an invocation.
type Result struct {
	Status ResultStatus

	// Output is the JSON encoding of the value the program returned, with
	// JSON.stringify semantics. It is nil when the program returned
	// undefined or was interrupted.
	Output json.RawMessage

	// Interrupt is set when Status is ResultStatusInterrupted.
	Interrupt *Interrupt

	// Console is the console output of this execution, within
	// ExecutionPolicy.MaxConsoleOutputBytes. ConsoleTruncated reports
	// that later output was dropped.
	Console          []ConsoleEntry
	ConsoleTruncated bool
}

// Run executes one program in a fresh sandbox with the given host tools and
// returns the JSON value it returns, or the interrupt it stopped at.
//
// The program's TypeScript syntax is stripped before it runs. Host tool
// calls run concurrently on their own goroutines; their inputs are
// validated against the tools' input schemas, and tools that need approval
// follow Options.Approval. Cancelling ctx or exceeding the timeout
// interrupts the program, cancels the context of running host tools and
// waits for them to return. An invocation that would exceed the
// process-wide cap of SetMaxWorkers fails with ErrorCodeConcurrency.
func Run(ctx context.Context, options RunOptions) (Result, error) {
	limits := resolveExecutionLimits(options.Options.ExecutionPolicy)

	if err := checkSourceSize(options.JS, limits.maxSourceBytes); err != nil {
		return Result{}, err
	}

	security, err := resolveContinuationSecurity(options.Options.ContinuationSecurity)
	if err != nil {
		return Result{}, fmt.Errorf("code mode: %w", err)
	}

	tools, toolNames, err := indexTools(options.Tools)
	if err != nil {
		return Result{}, err
	}

	resumption, err := prepareContinuation(continuationInput{
		options:   options,
		toolNames: toolNames,
		limits:    limits,
		security:  security,
	})
	if err != nil {
		return Result{}, err
	}

	if resumption.nextInterrupt != nil {
		return Result{
			Status:    ResultStatusInterrupted,
			Interrupt: resumption.nextInterrupt,
		}, nil
	}

	releaseWorker, err := acquireWorker()
	if err != nil {
		return Result{}, err
	}
	defer releaseWorker()

	outerToolCallID := outerToolCallID(options)

	scope, err := newContinuationScope(options.JS, outerToolCallID, toolNames)
	if err != nil {
		return Result{}, err
	}

	replay, err := newReplay(replayInput{
		resumption: resumption,
		scope:      scope,
		limits:     limits,
		security:   security,
	})
	if err != nil {
		return Result{}, err
	}

	outcome, err := runSandbox(ctx, sandboxInput{
		js: options.JS,
		host: &hostTools{
			tools:           tools,
			names:           toolNames,
			validators:      compileValidators(tools),
			options:         options.Options,
			limits:          limits,
			messages:        options.Messages,
			outerToolCallID: outerToolCallID,
		},
		replay:    replay,
		limits:    limits,
		onConsole: options.Options.OnConsole,
	})
	if err != nil {
		return Result{}, err
	}

	result := Result{
		Status:           ResultStatusCompleted,
		Output:           outcome.output,
		Console:          outcome.console.entries,
		ConsoleTruncated: outcome.console.truncated,
	}

	if len(outcome.interruptions) == 0 {
		return result, nil
	}

	interrupt, err := interruptOutcome(interruptInput{
		js:              options.JS,
		outerToolCallID: outerToolCallID,
		toolNames:       toolNames,
		outcome:         outcome,
		state:           replay.state,
		scope:           scope,
		security:        security,
	})
	if err != nil {
		return Result{}, err
	}

	result.Status = ResultStatusInterrupted
	result.Output = nil
	result.Interrupt = interrupt

	return result, nil
}

func checkSourceSize(js string, maxBytes int) error {
	if len(js) <= maxBytes {
		return nil
	}

	return &Error{
		Code:    ErrorCodeSourceTooLarge,
		Message: fmt.Sprintf("code mode source exceeds the %d byte size limit", maxBytes),
		Limit:   int64(maxBytes),
		Size:    int64(len(js)),
	}
}

func outerToolCallID(options RunOptions) string {
	if options.ToolCallID != "" {
		return options.ToolCallID
	}

	if options.Continuation != nil {
		return options.Continuation.OuterToolCallID
	}

	return fmt.Sprintf("code-mode-%d", invocationCounter.Add(1))
}

// interruptInput is what interruptOutcome needs to describe an
// interrupted execution.
type interruptInput struct {
	js              string
	outerToolCallID string
	toolNames       []string
	outcome         sandboxOutcome
	state           runState
	scope           continuationScope
	security        continuationSecurity
}

// interruptOutcome signs a continuation for the interrupted batch and
// exposes its first interruption.
func interruptOutcome(input interruptInput) (*Interrupt, error) {
	state := input.state
	state.Ledger = input.outcome.ledger
	state.ScopeHash = input.scope.scopeHash
	state.Source = input.scope.source

	token, err := encodeRunToken(state, input.security)
	if err != nil {
		return nil, err
	}

	pending := make([]PendingInterruption, 0, len(input.outcome.interruptions))

	for _, interruption := range input.outcome.interruptions {
		toolCallID := nestedToolCallID(input.outerToolCallID, interruption.request)

		pending = append(pending, PendingInterruption{
			RunInterruptionID: interruption.interruptionID,
			InterruptID:       toolCallID + ":interrupt",
			ToolName:          interruption.toolName,
			ToolCallID:        toolCallID,
			Input:             interruption.input,
			Payload:           interruption.payload,
		})
	}

	continuation, err := signContinuation(Continuation{
		Version:              continuationVersion,
		JS:                   input.js,
		OuterToolCallID:      input.outerToolCallID,
		ToolNames:            slices.Clone(input.toolNames),
		Token:                token,
		PendingInterruptions: pending,
		Resolutions:          []PendingResolution{},
	}, input.security)
	if err != nil {
		return nil, err
	}

	return interruptAt(continuation, 0)
}
