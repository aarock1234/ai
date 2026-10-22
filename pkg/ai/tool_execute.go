package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

// ToolExecutionStartEvent describes a tool call about to run.
type ToolExecutionStartEvent struct {
	CallID   string
	ToolCall ToolCall

	// Messages are the messages sent to the model in the step that produced
	// the call.
	Messages []model.Message
}

// ToolExecutionEndEvent describes a finished tool call.
type ToolExecutionEndEvent struct {
	CallID   string
	ToolCall ToolCall
	Messages []model.Message

	// Output is the ToolResult or ToolError of the call.
	Output ToolOutput

	ExecutionTime time.Duration
}

// toolExecution describes how the tool calls of one step run.
type toolExecution struct {
	callID   string
	tools    []Tool
	messages []model.Message

	// concurrency limits the calls running at once; zero means no limit.
	concurrency int

	// timeout holds the tool timeouts.
	timeout   TimeoutConfiguration
	telemetry telemetryDispatcher

	onStart func(ctx context.Context, event ToolExecutionStartEvent)
	onEnd   func(ctx context.Context, event ToolExecutionEndEvent)

	// onPreliminary receives preliminary results. Nil discards them. It is
	// called from the goroutine running the tool.
	onPreliminary func(result ToolResult)
}

// executedToolCall is the outcome of one tool execution.
type executedToolCall struct {
	output        ToolOutput
	executionTime time.Duration
}

// executeToolCalls runs the calls of tools that have an Execute function,
// with the configured concurrency, and returns their outcomes in call order.
// Tool failures become ToolError outputs; the only error is the
// cancellation of ctx.
func executeToolCalls(ctx context.Context, execution toolExecution, calls []ToolCall) ([]executedToolCall, error) {
	results, err := mapConcurrently(ctx, calls, execution.concurrency, func(ctx context.Context, call ToolCall) (*executedToolCall, error) {
		return executeToolCall(ctx, execution, call), nil
	})
	if err != nil {
		return nil, err
	}

	executed := make([]executedToolCall, 0, len(results))
	for _, result := range results {
		if result != nil {
			executed = append(executed, *result)
		}
	}

	return executed, nil
}

// executeToolCall runs one tool call. It returns nil when the tool has no
// Execute function.
func executeToolCall(ctx context.Context, execution toolExecution, call ToolCall) *executedToolCall {
	tool, found := findTool(execution.tools, call.ToolName)
	if !found || tool.Execute == nil {
		return nil
	}

	startEvent := ToolExecutionStartEvent{
		CallID:   execution.callID,
		ToolCall: call,
		Messages: execution.messages,
	}

	if execution.onStart != nil {
		execution.onStart(ctx, startEvent)
	}

	execution.telemetry.onToolExecutionStart(ctx, startEvent)

	run := runTool(ctx, toolRun{
		execution: execution,
		tool:      tool,
		call:      call,
		event:     startEvent,
	})

	outcome := &executedToolCall{
		output:        newToolOutput(tool, call, run.output, run.err),
		executionTime: run.executionTime,
	}

	endEvent := ToolExecutionEndEvent{
		CallID:        execution.callID,
		ToolCall:      call,
		Messages:      execution.messages,
		Output:        outcome.output,
		ExecutionTime: run.executionTime,
	}

	if execution.onEnd != nil {
		execution.onEnd(ctx, endEvent)
	}

	execution.telemetry.onToolExecutionEnd(ctx, endEvent)

	return outcome
}

// toolRun is one execution of a tool's Execute function.
type toolRun struct {
	execution toolExecution
	tool      Tool
	call      ToolCall
	event     ToolExecutionStartEvent
}

// toolRunResult is the outcome of a toolRun.
type toolRunResult struct {
	output        any
	err           error
	executionTime time.Duration
}

// runTool calls Execute inside the telemetry wrappers, limited by the
// tool's timeout. A tool that fails because its context ended reports the
// cancellation cause, such as a *TimeoutError.
func runTool(ctx context.Context, run toolRun) toolRunResult {
	timeout := run.execution.timeout.toolTimeout(run.call.ToolName)

	toolCtx, cancel := withTimeout(ctx, TimeoutTool, timeout)
	defer cancel()

	options := ExecuteToolOptions{
		Tool:       run.tool,
		Input:      run.call.Input,
		ToolCallID: run.call.ToolCallID,
		Messages:   run.execution.messages,
	}

	if onPreliminary := run.execution.onPreliminary; onPreliminary != nil {
		options.OnPreliminary = func(output any) {
			onPreliminary(preliminaryToolResult(run.tool, run.call, output))
		}
	}

	var result toolRunResult

	result.err = run.execution.telemetry.executeTool(toolCtx, run.event, func(spanCtx context.Context) error {
		started := time.Now()
		output, err := ExecuteTool(spanCtx, options)
		result.executionTime = time.Since(started)
		result.output = output

		return contextCauseOf(toolCtx, err)
	})

	return result
}

// contextCauseOf replaces err with the cancellation cause of ctx when err is
// the error of the ended ctx, so a tool timeout surfaces as *TimeoutError.
func contextCauseOf(ctx context.Context, err error) error {
	if err == nil || ctx.Err() == nil {
		return err
	}

	if !errors.Is(err, ctx.Err()) {
		return err
	}

	return context.Cause(ctx)
}

// ExecuteToolOptions configure ExecuteTool.
type ExecuteToolOptions struct {
	Tool Tool

	// Input is the JSON input of the call.
	Input json.RawMessage

	ToolCallID string

	// Messages are the messages sent to the model in the step that produced
	// the call.
	Messages []model.Message

	// OnPreliminary receives each preliminary output the tool reports with
	// ToolExecutionOptions.Preliminary. It is called from the goroutine
	// running Execute. Nil discards preliminary outputs.
	OnPreliminary func(output any)
}

// ExecuteTool runs a tool's Execute function outside a generation, such as
// in an agent runtime that executes tools on the host, and returns its final
// output. It is the Go form of upstream's executeTool: preliminary outputs
// go to OnPreliminary instead of a stream of parts.
func ExecuteTool(ctx context.Context, options ExecuteToolOptions) (any, error) {
	if options.Tool.err != nil {
		return nil, options.Tool.err
	}

	if options.Tool.Execute == nil {
		return nil, fmt.Errorf("execute tool %q: tool has no execute function", options.Tool.Name)
	}

	return options.Tool.Execute(ctx, options.Input, ToolExecutionOptions{
		ToolCallID:  options.ToolCallID,
		Messages:    options.Messages,
		preliminary: options.OnPreliminary,
	})
}

func newToolOutput(tool Tool, call ToolCall, output any, err error) ToolOutput {
	if err != nil {
		return ToolError{
			ToolCallID:       call.ToolCallID,
			ToolName:         call.ToolName,
			Input:            call.Input,
			Error:            err,
			Dynamic:          tool.kind() == ToolKindDynamic,
			Title:            call.Title,
			ProviderMetadata: call.ProviderMetadata,
			ToolMetadata:     call.ToolMetadata,
		}
	}

	return newToolResult(tool, call, output)
}

func newToolResult(tool Tool, call ToolCall, output any) ToolResult {
	return ToolResult{
		ToolCallID:       call.ToolCallID,
		ToolName:         call.ToolName,
		Input:            call.Input,
		Output:           output,
		Dynamic:          tool.kind() == ToolKindDynamic,
		Title:            call.Title,
		ProviderMetadata: call.ProviderMetadata,
		ToolMetadata:     call.ToolMetadata,
	}
}

func preliminaryToolResult(tool Tool, call ToolCall, output any) ToolResult {
	result := newToolResult(tool, call, output)
	result.Preliminary = true

	return result
}
