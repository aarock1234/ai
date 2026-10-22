package ai

import (
	"context"
	"sync"
	"time"
)

// streamedToolExecution describes tool calls whose outcomes are streamed.
type streamedToolExecution struct {
	execution toolExecution
	calls     []ToolCall
	yield     func(StreamPart) bool
}

// streamedToolOutputs are the final outputs of streamed tool executions.
type streamedToolOutputs struct {
	outputs        []ToolOutput
	executionTimes map[string]time.Duration
}

// toolStreamEvent is a preliminary or final tool outcome sent from a tool
// goroutine to the consumer.
type toolStreamEvent struct {
	part StreamPart

	// executed is set for final outcomes.
	executed *executedToolCall
}

// streamToolExecutions runs the calls concurrently and streams preliminary
// results and outcomes as they arrive. The consumer's goroutine owns the
// tool goroutines: when the consumer stops, the tools are cancelled and
// awaited before it returns false.
func (r *streamTextRun) streamToolExecutions(ctx context.Context, request streamedToolExecution) (streamedToolOutputs, bool) {
	executed := streamedToolOutputs{
		executionTimes: make(map[string]time.Duration),
	}

	if len(request.calls) == 0 {
		return executed, true
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	events := make(chan toolStreamEvent)

	execution := request.execution
	execution.onPreliminary = func(result ToolResult) {
		sendToolStreamEvent(ctx, events, toolStreamEvent{
			part: result,
		})
	}

	var group sync.WaitGroup
	defer group.Wait()

	group.Go(func() {
		defer close(events)

		// The outcomes are sent as events, and the work never fails, so the
		// results and error of mapConcurrently carry nothing.
		_, _ = mapConcurrently(ctx, request.calls, execution.concurrency, func(ctx context.Context, call ToolCall) (struct{}, error) {
			outcome := executeToolCall(ctx, execution, call)
			if outcome != nil {
				sendToolStreamEvent(ctx, events, toolStreamEvent{
					part:     outcome.output,
					executed: outcome,
				})
			}

			return struct{}{}, nil
		})
	})

	for event := range events {
		if event.executed != nil {
			executed.outputs = append(executed.outputs, event.executed.output)
			executed.executionTimes[toolOutputCallID(event.executed.output)] = event.executed.executionTime
		}

		if !request.yield(event.part) {
			cancel()
			drainToolStreamEvents(events)

			return executed, false
		}
	}

	return executed, true
}

func sendToolStreamEvent(ctx context.Context, events chan<- toolStreamEvent, event toolStreamEvent) {
	select {
	case events <- event:
	case <-ctx.Done():
	}
}

func drainToolStreamEvents(events <-chan toolStreamEvent) {
	for range events {
	}
}
