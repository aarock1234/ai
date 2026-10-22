package ai

import (
	"context"
	"slices"
)

// StopCondition decides whether a tool-calling loop stops after the latest
// step. The loop also stops when the model finishes without tool calls, when
// a called tool has no Execute function and when a tool call needs approval.
type StopCondition func(ctx context.Context, event StopConditionEvent) bool

// StopConditionEvent describes the completed steps.
type StopConditionEvent struct {
	Steps []StepResult
}

// IsStepCount stops the loop once stepCount steps have completed.
func IsStepCount(stepCount int) StopCondition {
	return func(_ context.Context, event StopConditionEvent) bool {
		return len(event.Steps) == stepCount
	}
}

// IsLoopFinished never stops the loop, so that it runs until one of its
// natural ends.
func IsLoopFinished() StopCondition {
	return func(context.Context, StopConditionEvent) bool {
		return false
	}
}

// HasToolCall stops the loop when the latest step calls one of the named
// tools.
func HasToolCall(toolNames ...string) StopCondition {
	return func(_ context.Context, event StopConditionEvent) bool {
		if len(event.Steps) == 0 {
			return false
		}

		latest := event.Steps[len(event.Steps)-1]

		return slices.ContainsFunc(latest.ToolCalls(), func(call ToolCall) bool {
			return slices.Contains(toolNames, call.ToolName)
		})
	}
}

// isStopConditionMet reports whether any condition is met. Every condition
// is evaluated, as upstream does.
func isStopConditionMet(ctx context.Context, conditions []StopCondition, steps []StepResult) bool {
	met := false

	for _, condition := range conditions {
		if condition(ctx, StopConditionEvent{
			Steps: steps,
		}) {
			met = true
		}
	}

	return met
}

// stopConditionsOrDefault returns the conditions, defaulting to a single
// step.
func stopConditionsOrDefault(conditions []StopCondition) []StopCondition {
	if len(conditions) == 0 {
		return []StopCondition{
			IsStepCount(1),
		}
	}

	return conditions
}
