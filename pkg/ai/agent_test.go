package ai_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// countingToolModel requests the counter tool until maxCalls is
// reached, then answers with plain text.
func countingToolModel(calls *int, maxCalls int) *mockModel {
	return &mockModel{
		generateFn: func(_ context.Context, _ *model.Request) (*model.Response, error) {
			if *calls < maxCalls {
				*calls++

				return &model.Response{
					Content: []model.Content{
						model.ToolCallContent{
							ToolCallID: "call_1",
							ToolName:   "counter",
							Args:       `{}`,
						},
					},
					FinishReason: model.FinishReasonToolCalls,
				}, nil
			}

			return &model.Response{
				Content:      []model.Content{model.TextContent{Text: "done"}},
				FinishReason: model.FinishReasonStop,
			}, nil
		},
	}
}

func counterTool() ai.Tool {
	return ai.Tool{
		Name:        "counter",
		Description: "counts",
		InputSchema: json.RawMessage(`{"type":"object"}`),
		Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
			return "counted", nil
		},
	}
}

// TestAgentDefaultMaxSteps verifies the agent runs the tool loop with
// the 20-step default rather than GenerateText's single step.
func TestAgentDefaultMaxSteps(t *testing.T) {
	calls := 0
	agent := ai.NewAgent(countingToolModel(&calls, 3),
		ai.WithTools(counterTool()),
	)

	result, err := agent.Generate(context.Background(), ai.WithPrompt("count to 3"))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	if result.Text != "done" {
		t.Errorf("text = %q, want done", result.Text)
	}
	if len(result.Steps) != 4 {
		t.Errorf("steps = %d, want 4 (three tool rounds + final)", len(result.Steps))
	}
}

// TestAgentPerCallOverride verifies per-call options override agent
// options for that call only.
func TestAgentPerCallOverride(t *testing.T) {
	calls := 0
	agent := ai.NewAgent(countingToolModel(&calls, 5),
		ai.WithTools(counterTool()),
	)

	result, err := agent.Generate(context.Background(),
		ai.WithPrompt("count"),
		ai.WithMaxSteps(2),
	)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	if len(result.Steps) != 2 {
		t.Errorf("steps = %d, want 2 (per-call cap)", len(result.Steps))
	}
}

// TestAgentStopWhen verifies stop-condition helpers halt the loop.
func TestAgentStopWhen(t *testing.T) {
	calls := 0
	agent := ai.NewAgent(countingToolModel(&calls, 10),
		ai.WithTools(counterTool()),
		ai.WithStopWhen(ai.StepCountIs(3)),
	)

	result, err := agent.Generate(context.Background(), ai.WithPrompt("count"))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	if len(result.Steps) != 3 {
		t.Errorf("steps = %d, want 3 (StepCountIs)", len(result.Steps))
	}
}

// TestAgentStream verifies the streaming path layers options the same
// way as Generate.
func TestAgentStream(t *testing.T) {
	step := 0
	mock := &mockModel{
		streamFn: func(_ context.Context, _ *model.Request) (*model.StreamResponse, error) {
			step++
			if step == 1 {
				return chunkStream(
					model.ToolCallDelta{ToolCallID: "call_1", ToolName: "counter", ArgsJSON: `{}`},
					model.FinishChunk{FinishReason: model.FinishReasonToolCalls},
				), nil
			}

			return chunkStream(
				model.TextDelta{Text: "streamed done"},
				model.FinishChunk{FinishReason: model.FinishReasonStop},
			), nil
		},
	}

	agent := ai.NewAgent(mock, ai.WithTools(counterTool()))

	stream, err := agent.Stream(context.Background(), ai.WithPrompt("count"))
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	result, err := stream.Result()
	if err != nil {
		t.Fatalf("Result: %v", err)
	}

	if result.Text != "streamed done" {
		t.Errorf("text = %q, want streamed done", result.Text)
	}
	if len(result.Steps) != 2 {
		t.Errorf("steps = %d, want 2", len(result.Steps))
	}
}

// TestStopConditionHelpers exercises HasToolCall and NoToolCalls.
func TestStopConditionHelpers(t *testing.T) {
	withCall := ai.Step{
		ToolCalls: []ai.ToolCall{{ToolName: "search"}},
	}
	without := ai.Step{}

	if !ai.HasToolCall("search")(withCall) {
		t.Error("HasToolCall(search) = false, want true")
	}
	if ai.HasToolCall("other")(withCall) {
		t.Error("HasToolCall(other) = true, want false")
	}
	if ai.NoToolCalls()(withCall) {
		t.Error("NoToolCalls = true for step with calls")
	}
	if !ai.NoToolCalls()(without) {
		t.Error("NoToolCalls = false for empty step")
	}
	if !ai.StepCountIs(1)(ai.Step{Index: 0}) {
		t.Error("StepCountIs(1) on step 0 = false, want true")
	}
	if ai.StepCountIs(3)(ai.Step{Index: 1}) {
		t.Error("StepCountIs(3) on step 1 = true, want false")
	}
}
