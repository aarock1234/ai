package ai_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// approvalModel requests the guarded tool once, then answers.
func approvalModel() *mockModel {
	first := true

	return &mockModel{
		generateFn: func(_ context.Context, _ *model.Request) (*model.Response, error) {
			if first {
				first = false

				return &model.Response{
					Content: []model.Content{
						model.ToolCallContent{
							ToolCallID: "call_1",
							ToolName:   "delete_file",
							Args:       `{"path":"/tmp/x"}`,
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

func guardedTool(executed *bool) ai.Tool {
	return ai.Tool{
		Name:          "delete_file",
		Description:   "Deletes a file.",
		InputSchema:   json.RawMessage(`{"type":"object"}`),
		NeedsApproval: ai.AlwaysNeedsApproval,
		Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
			*executed = true

			return "deleted", nil
		},
	}
}

func TestToolApprovalGranted(t *testing.T) {
	executed := false

	result, err := ai.GenerateText(context.Background(), approvalModel(),
		ai.WithPrompt("clean up"),
		ai.WithTools(guardedTool(&executed)),
		ai.WithMaxSteps(3),
		ai.WithToolApproval(func(_ context.Context, call ai.ToolCall) (ai.ApprovalDecision, error) {
			if call.ToolName != "delete_file" {
				t.Errorf("approval for %q, want delete_file", call.ToolName)
			}

			return ai.ApprovalDecision{Approved: true}, nil
		}),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	if !executed {
		t.Error("tool not executed after approval")
	}
	if result.ToolResults[0].IsError {
		t.Errorf("result = %+v, want success", result.ToolResults[0])
	}
}

func TestToolApprovalDenied(t *testing.T) {
	executed := false

	result, err := ai.GenerateText(context.Background(), approvalModel(),
		ai.WithPrompt("clean up"),
		ai.WithTools(guardedTool(&executed)),
		ai.WithMaxSteps(3),
		ai.WithToolApproval(func(_ context.Context, _ ai.ToolCall) (ai.ApprovalDecision, error) {
			return ai.ApprovalDecision{
				Approved: false,
				Reason:   "not in maintenance window",
			}, nil
		}),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	if executed {
		t.Error("tool executed despite denial")
	}
	tr := result.ToolResults[0]
	if !tr.IsError || !strings.Contains(tr.Content, "not in maintenance window") {
		t.Errorf("result = %+v, want denial with reason", tr)
	}
}

func TestToolApprovalMissingHandler(t *testing.T) {
	executed := false

	result, err := ai.GenerateText(context.Background(), approvalModel(),
		ai.WithPrompt("clean up"),
		ai.WithTools(guardedTool(&executed)),
		ai.WithMaxSteps(3),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	if executed {
		t.Error("tool executed without approval handler")
	}
	tr := result.ToolResults[0]
	if !tr.IsError || !strings.Contains(tr.Content, "no approval handler") {
		t.Errorf("result = %+v, want missing-handler rejection", tr)
	}
}

// TestProviderExecutedToolsSkipped verifies the loop neither executes
// provider-executed calls locally nor loops on them.
func TestProviderExecutedToolsSkipped(t *testing.T) {
	calls := 0
	mock := &mockModel{
		generateFn: func(_ context.Context, _ *model.Request) (*model.Response, error) {
			calls++

			return &model.Response{
				Content: []model.Content{
					model.ToolCallContent{
						ToolCallID:       "srv_1",
						ToolName:         "web_search",
						Args:             `{"query":"go"}`,
						ProviderExecuted: true,
					},
					model.ToolResultContent{
						ToolCallID: "srv_1",
						ToolName:   "web_search",
						Content:    "search results",
					},
					model.TextContent{Text: "based on the search: answer"},
				},
				FinishReason: model.FinishReasonToolCalls,
			}, nil
		},
	}

	executed := false
	result, err := ai.GenerateText(context.Background(), mock,
		ai.WithPrompt("search"),
		ai.WithTools(ai.Tool{
			Name:        "web_search",
			InputSchema: json.RawMessage(`{"type":"object"}`),
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				executed = true

				return "local result", nil
			},
		}),
		ai.WithMaxSteps(5),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	if calls != 1 {
		t.Errorf("model calls = %d, want 1 (no loop on provider-executed calls)", calls)
	}
	if executed {
		t.Error("provider-executed tool ran locally")
	}
	if len(result.ToolResults) != 1 || !result.ToolResults[0].ProviderExecuted {
		t.Errorf("tool results = %+v, want single provider-executed result", result.ToolResults)
	}
}

// TestToModelOutput verifies output mapping runs before the result is
// recorded.
func TestToModelOutput(t *testing.T) {
	mock := approvalModel()

	var sawContent string
	result, err := ai.GenerateText(context.Background(), mock,
		ai.WithPrompt("clean up"),
		ai.WithTools(ai.Tool{
			Name:        "delete_file",
			InputSchema: json.RawMessage(`{"type":"object"}`),
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				return strings.Repeat("verbose ", 100), nil
			},
			ToModelOutput: func(string) string {
				return "trimmed"
			},
		}),
		ai.WithMaxSteps(3),
		ai.WithOnStepFinish(func(_ context.Context, step ai.Step) error {
			for _, tr := range step.ToolResults {
				sawContent = tr.Content
			}

			return nil
		}),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	if result.ToolResults[0].Content != "trimmed" {
		t.Errorf("content = %q, want trimmed", result.ToolResults[0].Content)
	}
	_ = sawContent
}
