package ai_test

import (
	"context"
	"iter"
	"sync/atomic"
	"testing"
	"time"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

func chunkStream(chunks ...model.Chunk) *model.StreamResponse {
	return &model.StreamResponse{
		Stream: iter.Seq[model.Chunk](func(yield func(model.Chunk) bool) {
			for _, c := range chunks {
				if !yield(c) {
					return
				}
			}
		}),
	}
}

// TestStreamToolCallOrder verifies that streamed tool calls preserve
// arrival order even when deltas for multiple calls interleave.
func TestStreamToolCallOrder(t *testing.T) {
	mock := &mockModel{
		streamFn: func(_ context.Context, _ *model.Request) (*model.StreamResponse, error) {
			return chunkStream(
				model.ToolCallDelta{ToolCallID: "call_b", ToolName: "beta", ArgsJSON: `{"x":`},
				model.ToolCallDelta{ToolCallID: "call_a", ToolName: "alpha", ArgsJSON: `{}`},
				model.ToolCallDelta{ToolCallID: "call_b", ArgsJSON: `1}`},
				model.ToolCallDelta{ToolCallID: "call_c", ToolName: "gamma", ArgsJSON: `{}`},
				model.FinishChunk{FinishReason: model.FinishReasonToolCalls},
			), nil
		},
	}

	stream, err := ai.StreamText(context.Background(), mock, ai.WithPrompt("hi"))
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	result, err := stream.Result()
	if err != nil {
		t.Fatalf("Result: %v", err)
	}

	wantOrder := []string{"call_b", "call_a", "call_c"}
	if len(result.ToolCalls) != len(wantOrder) {
		t.Fatalf("tool calls = %d, want %d", len(result.ToolCalls), len(wantOrder))
	}
	for i, want := range wantOrder {
		if result.ToolCalls[i].ToolCallID != want {
			t.Errorf("tool call %d = %q, want %q", i, result.ToolCalls[i].ToolCallID, want)
		}
	}
	if string(result.ToolCalls[0].Args) != `{"x":1}` {
		t.Errorf("call_b args = %q, want accumulated json", result.ToolCalls[0].Args)
	}
}

// TestStreamResultSecondConsumeIsNoOp verifies that iterating a second
// time does not clobber the result accumulated by the first pass.
func TestStreamResultSecondConsumeIsNoOp(t *testing.T) {
	mock := &mockModel{
		streamFn: func(_ context.Context, _ *model.Request) (*model.StreamResponse, error) {
			return chunkStream(
				model.TextDelta{Text: "hello"},
				model.FinishChunk{FinishReason: model.FinishReasonStop},
			), nil
		},
	}

	stream, err := ai.StreamText(context.Background(), mock, ai.WithPrompt("hi"))
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	var text string
	for delta := range stream.Text() {
		text += delta
	}
	if text != "hello" {
		t.Fatalf("text = %q, want hello", text)
	}

	// Second iterator must yield nothing and leave the result intact.
	for range stream.Files() {
		t.Fatal("second consumption yielded a chunk")
	}

	result, err := stream.Result()
	if err != nil {
		t.Fatalf("Result: %v", err)
	}
	if result.Text != "hello" {
		t.Errorf("result text = %q, want hello (clobbered by re-consumption?)", result.Text)
	}
}

// TestRetryHonorsRetryAfter verifies that the retry loop waits at least
// the provider-requested Retry-After duration.
func TestRetryHonorsRetryAfter(t *testing.T) {
	const retryAfter = 30 * time.Millisecond

	var attempts atomic.Int32
	mock := &mockModel{
		generateFn: func(_ context.Context, _ *model.Request) (*model.Response, error) {
			if attempts.Add(1) <= 2 {
				return nil, &ai.APIError{
					StatusCode:  429,
					Message:     "slow down",
					IsRetryable: true,
					RetryAfter:  retryAfter,
				}
			}

			return &model.Response{
				Content:      []model.Content{model.TextContent{Text: "ok"}},
				FinishReason: model.FinishReasonStop,
			}, nil
		},
	}

	start := time.Now()
	result, err := ai.GenerateText(context.Background(), mock,
		ai.WithPrompt("hi"),
		ai.WithMaxRetries(3),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	if got := attempts.Load(); got != 3 {
		t.Errorf("attempts = %d, want 3", got)
	}
	if result.Text != "ok" {
		t.Errorf("text = %q, want ok", result.Text)
	}
	if elapsed := time.Since(start); elapsed < 2*retryAfter {
		t.Errorf("elapsed = %v, want at least %v (two Retry-After waits)", elapsed, 2*retryAfter)
	}
}
