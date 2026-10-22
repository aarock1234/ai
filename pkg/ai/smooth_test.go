package ai_test

import (
	"context"
	"testing"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// TestSmoothStreamByWord verifies text is re-chunked word by word and
// non-text chunks flush pending text in order.
func TestSmoothStreamByWord(t *testing.T) {
	mock := &mockModel{
		streamFn: func(_ context.Context, _ *model.Request) (*model.StreamResponse, error) {
			return chunkStream(
				model.TextDelta{Text: "hello wor"},
				model.TextDelta{Text: "ld and more"},
				model.FinishChunk{FinishReason: model.FinishReasonStop},
			), nil
		},
	}

	smoothed := ai.WrapLanguageModel(mock, ai.SmoothStreamMiddleware(0, nil))

	stream, err := ai.StreamText(context.Background(), smoothed, ai.WithPrompt("hi"))
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	var pieces []string
	for delta := range stream.Text() {
		pieces = append(pieces, delta)
	}

	want := []string{"hello ", "world ", "and ", "more"}
	if len(pieces) != len(want) {
		t.Fatalf("pieces = %q, want %q", pieces, want)
	}
	for i, piece := range want {
		if pieces[i] != piece {
			t.Errorf("piece %d = %q, want %q", i, pieces[i], piece)
		}
	}

	result, err := stream.Result()
	if err != nil {
		t.Fatalf("Result: %v", err)
	}
	if result.Text != "hello world and more" {
		t.Errorf("text = %q, want full text", result.Text)
	}
}

// TestSmoothStreamByLine verifies line chunking.
func TestSmoothStreamByLine(t *testing.T) {
	mock := &mockModel{
		streamFn: func(_ context.Context, _ *model.Request) (*model.StreamResponse, error) {
			return chunkStream(
				model.TextDelta{Text: "line one\nline "},
				model.TextDelta{Text: "two\ntail"},
				model.FinishChunk{FinishReason: model.FinishReasonStop},
			), nil
		},
	}

	smoothed := ai.WrapLanguageModel(mock, ai.SmoothStreamMiddleware(0, ai.ChunkByLine()))

	stream, err := ai.StreamText(context.Background(), smoothed, ai.WithPrompt("hi"))
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	var pieces []string
	for delta := range stream.Text() {
		pieces = append(pieces, delta)
	}

	want := []string{"line one\n", "line two\n", "tail"}
	if len(pieces) != len(want) {
		t.Fatalf("pieces = %q, want %q", pieces, want)
	}
}

// TestStepBoundaryChunks verifies StepStart/StepFinish chunks bracket
// each step in both single-step and tool-loop streams.
func TestStepBoundaryChunks(t *testing.T) {
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
				model.TextDelta{Text: "done"},
				model.FinishChunk{FinishReason: model.FinishReasonStop},
			), nil
		},
	}

	stream, err := ai.StreamText(context.Background(), mock,
		ai.WithPrompt("count"),
		ai.WithTools(counterTool()),
		ai.WithMaxSteps(3),
	)
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	var starts, finishes []int
	for chunk := range stream.Chunks() {
		switch c := chunk.(type) {
		case model.StepStartChunk:
			starts = append(starts, c.Step)
		case model.StepFinishChunk:
			finishes = append(finishes, c.Step)
		}
	}

	if len(starts) != 2 || starts[0] != 0 || starts[1] != 1 {
		t.Errorf("step starts = %v, want [0 1]", starts)
	}
	if len(finishes) != 2 || finishes[0] != 0 || finishes[1] != 1 {
		t.Errorf("step finishes = %v, want [0 1]", finishes)
	}
}

// TestSingleStepBoundaryChunks verifies the plain stream path is
// bracketed the same way.
func TestSingleStepBoundaryChunks(t *testing.T) {
	mock := &mockModel{
		streamFn: func(_ context.Context, _ *model.Request) (*model.StreamResponse, error) {
			return chunkStream(
				model.TextDelta{Text: "hi"},
				model.FinishChunk{FinishReason: model.FinishReasonStop},
			), nil
		},
	}

	stream, err := ai.StreamText(context.Background(), mock, ai.WithPrompt("hi"))
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	var sawStart, sawFinish bool
	for chunk := range stream.Chunks() {
		switch c := chunk.(type) {
		case model.StepStartChunk:
			sawStart = true
		case model.StepFinishChunk:
			sawFinish = true
			if c.FinishReason != model.FinishReasonStop {
				t.Errorf("finish reason = %q, want stop", c.FinishReason)
			}
		}
	}

	if !sawStart || !sawFinish {
		t.Errorf("start=%v finish=%v, want both", sawStart, sawFinish)
	}
}
