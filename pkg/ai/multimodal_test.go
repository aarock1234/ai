package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"iter"
	"testing"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// drainChunks exhausts a chunk iterator without inspecting the values.
// Used by tests that only care about the accumulated final result.
func drainChunks(seq iter.Seq[model.Chunk]) {
	next, stop := iter.Pull(seq)
	defer stop()
	for {
		if _, ok := next(); !ok {
			return
		}
	}
}

// TestGenerateTextExtractsFilesSourcesReasoning verifies that
// GenerateText populates the new File/Source/Reasoning result fields
// from a response containing those content segments.
func TestGenerateTextExtractsFilesSourcesReasoning(t *testing.T) {
	mock := &mockModel{
		generateFn: func(_ context.Context, _ *model.Request) (*model.Response, error) {
			return &model.Response{
				Content: []model.Content{
					model.ReasoningContent{
						Text:      "thinking step",
						Signature: "sig-1",
						Format:    "anthropic-thinking",
					},
					model.TextContent{Text: "here is the answer."},
					model.FileContent{
						Data:     []byte{0x89, 0x50, 0x4e, 0x47},
						MIMEType: "image/png",
					},
					model.SourceContent{
						SourceType: model.SourceTypeURL,
						URL:        "https://example.com/source",
						Title:      "Source",
					},
				},
				FinishReason: model.FinishReasonStop,
				Usage: model.Usage{
					InputTokens:       10,
					OutputTokens:      20,
					ReasoningTokens:   5,
					ImageOutputTokens: 2,
				},
			}, nil
		},
	}

	result, err := ai.GenerateText(context.Background(), mock, ai.WithPrompt("hi"))
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	if result.Text != "here is the answer." {
		t.Errorf("Text = %q, want 'here is the answer.'", result.Text)
	}

	if len(result.Reasoning) != 1 {
		t.Fatalf("Reasoning blocks = %d, want 1", len(result.Reasoning))
	}
	if result.Reasoning[0].Signature != "sig-1" {
		t.Errorf("reasoning signature = %q, want sig-1", result.Reasoning[0].Signature)
	}
	if result.ReasoningText() != "thinking step" {
		t.Errorf("ReasoningText() = %q, want 'thinking step'", result.ReasoningText())
	}

	if len(result.Files) != 1 {
		t.Fatalf("Files = %d, want 1", len(result.Files))
	}
	if result.Files[0].MIMEType != "image/png" {
		t.Errorf("file mime = %q, want image/png", result.Files[0].MIMEType)
	}

	if len(result.Sources) != 1 {
		t.Fatalf("Sources = %d, want 1", len(result.Sources))
	}
	if result.Sources[0].URL != "https://example.com/source" {
		t.Errorf("source url = %q, want example.com", result.Sources[0].URL)
	}

	if result.Usage.ReasoningTokens != 5 {
		t.Errorf("reasoning tokens = %d, want 5", result.Usage.ReasoningTokens)
	}
	if result.Usage.ImageOutputTokens != 2 {
		t.Errorf("image output tokens = %d, want 2", result.Usage.ImageOutputTokens)
	}
}

// TestStreamTextAccumulatesNewChunkTypes verifies that the stream
// accumulator builds a GenerateResult with files/sources/reasoning
// from streamed chunks.
func TestStreamTextAccumulatesNewChunkTypes(t *testing.T) {
	mock := &mockModel{
		streamFn: func(_ context.Context, _ *model.Request) (*model.StreamResponse, error) {
			chunks := []model.Chunk{
				model.StreamStartChunk{Warnings: []string{"example warn"}},
				model.ReasoningDelta{
					Text:   "think ",
					Format: "reasoning.text",
					ID:     "r-1",
				},
				model.ReasoningDelta{
					Text:      "harder",
					Format:    "reasoning.text",
					ID:        "r-1",
					Signature: "sig",
				},
				model.TextDelta{Text: "answer"},
				model.FileChunk{
					Data:     []byte("img-bytes"),
					MIMEType: "image/png",
				},
				model.SourceChunk{
					SourceType: model.SourceTypeURL,
					URL:        "https://example.com",
				},
				model.FinishChunk{FinishReason: model.FinishReasonStop},
				model.UsageChunk{Usage: model.Usage{
					InputTokens:  3,
					OutputTokens: 4,
				}},
			}

			return &model.StreamResponse{
				Stream: func(yield func(model.Chunk) bool) {
					for _, c := range chunks {
						if !yield(c) {
							return
						}
					}
				},
			}, nil
		},
	}

	stream, err := ai.StreamText(context.Background(), mock, ai.WithPrompt("hi"))
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	// Drain the stream.
	drainChunks(stream.Chunks())

	result, err := stream.Result()
	if err != nil {
		t.Fatalf("Result: %v", err)
	}

	if result.Text != "answer" {
		t.Errorf("Text = %q, want answer", result.Text)
	}

	if len(result.Reasoning) != 1 {
		t.Fatalf("Reasoning blocks = %d, want 1 (merged by ID)", len(result.Reasoning))
	}
	if result.Reasoning[0].Text != "think harder" {
		t.Errorf("reasoning = %q, want 'think harder'", result.Reasoning[0].Text)
	}
	if result.Reasoning[0].Signature != "sig" {
		t.Errorf("signature = %q, want sig", result.Reasoning[0].Signature)
	}

	if len(result.Files) != 1 || result.Files[0].MIMEType != "image/png" {
		t.Errorf("files = %+v, want one image/png", result.Files)
	}
	if len(result.Sources) != 1 || result.Sources[0].URL != "https://example.com" {
		t.Errorf("sources = %+v, want one example.com", result.Sources)
	}

	if len(result.Warnings) == 0 || result.Warnings[0] != "example warn" {
		t.Errorf("warnings = %v, want 'example warn'", result.Warnings)
	}
}

// TestStreamTextFilesIterator verifies StreamResult.Files() yields the
// same files that accumulate into the final result.
func TestStreamTextFilesIterator(t *testing.T) {
	mock := &mockModel{
		streamFn: func(_ context.Context, _ *model.Request) (*model.StreamResponse, error) {
			chunks := []model.Chunk{
				model.FileChunk{
					Data:     []byte("a"),
					MIMEType: "image/png",
				},
				model.FileChunk{
					Data:     []byte("b"),
					MIMEType: "image/jpeg",
				},
				model.FinishChunk{FinishReason: model.FinishReasonStop},
			}

			return &model.StreamResponse{
				Stream: func(yield func(model.Chunk) bool) {
					for _, c := range chunks {
						if !yield(c) {
							return
						}
					}
				},
			}, nil
		},
	}

	stream, err := ai.StreamText(context.Background(), mock, ai.WithPrompt("hi"))
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	var mimes []string
	for file := range stream.Files() {
		mimes = append(mimes, file.MIMEType)
	}

	if len(mimes) != 2 {
		t.Fatalf("Files() yielded %d, want 2", len(mimes))
	}
	if mimes[0] != "image/png" || mimes[1] != "image/jpeg" {
		t.Errorf("mimes = %v, want [image/png image/jpeg]", mimes)
	}
}

// TestAppendToolMessagesIncludesReasoning verifies the multi-step tool
// loop round-trips reasoning blocks back to the provider.
func TestAppendToolMessagesIncludesReasoning(t *testing.T) {
	calls := 0
	var secondRequest *model.Request
	mock := &mockModel{
		generateFn: func(_ context.Context, req *model.Request) (*model.Response, error) {
			calls++
			if calls == 1 {
				return &model.Response{
					Content: []model.Content{
						model.ReasoningContent{
							Text:      "considering tools",
							Signature: "sig-step1",
							Format:    "anthropic-thinking",
						},
						model.ToolCallContent{
							ToolCallID: "call-1",
							ToolName:   "echo",
							Args:       `{"msg":"hi"}`,
						},
					},
					FinishReason: model.FinishReasonToolCalls,
				}, nil
			}

			secondRequest = req

			return &model.Response{
				Content:      []model.Content{model.TextContent{Text: "done"}},
				FinishReason: model.FinishReasonStop,
			}, nil
		},
	}

	echoTool := ai.Tool{
		Name:        "echo",
		Description: "echo tool",
		Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
			return "hi", nil
		},
	}

	_, err := ai.GenerateText(context.Background(), mock,
		ai.WithPrompt("run tool"),
		ai.WithTools(echoTool),
		ai.WithMaxSteps(3),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	if secondRequest == nil {
		t.Fatal("expected second request")
	}

	// The assistant message that went back to the provider on step 2
	// must include a ReasoningPart preserving the prior signature.
	var assistantMsg *model.Message
	for i := range secondRequest.Messages {
		if secondRequest.Messages[i].Role == model.RoleAssistant {
			assistantMsg = &secondRequest.Messages[i]
		}
	}

	if assistantMsg == nil {
		t.Fatal("expected assistant message in second request")
	}

	var reasoning *model.ReasoningPart
	for _, part := range assistantMsg.Parts {
		if rp, ok := part.(model.ReasoningPart); ok {
			reasoning = &rp
		}
	}

	if reasoning == nil {
		t.Fatal("expected reasoning part in round-tripped assistant message")
	}
	if reasoning.Signature != "sig-step1" {
		t.Errorf("signature = %q, want sig-step1", reasoning.Signature)
	}
	if reasoning.Text != "considering tools" {
		t.Errorf("text = %q, want 'considering tools'", reasoning.Text)
	}
}

// TestReasoningOptionsOption verifies the WithReasoning option flows
// through to model.Request.
func TestReasoningOptionsOption(t *testing.T) {
	var received *model.ReasoningOptions
	mock := &mockModel{
		generateFn: func(_ context.Context, req *model.Request) (*model.Response, error) {
			received = req.Reasoning

			return &model.Response{
				Content:      []model.Content{model.TextContent{Text: "ok"}},
				FinishReason: model.FinishReasonStop,
			}, nil
		},
	}

	_, err := ai.GenerateText(context.Background(), mock,
		ai.WithPrompt("hi"),
		ai.WithReasoning(model.ReasoningOptions{
			Effort:    "high",
			MaxTokens: 5000,
		}),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	if received == nil {
		t.Fatal("reasoning options not attached to request")
	}
	if received.Effort != "high" {
		t.Errorf("effort = %q, want high", received.Effort)
	}
	if received.MaxTokens != 5000 {
		t.Errorf("max tokens = %d, want 5000", received.MaxTokens)
	}
}

// TestWithModalitiesAndAudio verifies Modalities/Audio/Prediction flow
// through to model.Request.
func TestWithModalitiesAndAudio(t *testing.T) {
	var received *model.Request
	mock := &mockModel{
		generateFn: func(_ context.Context, req *model.Request) (*model.Response, error) {
			received = req

			return &model.Response{
				Content:      []model.Content{model.TextContent{Text: "ok"}},
				FinishReason: model.FinishReasonStop,
			}, nil
		},
	}

	_, err := ai.GenerateText(context.Background(), mock,
		ai.WithPrompt("hi"),
		ai.WithModalities(model.ModalityText, model.ModalityAudio),
		ai.WithAudio("alloy", "wav"),
		ai.WithPrediction("predicted content"),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	if len(received.Modalities) != 2 {
		t.Fatalf("modalities = %v, want 2", received.Modalities)
	}
	if received.Audio == nil || received.Audio.Voice != "alloy" {
		t.Errorf("audio = %+v, want voice=alloy", received.Audio)
	}
	if received.Prediction == nil || received.Prediction.Content != "predicted content" {
		t.Errorf("prediction = %+v, want predicted content", received.Prediction)
	}
}

// TestErrorChunkSurfacesAsWarning verifies that an ErrorChunk in the
// stream ends up as a warning on the final result.
func TestErrorChunkSurfacesAsWarning(t *testing.T) {
	mock := &mockModel{
		streamFn: func(_ context.Context, _ *model.Request) (*model.StreamResponse, error) {
			return &model.StreamResponse{
				Stream: iter.Seq[model.Chunk](func(yield func(model.Chunk) bool) {
					if !yield(model.ErrorChunk{Err: errors.New("transient glitch")}) {
						return
					}
					if !yield(model.FinishChunk{FinishReason: model.FinishReasonError}) {
						return
					}
				}),
			}, nil
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

	if len(result.Warnings) == 0 {
		t.Fatal("expected warning from ErrorChunk")
	}
	if result.Warnings[0] != "transient glitch" {
		t.Errorf("warning = %q, want 'transient glitch'", result.Warnings[0])
	}
}
