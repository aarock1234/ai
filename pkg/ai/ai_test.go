package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"strings"
	"testing"
	"time"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// mockModel is a minimal model.Model implementation for testing.
type mockModel struct {
	generateFn             func(ctx context.Context, req *model.Request) (*model.Response, error)
	streamFn               func(ctx context.Context, req *model.Request) (*model.StreamResponse, error)
	supportsResponseFormat bool
}

func (m *mockModel) Provider() string { return "mock" }
func (m *mockModel) ModelID() string  { return "mock-model" }

func (m *mockModel) DoGenerate(ctx context.Context, req *model.Request) (*model.Response, error) {
	return m.generateFn(ctx, req)
}

func (m *mockModel) DoStream(ctx context.Context, req *model.Request) (*model.StreamResponse, error) {
	return m.streamFn(ctx, req)
}

func (m *mockModel) SupportsResponseFormat(_ *model.ResponseFormat) bool {
	return m.supportsResponseFormat
}

// mockEmbeddingModel is a minimal model.EmbeddingModel for testing.
type mockEmbeddingModel struct {
	embedFn func(ctx context.Context, texts []string) (*model.EmbedResponse, error)
}

func (m *mockEmbeddingModel) Provider() string { return "mock" }
func (m *mockEmbeddingModel) ModelID() string  { return "mock-embed" }

func (m *mockEmbeddingModel) DoEmbed(ctx context.Context, texts []string) (*model.EmbedResponse, error) {
	return m.embedFn(ctx, texts)
}

type testContextKey string

func TestGenerateTextBasic(t *testing.T) {
	mock := &mockModel{
		generateFn: func(_ context.Context, req *model.Request) (*model.Response, error) {
			// Verify the request was constructed correctly.
			if len(req.Messages) != 1 {
				t.Errorf("messages length = %d, want 1", len(req.Messages))
			}

			return &model.Response{
				Content:      []model.Content{model.TextContent{Text: "Hello back!"}},
				FinishReason: model.FinishReasonStop,
				Usage:        model.Usage{InputTokens: 5, OutputTokens: 3},
				Metadata: model.ProviderMetadata{
					RequestID: "req_basic",
					Provider: map[string]string{
						"trace_id": "trace-basic",
					},
				},
			}, nil
		},
	}

	result, err := ai.GenerateText(context.Background(), mock,
		ai.WithPrompt("Hi"),
		ai.WithProviderOption("mock", "metadata", map[string]string{"trace_id": "trace-basic"}),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	if result.Text != "Hello back!" {
		t.Errorf("Text = %q, want %q", result.Text, "Hello back!")
	}

	if result.FinishReason != model.FinishReasonStop {
		t.Errorf("FinishReason = %v, want stop", result.FinishReason)
	}

	if result.Usage.InputTokens != 5 {
		t.Errorf("InputTokens = %d, want 5", result.Usage.InputTokens)
	}

	if result.Usage.OutputTokens != 3 {
		t.Errorf("OutputTokens = %d, want 3", result.Usage.OutputTokens)
	}

	if result.Metadata.RequestID != "req_basic" {
		t.Errorf("Metadata.RequestID = %q, want req_basic", result.Metadata.RequestID)
	}

	if result.Metadata.Provider["trace_id"] != "trace-basic" {
		t.Errorf("Metadata.Provider[trace_id] = %q, want trace-basic", result.Metadata.Provider["trace_id"])
	}
}

func TestGenerateTextWithProviderOption(t *testing.T) {
	mock := &mockModel{
		generateFn: func(_ context.Context, req *model.Request) (*model.Response, error) {
			raw, ok := req.ProviderOptions.Lookup("anthropic", "metadata")
			if !ok {
				t.Fatal("provider option metadata not found")
			}

			if string(raw) != `{"trace_id":"trace-123"}` {
				t.Errorf("provider option metadata = %s, want trace JSON", raw)
			}

			return &model.Response{
				Content:      []model.Content{model.TextContent{Text: "ok"}},
				FinishReason: model.FinishReasonStop,
				Metadata: model.ProviderMetadata{
					RequestID: "req_1",
					ModelID:   "claude-test",
				},
			}, nil
		},
	}

	result, err := ai.GenerateText(context.Background(), mock,
		ai.WithPrompt("Hi"),
		ai.WithProviderOption("anthropic", "metadata", map[string]string{"trace_id": "trace-123"}),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	if result.Metadata.RequestID != "req_1" {
		t.Errorf("Metadata.RequestID = %q, want req_1", result.Metadata.RequestID)
	}
}

func TestGenerateTextWithInvalidProviderOption(t *testing.T) {
	mock := &mockModel{
		generateFn: func(_ context.Context, _ *model.Request) (*model.Response, error) {
			t.Fatal("generateFn should not be called when option resolution fails")

			return nil, nil
		},
	}

	_, err := ai.GenerateText(context.Background(), mock,
		ai.WithPrompt("Hi"),
		ai.WithProviderOption("openai", "metadata", func() {}),
	)
	if err == nil {
		t.Fatal("expected provider option error")
	}
	if !strings.Contains(err.Error(), "provider option openai.metadata") {
		t.Fatalf("error = %q, want provider option context", err)
	}
}

func TestGenerateTextWithSystemAndMessages(t *testing.T) {
	mock := &mockModel{
		generateFn: func(_ context.Context, req *model.Request) (*model.Response, error) {
			// System + 1 user message + prompt = 3 messages.
			if len(req.Messages) != 3 {
				t.Errorf("messages length = %d, want 3", len(req.Messages))
			}

			if req.Messages[0].Role != model.RoleSystem {
				t.Errorf("messages[0].Role = %v, want system", req.Messages[0].Role)
			}

			return &model.Response{
				Content:      []model.Content{model.TextContent{Text: "OK"}},
				FinishReason: model.FinishReasonStop,
			}, nil
		},
	}

	result, err := ai.GenerateText(context.Background(), mock,
		ai.WithSystem("You are helpful."),
		ai.WithMessages(ai.Message{Role: ai.RoleUser, Content: "Context message"}),
		ai.WithPrompt("And now the question"),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	if result.Text != "OK" {
		t.Errorf("Text = %q, want %q", result.Text, "OK")
	}
}

func TestGenerateTextWithTemperature(t *testing.T) {
	mock := &mockModel{
		generateFn: func(_ context.Context, req *model.Request) (*model.Response, error) {
			if req.Temperature == nil || *req.Temperature != 0.5 {
				t.Error("temperature not set to 0.5")
			}

			if req.MaxTokens == nil || *req.MaxTokens != 200 {
				t.Error("max tokens not set to 200")
			}

			return &model.Response{
				Content:      []model.Content{model.TextContent{Text: "done"}},
				FinishReason: model.FinishReasonStop,
			}, nil
		},
	}

	_, err := ai.GenerateText(context.Background(), mock,
		ai.WithPrompt("Hi"),
		ai.WithTemperature(0.5),
		ai.WithMaxTokens(200),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}
}

func TestGenerateTextToolLoop(t *testing.T) {
	callCount := 0

	mock := &mockModel{
		generateFn: func(_ context.Context, req *model.Request) (*model.Response, error) {
			callCount++

			if callCount == 1 {
				// First call: model requests a tool call.
				return &model.Response{
					Content: []model.Content{
						model.ToolCallContent{
							ToolCallID: "call_1",
							ToolName:   "get_weather",
							Args:       `{"city":"NYC"}`,
						},
					},
					FinishReason: model.FinishReasonToolCalls,
					Usage:        model.Usage{InputTokens: 10, OutputTokens: 5},
				}, nil
			}

			// Second call: model generates final response.
			// Verify tool result was included in messages.
			lastMsg := req.Messages[len(req.Messages)-1]
			if lastMsg.Role != model.RoleTool {
				t.Errorf("last message role = %v, want tool", lastMsg.Role)
			}

			return &model.Response{
				Content:      []model.Content{model.TextContent{Text: "It's sunny in NYC!"}},
				FinishReason: model.FinishReasonStop,
				Usage:        model.Usage{InputTokens: 20, OutputTokens: 10},
			}, nil
		},
	}

	weatherTool := ai.Tool{
		Name:        "get_weather",
		Description: "Get current weather",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
		Execute: func(_ context.Context, input json.RawMessage) (string, error) {
			var args struct {
				City string `json:"city"`
			}
			if err := json.Unmarshal(input, &args); err != nil {
				return "", err
			}

			return fmt.Sprintf("Sunny, 72°F in %s", args.City), nil
		},
	}

	result, err := ai.GenerateText(context.Background(), mock,
		ai.WithPrompt("What's the weather in NYC?"),
		ai.WithTools(weatherTool),
		ai.WithMaxSteps(3),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	if callCount != 2 {
		t.Errorf("model called %d times, want 2", callCount)
	}

	if result.Text != "It's sunny in NYC!" {
		t.Errorf("Text = %q, want %q", result.Text, "It's sunny in NYC!")
	}

	if len(result.Steps) != 2 {
		t.Errorf("Steps = %d, want 2", len(result.Steps))
	}

	if result.Usage.InputTokens != 30 {
		t.Errorf("total InputTokens = %d, want 30", result.Usage.InputTokens)
	}

	if result.Usage.OutputTokens != 15 {
		t.Errorf("total OutputTokens = %d, want 15", result.Usage.OutputTokens)
	}

	if len(result.ToolResults) != 1 {
		t.Errorf("ToolResults = %d, want 1", len(result.ToolResults))
	}
}

func TestGenerateTextToolLoopUnknownTool(t *testing.T) {
	callCount := 0

	mock := &mockModel{
		generateFn: func(_ context.Context, _ *model.Request) (*model.Response, error) {
			callCount++

			if callCount == 1 {
				return &model.Response{
					Content: []model.Content{
						model.ToolCallContent{
							ToolCallID: "call_1",
							ToolName:   "unknown_tool",
							Args:       `{}`,
						},
					},
					FinishReason: model.FinishReasonToolCalls,
					Usage:        model.Usage{InputTokens: 5, OutputTokens: 5},
				}, nil
			}

			return &model.Response{
				Content:      []model.Content{model.TextContent{Text: "sorry"}},
				FinishReason: model.FinishReasonStop,
				Usage:        model.Usage{InputTokens: 10, OutputTokens: 5},
			}, nil
		},
	}

	knownTool := ai.Tool{
		Name: "known_tool",
		Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
			return "ok", nil
		},
	}

	result, err := ai.GenerateText(context.Background(), mock,
		ai.WithPrompt("Test"),
		ai.WithTools(knownTool),
		ai.WithMaxSteps(3),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	// The unknown tool should be reported as an error result.
	if len(result.ToolResults) != 1 {
		t.Fatalf("ToolResults = %d, want 1", len(result.ToolResults))
	}

	if !result.ToolResults[0].IsError {
		t.Error("unknown tool result should be an error")
	}
}

func TestGenerateTextModelError(t *testing.T) {
	mock := &mockModel{
		generateFn: func(_ context.Context, _ *model.Request) (*model.Response, error) {
			return nil, fmt.Errorf("connection refused")
		},
	}

	_, err := ai.GenerateText(context.Background(), mock, ai.WithPrompt("Hi"))
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestStreamTextBasic(t *testing.T) {
	mock := &mockModel{
		streamFn: func(_ context.Context, _ *model.Request) (*model.StreamResponse, error) {
			stream := func(yield func(model.Chunk) bool) {
				if !yield(model.ProviderMetadataChunk{Metadata: model.ProviderMetadata{
					RequestID: "req_stream",
					ModelID:   "mock-model-stream",
					Provider: map[string]string{
						"trace_id": "trace-stream",
					},
				}}) {
					return
				}
				if !yield(model.TextDelta{Text: "Hello"}) {
					return
				}
				if !yield(model.TextDelta{Text: " world"}) {
					return
				}
				if !yield(model.FinishChunk{
					FinishReason:    model.FinishReasonStop,
					RawFinishReason: "provider_stop",
				}) {
					return
				}
				yield(model.UsageChunk{Usage: model.Usage{InputTokens: 5, OutputTokens: 2}})
			}

			return &model.StreamResponse{
				Stream: stream,
				Metadata: model.ProviderMetadata{
					ModelID: "mock-model-initial",
					Provider: map[string]string{
						"trace_id": "trace-initial",
					},
				},
			}, nil
		},
	}

	streamResult, err := ai.StreamText(context.Background(), mock, ai.WithPrompt("Hi"))
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	var collected string
	for delta := range streamResult.Text() {
		collected += delta
	}

	if collected != "Hello world" {
		t.Errorf("collected = %q, want %q", collected, "Hello world")
	}

	result, err := streamResult.Result()
	if err != nil {
		t.Fatalf("Result: %v", err)
	}

	if result.Text != "Hello world" {
		t.Errorf("result.Text = %q, want %q", result.Text, "Hello world")
	}

	if result.FinishReason != model.FinishReasonStop {
		t.Errorf("FinishReason = %v, want stop", result.FinishReason)
	}

	if result.RawFinishReason != "provider_stop" {
		t.Errorf("RawFinishReason = %q, want provider_stop", result.RawFinishReason)
	}

	if result.Usage.InputTokens != 5 {
		t.Errorf("InputTokens = %d, want 5", result.Usage.InputTokens)
	}

	if result.Metadata.RequestID != "req_stream" {
		t.Errorf("Metadata.RequestID = %q, want req_stream", result.Metadata.RequestID)
	}

	if result.Metadata.Provider["trace_id"] != "trace-stream" {
		t.Errorf("Metadata.Provider[trace_id] = %q, want trace-stream", result.Metadata.Provider["trace_id"])
	}

	if streamResult.Metadata().RequestID != "req_stream" {
		t.Errorf("stream metadata RequestID = %q, want req_stream", streamResult.Metadata().RequestID)
	}

	if streamResult.Metadata().Provider["trace_id"] != "trace-stream" {
		t.Errorf("stream metadata trace_id = %q, want trace-stream", streamResult.Metadata().Provider["trace_id"])
	}
}

func TestStreamTextResultConsumesFull(t *testing.T) {
	mock := &mockModel{
		streamFn: func(_ context.Context, _ *model.Request) (*model.StreamResponse, error) {
			stream := func(yield func(model.Chunk) bool) {
				if !yield(model.TextDelta{Text: "auto"}) {
					return
				}
				yield(model.FinishChunk{FinishReason: model.FinishReasonStop})
			}

			return &model.StreamResponse{
				Stream:   stream,
				Metadata: model.ProviderMetadata{RequestID: "req_auto"},
			}, nil
		},
	}

	streamResult, err := ai.StreamText(context.Background(), mock, ai.WithPrompt("Hi"))
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	// Call Result() directly without consuming Text() — it should
	// auto-consume the stream.
	result, err := streamResult.Result()
	if err != nil {
		t.Fatalf("Result: %v", err)
	}

	if result.Text != "auto" {
		t.Errorf("result.Text = %q, want %q", result.Text, "auto")
	}

	if result.Metadata.RequestID != "req_auto" {
		t.Errorf("Metadata.RequestID = %q, want req_auto", result.Metadata.RequestID)
	}
}

func TestStreamTextChunks(t *testing.T) {
	mock := &mockModel{
		streamFn: func(_ context.Context, _ *model.Request) (*model.StreamResponse, error) {
			stream := func(yield func(model.Chunk) bool) {
				if !yield(model.TextDelta{Text: "chunk"}) {
					return
				}
				yield(model.FinishChunk{FinishReason: model.FinishReasonStop})
			}

			return &model.StreamResponse{
				Stream:   stream,
				Metadata: model.ProviderMetadata{RequestID: "req_chunk"},
			}, nil
		},
	}

	streamResult, err := ai.StreamText(context.Background(), mock, ai.WithPrompt("Hi"))
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	var count int
	for range streamResult.Chunks() {
		count++
	}

	if count != 2 {
		t.Errorf("chunk count = %d, want 2", count)
	}
}

func TestStreamTextResultAfterEarlyStop(t *testing.T) {
	mock := &mockModel{
		streamFn: func(_ context.Context, _ *model.Request) (*model.StreamResponse, error) {
			stream := func(yield func(model.Chunk) bool) {
				if !yield(model.TextDelta{Text: "first"}) {
					return
				}
				yield(model.TextDelta{Text: " second"})
			}

			return &model.StreamResponse{
				Stream: stream,
			}, nil
		},
	}

	streamResult, err := ai.StreamText(context.Background(), mock, ai.WithPrompt("Hi"))
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	for delta := range streamResult.Text() {
		if delta == "first" {
			break
		}
	}

	result, err := streamResult.Result()
	if err != nil {
		t.Fatalf("Result: %v", err)
	}

	if result.Text != "first" {
		t.Fatalf("result.Text = %q, want first", result.Text)
	}
}

func TestStreamTextOnFinishUsesRequestContext(t *testing.T) {
	ctx := context.WithValue(context.Background(), testContextKey("trace"), "stream-text")

	mock := &mockModel{
		streamFn: func(_ context.Context, _ *model.Request) (*model.StreamResponse, error) {
			return &model.StreamResponse{
				Stream: func(yield func(model.Chunk) bool) {
					if !yield(model.TextDelta{Text: "done"}) {
						return
					}

					yield(model.FinishChunk{
						FinishReason:    model.FinishReasonStop,
						RawFinishReason: "provider_stop",
					})
				},
			}, nil
		},
	}

	streamResult, err := ai.StreamText(ctx, mock,
		ai.WithPrompt("Hi"),
		ai.WithOnFinish(func(hookCtx context.Context, result *ai.GenerateResult) error {
			if got := hookCtx.Value(testContextKey("trace")); got != "stream-text" {
				t.Fatalf("hook context value = %v, want stream-text", got)
			}

			if result.RawFinishReason != "provider_stop" {
				t.Fatalf("RawFinishReason = %q, want provider_stop", result.RawFinishReason)
			}

			return nil
		}),
	)
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	result, err := streamResult.Result()
	if err != nil {
		t.Fatalf("Result: %v", err)
	}

	if result.RawFinishReason != "provider_stop" {
		t.Fatalf("result.RawFinishReason = %q, want provider_stop", result.RawFinishReason)
	}
}

func TestStreamTextModelError(t *testing.T) {
	mock := &mockModel{
		streamFn: func(_ context.Context, _ *model.Request) (*model.StreamResponse, error) {
			return nil, fmt.Errorf("stream error")
		},
	}

	_, err := ai.StreamText(context.Background(), mock, ai.WithPrompt("Hi"))
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestStreamTextToolLoop(t *testing.T) {
	streamCallCount := 0

	mock := &mockModel{
		streamFn: func(_ context.Context, req *model.Request) (*model.StreamResponse, error) {
			streamCallCount++

			if streamCallCount == 1 {
				// Step 1: model streams a tool call.
				return &model.StreamResponse{
					Stream: func(yield func(model.Chunk) bool) {
						if !yield(model.ToolCallDelta{
							ToolCallID: "call_1",
							ToolName:   "get_weather",
							ArgsJSON:   `{"city":`,
						}) {
							return
						}
						if !yield(model.ToolCallDelta{
							ToolCallID: "call_1",
							ToolName:   "get_weather",
							ArgsJSON:   `"NYC"}`,
						}) {
							return
						}
						if !yield(model.FinishChunk{
							FinishReason:    model.FinishReasonToolCalls,
							RawFinishReason: "tool_calls",
						}) {
							return
						}
						yield(model.UsageChunk{Usage: model.Usage{InputTokens: 10, OutputTokens: 5}})
					},
				}, nil
			}

			// Step 2: verify tool result was appended, stream final answer.
			lastMsg := req.Messages[len(req.Messages)-1]
			if lastMsg.Role != model.RoleTool {
				t.Errorf("step 2: last message role = %v, want tool", lastMsg.Role)
			}

			return &model.StreamResponse{
				Stream: func(yield func(model.Chunk) bool) {
					if !yield(model.TextDelta{Text: "It's sunny"}) {
						return
					}
					if !yield(model.TextDelta{Text: " in NYC!"}) {
						return
					}
					if !yield(model.FinishChunk{
						FinishReason:    model.FinishReasonStop,
						RawFinishReason: "stop",
					}) {
						return
					}
					yield(model.UsageChunk{Usage: model.Usage{InputTokens: 20, OutputTokens: 10}})
				},
			}, nil
		},
	}

	weatherTool := ai.Tool{
		Name:        "get_weather",
		Description: "Get current weather",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
		Execute: func(_ context.Context, input json.RawMessage) (string, error) {
			var args struct {
				City string `json:"city"`
			}
			if err := json.Unmarshal(input, &args); err != nil {
				return "", err
			}

			return fmt.Sprintf("Sunny, 72°F in %s", args.City), nil
		},
	}

	streamResult, err := ai.StreamText(context.Background(), mock,
		ai.WithPrompt("What's the weather in NYC?"),
		ai.WithTools(weatherTool),
		ai.WithMaxSteps(3),
	)
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	// Collect text deltas from the stream.
	var collected string
	for delta := range streamResult.Text() {
		collected += delta
	}

	if collected != "It's sunny in NYC!" {
		t.Errorf("collected text = %q, want %q", collected, "It's sunny in NYC!")
	}

	result, err := streamResult.Result()
	if err != nil {
		t.Fatalf("Result: %v", err)
	}

	if streamCallCount != 2 {
		t.Errorf("stream called %d times, want 2", streamCallCount)
	}

	if result.Text != "It's sunny in NYC!" {
		t.Errorf("result.Text = %q, want %q", result.Text, "It's sunny in NYC!")
	}

	if result.FinishReason != model.FinishReasonStop {
		t.Errorf("FinishReason = %v, want stop", result.FinishReason)
	}

	if len(result.Steps) != 2 {
		t.Fatalf("Steps = %d, want 2", len(result.Steps))
	}

	if result.Steps[0].FinishReason != model.FinishReasonToolCalls {
		t.Errorf("Steps[0].FinishReason = %v, want tool_calls", result.Steps[0].FinishReason)
	}

	if result.Steps[1].FinishReason != model.FinishReasonStop {
		t.Errorf("Steps[1].FinishReason = %v, want stop", result.Steps[1].FinishReason)
	}

	if result.Usage.InputTokens != 30 {
		t.Errorf("total InputTokens = %d, want 30", result.Usage.InputTokens)
	}

	if result.Usage.OutputTokens != 15 {
		t.Errorf("total OutputTokens = %d, want 15", result.Usage.OutputTokens)
	}

	if len(result.ToolResults) != 1 {
		t.Fatalf("ToolResults = %d, want 1", len(result.ToolResults))
	}

	if result.ToolResults[0].IsError {
		t.Error("tool result should not be an error")
	}

	if !strings.Contains(result.ToolResults[0].Content, "Sunny") {
		t.Errorf("tool result content = %q, want weather string", result.ToolResults[0].Content)
	}
}

func TestStreamTextToolLoopResultWithoutConsuming(t *testing.T) {
	// Verify that calling Result() directly (without ranging Text())
	// still runs the full tool loop.
	streamCallCount := 0

	mock := &mockModel{
		streamFn: func(_ context.Context, _ *model.Request) (*model.StreamResponse, error) {
			streamCallCount++

			if streamCallCount == 1 {
				return &model.StreamResponse{
					Stream: func(yield func(model.Chunk) bool) {
						if !yield(model.ToolCallDelta{
							ToolCallID: "call_1",
							ToolName:   "ping",
							ArgsJSON:   `{}`,
						}) {
							return
						}
						yield(model.FinishChunk{FinishReason: model.FinishReasonToolCalls})
					},
				}, nil
			}

			return &model.StreamResponse{
				Stream: func(yield func(model.Chunk) bool) {
					if !yield(model.TextDelta{Text: "pong"}) {
						return
					}
					yield(model.FinishChunk{FinishReason: model.FinishReasonStop})
				},
			}, nil
		},
	}

	pingTool := ai.Tool{
		Name: "ping",
		Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
			return "pong", nil
		},
	}

	streamResult, err := ai.StreamText(context.Background(), mock,
		ai.WithPrompt("Ping"),
		ai.WithTools(pingTool),
		ai.WithMaxSteps(3),
	)
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	// Call Result() directly without consuming Text().
	result, err := streamResult.Result()
	if err != nil {
		t.Fatalf("Result: %v", err)
	}

	if result.Text != "pong" {
		t.Errorf("result.Text = %q, want pong", result.Text)
	}

	if streamCallCount != 2 {
		t.Errorf("stream called %d times, want 2", streamCallCount)
	}

	if len(result.Steps) != 2 {
		t.Errorf("Steps = %d, want 2", len(result.Steps))
	}
}

func TestStreamTextToolLoopHooksAndPrepareStep(t *testing.T) {
	ctx := context.WithValue(context.Background(), testContextKey("trace"), "stream-tool-loop")
	var hookCalls []string

	streamCallCount := 0
	mock := &mockModel{
		streamFn: func(_ context.Context, _ *model.Request) (*model.StreamResponse, error) {
			streamCallCount++

			if streamCallCount == 1 {
				return &model.StreamResponse{
					Stream: func(yield func(model.Chunk) bool) {
						if !yield(model.ToolCallDelta{
							ToolCallID: "call_1",
							ToolName:   "echo",
							ArgsJSON:   `{"msg":"hi"}`,
						}) {
							return
						}
						yield(model.FinishChunk{FinishReason: model.FinishReasonToolCalls})
					},
				}, nil
			}

			return &model.StreamResponse{
				Stream: func(yield func(model.Chunk) bool) {
					if !yield(model.TextDelta{Text: "done"}) {
						return
					}
					yield(model.FinishChunk{FinishReason: model.FinishReasonStop})
				},
			}, nil
		},
	}

	echoTool := ai.Tool{
		Name: "echo",
		Execute: func(_ context.Context, input json.RawMessage) (string, error) {
			return string(input), nil
		},
	}

	streamResult, err := ai.StreamText(ctx, mock,
		ai.WithPrompt("echo"),
		ai.WithTools(echoTool),
		ai.WithMaxSteps(3),
		ai.WithPrepareStep(func(hookCtx context.Context, step int, _ *model.Request) error {
			if got := hookCtx.Value(testContextKey("trace")); got != "stream-tool-loop" {
				t.Fatalf("prepare context = %v, want stream-tool-loop", got)
			}
			hookCalls = append(hookCalls, fmt.Sprintf("prepare:%d", step))

			return nil
		}),
		ai.WithOnStepFinish(func(hookCtx context.Context, step ai.Step) error {
			if got := hookCtx.Value(testContextKey("trace")); got != "stream-tool-loop" {
				t.Fatalf("step context = %v, want stream-tool-loop", got)
			}
			hookCalls = append(hookCalls, fmt.Sprintf("step:%s", step.FinishReason))

			return nil
		}),
		ai.WithOnFinish(func(hookCtx context.Context, _ *ai.GenerateResult) error {
			if got := hookCtx.Value(testContextKey("trace")); got != "stream-tool-loop" {
				t.Fatalf("finish context = %v, want stream-tool-loop", got)
			}
			hookCalls = append(hookCalls, "finish")

			return nil
		}),
	)
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	result, err := streamResult.Result()
	if err != nil {
		t.Fatalf("Result: %v", err)
	}

	if result.Text != "done" {
		t.Errorf("result.Text = %q, want done", result.Text)
	}

	expected := "[prepare:0 step:tool_calls prepare:1 step:stop finish]"
	if got := fmt.Sprint(hookCalls); got != expected {
		t.Errorf("hook calls = %s, want %s", got, expected)
	}
}

func TestStreamTextToolLoopNoToolsConfigured(t *testing.T) {
	// When no tools are configured, StreamText should behave exactly
	// like the single-step path even with MaxSteps > 1.
	mock := &mockModel{
		streamFn: func(_ context.Context, _ *model.Request) (*model.StreamResponse, error) {
			return &model.StreamResponse{
				Stream: func(yield func(model.Chunk) bool) {
					if !yield(model.TextDelta{Text: "plain"}) {
						return
					}
					yield(model.FinishChunk{FinishReason: model.FinishReasonStop})
				},
			}, nil
		},
	}

	streamResult, err := ai.StreamText(context.Background(), mock,
		ai.WithPrompt("Hi"),
		ai.WithMaxSteps(5),
	)
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	result, err := streamResult.Result()
	if err != nil {
		t.Fatalf("Result: %v", err)
	}

	if result.Text != "plain" {
		t.Errorf("result.Text = %q, want plain", result.Text)
	}

	// Single-step path: no Steps populated.
	if len(result.Steps) != 0 {
		t.Errorf("Steps = %d, want 0 (single-step path)", len(result.Steps))
	}
}

func TestStreamTextToolLoopChunksFromAllSteps(t *testing.T) {
	// Verify that Chunks() yields chunks from all steps.
	streamCallCount := 0
	mock := &mockModel{
		streamFn: func(_ context.Context, _ *model.Request) (*model.StreamResponse, error) {
			streamCallCount++

			if streamCallCount == 1 {
				return &model.StreamResponse{
					Stream: func(yield func(model.Chunk) bool) {
						if !yield(model.TextDelta{Text: "thinking..."}) {
							return
						}
						if !yield(model.ToolCallDelta{
							ToolCallID: "call_1",
							ToolName:   "calc",
							ArgsJSON:   `{"x":1}`,
						}) {
							return
						}
						if !yield(model.FinishChunk{FinishReason: model.FinishReasonToolCalls}) {
							return
						}
						yield(model.UsageChunk{Usage: model.Usage{InputTokens: 5, OutputTokens: 3}})
					},
				}, nil
			}

			return &model.StreamResponse{
				Stream: func(yield func(model.Chunk) bool) {
					if !yield(model.TextDelta{Text: "result: 1"}) {
						return
					}
					if !yield(model.FinishChunk{FinishReason: model.FinishReasonStop}) {
						return
					}
					yield(model.UsageChunk{Usage: model.Usage{InputTokens: 10, OutputTokens: 5}})
				},
			}, nil
		},
	}

	calcTool := ai.Tool{
		Name: "calc",
		Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
			return "1", nil
		},
	}

	streamResult, err := ai.StreamText(context.Background(), mock,
		ai.WithPrompt("calc"),
		ai.WithTools(calcTool),
		ai.WithMaxSteps(3),
	)
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	var (
		textChunks     int
		toolCallChunks int
		finishChunks   int
		usageChunks    int
	)
	for chunk := range streamResult.Chunks() {
		switch chunk.(type) {
		case model.TextDelta:
			textChunks++
		case model.ToolCallDelta:
			toolCallChunks++
		case model.FinishChunk:
			finishChunks++
		case model.UsageChunk:
			usageChunks++
		}
	}

	if textChunks != 2 {
		t.Errorf("textChunks = %d, want 2 (one from each step)", textChunks)
	}

	if toolCallChunks != 1 {
		t.Errorf("toolCallChunks = %d, want 1", toolCallChunks)
	}

	if finishChunks != 2 {
		t.Errorf("finishChunks = %d, want 2 (one per step)", finishChunks)
	}

	if usageChunks != 2 {
		t.Errorf("usageChunks = %d, want 2 (one per step)", usageChunks)
	}
}

func TestGenerateObjectBasic(t *testing.T) {
	type Recipe struct {
		Name        string   `json:"name"`
		Ingredients []string `json:"ingredients"`
	}

	mock := &mockModel{
		supportsResponseFormat: true,
		generateFn: func(_ context.Context, req *model.Request) (*model.Response, error) {
			// Verify JSON schema response format was set.
			if req.ResponseFormat == nil {
				t.Fatal("ResponseFormat not set")
			}

			if req.ResponseFormat.Type != "json_schema" {
				t.Errorf("ResponseFormat.Type = %q, want json_schema", req.ResponseFormat.Type)
			}

			return &model.Response{
				Content: []model.Content{
					model.TextContent{Text: `{"name":"Pasta","ingredients":["noodles","sauce"]}`},
				},
				FinishReason: model.FinishReasonStop,
				Usage:        model.Usage{InputTokens: 10, OutputTokens: 20},
				Metadata: model.ProviderMetadata{
					RequestID: "req_object",
					Provider: map[string]string{
						"schema_mode": "strict",
					},
				},
			}, nil
		},
	}

	result, err := ai.GenerateObject[Recipe](context.Background(), mock,
		ai.WithPrompt("Give me a pasta recipe"),
	)
	if err != nil {
		t.Fatalf("GenerateObject: %v", err)
	}

	if result.Object.Name != "Pasta" {
		t.Errorf("Name = %q, want Pasta", result.Object.Name)
	}

	if len(result.Object.Ingredients) != 2 {
		t.Errorf("Ingredients length = %d, want 2", len(result.Object.Ingredients))
	}

	if result.Usage.InputTokens != 10 {
		t.Errorf("InputTokens = %d, want 10", result.Usage.InputTokens)
	}

	if result.Metadata.RequestID != "req_object" {
		t.Errorf("Metadata.RequestID = %q, want req_object", result.Metadata.RequestID)
	}

	if result.Metadata.Provider["schema_mode"] != "strict" {
		t.Errorf("Metadata.Provider[schema_mode] = %q, want strict", result.Metadata.Provider["schema_mode"])
	}
}

func TestGenerateObjectInvalidJSON(t *testing.T) {
	type Item struct {
		Name string `json:"name"`
	}

	mock := &mockModel{
		supportsResponseFormat: true,
		generateFn: func(_ context.Context, _ *model.Request) (*model.Response, error) {
			return &model.Response{
				Content:      []model.Content{model.TextContent{Text: "not valid json"}},
				FinishReason: model.FinishReasonStop,
			}, nil
		},
	}

	_, err := ai.GenerateObject[Item](context.Background(), mock,
		ai.WithPrompt("Bad response"),
	)
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestGenerateObjectEmptyResponse(t *testing.T) {
	type Item struct {
		Name string `json:"name"`
	}

	mock := &mockModel{
		supportsResponseFormat: true,
		generateFn: func(_ context.Context, _ *model.Request) (*model.Response, error) {
			return &model.Response{
				Content:      []model.Content{},
				FinishReason: model.FinishReasonStop,
			}, nil
		},
	}

	_, err := ai.GenerateObject[Item](context.Background(), mock,
		ai.WithPrompt("Empty"),
	)
	if err == nil {
		t.Fatal("expected error for empty response")
	}
}

func TestEmbedBasic(t *testing.T) {
	mock := &mockEmbeddingModel{
		embedFn: func(_ context.Context, texts []string) (*model.EmbedResponse, error) {
			if len(texts) != 1 {
				t.Errorf("texts length = %d, want 1", len(texts))
			}

			return &model.EmbedResponse{
				Embeddings: [][]float64{{0.1, 0.2, 0.3}},
				Usage:      model.Usage{InputTokens: 3},
			}, nil
		},
	}

	result, err := ai.Embed(context.Background(), mock, "hello world")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if len(result.Embedding) != 3 {
		t.Errorf("Embedding length = %d, want 3", len(result.Embedding))
	}

	if result.Embedding[0] != 0.1 {
		t.Errorf("Embedding[0] = %f, want 0.1", result.Embedding[0])
	}

	if result.Usage.InputTokens != 3 {
		t.Errorf("InputTokens = %d, want 3", result.Usage.InputTokens)
	}
}

func TestEmbedEmptyResponse(t *testing.T) {
	mock := &mockEmbeddingModel{
		embedFn: func(_ context.Context, _ []string) (*model.EmbedResponse, error) {
			return &model.EmbedResponse{Embeddings: nil}, nil
		},
	}

	_, err := ai.Embed(context.Background(), mock, "hello")
	if err == nil {
		t.Fatal("expected error for empty embeddings")
	}
}

func TestEmbedManyBasic(t *testing.T) {
	mock := &mockEmbeddingModel{
		embedFn: func(_ context.Context, texts []string) (*model.EmbedResponse, error) {
			embeddings := make([][]float64, len(texts))
			for i := range texts {
				embeddings[i] = []float64{float64(i), float64(i + 1)}
			}

			return &model.EmbedResponse{
				Embeddings: embeddings,
				Usage:      model.Usage{InputTokens: len(texts) * 2},
			}, nil
		},
	}

	result, err := ai.EmbedMany(context.Background(), mock, []string{"hello", "world", "test"})
	if err != nil {
		t.Fatalf("EmbedMany: %v", err)
	}

	if len(result.Embeddings) != 3 {
		t.Errorf("Embeddings length = %d, want 3", len(result.Embeddings))
	}

	if result.Usage.InputTokens != 6 {
		t.Errorf("InputTokens = %d, want 6", result.Usage.InputTokens)
	}
}

func TestGenerateTextRejectsInvalidMaxSteps(t *testing.T) {
	mock := &mockModel{
		generateFn: func(_ context.Context, _ *model.Request) (*model.Response, error) {
			return &model.Response{}, nil
		},
	}

	_, err := ai.GenerateText(context.Background(), mock,
		ai.WithPrompt("hello"),
		ai.WithMaxSteps(0),
	)
	if !errors.Is(err, ai.ErrInvalidMaxSteps) {
		t.Fatalf("err = %v, want ErrInvalidMaxSteps", err)
	}
}

func TestGenerateObjectFallbackUsesToolSchema(t *testing.T) {
	type Recipe struct {
		Name string `json:"name"`
	}

	mock := &mockModel{
		generateFn: func(_ context.Context, req *model.Request) (*model.Response, error) {
			if req.ResponseFormat != nil {
				t.Fatal("response format should not be set for fallback path")
			}

			if len(req.Tools) != 1 {
				t.Fatalf("tools length = %d, want 1", len(req.Tools))
			}

			if req.ToolChoice == nil || req.ToolChoice.ToolName != "structured_output" {
				t.Fatalf("tool choice = %#v, want structured_output", req.ToolChoice)
			}

			return &model.Response{
				Content: []model.Content{
					model.ToolCallContent{
						ToolCallID: "call_1",
						ToolName:   "structured_output",
						Args:       `{"name":"Soup"}`,
					},
				},
				FinishReason: model.FinishReasonToolCalls,
			}, nil
		},
	}

	result, err := ai.GenerateObject[Recipe](context.Background(), mock,
		ai.WithPrompt("Give me a recipe"),
	)
	if err != nil {
		t.Fatalf("GenerateObject fallback: %v", err)
	}

	if result.Object.Name != "Soup" {
		t.Fatalf("object name = %q, want Soup", result.Object.Name)
	}

	if result.RawJSON != `{"name":"Soup"}` {
		t.Fatalf("raw json = %q, want JSON payload", result.RawJSON)
	}
}

func TestGenerateObjectFallbackRunsHooks(t *testing.T) {
	type Recipe struct {
		Name string `json:"name"`
	}

	ctx := context.WithValue(context.Background(), testContextKey("trace"), "generate-object")
	var calls []string

	mock := &mockModel{
		generateFn: func(hookCtx context.Context, req *model.Request) (*model.Response, error) {
			if got := hookCtx.Value(testContextKey("trace")); got != "generate-object" {
				t.Fatalf("generate context value = %v, want generate-object", got)
			}

			if req.ToolChoice == nil || req.ToolChoice.ToolName != "structured_output" {
				t.Fatalf("tool choice = %#v, want structured_output", req.ToolChoice)
			}

			return &model.Response{
				Content: []model.Content{
					model.ToolCallContent{
						ToolCallID: "call_1",
						ToolName:   "structured_output",
						Args:       `{"name":"Soup"}`,
					},
				},
				FinishReason: model.FinishReasonToolCalls,
			}, nil
		},
	}

	result, err := ai.GenerateObject[Recipe](ctx, mock,
		ai.WithPrompt("Give me a recipe"),
		ai.WithPrepareStep(func(hookCtx context.Context, step int, req *model.Request) error {
			calls = append(calls, "prepare")
			if got := hookCtx.Value(testContextKey("trace")); got != "generate-object" {
				t.Fatalf("prepare context value = %v, want generate-object", got)
			}

			if step != 0 {
				t.Fatalf("prepare step = %d, want 0", step)
			}

			if req.ToolChoice == nil || req.ToolChoice.ToolName != "structured_output" {
				t.Fatalf("prepare tool choice = %#v, want structured_output", req.ToolChoice)
			}

			return nil
		}),
		ai.WithOnStepFinish(func(hookCtx context.Context, step ai.Step) error {
			calls = append(calls, "step")
			if got := hookCtx.Value(testContextKey("trace")); got != "generate-object" {
				t.Fatalf("step context value = %v, want generate-object", got)
			}

			if len(step.ToolCalls) != 1 || step.ToolCalls[0].ToolName != "structured_output" {
				t.Fatalf("step tool calls = %#v, want structured_output", step.ToolCalls)
			}

			return nil
		}),
		ai.WithOnFinish(func(hookCtx context.Context, result *ai.GenerateResult) error {
			calls = append(calls, "finish")
			if got := hookCtx.Value(testContextKey("trace")); got != "generate-object" {
				t.Fatalf("finish context value = %v, want generate-object", got)
			}

			if len(result.Steps) != 1 || len(result.Steps[0].ToolCalls) != 1 {
				t.Fatalf("result steps = %#v, want one structured-output step", result.Steps)
			}

			return nil
		}),
	)
	if err != nil {
		t.Fatalf("GenerateObject: %v", err)
	}

	if result.Object.Name != "Soup" {
		t.Fatalf("result.Object.Name = %q, want Soup", result.Object.Name)
	}

	if got := fmt.Sprint(calls); got != "[prepare step finish]" {
		t.Fatalf("hook calls = %s, want [prepare step finish]", got)
	}
}

func TestStreamObjectConsumesUsageBeforeTimeoutCleanup(t *testing.T) {
	type Recipe struct {
		Name string `json:"name"`
	}

	mock := &mockModel{
		supportsResponseFormat: true,
		streamFn: func(ctx context.Context, _ *model.Request) (*model.StreamResponse, error) {
			return &model.StreamResponse{
				Stream: func(yield func(model.Chunk) bool) {
					if ctx.Err() != nil {
						return
					}

					if !yield(model.TextDelta{Text: `{"name":"Soup"}`}) {
						return
					}

					yield(model.UsageChunk{Usage: model.Usage{InputTokens: 3, OutputTokens: 4}})
				},
			}, nil
		},
	}

	streamResult, err := ai.StreamObject[Recipe](context.Background(), mock,
		ai.WithPrompt("Give me soup"),
		ai.WithTimeout(50*time.Millisecond),
	)
	if err != nil {
		t.Fatalf("StreamObject: %v", err)
	}

	result, err := streamResult.Result()
	if err != nil {
		t.Fatalf("Result: %v", err)
	}

	if result.Object.Name != "Soup" {
		t.Fatalf("result.Object.Name = %q, want Soup", result.Object.Name)
	}

	if result.Usage.InputTokens != 3 || result.Usage.OutputTokens != 4 {
		t.Fatalf("usage = %+v, want input=3 output=4", result.Usage)
	}
}

func TestStreamObjectRunsHooksWithRequestContext(t *testing.T) {
	type Recipe struct {
		Name string `json:"name"`
	}

	ctx := context.WithValue(context.Background(), testContextKey("trace"), "stream-object")
	var calls []string

	mock := &mockModel{
		supportsResponseFormat: true,
		streamFn: func(hookCtx context.Context, req *model.Request) (*model.StreamResponse, error) {
			if got := hookCtx.Value(testContextKey("trace")); got != "stream-object" {
				t.Fatalf("stream context value = %v, want stream-object", got)
			}

			if req.ResponseFormat == nil || req.ResponseFormat.Type != "json_schema" {
				t.Fatalf("response format = %#v, want json_schema", req.ResponseFormat)
			}

			return &model.StreamResponse{
				Stream: func(yield func(model.Chunk) bool) {
					if !yield(model.TextDelta{Text: `{"name":"Soup"}`}) {
						return
					}

					if !yield(model.FinishChunk{
						FinishReason:    model.FinishReasonStop,
						RawFinishReason: "provider_stop",
					}) {
						return
					}

					yield(model.UsageChunk{Usage: model.Usage{InputTokens: 2, OutputTokens: 3}})
				},
			}, nil
		},
	}

	streamResult, err := ai.StreamObject[Recipe](ctx, mock,
		ai.WithPrompt("Give me a recipe"),
		ai.WithPrepareStep(func(hookCtx context.Context, step int, req *model.Request) error {
			calls = append(calls, "prepare")
			if got := hookCtx.Value(testContextKey("trace")); got != "stream-object" {
				t.Fatalf("prepare context value = %v, want stream-object", got)
			}

			if step != 0 {
				t.Fatalf("prepare step = %d, want 0", step)
			}

			if req.ResponseFormat == nil || req.ResponseFormat.Type != "json_schema" {
				t.Fatalf("prepare response format = %#v, want json_schema", req.ResponseFormat)
			}

			return nil
		}),
		ai.WithOnStepFinish(func(hookCtx context.Context, step ai.Step) error {
			calls = append(calls, "step")
			if got := hookCtx.Value(testContextKey("trace")); got != "stream-object" {
				t.Fatalf("step context value = %v, want stream-object", got)
			}

			if step.Text != `{"name":"Soup"}` {
				t.Fatalf("step.Text = %q, want JSON body", step.Text)
			}

			if step.RawFinishReason != "provider_stop" {
				t.Fatalf("step.RawFinishReason = %q, want provider_stop", step.RawFinishReason)
			}

			return nil
		}),
		ai.WithOnFinish(func(hookCtx context.Context, result *ai.GenerateResult) error {
			calls = append(calls, "finish")
			if got := hookCtx.Value(testContextKey("trace")); got != "stream-object" {
				t.Fatalf("finish context value = %v, want stream-object", got)
			}

			if result.Text != `{"name":"Soup"}` {
				t.Fatalf("result.Text = %q, want JSON body", result.Text)
			}

			if result.RawFinishReason != "provider_stop" {
				t.Fatalf("result.RawFinishReason = %q, want provider_stop", result.RawFinishReason)
			}

			return nil
		}),
	)
	if err != nil {
		t.Fatalf("StreamObject: %v", err)
	}

	result, err := streamResult.Result()
	if err != nil {
		t.Fatalf("Result: %v", err)
	}

	if result.Object.Name != "Soup" {
		t.Fatalf("result.Object.Name = %q, want Soup", result.Object.Name)
	}

	if got := fmt.Sprint(calls); got != "[prepare step finish]" {
		t.Fatalf("hook calls = %s, want [prepare step finish]", got)
	}
}

func TestExtractJSONMiddleware(t *testing.T) {
	base := &mockModel{
		generateFn: func(_ context.Context, _ *model.Request) (*model.Response, error) {
			return &model.Response{
				Content: []model.Content{
					model.TextContent{Text: "```json\n{\"ok\":true}\n```"},
				},
				FinishReason: model.FinishReasonStop,
			}, nil
		},
	}

	modelWithMiddleware := ai.WrapLanguageModel(base, ai.ExtractJSONMiddleware())
	result, err := ai.GenerateText(context.Background(), modelWithMiddleware, ai.WithPrompt("Hi"))
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	if result.Text != `{"ok":true}` {
		t.Fatalf("result.Text = %q, want stripped JSON", result.Text)
	}
}

func TestExtractJSONMiddlewareStreamYieldsBeforeCompletion(t *testing.T) {
	continueStream := make(chan struct{})
	base := &mockModel{
		streamFn: func(_ context.Context, _ *model.Request) (*model.StreamResponse, error) {
			return &model.StreamResponse{
				Stream: func(yield func(model.Chunk) bool) {
					if !yield(model.TextDelta{Text: "```json\n{\"ok\":"}) {
						return
					}

					<-continueStream

					if !yield(model.TextDelta{Text: "true}\n```"}) {
						return
					}

					yield(model.FinishChunk{FinishReason: model.FinishReasonStop})
				},
			}, nil
		},
	}

	modelWithMiddleware := ai.WrapLanguageModel(base, ai.ExtractJSONMiddleware())
	streamResult, err := ai.StreamText(context.Background(), modelWithMiddleware, ai.WithPrompt("Hi"))
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	deltas := make(chan string, 4)
	done := make(chan struct{})
	go func() {
		defer close(done)
		for delta := range streamResult.Text() {
			deltas <- delta
		}
	}()

	select {
	case delta := <-deltas:
		if delta != `{"ok":` {
			t.Fatalf("first delta = %q, want JSON body prefix", delta)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for streamed JSON delta")
	}

	close(continueStream)
	<-done

	result, err := streamResult.Result()
	if err != nil {
		t.Fatalf("Result: %v", err)
	}

	if result.Text != `{"ok":true}` {
		t.Fatalf("result.Text = %q, want stripped JSON", result.Text)
	}
}

func TestExtractReasoningMiddlewareStreamYieldsBeforeCompletion(t *testing.T) {
	continueStream := make(chan struct{})
	base := &mockModel{
		streamFn: func(_ context.Context, _ *model.Request) (*model.StreamResponse, error) {
			return &model.StreamResponse{
				Stream: func(yield func(model.Chunk) bool) {
					if !yield(model.TextDelta{Text: "Answer: "}) {
						return
					}

					<-continueStream

					if !yield(model.TextDelta{Text: "<reasoning>internal</reasoning>done"}) {
						return
					}

					yield(model.FinishChunk{FinishReason: model.FinishReasonStop})
				},
			}, nil
		},
	}

	modelWithMiddleware := ai.WrapLanguageModel(base, ai.ExtractReasoningMiddleware())
	streamResult, err := ai.StreamText(context.Background(), modelWithMiddleware, ai.WithPrompt("Hi"))
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	deltas := make(chan string, 4)
	done := make(chan struct{})
	go func() {
		defer close(done)
		for delta := range streamResult.Text() {
			deltas <- delta
		}
	}()

	select {
	case delta := <-deltas:
		if delta != "Answer: " {
			t.Fatalf("first delta = %q, want text before reasoning block", delta)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for streamed text delta")
	}

	close(continueStream)
	<-done

	result, err := streamResult.Result()
	if err != nil {
		t.Fatalf("Result: %v", err)
	}

	if result.Text != "Answer: done" {
		t.Fatalf("result.Text = %q, want Answer: done", result.Text)
	}

	if result.ReasoningText() != "internal" {
		t.Fatalf("result.ReasoningText() = %q, want internal", result.ReasoningText())
	}
}

func TestSimulateStreamingMiddlewarePreservesRawFinishReason(t *testing.T) {
	base := &mockModel{
		generateFn: func(_ context.Context, _ *model.Request) (*model.Response, error) {
			return &model.Response{
				Content: []model.Content{
					model.TextContent{Text: "done"},
				},
				FinishReason:    model.FinishReasonStop,
				RawFinishReason: "provider_stop",
			}, nil
		},
	}

	modelWithMiddleware := ai.WrapLanguageModel(base, ai.SimulateStreamingMiddleware(0))

	streamResult, err := ai.StreamText(context.Background(), modelWithMiddleware, ai.WithPrompt("Hi"))
	if err != nil {
		t.Fatalf("StreamText: %v", err)
	}

	result, err := streamResult.Result()
	if err != nil {
		t.Fatalf("Result: %v", err)
	}

	if result.RawFinishReason != "provider_stop" {
		t.Fatalf("RawFinishReason = %q, want provider_stop", result.RawFinishReason)
	}
}

func TestDefaultSettingsMiddlewareMergesProviderOptions(t *testing.T) {
	base := &mockModel{
		generateFn: func(_ context.Context, req *model.Request) (*model.Response, error) {
			metadata, ok := req.ProviderOptions.Lookup("openai", "metadata")
			if !ok {
				t.Fatal("openai metadata default missing")
			}

			if string(metadata) != `{"trace_id":"default-trace"}` {
				t.Fatalf("metadata = %s, want default metadata", metadata)
			}

			reasoning, ok := req.ProviderOptions.Lookup("openai", "reasoning_effort")
			if !ok {
				t.Fatal("openai reasoning_effort override missing")
			}

			if string(reasoning) != `"high"` {
				t.Fatalf("reasoning_effort = %s, want \"high\"", reasoning)
			}

			anthropicMetadata, ok := req.ProviderOptions.Lookup("anthropic", "metadata")
			if !ok {
				t.Fatal("anthropic metadata default missing")
			}

			if string(anthropicMetadata) != `{"trace_id":"anthropic-trace"}` {
				t.Fatalf("anthropic metadata = %s, want anthropic default", anthropicMetadata)
			}

			return &model.Response{
				Content:      []model.Content{model.TextContent{Text: "ok"}},
				FinishReason: model.FinishReasonStop,
			}, nil
		},
	}

	modelWithMiddleware := ai.WrapLanguageModel(base, ai.DefaultSettingsMiddleware(model.Request{
		ProviderOptions: model.ProviderOptions{
			"openai": {
				"metadata": json.RawMessage(`{"trace_id":"default-trace"}`),
			},
			"anthropic": {
				"metadata": json.RawMessage(`{"trace_id":"anthropic-trace"}`),
			},
		},
	}))

	_, err := ai.GenerateText(context.Background(), modelWithMiddleware,
		ai.WithPrompt("Hi"),
		ai.WithProviderOption("openai", "reasoning_effort", "high"),
	)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}
}

func TestSchemaFor(t *testing.T) {
	type TestStruct struct {
		Name string `json:"name"`
		Age  int    `json:"age"`
	}

	raw := ai.SchemaFor[TestStruct]()
	if raw == nil {
		t.Fatal("SchemaFor returned nil")
	}

	var s map[string]any
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if s["type"] != "object" {
		t.Errorf("type = %v, want object", s["type"])
	}
}

func TestUsageTotalTokens(t *testing.T) {
	u := ai.Usage{InputTokens: 100, OutputTokens: 50}

	if got := u.TotalTokens(); got != 150 {
		t.Errorf("TotalTokens() = %d, want 150", got)
	}
}

func TestAPIError(t *testing.T) {
	err := ai.NewAPIError(429, "rate limited")

	if err.StatusCode != 429 {
		t.Errorf("StatusCode = %d, want 429", err.StatusCode)
	}

	if !err.IsRetryable {
		t.Error("429 should be retryable")
	}

	if err.Error() != "api error 429: rate limited" {
		t.Errorf("Error() = %q", err.Error())
	}

	err500 := ai.NewAPIError(500, "internal error")
	if !err500.IsRetryable {
		t.Error("500 should be retryable")
	}

	err400 := ai.NewAPIError(400, "bad request")
	if err400.IsRetryable {
		t.Error("400 should not be retryable")
	}
}

// Verify mock implements model.Model in <15 lines (per verification criteria).
var _ model.Model = (*mockModel)(nil)
var _ model.EmbeddingModel = (*mockEmbeddingModel)(nil)

// Suppress unused import warning for iter.
var _ iter.Seq[string]
