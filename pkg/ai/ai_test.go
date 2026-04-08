package ai_test

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"testing"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// mockModel is a minimal model.Model implementation for testing.
type mockModel struct {
	generateFn func(ctx context.Context, req *model.Request) (*model.Response, error)
	streamFn   func(ctx context.Context, req *model.Request) (*model.StreamResponse, error)
}

func (m *mockModel) Provider() string { return "mock" }
func (m *mockModel) ModelID() string  { return "mock-model" }

func (m *mockModel) DoGenerate(ctx context.Context, req *model.Request) (*model.Response, error) {
	return m.generateFn(ctx, req)
}

func (m *mockModel) DoStream(ctx context.Context, req *model.Request) (*model.StreamResponse, error) {
	return m.streamFn(ctx, req)
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
			}, nil
		},
	}

	result, err := ai.GenerateText(context.Background(), mock, ai.WithPrompt("Hi"))
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
				if !yield(model.TextDelta{Text: "Hello"}) {
					return
				}
				if !yield(model.TextDelta{Text: " world"}) {
					return
				}
				if !yield(model.FinishChunk{FinishReason: model.FinishReasonStop}) {
					return
				}
				yield(model.UsageChunk{Usage: model.Usage{InputTokens: 5, OutputTokens: 2}})
			}

			return &model.StreamResponse{Stream: stream}, nil
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

	if result.Usage.InputTokens != 5 {
		t.Errorf("InputTokens = %d, want 5", result.Usage.InputTokens)
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

			return &model.StreamResponse{Stream: stream}, nil
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

			return &model.StreamResponse{Stream: stream}, nil
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

func TestGenerateObjectBasic(t *testing.T) {
	type Recipe struct {
		Name        string   `json:"name"`
		Ingredients []string `json:"ingredients"`
	}

	mock := &mockModel{
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
}

func TestGenerateObjectInvalidJSON(t *testing.T) {
	type Item struct {
		Name string `json:"name"`
	}

	mock := &mockModel{
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
