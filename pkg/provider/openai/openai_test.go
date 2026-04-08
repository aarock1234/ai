package openai_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

func TestChatModelDoGenerate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}

		if req["model"] != "gpt-4o" {
			t.Errorf("model = %v, want gpt-4o", req["model"])
		}

		resp := `{
			"id": "chatcmpl-test",
			"choices": [{
				"index": 0,
				"message": {"role": "assistant", "content": "Hello!"},
				"finish_reason": "stop"
			}],
			"usage": {"prompt_tokens": 5, "completion_tokens": 3}
		}`
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(resp))
	}))
	defer server.Close()

	m := openai.Chat("gpt-4o",
		openai.WithAPIKey("test-key"),
		openai.WithBaseURL(server.URL),
	)

	if m.Provider() != "openai" {
		t.Errorf("Provider() = %q, want openai", m.Provider())
	}

	if m.ModelID() != "gpt-4o" {
		t.Errorf("ModelID() = %q, want gpt-4o", m.ModelID())
	}

	resp, err := m.DoGenerate(context.Background(), &model.Request{
		Messages: []model.Message{
			{
				Role:  model.RoleUser,
				Parts: []model.Part{model.TextPart{Text: "Hi"}},
			},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}

	if len(resp.Content) != 1 {
		t.Fatalf("content length = %d, want 1", len(resp.Content))
	}

	tc, ok := resp.Content[0].(model.TextContent)
	if !ok {
		t.Fatal("content[0] not TextContent")
	}

	if tc.Text != "Hello!" {
		t.Errorf("text = %q, want Hello!", tc.Text)
	}

	if resp.FinishReason != model.FinishReasonStop {
		t.Errorf("finish reason = %v, want stop", resp.FinishReason)
	}
}

func TestChatModelDoStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		lines := []string{
			`data: {"id":"1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"}}]}`,
			``,
			`data: {"id":"1","choices":[{"index":0,"delta":{"content":" there"}}]}`,
			``,
			`data: {"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			``,
			`data: [DONE]`,
			``,
		}

		for _, line := range lines {
			_, _ = w.Write([]byte(line + "\n"))
			flusher.Flush()
		}
	}))
	defer server.Close()

	m := openai.Chat("gpt-4o",
		openai.WithAPIKey("test-key"),
		openai.WithBaseURL(server.URL),
	)

	streamResp, err := m.DoStream(context.Background(), &model.Request{
		Messages: []model.Message{
			{
				Role:  model.RoleUser,
				Parts: []model.Part{model.TextPart{Text: "Hi"}},
			},
		},
	})
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}

	var text string
	for chunk := range streamResp.Stream {
		if td, ok := chunk.(model.TextDelta); ok {
			text += td.Text
		}
	}

	if text != "Hi there" {
		t.Errorf("streamed text = %q, want %q", text, "Hi there")
	}
}

func TestEmbeddingModelDoEmbed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}

		if req["model"] != "text-embedding-3-small" {
			t.Errorf("model = %v, want text-embedding-3-small", req["model"])
		}

		resp := `{
			"data": [
				{"embedding": [0.1, 0.2, 0.3], "index": 0},
				{"embedding": [0.4, 0.5, 0.6], "index": 1}
			],
			"usage": {"prompt_tokens": 5, "total_tokens": 5}
		}`
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(resp))
	}))
	defer server.Close()

	m := openai.Embedding("text-embedding-3-small",
		openai.WithAPIKey("test-key"),
		openai.WithBaseURL(server.URL),
	)

	if m.Provider() != "openai" {
		t.Errorf("Provider() = %q, want openai", m.Provider())
	}

	if m.ModelID() != "text-embedding-3-small" {
		t.Errorf("ModelID() = %q, want text-embedding-3-small", m.ModelID())
	}

	resp, err := m.DoEmbed(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("DoEmbed: %v", err)
	}

	if len(resp.Embeddings) != 2 {
		t.Fatalf("embeddings length = %d, want 2", len(resp.Embeddings))
	}

	if resp.Embeddings[0][0] != 0.1 {
		t.Errorf("embedding[0][0] = %f, want 0.1", resp.Embeddings[0][0])
	}

	if resp.Embeddings[1][0] != 0.4 {
		t.Errorf("embedding[1][0] = %f, want 0.4", resp.Embeddings[1][0])
	}

	if resp.Usage.InputTokens != 5 {
		t.Errorf("input tokens = %d, want 5", resp.Usage.InputTokens)
	}
}

func TestProviderFactoryChat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		resp := `{"id":"1","choices":[{"index":0,"message":{"role":"assistant","content":"from provider"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(resp))
	}))
	defer server.Close()

	p := openai.New(
		openai.WithAPIKey("test-key"),
		openai.WithBaseURL(server.URL),
	)

	m := p.Chat("gpt-4o")

	if m.Provider() != "openai" {
		t.Errorf("Provider() = %q, want openai", m.Provider())
	}

	resp, err := m.DoGenerate(context.Background(), &model.Request{
		Messages: []model.Message{
			{
				Role:  model.RoleUser,
				Parts: []model.Part{model.TextPart{Text: "Hi"}},
			},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}

	tc := resp.Content[0].(model.TextContent)
	if tc.Text != "from provider" {
		t.Errorf("text = %q, want %q", tc.Text, "from provider")
	}
}

func TestProviderFactoryEmbedding(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		resp := `{"data":[{"embedding":[0.1],"index":0}],"usage":{"prompt_tokens":1,"total_tokens":1}}`
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(resp))
	}))
	defer server.Close()

	p := openai.New(
		openai.WithAPIKey("test-key"),
		openai.WithBaseURL(server.URL),
	)

	em := p.Embedding("text-embedding-3-small")

	resp, err := em.DoEmbed(context.Background(), []string{"hello"})
	if err != nil {
		t.Fatalf("DoEmbed: %v", err)
	}

	if len(resp.Embeddings) != 1 {
		t.Fatalf("embeddings length = %d, want 1", len(resp.Embeddings))
	}
}

func TestProviderImplementsInterface(t *testing.T) {
	var p provider.Provider = openai.New(openai.WithAPIKey("test"))

	m := p.Chat("gpt-4o")
	if m == nil {
		t.Fatal("Chat returned nil")
	}

	em := p.Embedding("text-embedding-3-small")
	if em == nil {
		t.Fatal("Embedding returned nil")
	}
}

func TestOrganizationHeader(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("OpenAI-Organization") != "org-123" {
			t.Errorf("OpenAI-Organization = %q, want org-123", r.Header.Get("OpenAI-Organization"))
		}

		resp := `{"id":"1","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(resp))
	}))
	defer server.Close()

	m := openai.Chat("gpt-4o",
		openai.WithAPIKey("test-key"),
		openai.WithBaseURL(server.URL),
		openai.WithOrganization("org-123"),
	)

	_, err := m.DoGenerate(context.Background(), &model.Request{
		Messages: []model.Message{
			{
				Role:  model.RoleUser,
				Parts: []model.Part{model.TextPart{Text: "Hi"}},
			},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
}
