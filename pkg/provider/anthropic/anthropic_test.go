package anthropic_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider"
	"github.com/aarock1234/ai/pkg/provider/anthropic"
)

func TestChatModelDoGenerate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/messages" {
			t.Errorf("path = %q, want /messages", r.URL.Path)
		}

		if r.Header.Get("x-api-key") != "test-key" {
			t.Errorf("x-api-key = %q, want test-key", r.Header.Get("x-api-key"))
		}

		if r.Header.Get("anthropic-version") != "2023-06-01" {
			t.Errorf("anthropic-version = %q, want 2023-06-01", r.Header.Get("anthropic-version"))
		}

		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("unmarshal request: %v", err)
		}

		if req["model"] != "claude-sonnet-4-5" {
			t.Errorf("model = %v, want claude-sonnet-4-5", req["model"])
		}

		resp := `{
			"id": "msg_123",
			"type": "message",
			"role": "assistant",
			"content": [
				{"type": "text", "text": "Hello from Claude!"}
			],
			"model": "claude-sonnet-4-5",
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 7, "output_tokens": 4}
		}`
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(resp))
	}))
	defer server.Close()

	m := anthropic.Chat(
		"claude-sonnet-4-5",
		anthropic.WithAPIKey("test-key"),
		anthropic.WithBaseURL(server.URL),
	)

	if m.Provider() != "anthropic" {
		t.Errorf("Provider() = %q, want anthropic", m.Provider())
	}

	if m.ModelID() != "claude-sonnet-4-5" {
		t.Errorf("ModelID() = %q, want claude-sonnet-4-5", m.ModelID())
	}

	resp, err := m.DoGenerate(context.Background(), &model.Request{
		Messages: []model.Message{
			{
				Role: model.RoleUser,
				Parts: []model.Part{
					model.TextPart{Text: "Hi"},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}

	if len(resp.Content) != 1 {
		t.Fatalf("content length = %d, want 1", len(resp.Content))
	}

	text, ok := resp.Content[0].(model.TextContent)
	if !ok {
		t.Fatalf("content[0] type = %T, want TextContent", resp.Content[0])
	}

	if text.Text != "Hello from Claude!" {
		t.Errorf("text = %q, want Hello from Claude!", text.Text)
	}

	if resp.FinishReason != model.FinishReasonStop {
		t.Errorf("finish reason = %v, want stop", resp.FinishReason)
	}

	if resp.Usage.InputTokens != 7 {
		t.Errorf("input tokens = %d, want 7", resp.Usage.InputTokens)
	}

	if resp.Usage.OutputTokens != 4 {
		t.Errorf("output tokens = %d, want 4", resp.Usage.OutputTokens)
	}
}

func TestChatModelDoStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected http.Flusher")
		}

		events := []string{
			"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-5\",\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\n",
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" there\"}}\n\n",
			"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":5,\"output_tokens\":2}}\n\n",
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
		}

		for _, event := range events {
			_, _ = w.Write([]byte(event))
			flusher.Flush()
		}
	}))
	defer server.Close()

	m := anthropic.Chat(
		"claude-sonnet-4-5",
		anthropic.WithAPIKey("test-key"),
		anthropic.WithBaseURL(server.URL),
	)

	streamResp, err := m.DoStream(context.Background(), &model.Request{
		Messages: []model.Message{
			{
				Role: model.RoleUser,
				Parts: []model.Part{
					model.TextPart{Text: "Hello"},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("DoStream: %v", err)
	}

	var text string
	var finishReason model.FinishReason
	var usage model.Usage
	for chunk := range streamResp.Stream {
		switch c := chunk.(type) {
		case model.TextDelta:
			text += c.Text
		case model.FinishChunk:
			finishReason = c.FinishReason
		case model.UsageChunk:
			usage = c.Usage
		}
	}

	if text != "Hi there" {
		t.Errorf("streamed text = %q, want Hi there", text)
	}

	if finishReason != model.FinishReasonStop {
		t.Errorf("finish reason = %v, want stop", finishReason)
	}

	if usage.InputTokens != 5 {
		t.Errorf("input tokens = %d, want 5", usage.InputTokens)
	}

	if usage.OutputTokens != 2 {
		t.Errorf("output tokens = %d, want 2", usage.OutputTokens)
	}
}

func TestProviderFactoryAndRegistry(t *testing.T) {
	p := anthropic.New(
		anthropic.WithAPIKey("test-key"),
		anthropic.WithBaseURL("https://example.com/v1"),
	)

	var iface provider.Provider = p

	m := iface.Chat("claude-sonnet-4-5")
	if m == nil {
		t.Fatal("Chat returned nil")
	}

	if m.Provider() != "anthropic" {
		t.Errorf("Provider() = %q, want anthropic", m.Provider())
	}

	if iface.Embedding("voyage-3") != nil {
		t.Fatal("Embedding() should return nil for anthropic")
	}

	r := provider.NewRegistry()
	r.Register("anthropic", p)

	got, ok := r.Get("anthropic")
	if !ok {
		t.Fatal("registry get returned false")
	}

	if got != p {
		t.Fatal("registry returned different provider instance")
	}
}
