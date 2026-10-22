package google_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aarock1234/ai/internal/apierror"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider"
	"github.com/aarock1234/ai/pkg/provider/google"
)

func TestChatModelDoGenerate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models/gemini-3-flash-preview:generateContent" {
			t.Errorf("path = %q, want /models/gemini-3-flash-preview:generateContent", r.URL.Path)
		}

		if r.Header.Get("x-goog-api-key") != "test-key" {
			t.Errorf("x-goog-api-key = %q, want test-key", r.Header.Get("x-goog-api-key"))
		}

		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("unmarshal request: %v", err)
		}

		if _, ok := req["contents"]; !ok {
			t.Error("request body missing contents")
		}

		resp := `{
			"candidates": [
				{
					"content": {"role": "model", "parts": [{"text": "Hello from Gemini!"}]},
					"finishReason": "STOP"
				}
			],
			"usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 4, "totalTokenCount": 11},
			"modelVersion": "gemini-3-flash-preview",
			"responseId": "resp_123"
		}`
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(resp))
	}))
	defer server.Close()

	m := google.Chat(
		"gemini-3-flash-preview",
		google.WithAPIKey("test-key"),
		google.WithBaseURL(server.URL),
	)

	if m.Provider() != "google" {
		t.Errorf("Provider() = %q, want google", m.Provider())
	}

	if m.ModelID() != "gemini-3-flash-preview" {
		t.Errorf("ModelID() = %q, want gemini-3-flash-preview", m.ModelID())
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

	if text.Text != "Hello from Gemini!" {
		t.Errorf("text = %q, want Hello from Gemini!", text.Text)
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

	if resp.Metadata.RequestID != "resp_123" {
		t.Errorf("request id = %q, want resp_123", resp.Metadata.RequestID)
	}
}

func TestChatModelDoStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models/gemini-3-flash-preview:streamGenerateContent" {
			t.Errorf("path = %q, want /models/gemini-3-flash-preview:streamGenerateContent", r.URL.Path)
		}

		if r.URL.Query().Get("alt") != "sse" {
			t.Errorf("alt = %q, want sse", r.URL.Query().Get("alt"))
		}

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected http.Flusher")
		}

		events := []string{
			"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hi\"}]}}],\"usageMetadata\":{\"promptTokenCount\":5,\"candidatesTokenCount\":1,\"totalTokenCount\":6},\"modelVersion\":\"gemini-3-flash-preview\",\"responseId\":\"resp_1\"}\n\n",
			"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\" there\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":5,\"candidatesTokenCount\":2,\"totalTokenCount\":7},\"modelVersion\":\"gemini-3-flash-preview\",\"responseId\":\"resp_1\"}\n\n",
		}

		for _, event := range events {
			_, _ = w.Write([]byte(event))
			flusher.Flush()
		}
	}))
	defer server.Close()

	m := google.Chat(
		"gemini-3-flash-preview",
		google.WithAPIKey("test-key"),
		google.WithBaseURL(server.URL),
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
	var usageChunks []model.Usage
	for chunk := range streamResp.Stream {
		switch c := chunk.(type) {
		case model.TextDelta:
			text += c.Text
		case model.FinishChunk:
			finishReason = c.FinishReason
		case model.UsageChunk:
			usageChunks = append(usageChunks, c.Usage)
		}
	}

	if text != "Hi there" {
		t.Errorf("streamed text = %q, want Hi there", text)
	}

	if finishReason != model.FinishReasonStop {
		t.Errorf("finish reason = %v, want stop", finishReason)
	}

	if len(usageChunks) != 1 {
		t.Fatalf("usage chunks = %d, want 1 (cumulative usage must be emitted once)", len(usageChunks))
	}

	if usageChunks[0].InputTokens != 5 {
		t.Errorf("input tokens = %d, want 5", usageChunks[0].InputTokens)
	}

	if usageChunks[0].OutputTokens != 2 {
		t.Errorf("output tokens = %d, want 2", usageChunks[0].OutputTokens)
	}
}

func TestProviderFactoryAndRegistry(t *testing.T) {
	p := google.New(
		google.WithAPIKey("test-key"),
		google.WithBaseURL("https://example.com/v1beta"),
	)

	var iface provider.Provider = p

	m := iface.Chat("gemini-3-flash-preview")
	if m == nil {
		t.Fatal("Chat returned nil")
	}

	if m.Provider() != "google" {
		t.Errorf("Provider() = %q, want google", m.Provider())
	}

	e := iface.Embedding("gemini-embedding-001")
	if e == nil {
		t.Fatal("Embedding returned nil")
	}

	if e.Provider() != "google" {
		t.Errorf("embedding Provider() = %q, want google", e.Provider())
	}

	r := provider.NewRegistry()
	r.Register("google", p)

	got, ok := r.Get("google")
	if !ok {
		t.Fatal("registry get returned false")
	}

	if got != p {
		t.Fatal("registry returned different provider instance")
	}
}

func TestEmbeddingModelSingle(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models/gemini-embedding-001:embedContent" {
			t.Errorf("path = %q, want /models/gemini-embedding-001:embedContent", r.URL.Path)
		}

		if r.Header.Get("x-goog-api-key") != "test-key" {
			t.Errorf("x-goog-api-key = %q, want test-key", r.Header.Get("x-goog-api-key"))
		}

		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("unmarshal request: %v", err)
		}

		if _, ok := req["content"]; !ok {
			t.Error("request body missing content")
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"embedding":{"values":[0.1,0.2,0.3]}}`))
	}))
	defer server.Close()

	m := google.Embedding(
		"gemini-embedding-001",
		google.WithAPIKey("test-key"),
		google.WithBaseURL(server.URL),
	)

	resp, err := m.DoEmbed(context.Background(), []string{"hello"})
	if err != nil {
		t.Fatalf("DoEmbed: %v", err)
	}

	if len(resp.Embeddings) != 1 {
		t.Fatalf("embeddings length = %d, want 1", len(resp.Embeddings))
	}

	if len(resp.Embeddings[0]) != 3 || resp.Embeddings[0][0] != 0.1 {
		t.Errorf("embedding = %v, want [0.1 0.2 0.3]", resp.Embeddings[0])
	}

	if resp.Metadata.ModelID != "gemini-embedding-001" {
		t.Errorf("model id = %q, want gemini-embedding-001", resp.Metadata.ModelID)
	}
}

func TestEmbeddingModelBatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models/gemini-embedding-001:batchEmbedContents" {
			t.Errorf("path = %q, want /models/gemini-embedding-001:batchEmbedContents", r.URL.Path)
		}

		body, _ := io.ReadAll(r.Body)
		var req struct {
			Requests []struct {
				Model string `json:"model"`
			} `json:"requests"`
		}
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("unmarshal request: %v", err)
		}

		if len(req.Requests) != 2 {
			t.Fatalf("requests length = %d, want 2", len(req.Requests))
		}

		if req.Requests[0].Model != "models/gemini-embedding-001" {
			t.Errorf("request model = %q, want models/gemini-embedding-001", req.Requests[0].Model)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"embeddings":[{"values":[0.1,0.2]},{"values":[0.3,0.4]}]}`))
	}))
	defer server.Close()

	m := google.Embedding(
		"gemini-embedding-001",
		google.WithAPIKey("test-key"),
		google.WithBaseURL(server.URL),
	)

	resp, err := m.DoEmbed(context.Background(), []string{"one", "two"})
	if err != nil {
		t.Fatalf("DoEmbed: %v", err)
	}

	if len(resp.Embeddings) != 2 {
		t.Fatalf("embeddings length = %d, want 2", len(resp.Embeddings))
	}

	if resp.Embeddings[1][1] != 0.4 {
		t.Errorf("embeddings[1][1] = %v, want 0.4", resp.Embeddings[1][1])
	}
}

func TestErrorResponsesReturnAPIError(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		body       string
		wantMsg    string
	}{
		{
			name:       "gemini error envelope",
			statusCode: http.StatusTooManyRequests,
			body:       `{"error":{"code":429,"message":"quota exceeded","status":"RESOURCE_EXHAUSTED"}}`,
			wantMsg:    "quota exceeded",
		},
		{
			name:       "plain body fallback",
			statusCode: http.StatusInternalServerError,
			body:       "internal error",
			wantMsg:    "internal error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tt.statusCode)
				_, _ = w.Write([]byte(tt.body))
			}))
			defer server.Close()

			m := google.Chat(
				"gemini-3-flash-preview",
				google.WithAPIKey("test-key"),
				google.WithBaseURL(server.URL),
			)

			_, err := m.DoGenerate(context.Background(), &model.Request{
				Messages: []model.Message{
					{
						Role:  model.RoleUser,
						Parts: []model.Part{model.TextPart{Text: "Hi"}},
					},
				},
			})
			if err == nil {
				t.Fatal("expected error")
			}

			var apiErr *apierror.APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("error type = %T, want *apierror.APIError", err)
			}

			if apiErr.StatusCode != tt.statusCode {
				t.Errorf("status code = %d, want %d", apiErr.StatusCode, tt.statusCode)
			}

			if apiErr.Message != tt.wantMsg {
				t.Errorf("message = %q, want %q", apiErr.Message, tt.wantMsg)
			}
		})
	}
}
