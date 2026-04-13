package openai_test

import (
	"context"
	"encoding/base64"
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

		if req["model"] != "gpt-5.4" {
			t.Errorf("model = %v, want gpt-5.4", req["model"])
		}

		resp := `{
			"id": "chatcmpl-test",
			"choices": [{
				"index": 0,
				"message": {"role": "assistant", "content": "Hello!"},
				"finish_reason": "stop"
			}],
			"usage": {"prompt_tokens": 5, "completion_tokens": 3},
			"metadata": {"trace_id": "trace-openai"}
		}`
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(resp))
	}))
	defer server.Close()

	m := openai.Chat("gpt-5.4",
		openai.WithAPIKey("test-key"),
		openai.WithBaseURL(server.URL),
	)

	if m.Provider() != "openai" {
		t.Errorf("Provider() = %q, want openai", m.Provider())
	}

	if m.ModelID() != "gpt-5.4" {
		t.Errorf("ModelID() = %q, want gpt-5.4", m.ModelID())
	}

	resp, err := m.DoGenerate(context.Background(), &model.Request{
		Messages: []model.Message{
			{
				Role:  model.RoleUser,
				Parts: []model.Part{model.TextPart{Text: "Hi"}},
			},
		},
		ProviderOptions: model.ProviderOptions{
			"openai": {
				"metadata": json.RawMessage(`{"trace_id":"trace-openai"}`),
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

	if resp.RawFinishReason != "stop" {
		t.Errorf("raw finish reason = %q, want stop", resp.RawFinishReason)
	}

	if resp.Metadata.RequestID != "chatcmpl-test" {
		t.Errorf("Metadata.RequestID = %q, want chatcmpl-test", resp.Metadata.RequestID)
	}

	if resp.Metadata.Provider["trace_id"] != "trace-openai" {
		t.Errorf("Metadata.Provider[trace_id] = %q, want trace-openai", resp.Metadata.Provider["trace_id"])
	}
}

func TestChatModelDoStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		lines := []string{
			`data: {"id":"1","model":"gpt-5.4","metadata":{"trace_id":"trace-stream"},"choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"}}]}`,
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

	m := openai.Chat("gpt-5.4",
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
	var metadata model.ProviderMetadata
	for chunk := range streamResp.Stream {
		switch c := chunk.(type) {
		case model.TextDelta:
			text += c.Text
		case model.ProviderMetadataChunk:
			metadata = c.Metadata
		}
	}

	if text != "Hi there" {
		t.Errorf("streamed text = %q, want %q", text, "Hi there")
	}

	if metadata.RequestID != "1" {
		t.Errorf("stream metadata RequestID = %q, want 1", metadata.RequestID)
	}

	if metadata.ModelID != "gpt-5.4" {
		t.Errorf("stream metadata ModelID = %q, want gpt-5.4", metadata.ModelID)
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

	m := p.Chat("gpt-5.4")

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

	m := p.Chat("gpt-5.4")
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

	m := openai.Chat("gpt-5.4",
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

func TestImageModelEditUsesImageFieldAndOutputFormatMIME(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/images/edits" {
			t.Fatalf("path = %q, want /images/edits", r.URL.Path)
		}

		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatalf("ParseMultipartForm: %v", err)
		}

		if len(r.MultipartForm.File["image"]) != 1 {
			t.Fatalf("image files = %d, want 1", len(r.MultipartForm.File["image"]))
		}

		if len(r.MultipartForm.File["image[]"]) != 0 {
			t.Fatalf("image[] files = %d, want 0", len(r.MultipartForm.File["image[]"]))
		}

		if got := r.MultipartForm.Value["output_format"]; len(got) != 1 || got[0] != "jpeg" {
			t.Fatalf("output_format = %v, want [jpeg]", got)
		}

		payload := map[string]any{
			"data": []map[string]string{
				{
					"b64_json": base64.StdEncoding.EncodeToString([]byte("jpeg-bytes")),
				},
			},
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(payload); err != nil {
			t.Fatalf("Encode: %v", err)
		}
	}))
	defer server.Close()

	m := openai.Image("gpt-image-1",
		openai.WithAPIKey("test-key"),
		openai.WithBaseURL(server.URL),
	)

	resp, err := m.DoGenerateImage(context.Background(), &model.ImageRequest{
		Prompt:       "Edit this",
		OutputFormat: "jpeg",
		Images: []model.ImageInput{
			{
				Data:     []byte("png-data"),
				MIMEType: "image/png",
			},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerateImage: %v", err)
	}

	if len(resp.Images) != 1 {
		t.Fatalf("images length = %d, want 1", len(resp.Images))
	}

	if resp.Images[0].MIMEType != "image/jpeg" {
		t.Fatalf("MIMEType = %q, want image/jpeg", resp.Images[0].MIMEType)
	}

	if string(resp.Images[0].Data) != "jpeg-bytes" {
		t.Fatalf("image bytes = %q, want jpeg-bytes", string(resp.Images[0].Data))
	}
}

func TestImageModelGeneratePassesProviderOptions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/images/generations" {
			t.Fatalf("path = %q, want /images/generations", r.URL.Path)
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("ReadAll: %v", err)
		}

		var req map[string]any
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("Unmarshal: %v", err)
		}

		if req["user"] != "user-123" {
			t.Fatalf("user = %v, want user-123", req["user"])
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"url":"https://example.com/image.png"}]}`))
	}))
	defer server.Close()

	m := openai.Image("gpt-image-1",
		openai.WithAPIKey("test-key"),
		openai.WithBaseURL(server.URL),
	)

	resp, err := m.DoGenerateImage(context.Background(), &model.ImageRequest{
		Prompt: "Draw a cat",
		ProviderOptions: model.ProviderOptions{
			"openai": {
				"user": json.RawMessage(`"user-123"`),
			},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerateImage: %v", err)
	}

	if len(resp.Images) != 1 || resp.Images[0].URL != "https://example.com/image.png" {
		t.Fatalf("images = %#v, want generated image URL", resp.Images)
	}
}

func TestTranscriptionModelSupportsTextFormats(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/audio/transcriptions" {
			t.Fatalf("path = %q, want /audio/transcriptions", r.URL.Path)
		}

		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatalf("ParseMultipartForm: %v", err)
		}

		if got := r.MultipartForm.Value["response_format"]; len(got) != 1 || got[0] != "vtt" {
			t.Fatalf("response_format = %v, want [vtt]", got)
		}

		w.Header().Set("Content-Type", "text/vtt")
		_, _ = w.Write([]byte("WEBVTT\n\n00:00.000 --> 00:01.000\nHello\n"))
	}))
	defer server.Close()

	m := openai.Transcription("gpt-4o-mini-transcribe",
		openai.WithAPIKey("test-key"),
		openai.WithBaseURL(server.URL),
	)

	resp, err := m.DoTranscribe(context.Background(), &model.TranscriptionRequest{
		ResponseFormat: "vtt",
		Audio: model.AudioInput{
			Data:     []byte("audio-bytes"),
			Filename: "sample.wav",
		},
	})
	if err != nil {
		t.Fatalf("DoTranscribe: %v", err)
	}

	if resp.Text != "WEBVTT\n\n00:00.000 --> 00:01.000\nHello\n" {
		t.Fatalf("resp.Text = %q, want raw VTT body", resp.Text)
	}
}

func TestTranscriptionModelPassesProviderOptions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatalf("ParseMultipartForm: %v", err)
		}

		got := r.MultipartForm.Value["timestamp_granularities[]"]
		if len(got) != 2 || got[0] != "segment" || got[1] != "word" {
			t.Fatalf("timestamp_granularities[] = %v, want [segment word]", got)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"text":"Hello"}`))
	}))
	defer server.Close()

	m := openai.Transcription("gpt-4o-mini-transcribe",
		openai.WithAPIKey("test-key"),
		openai.WithBaseURL(server.URL),
	)

	resp, err := m.DoTranscribe(context.Background(), &model.TranscriptionRequest{
		Audio: model.AudioInput{
			Data:     []byte("audio-bytes"),
			Filename: "sample.wav",
		},
		ProviderOptions: model.ProviderOptions{
			"openai": {
				"timestamp_granularities[]": json.RawMessage(`["segment","word"]`),
			},
		},
	})
	if err != nil {
		t.Fatalf("DoTranscribe: %v", err)
	}

	if resp.Text != "Hello" {
		t.Fatalf("resp.Text = %q, want Hello", resp.Text)
	}
}

func TestTranscriptionModelVerboseJSONSupportsSpeakerSegments(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatalf("ParseMultipartForm: %v", err)
		}

		if got := r.MultipartForm.Value["response_format"]; len(got) != 1 || got[0] != "verbose_json" {
			t.Fatalf("response_format = %v, want [verbose_json]", got)
		}

		payload := map[string]any{
			"text":     "Hello there",
			"language": "en",
			"duration": 1.25,
			"segments": []map[string]any{
				{
					"id":      "segment_1",
					"start":   0.0,
					"end":     1.25,
					"text":    "Hello there",
					"speaker": "A",
				},
			},
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(payload); err != nil {
			t.Fatalf("Encode: %v", err)
		}
	}))
	defer server.Close()

	m := openai.Transcription("gpt-4o-mini-transcribe",
		openai.WithAPIKey("test-key"),
		openai.WithBaseURL(server.URL),
	)

	resp, err := m.DoTranscribe(context.Background(), &model.TranscriptionRequest{
		ResponseFormat: "verbose_json",
		Audio: model.AudioInput{
			Data:     []byte("audio-bytes"),
			Filename: "sample.wav",
		},
	})
	if err != nil {
		t.Fatalf("DoTranscribe: %v", err)
	}

	if resp.Text != "Hello there" {
		t.Fatalf("resp.Text = %q, want Hello there", resp.Text)
	}

	if len(resp.Segments) != 1 {
		t.Fatalf("segments length = %d, want 1", len(resp.Segments))
	}

	if resp.Segments[0].ID != "segment_1" {
		t.Fatalf("segment ID = %q, want segment_1", resp.Segments[0].ID)
	}

	if resp.Segments[0].Speaker != "A" {
		t.Fatalf("speaker = %q, want A", resp.Segments[0].Speaker)
	}
}
