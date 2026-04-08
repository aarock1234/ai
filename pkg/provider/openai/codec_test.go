package openai_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

func TestEncodeRequestBasic(t *testing.T) {
	req := &model.Request{
		Messages: []model.Message{
			{
				Role:  model.RoleUser,
				Parts: []model.Part{model.TextPart{Text: "Hello"}},
			},
		},
	}

	chatReq, err := openai.EncodeRequest("gpt-5.4", req, false)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	body, err := openai.MarshalRequest(chatReq)
	if err != nil {
		t.Fatalf("MarshalRequest: %v", err)
	}

	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if raw["model"] != "gpt-5.4" {
		t.Errorf("model = %v, want gpt-5.4", raw["model"])
	}

	msgs, ok := raw["messages"].([]any)
	if !ok || len(msgs) != 1 {
		t.Fatalf("messages: got %v, want 1 message", raw["messages"])
	}

	msg := msgs[0].(map[string]any)
	if msg["role"] != "user" {
		t.Errorf("role = %v, want user", msg["role"])
	}

	if msg["content"] != "Hello" {
		t.Errorf("content = %v, want Hello", msg["content"])
	}
}

func TestEncodeRequestWithTools(t *testing.T) {
	req := &model.Request{
		Messages: []model.Message{
			{
				Role:  model.RoleUser,
				Parts: []model.Part{model.TextPart{Text: "What's the weather?"}},
			},
		},
		Tools: []model.ToolDefinition{
			{
				Name:        "get_weather",
				Description: "Get current weather",
				InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
			},
		},
	}

	chatReq, err := openai.EncodeRequest("gpt-5.4", req, false)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	body, err := openai.MarshalRequest(chatReq)
	if err != nil {
		t.Fatalf("MarshalRequest: %v", err)
	}

	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	tools, ok := raw["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools: got %v, want 1 tool", raw["tools"])
	}

	tool := tools[0].(map[string]any)
	if tool["type"] != "function" {
		t.Errorf("tool type = %v, want function", tool["type"])
	}

	fn := tool["function"].(map[string]any)
	if fn["name"] != "get_weather" {
		t.Errorf("function name = %v, want get_weather", fn["name"])
	}
}

func TestEncodeRequestStreaming(t *testing.T) {
	req := &model.Request{
		Messages: []model.Message{
			{
				Role:  model.RoleUser,
				Parts: []model.Part{model.TextPart{Text: "Hi"}},
			},
		},
	}

	chatReq, err := openai.EncodeRequest("gpt-5.4", req, true)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	body, err := openai.MarshalRequest(chatReq)
	if err != nil {
		t.Fatalf("MarshalRequest: %v", err)
	}

	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if raw["stream"] != true {
		t.Errorf("stream = %v, want true", raw["stream"])
	}

	streamOpts, ok := raw["stream_options"].(map[string]any)
	if !ok {
		t.Fatal("stream_options missing")
	}

	if streamOpts["include_usage"] != true {
		t.Errorf("include_usage = %v, want true", streamOpts["include_usage"])
	}
}

func TestEncodeRequestWithTemperature(t *testing.T) {
	temp := 0.7
	maxTokens := 100
	req := &model.Request{
		Messages: []model.Message{
			{
				Role:  model.RoleUser,
				Parts: []model.Part{model.TextPart{Text: "Hi"}},
			},
		},
		Temperature: &temp,
		MaxTokens:   &maxTokens,
	}

	chatReq, err := openai.EncodeRequest("gpt-5.4", req, false)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	body, err := openai.MarshalRequest(chatReq)
	if err != nil {
		t.Fatalf("MarshalRequest: %v", err)
	}

	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if raw["temperature"] != 0.7 {
		t.Errorf("temperature = %v, want 0.7", raw["temperature"])
	}

	if raw["max_completion_tokens"] != float64(100) {
		t.Errorf("max_completion_tokens = %v, want 100", raw["max_completion_tokens"])
	}
}

func TestEncodeRequestWithResponseFormat(t *testing.T) {
	req := &model.Request{
		Messages: []model.Message{
			{
				Role:  model.RoleUser,
				Parts: []model.Part{model.TextPart{Text: "Hi"}},
			},
		},
		ResponseFormat: &model.ResponseFormat{
			Type: "json_schema",
			JSONSchema: &model.JSONSchema{
				Name:   "response",
				Schema: json.RawMessage(`{"type":"object"}`),
				Strict: true,
			},
		},
	}

	chatReq, err := openai.EncodeRequest("gpt-5.4", req, false)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	body, err := openai.MarshalRequest(chatReq)
	if err != nil {
		t.Fatalf("MarshalRequest: %v", err)
	}

	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	rf, ok := raw["response_format"].(map[string]any)
	if !ok {
		t.Fatal("response_format missing")
	}

	if rf["type"] != "json_schema" {
		t.Errorf("response_format.type = %v, want json_schema", rf["type"])
	}

	js, ok := rf["json_schema"].(map[string]any)
	if !ok {
		t.Fatal("json_schema missing")
	}

	if js["name"] != "response" {
		t.Errorf("json_schema.name = %v, want response", js["name"])
	}

	if js["strict"] != true {
		t.Errorf("json_schema.strict = %v, want true", js["strict"])
	}
}

func TestEncodeRequestToolCallMessage(t *testing.T) {
	req := &model.Request{
		Messages: []model.Message{
			{
				Role:  model.RoleUser,
				Parts: []model.Part{model.TextPart{Text: "Hello"}},
			},
			{
				Role: model.RoleAssistant,
				Parts: []model.Part{
					model.ToolCallPart{
						ToolCallID: "call_1",
						ToolName:   "get_weather",
						Args:       json.RawMessage(`{"city":"NYC"}`),
					},
				},
			},
			{
				Role: model.RoleTool,
				Parts: []model.Part{
					model.ToolResultPart{
						ToolCallID: "call_1",
						ToolName:   "get_weather",
						Content:    "Sunny, 72°F",
					},
				},
			},
		},
	}

	chatReq, err := openai.EncodeRequest("gpt-5.4", req, false)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	body, err := openai.MarshalRequest(chatReq)
	if err != nil {
		t.Fatalf("MarshalRequest: %v", err)
	}

	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	msgs, ok := raw["messages"].([]any)
	if !ok {
		t.Fatal("messages not a slice")
	}

	if len(msgs) != 3 {
		t.Fatalf("messages length = %d, want 3", len(msgs))
	}

	// Assistant message with tool calls.
	assistantMsg := msgs[1].(map[string]any)
	if assistantMsg["role"] != "assistant" {
		t.Errorf("assistant role = %v, want assistant", assistantMsg["role"])
	}

	toolCalls, ok := assistantMsg["tool_calls"].([]any)
	if !ok || len(toolCalls) != 1 {
		t.Fatalf("tool_calls: got %v, want 1", assistantMsg["tool_calls"])
	}

	tc := toolCalls[0].(map[string]any)
	if tc["id"] != "call_1" {
		t.Errorf("tool_call id = %v, want call_1", tc["id"])
	}

	// Tool result message.
	toolMsg := msgs[2].(map[string]any)
	if toolMsg["role"] != "tool" {
		t.Errorf("tool role = %v, want tool", toolMsg["role"])
	}

	if toolMsg["tool_call_id"] != "call_1" {
		t.Errorf("tool_call_id = %v, want call_1", toolMsg["tool_call_id"])
	}
}

func TestDoRequestMockServer(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("method = %s, want POST", r.Method)
		}

		if !strings.HasSuffix(r.URL.Path, "/chat/completions") {
			t.Errorf("path = %s, want /chat/completions", r.URL.Path)
		}

		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("auth = %s, want Bearer test-key", r.Header.Get("Authorization"))
		}

		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("content-type = %s, want application/json", r.Header.Get("Content-Type"))
		}

		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("unmarshal request: %v", err)
		}

		resp := `{
			"id": "chatcmpl-123",
			"choices": [{
				"index": 0,
				"message": {
					"role": "assistant",
					"content": "Hello there!"
				},
				"finish_reason": "stop"
			}],
			"usage": {
				"prompt_tokens": 10,
				"completion_tokens": 5
			}
		}`
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(resp))
	}))
	defer server.Close()

	cfg := &openai.ClientConfig{
		BaseURL: server.URL,
		APIKey:  "test-key",
	}

	req := &model.Request{
		Messages: []model.Message{
			{
				Role:  model.RoleUser,
				Parts: []model.Part{model.TextPart{Text: "Hello"}},
			},
		},
	}

	chatReq, err := openai.EncodeRequest("gpt-5.4", req, false)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	chatResp, err := openai.DoRequest(context.Background(), cfg, chatReq)
	if err != nil {
		t.Fatalf("DoRequest: %v", err)
	}

	resp := openai.DecodeResponse(chatResp)

	if len(resp.Content) != 1 {
		t.Fatalf("content length = %d, want 1", len(resp.Content))
	}

	textContent, ok := resp.Content[0].(model.TextContent)
	if !ok {
		t.Fatal("content[0] is not TextContent")
	}

	if textContent.Text != "Hello there!" {
		t.Errorf("text = %q, want %q", textContent.Text, "Hello there!")
	}

	if resp.FinishReason != model.FinishReasonStop {
		t.Errorf("finish reason = %v, want stop", resp.FinishReason)
	}

	if resp.Usage.InputTokens != 10 {
		t.Errorf("input tokens = %d, want 10", resp.Usage.InputTokens)
	}

	if resp.Usage.OutputTokens != 5 {
		t.Errorf("output tokens = %d, want 5", resp.Usage.OutputTokens)
	}
}

func TestDoRequestErrorStatus(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error": {"message": "rate limited"}}`))
	}))
	defer server.Close()

	cfg := &openai.ClientConfig{
		BaseURL: server.URL,
		APIKey:  "test-key",
	}

	req := &model.Request{
		Messages: []model.Message{
			{
				Role:  model.RoleUser,
				Parts: []model.Part{model.TextPart{Text: "Hello"}},
			},
		},
	}

	chatReq, err := openai.EncodeRequest("gpt-5.4", req, false)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	_, err = openai.DoRequest(context.Background(), cfg, chatReq)
	if err == nil {
		t.Fatal("expected error for 429 status")
	}
}

func TestDoRequestToolCallResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		resp := `{
			"id": "chatcmpl-456",
			"choices": [{
				"index": 0,
				"message": {
					"role": "assistant",
					"content": null,
					"tool_calls": [{
						"id": "call_abc",
						"type": "function",
						"function": {
							"name": "get_weather",
							"arguments": "{\"city\":\"NYC\"}"
						}
					}]
				},
				"finish_reason": "tool_calls"
			}],
			"usage": {
				"prompt_tokens": 20,
				"completion_tokens": 10
			}
		}`
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(resp))
	}))
	defer server.Close()

	cfg := &openai.ClientConfig{
		BaseURL: server.URL,
		APIKey:  "test-key",
	}

	req := &model.Request{
		Messages: []model.Message{
			{
				Role:  model.RoleUser,
				Parts: []model.Part{model.TextPart{Text: "Weather?"}},
			},
		},
	}

	chatReq, err := openai.EncodeRequest("gpt-5.4", req, false)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	chatResp, err := openai.DoRequest(context.Background(), cfg, chatReq)
	if err != nil {
		t.Fatalf("DoRequest: %v", err)
	}

	resp := openai.DecodeResponse(chatResp)

	if resp.FinishReason != model.FinishReasonToolCalls {
		t.Errorf("finish reason = %v, want tool_calls", resp.FinishReason)
	}

	if len(resp.Content) != 1 {
		t.Fatalf("content length = %d, want 1", len(resp.Content))
	}

	tc, ok := resp.Content[0].(model.ToolCallContent)
	if !ok {
		t.Fatal("content[0] is not ToolCallContent")
	}

	if tc.ToolCallID != "call_abc" {
		t.Errorf("tool call id = %q, want call_abc", tc.ToolCallID)
	}

	if tc.ToolName != "get_weather" {
		t.Errorf("tool name = %q, want get_weather", tc.ToolName)
	}
}

func TestDecodeStreamChunks(t *testing.T) {
	data := `data: {"id":"cmpl-1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}

data: {"id":"cmpl-1","choices":[{"index":0,"delta":{"content":" world"}}]}

data: {"id":"cmpl-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"cmpl-1","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}

data: [DONE]

`
	reader := strings.NewReader(data)
	var chunks []model.Chunk

	for chunk := range openai.DecodeStream(reader) {
		chunks = append(chunks, chunk)
	}

	// Expect: TextDelta("Hello"), TextDelta(" world"), FinishChunk(stop), UsageChunk.
	if len(chunks) != 4 {
		t.Fatalf("chunks count = %d, want 4, got: %v", len(chunks), chunks)
	}

	td1, ok := chunks[0].(model.TextDelta)
	if !ok {
		t.Fatalf("chunk[0] type = %T, want TextDelta", chunks[0])
	}

	if td1.Text != "Hello" {
		t.Errorf("chunk[0] text = %q, want Hello", td1.Text)
	}

	td2, ok := chunks[1].(model.TextDelta)
	if !ok {
		t.Fatalf("chunk[1] type = %T, want TextDelta", chunks[1])
	}

	if td2.Text != " world" {
		t.Errorf("chunk[1] text = %q, want ' world'", td2.Text)
	}

	fc, ok := chunks[2].(model.FinishChunk)
	if !ok {
		t.Fatalf("chunk[2] type = %T, want FinishChunk", chunks[2])
	}

	if fc.FinishReason != model.FinishReasonStop {
		t.Errorf("finish reason = %v, want stop", fc.FinishReason)
	}

	uc, ok := chunks[3].(model.UsageChunk)
	if !ok {
		t.Fatalf("chunk[3] type = %T, want UsageChunk", chunks[3])
	}

	if uc.Usage.InputTokens != 5 {
		t.Errorf("input tokens = %d, want 5", uc.Usage.InputTokens)
	}

	if uc.Usage.OutputTokens != 2 {
		t.Errorf("output tokens = %d, want 2", uc.Usage.OutputTokens)
	}
}

func TestDecodeStreamToolCallDeltas(t *testing.T) {
	data := `data: {"id":"cmpl-1","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}

data: {"id":"cmpl-1","choices":[{"index":0,"delta":{"tool_calls":[{"id":"","type":"function","function":{"name":"","arguments":"{\"city\":"}}]}}]}

data: {"id":"cmpl-1","choices":[{"index":0,"delta":{"tool_calls":[{"id":"","type":"function","function":{"name":"","arguments":"\"NYC\"}"}}]}}]}

data: {"id":"cmpl-1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]

`
	reader := strings.NewReader(data)
	var chunks []model.Chunk

	for chunk := range openai.DecodeStream(reader) {
		chunks = append(chunks, chunk)
	}

	// First chunk: ToolCallDelta with name
	tcd, ok := chunks[0].(model.ToolCallDelta)
	if !ok {
		t.Fatalf("chunk[0] type = %T, want ToolCallDelta", chunks[0])
	}

	if tcd.ToolName != "get_weather" {
		t.Errorf("tool name = %q, want get_weather", tcd.ToolName)
	}

	if tcd.ToolCallID != "call_1" {
		t.Errorf("tool call id = %q, want call_1", tcd.ToolCallID)
	}
}

func TestDecodeResponseEmptyChoices(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		resp := `{
			"id": "chatcmpl-empty",
			"choices": [],
			"usage": {
				"prompt_tokens": 10,
				"completion_tokens": 0
			}
		}`
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(resp))
	}))
	defer server.Close()

	cfg := &openai.ClientConfig{
		BaseURL: server.URL,
		APIKey:  "test-key",
	}

	req := &model.Request{
		Messages: []model.Message{
			{
				Role:  model.RoleUser,
				Parts: []model.Part{model.TextPart{Text: "Hello"}},
			},
		},
	}

	chatReq, err := openai.EncodeRequest("gpt-5.4", req, false)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	chatResp, err := openai.DoRequest(context.Background(), cfg, chatReq)
	if err != nil {
		t.Fatalf("DoRequest: %v", err)
	}

	resp := openai.DecodeResponse(chatResp)

	if resp.FinishReason != model.FinishReasonUnknown {
		t.Errorf("finish reason = %v, want unknown", resp.FinishReason)
	}

	if len(resp.Content) != 0 {
		t.Errorf("content length = %d, want 0", len(resp.Content))
	}
}

func TestDoStreamRequestMockServer(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected http.Flusher")
		}

		chunks := []string{
			`data: {"id":"cmpl-1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"}}]}`,
			``,
			`data: {"id":"cmpl-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			``,
			`data: [DONE]`,
			``,
		}

		for _, chunk := range chunks {
			_, _ = w.Write([]byte(chunk + "\n"))
			flusher.Flush()
		}
	}))
	defer server.Close()

	cfg := &openai.ClientConfig{
		BaseURL: server.URL,
		APIKey:  "test-key",
	}

	req := &model.Request{
		Messages: []model.Message{
			{
				Role:  model.RoleUser,
				Parts: []model.Part{model.TextPart{Text: "Hello"}},
			},
		},
	}

	chatReq, err := openai.EncodeRequest("gpt-5.4", req, true)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	body, err := openai.DoStreamRequest(context.Background(), cfg, chatReq)
	if err != nil {
		t.Fatalf("DoStreamRequest: %v", err)
	}
	defer func() { _ = body.Close() }()

	var chunks []model.Chunk
	for chunk := range openai.DecodeStream(body) {
		chunks = append(chunks, chunk)
	}

	if len(chunks) < 1 {
		t.Fatal("expected at least one chunk")
	}

	td, ok := chunks[0].(model.TextDelta)
	if !ok {
		t.Fatalf("chunk[0] type = %T, want TextDelta", chunks[0])
	}

	if td.Text != "Hi" {
		t.Errorf("text = %q, want Hi", td.Text)
	}
}

func TestExtraHeadersSent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Custom") != "test-value" {
			t.Errorf("X-Custom = %q, want test-value", r.Header.Get("X-Custom"))
		}

		resp := `{"id":"1","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(resp))
	}))
	defer server.Close()

	cfg := &openai.ClientConfig{
		BaseURL: server.URL,
		APIKey:  "test-key",
		ExtraHeaders: http.Header{
			"X-Custom": {"test-value"},
		},
	}

	req := &model.Request{
		Messages: []model.Message{
			{
				Role:  model.RoleUser,
				Parts: []model.Part{model.TextPart{Text: "Hi"}},
			},
		},
	}

	chatReq, err := openai.EncodeRequest("gpt-5.4", req, false)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	_, err = openai.DoRequest(context.Background(), cfg, chatReq)
	if err != nil {
		t.Fatalf("DoRequest: %v", err)
	}
}
