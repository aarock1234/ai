package anthropic_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/anthropic"
)

func TestEncodeRequestBasic(t *testing.T) {
	req := &model.Request{
		Messages: []model.Message{
			{
				Role: model.RoleSystem,
				Parts: []model.Part{
					model.TextPart{Text: "Be concise."},
				},
			},
			{
				Role: model.RoleUser,
				Parts: []model.Part{
					model.TextPart{Text: "Hello"},
				},
			},
		},
	}

	msgReq, err := anthropic.EncodeRequest("claude-sonnet-4-5", req, false)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	body, err := anthropic.MarshalRequest(msgReq)
	if err != nil {
		t.Fatalf("MarshalRequest: %v", err)
	}

	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if raw["model"] != "claude-sonnet-4-5" {
		t.Errorf("model = %v, want claude-sonnet-4-5", raw["model"])
	}

	if raw["system"] != "Be concise." {
		t.Errorf("system = %v, want Be concise.", raw["system"])
	}

	if raw["max_tokens"] != float64(1024) {
		t.Errorf("max_tokens = %v, want 1024", raw["max_tokens"])
	}

	msgs, ok := raw["messages"].([]any)
	if !ok || len(msgs) != 1 {
		t.Fatalf("messages = %v, want 1 message", raw["messages"])
	}

	msg := msgs[0].(map[string]any)
	if msg["role"] != "user" {
		t.Errorf("role = %v, want user", msg["role"])
	}

	content := msg["content"].([]any)
	part := content[0].(map[string]any)
	if part["type"] != "text" {
		t.Errorf("content[0].type = %v, want text", part["type"])
	}

	if part["text"] != "Hello" {
		t.Errorf("content[0].text = %v, want Hello", part["text"])
	}
}

func TestEncodeRequestSystemMessagePartsUseSeparators(t *testing.T) {
	req := &model.Request{
		Messages: []model.Message{
			{
				Role: model.RoleSystem,
				Parts: []model.Part{
					model.TextPart{Text: "First instruction."},
					model.TextPart{Text: "Second instruction."},
				},
			},
			{
				Role: model.RoleUser,
				Parts: []model.Part{
					model.TextPart{Text: "Hello"},
				},
			},
		},
	}

	msgReq, err := anthropic.EncodeRequest("claude-sonnet-4-5", req, false)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	if msgReq.System != "First instruction.\n\nSecond instruction." {
		t.Errorf("system = %q, want separated system parts", msgReq.System)
	}
}

func TestEncodeRequestToolConversation(t *testing.T) {
	req := &model.Request{
		Messages: []model.Message{
			{
				Role: model.RoleUser,
				Parts: []model.Part{
					model.TextPart{Text: "What's the weather?"},
				},
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
		Tools: []model.ToolDefinition{
			{
				Name:        "get_weather",
				Description: "Get current weather",
				InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
			},
		},
		ToolChoice: &model.ToolChoice{Type: "required"},
	}

	msgReq, err := anthropic.EncodeRequest("claude-sonnet-4-5", req, false)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	body, err := anthropic.MarshalRequest(msgReq)
	if err != nil {
		t.Fatalf("MarshalRequest: %v", err)
	}

	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if raw["tool_choice"].(map[string]any)["type"] != "any" {
		t.Errorf("tool_choice.type = %v, want any", raw["tool_choice"].(map[string]any)["type"])
	}

	tools, ok := raw["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools = %v, want 1 tool", raw["tools"])
	}

	messages := raw["messages"].([]any)
	if len(messages) != 3 {
		t.Fatalf("messages length = %d, want 3", len(messages))
	}

	assistantMsg := messages[1].(map[string]any)
	if assistantMsg["role"] != "assistant" {
		t.Errorf("assistant role = %v, want assistant", assistantMsg["role"])
	}

	assistantContent := assistantMsg["content"].([]any)
	toolUse := assistantContent[0].(map[string]any)
	if toolUse["type"] != "tool_use" {
		t.Errorf("assistant content type = %v, want tool_use", toolUse["type"])
	}

	if toolUse["id"] != "call_1" {
		t.Errorf("tool_use.id = %v, want call_1", toolUse["id"])
	}

	toolResultMsg := messages[2].(map[string]any)
	if toolResultMsg["role"] != "user" {
		t.Errorf("tool result role = %v, want user", toolResultMsg["role"])
	}

	toolResultContent := toolResultMsg["content"].([]any)
	toolResult := toolResultContent[0].(map[string]any)
	if toolResult["type"] != "tool_result" {
		t.Errorf("tool result type = %v, want tool_result", toolResult["type"])
	}

	if toolResult["tool_use_id"] != "call_1" {
		t.Errorf("tool_use_id = %v, want call_1", toolResult["tool_use_id"])
	}
}

func TestEncodeRequestRejectsResponseFormat(t *testing.T) {
	req := &model.Request{
		Messages: []model.Message{
			{
				Role: model.RoleUser,
				Parts: []model.Part{
					model.TextPart{Text: "Hi"},
				},
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

	_, err := anthropic.EncodeRequest("claude-sonnet-4-5", req, false)
	if err == nil {
		t.Fatal("expected response_format error")
	}
}

func TestDoRequestMockServer(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}

		if r.URL.Path != "/messages" {
			t.Errorf("path = %q, want /messages", r.URL.Path)
		}

		if r.Header.Get("x-api-key") != "test-key" {
			t.Errorf("x-api-key = %q, want test-key", r.Header.Get("x-api-key"))
		}

		if r.Header.Get("anthropic-version") != "2024-01-01" {
			t.Errorf("anthropic-version = %q, want 2024-01-01", r.Header.Get("anthropic-version"))
		}

		resp := `{
			"id": "msg_123",
			"type": "message",
			"role": "assistant",
			"content": [
				{"type": "text", "text": "Hello there!"}
			],
			"model": "claude-sonnet-4-5",
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 10, "output_tokens": 5}
		}`
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(resp))
	}))
	defer server.Close()

	cfg := &anthropic.ClientConfig{
		BaseURL:    server.URL,
		APIKey:     "test-key",
		APIVersion: "2024-01-01",
	}

	encoded, err := anthropic.EncodeRequest("claude-sonnet-4-5", &model.Request{
		Messages: []model.Message{
			{
				Role: model.RoleUser,
				Parts: []model.Part{
					model.TextPart{Text: "Hello"},
				},
			},
		},
		MaxTokens: intPtr(64),
	}, false)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	msgResp, err := anthropic.DoRequest(context.Background(), cfg, encoded)
	if err != nil {
		t.Fatalf("DoRequest: %v", err)
	}

	resp := anthropic.DecodeResponse(msgResp)
	if len(resp.Content) != 1 {
		t.Fatalf("content length = %d, want 1", len(resp.Content))
	}

	text, ok := resp.Content[0].(model.TextContent)
	if !ok {
		t.Fatalf("content[0] type = %T, want TextContent", resp.Content[0])
	}

	if text.Text != "Hello there!" {
		t.Errorf("text = %q, want Hello there!", text.Text)
	}

	if resp.FinishReason != model.FinishReasonStop {
		t.Errorf("finish reason = %v, want stop", resp.FinishReason)
	}
}

func TestDoRequestDoesNotMutateStreamFlag(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "msg_123",
			"type": "message",
			"role": "assistant",
			"content": [{"type": "text", "text": "ok"}],
			"model": "claude-sonnet-4-5",
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 1, "output_tokens": 1}
		}`))
	}))
	defer server.Close()

	cfg := &anthropic.ClientConfig{
		BaseURL:    server.URL,
		APIKey:     "test-key",
		APIVersion: "2023-06-01",
	}

	msgReq, err := anthropic.EncodeRequest("claude-sonnet-4-5", &model.Request{
		Messages: []model.Message{
			{
				Role:  model.RoleUser,
				Parts: []model.Part{model.TextPart{Text: "Hello"}},
			},
		},
	}, true)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	if !msgReq.Stream {
		t.Fatal("precondition failed: encoded request should be streaming")
	}

	if _, err := anthropic.DoRequest(context.Background(), cfg, msgReq); err != nil {
		t.Fatalf("DoRequest: %v", err)
	}

	if !msgReq.Stream {
		t.Fatal("DoRequest mutated request Stream flag")
	}
}

func TestDecodeResponseToolCall(t *testing.T) {
	const body = `{
		"id": "msg_456",
		"type": "message",
		"role": "assistant",
		"content": [
			{
				"type": "tool_use",
				"id": "call_abc",
				"name": "get_weather",
				"input": {"city":"NYC"}
			}
		],
		"stop_reason": "tool_use",
		"usage": {"input_tokens": 12, "output_tokens": 6}
	}`

	var msgResp anthropic.MessageResponse
	if err := json.Unmarshal([]byte(body), &msgResp); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}

	resp := anthropic.DecodeResponse(&msgResp)

	if resp.FinishReason != model.FinishReasonToolCalls {
		t.Errorf("finish reason = %v, want tool_calls", resp.FinishReason)
	}

	if len(resp.Content) != 1 {
		t.Fatalf("content length = %d, want 1", len(resp.Content))
	}

	toolCall, ok := resp.Content[0].(model.ToolCallContent)
	if !ok {
		t.Fatalf("content[0] type = %T, want ToolCallContent", resp.Content[0])
	}

	if toolCall.ToolCallID != "call_abc" {
		t.Errorf("tool call id = %q, want call_abc", toolCall.ToolCallID)
	}

	if toolCall.ToolName != "get_weather" {
		t.Errorf("tool name = %q, want get_weather", toolCall.ToolName)
	}

	if toolCall.Args != `{"city":"NYC"}` {
		t.Errorf("args = %q, want JSON args", toolCall.Args)
	}
}

func TestDecodeStreamChunks(t *testing.T) {
	data := strings.Join([]string{
		"event: message_start",
		"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-5\",\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}",
		"",
		"event: content_block_delta",
		"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}",
		"",
		"event: content_block_delta",
		"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" world\"}}",
		"",
		"event: message_delta",
		"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":5,\"output_tokens\":2}}",
		"",
		"event: message_stop",
		"data: {\"type\":\"message_stop\"}",
		"",
	}, "\n")

	var chunks []model.Chunk
	for chunk := range anthropic.DecodeStream(strings.NewReader(data)) {
		chunks = append(chunks, chunk)
	}

	if len(chunks) != 6 {
		t.Fatalf("chunks count = %d, want 6", len(chunks))
	}

	metadata, ok := chunks[0].(model.ProviderMetadataChunk)
	if !ok {
		t.Fatalf("chunk[0] type = %T, want ProviderMetadataChunk", chunks[0])
	}

	if metadata.Metadata.RequestID != "msg_1" {
		t.Errorf("request id = %q, want msg_1", metadata.Metadata.RequestID)
	}

	if metadata.Metadata.ModelID != "claude-sonnet-4-5" {
		t.Errorf("model id = %q, want claude-sonnet-4-5", metadata.Metadata.ModelID)
	}

	usageStart, ok := chunks[1].(model.UsageChunk)
	if !ok {
		t.Fatalf("chunk[1] type = %T, want UsageChunk", chunks[1])
	}

	if usageStart.Usage.InputTokens != 5 {
		t.Errorf("initial input tokens = %d, want 5", usageStart.Usage.InputTokens)
	}

	text1, ok := chunks[2].(model.TextDelta)
	if !ok {
		t.Fatalf("chunk[2] type = %T, want TextDelta", chunks[2])
	}

	if text1.Text != "Hello" {
		t.Errorf("chunk[2].Text = %q, want Hello", text1.Text)
	}

	text2, ok := chunks[3].(model.TextDelta)
	if !ok {
		t.Fatalf("chunk[3] type = %T, want TextDelta", chunks[3])
	}

	if text2.Text != " world" {
		t.Errorf("chunk[3].Text = %q, want ' world'", text2.Text)
	}

	usageEnd, ok := chunks[4].(model.UsageChunk)
	if !ok {
		t.Fatalf("chunk[4] type = %T, want UsageChunk", chunks[4])
	}

	if usageEnd.Usage.OutputTokens != 2 {
		t.Errorf("final output tokens = %d, want 2", usageEnd.Usage.OutputTokens)
	}

	finish, ok := chunks[5].(model.FinishChunk)
	if !ok {
		t.Fatalf("chunk[5] type = %T, want FinishChunk", chunks[5])
	}

	if finish.FinishReason != model.FinishReasonStop {
		t.Errorf("finish reason = %v, want stop", finish.FinishReason)
	}
}

func TestDecodeStreamToolCallDeltas(t *testing.T) {
	data := strings.Join([]string{
		"event: content_block_start",
		"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_1\",\"name\":\"get_weather\",\"input\":{}}}",
		"",
		"event: content_block_delta",
		"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\"}}",
		"",
		"event: content_block_delta",
		"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"NYC\\\"}\"}}",
		"",
		"event: message_delta",
		"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}",
		"",
	}, "\n")

	var chunks []model.Chunk
	for chunk := range anthropic.DecodeStream(strings.NewReader(data)) {
		chunks = append(chunks, chunk)
	}

	if len(chunks) != 4 {
		t.Fatalf("chunks count = %d, want 4", len(chunks))
	}

	start, ok := chunks[0].(model.ToolCallDelta)
	if !ok {
		t.Fatalf("chunk[0] type = %T, want ToolCallDelta", chunks[0])
	}

	if start.ToolCallID != "call_1" {
		t.Errorf("tool call id = %q, want call_1", start.ToolCallID)
	}

	if start.ToolName != "get_weather" {
		t.Errorf("tool name = %q, want get_weather", start.ToolName)
	}

	delta1, ok := chunks[1].(model.ToolCallDelta)
	if !ok {
		t.Fatalf("chunk[1] type = %T, want ToolCallDelta", chunks[1])
	}

	if delta1.ArgsJSON != `{"city":` {
		t.Errorf("chunk[1].ArgsJSON = %q, want partial JSON", delta1.ArgsJSON)
	}

	delta2, ok := chunks[2].(model.ToolCallDelta)
	if !ok {
		t.Fatalf("chunk[2] type = %T, want ToolCallDelta", chunks[2])
	}

	if delta2.ArgsJSON != `"NYC"}` {
		t.Errorf("chunk[2].ArgsJSON = %q, want closing JSON", delta2.ArgsJSON)
	}

	finish, ok := chunks[3].(model.FinishChunk)
	if !ok {
		t.Fatalf("chunk[3] type = %T, want FinishChunk", chunks[3])
	}

	if finish.FinishReason != model.FinishReasonToolCalls {
		t.Errorf("finish reason = %v, want tool_calls", finish.FinishReason)
	}
}

func TestDecodeStreamIgnoresUnknownToolDeltaState(t *testing.T) {
	data := strings.Join([]string{
		"event: content_block_delta",
		"data: {\"type\":\"content_block_delta\",\"index\":7,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"NYC\\\"}\"}}",
		"",
		"event: message_delta",
		"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}",
		"",
	}, "\n")

	var chunks []model.Chunk
	for chunk := range anthropic.DecodeStream(strings.NewReader(data)) {
		chunks = append(chunks, chunk)
	}

	if len(chunks) != 1 {
		t.Fatalf("chunks count = %d, want 1", len(chunks))
	}

	finish, ok := chunks[0].(model.FinishChunk)
	if !ok {
		t.Fatalf("chunk[0] type = %T, want FinishChunk", chunks[0])
	}

	if finish.FinishReason != model.FinishReasonToolCalls {
		t.Errorf("finish reason = %v, want tool_calls", finish.FinishReason)
	}
}

func TestDecodeStreamStopsTrackingToolStateOnBlockStop(t *testing.T) {
	data := strings.Join([]string{
		"event: content_block_start",
		"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_1\",\"name\":\"get_weather\",\"input\":{}}}",
		"",
		"event: content_block_stop",
		"data: {\"type\":\"content_block_stop\",\"index\":0}",
		"",
		"event: content_block_delta",
		"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"late\\\":true}\"}}",
		"",
		"event: message_delta",
		"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}",
		"",
	}, "\n")

	var chunks []model.Chunk
	for chunk := range anthropic.DecodeStream(strings.NewReader(data)) {
		chunks = append(chunks, chunk)
	}

	if len(chunks) != 2 {
		t.Fatalf("chunks count = %d, want 2", len(chunks))
	}

	start, ok := chunks[0].(model.ToolCallDelta)
	if !ok {
		t.Fatalf("chunk[0] type = %T, want ToolCallDelta", chunks[0])
	}

	if start.ToolCallID != "call_1" {
		t.Errorf("tool call id = %q, want call_1", start.ToolCallID)
	}

	finish, ok := chunks[1].(model.FinishChunk)
	if !ok {
		t.Fatalf("chunk[1] type = %T, want FinishChunk", chunks[1])
	}

	if finish.FinishReason != model.FinishReasonToolCalls {
		t.Errorf("finish reason = %v, want tool_calls", finish.FinishReason)
	}
}

func TestDoStreamRequestDoesNotMutateStreamFlag(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"))
	}))
	defer server.Close()

	cfg := &anthropic.ClientConfig{
		BaseURL:    server.URL,
		APIKey:     "test-key",
		APIVersion: "2023-06-01",
	}

	msgReq, err := anthropic.EncodeRequest("claude-sonnet-4-5", &model.Request{
		Messages: []model.Message{
			{
				Role:  model.RoleUser,
				Parts: []model.Part{model.TextPart{Text: "Hello"}},
			},
		},
	}, false)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	if msgReq.Stream {
		t.Fatal("precondition failed: encoded request should be non-streaming")
	}

	body, err := anthropic.DoStreamRequest(context.Background(), cfg, msgReq)
	if err != nil {
		t.Fatalf("DoStreamRequest: %v", err)
	}
	defer func() { _ = body.Close() }()

	if msgReq.Stream {
		t.Fatal("DoStreamRequest mutated request Stream flag")
	}
}

func TestDoStreamRequestMockServer(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-api-key") != "test-key" {
			t.Errorf("x-api-key = %q, want test-key", r.Header.Get("x-api-key"))
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected http.Flusher")
		}

		events := []string{
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
			"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
		}

		for _, event := range events {
			_, _ = w.Write([]byte(event))
			flusher.Flush()
		}
	}))
	defer server.Close()

	cfg := &anthropic.ClientConfig{
		BaseURL:    server.URL,
		APIKey:     "test-key",
		APIVersion: "2023-06-01",
	}

	msgReq, err := anthropic.EncodeRequest("claude-sonnet-4-5", &model.Request{
		Messages: []model.Message{
			{
				Role: model.RoleUser,
				Parts: []model.Part{
					model.TextPart{Text: "Hello"},
				},
			},
		},
	}, true)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	body, err := anthropic.DoStreamRequest(context.Background(), cfg, msgReq)
	if err != nil {
		t.Fatalf("DoStreamRequest: %v", err)
	}
	defer func() { _ = body.Close() }()

	var chunks []model.Chunk
	for chunk := range anthropic.DecodeStream(body) {
		chunks = append(chunks, chunk)
	}

	if len(chunks) != 2 {
		t.Fatalf("chunks count = %d, want 2", len(chunks))
	}

	text, ok := chunks[0].(model.TextDelta)
	if !ok {
		t.Fatalf("chunk[0] type = %T, want TextDelta", chunks[0])
	}

	if text.Text != "Hi" {
		t.Errorf("text = %q, want Hi", text.Text)
	}
}

func intPtr(v int) *int {
	return &v
}
