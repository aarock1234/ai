package google_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/google"
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

	genReq, err := google.EncodeRequest("gemini-3-flash-preview", req)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	body, err := google.MarshalRequest(genReq)
	if err != nil {
		t.Fatalf("MarshalRequest: %v", err)
	}

	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	system := raw["systemInstruction"].(map[string]any)
	systemParts := system["parts"].([]any)
	if systemParts[0].(map[string]any)["text"] != "Be concise." {
		t.Errorf("systemInstruction text = %v, want Be concise.", systemParts[0])
	}

	contents, ok := raw["contents"].([]any)
	if !ok || len(contents) != 1 {
		t.Fatalf("contents = %v, want 1 content", raw["contents"])
	}

	first := contents[0].(map[string]any)
	if first["role"] != "user" {
		t.Errorf("role = %v, want user", first["role"])
	}

	parts := first["parts"].([]any)
	if parts[0].(map[string]any)["text"] != "Hello" {
		t.Errorf("parts[0].text = %v, want Hello", parts[0])
	}

	if _, ok := raw["generationConfig"]; ok {
		t.Error("generationConfig should be omitted when no settings are set")
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
						ToolCallID: "call_0",
						ToolName:   "get_weather",
						Args:       json.RawMessage(`{"city":"NYC"}`),
					},
				},
			},
			{
				Role: model.RoleTool,
				Parts: []model.Part{
					model.ToolResultPart{
						ToolCallID: "call_0",
						ToolName:   "get_weather",
						Content:    "Sunny, 72F",
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
		ToolChoice: &model.ToolChoice{
			Type:     "tool",
			ToolName: "get_weather",
		},
	}

	genReq, err := google.EncodeRequest("gemini-3-flash-preview", req)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	body, err := google.MarshalRequest(genReq)
	if err != nil {
		t.Fatalf("MarshalRequest: %v", err)
	}

	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	toolCfg := raw["toolConfig"].(map[string]any)["functionCallingConfig"].(map[string]any)
	if toolCfg["mode"] != "ANY" {
		t.Errorf("mode = %v, want ANY", toolCfg["mode"])
	}

	allowed := toolCfg["allowedFunctionNames"].([]any)
	if len(allowed) != 1 || allowed[0] != "get_weather" {
		t.Errorf("allowedFunctionNames = %v, want [get_weather]", allowed)
	}

	tools := raw["tools"].([]any)
	declarations := tools[0].(map[string]any)["functionDeclarations"].([]any)
	if declarations[0].(map[string]any)["name"] != "get_weather" {
		t.Errorf("functionDeclarations[0].name = %v, want get_weather", declarations[0])
	}

	contents := raw["contents"].([]any)
	if len(contents) != 3 {
		t.Fatalf("contents length = %d, want 3", len(contents))
	}

	assistant := contents[1].(map[string]any)
	if assistant["role"] != "model" {
		t.Errorf("assistant role = %v, want model", assistant["role"])
	}

	fnCall := assistant["parts"].([]any)[0].(map[string]any)["functionCall"].(map[string]any)
	if fnCall["name"] != "get_weather" {
		t.Errorf("functionCall.name = %v, want get_weather", fnCall["name"])
	}

	if fnCall["args"].(map[string]any)["city"] != "NYC" {
		t.Errorf("functionCall.args.city = %v, want NYC", fnCall["args"])
	}

	toolResult := contents[2].(map[string]any)
	if toolResult["role"] != "user" {
		t.Errorf("tool result role = %v, want user", toolResult["role"])
	}

	fnResp := toolResult["parts"].([]any)[0].(map[string]any)["functionResponse"].(map[string]any)
	if fnResp["name"] != "get_weather" {
		t.Errorf("functionResponse.name = %v, want get_weather", fnResp["name"])
	}

	if fnResp["response"].(map[string]any)["result"] != "Sunny, 72F" {
		t.Errorf("functionResponse.response = %v, want wrapped result", fnResp["response"])
	}
}

func TestEncodeRequestThinkingConfig(t *testing.T) {
	tests := []struct {
		name                string
		reasoning           *model.ReasoningOptions
		wantBudget          int
		wantIncludeThoughts bool
	}{
		{
			name:                "effort low",
			reasoning:           &model.ReasoningOptions{Effort: "low"},
			wantBudget:          1024,
			wantIncludeThoughts: true,
		},
		{
			name:                "effort medium",
			reasoning:           &model.ReasoningOptions{Effort: "medium"},
			wantBudget:          8192,
			wantIncludeThoughts: true,
		},
		{
			name:                "effort high",
			reasoning:           &model.ReasoningOptions{Effort: "high"},
			wantBudget:          24576,
			wantIncludeThoughts: true,
		},
		{
			name: "max tokens overrides effort",
			reasoning: &model.ReasoningOptions{
				Effort:    "high",
				MaxTokens: 5000,
			},
			wantBudget:          5000,
			wantIncludeThoughts: true,
		},
		{
			name: "exclude disables thought output",
			reasoning: &model.ReasoningOptions{
				Effort:  "low",
				Exclude: true,
			},
			wantBudget:          1024,
			wantIncludeThoughts: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			genReq, err := google.EncodeRequest("gemini-3-flash-preview", &model.Request{
				Messages: []model.Message{
					{
						Role:  model.RoleUser,
						Parts: []model.Part{model.TextPart{Text: "Hi"}},
					},
				},
				Reasoning: tt.reasoning,
			})
			if err != nil {
				t.Fatalf("EncodeRequest: %v", err)
			}

			if genReq.GenerationConfig == nil || genReq.GenerationConfig.ThinkingConfig == nil {
				t.Fatal("thinking config not set")
			}

			thinking := genReq.GenerationConfig.ThinkingConfig
			if thinking.ThinkingBudget == nil || *thinking.ThinkingBudget != tt.wantBudget {
				t.Errorf("thinking budget = %v, want %d", thinking.ThinkingBudget, tt.wantBudget)
			}

			if thinking.IncludeThoughts != tt.wantIncludeThoughts {
				t.Errorf("include thoughts = %v, want %v", thinking.IncludeThoughts, tt.wantIncludeThoughts)
			}
		})
	}
}

func TestEncodeRequestUnsupportedReasoningEffort(t *testing.T) {
	_, err := google.EncodeRequest("gemini-3-flash-preview", &model.Request{
		Messages: []model.Message{
			{
				Role:  model.RoleUser,
				Parts: []model.Part{model.TextPart{Text: "Hi"}},
			},
		},
		Reasoning: &model.ReasoningOptions{Effort: "extreme"},
	})
	if err == nil {
		t.Fatal("expected unsupported reasoning effort error")
	}
}

func TestEncodeRequestResponseSchemaSanitized(t *testing.T) {
	schema := `{
		"$schema": "https://json-schema.org/draft/2020-12/schema",
		"type": "object",
		"additionalProperties": false,
		"properties": {
			"name": {"type": "string"},
			"nested": {
				"type": "object",
				"additionalProperties": false,
				"properties": {"id": {"type": "integer"}}
			}
		}
	}`

	genReq, err := google.EncodeRequest("gemini-3-flash-preview", &model.Request{
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
				Schema: json.RawMessage(schema),
			},
		},
		Tools: []model.ToolDefinition{
			{
				Name:        "lookup",
				InputSchema: json.RawMessage(`{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","additionalProperties":false}`),
			},
		},
	})
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	if genReq.GenerationConfig == nil {
		t.Fatal("generation config not set")
	}

	if genReq.GenerationConfig.ResponseMIMEType != "application/json" {
		t.Errorf("responseMimeType = %q, want application/json", genReq.GenerationConfig.ResponseMIMEType)
	}

	body, err := google.MarshalRequest(genReq)
	if err != nil {
		t.Fatalf("MarshalRequest: %v", err)
	}

	if strings.Contains(string(body), "$schema") {
		t.Error("marshalled request still contains $schema")
	}

	if strings.Contains(string(body), "additionalProperties") {
		t.Error("marshalled request still contains additionalProperties")
	}

	if !strings.Contains(string(body), `"responseSchema"`) {
		t.Error("marshalled request missing responseSchema")
	}
}

func TestEncodeRequestReasoningPartRoundTrip(t *testing.T) {
	genReq, err := google.EncodeRequest("gemini-3-flash-preview", &model.Request{
		Messages: []model.Message{
			{
				Role: model.RoleAssistant,
				Parts: []model.Part{
					model.ReasoningPart{
						Text:      "thinking about it",
						Signature: "sig_abc",
					},
					model.TextPart{Text: "The answer is 4."},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	if len(genReq.Contents) != 1 {
		t.Fatalf("contents length = %d, want 1", len(genReq.Contents))
	}

	parts := genReq.Contents[0].Parts
	if len(parts) != 2 {
		t.Fatalf("parts length = %d, want 2", len(parts))
	}

	if !parts[0].Thought {
		t.Error("parts[0].Thought = false, want true")
	}

	if parts[0].ThoughtSignature != "sig_abc" {
		t.Errorf("thought signature = %q, want sig_abc", parts[0].ThoughtSignature)
	}
}

func TestEncodeRequestProviderOptionsMerge(t *testing.T) {
	genReq, err := google.EncodeRequest("gemini-3-flash-preview", &model.Request{
		Messages: []model.Message{
			{
				Role:  model.RoleUser,
				Parts: []model.Part{model.TextPart{Text: "Hi"}},
			},
		},
		ProviderOptions: model.ProviderOptions{
			"google": model.ProviderOptionSet{
				"safetySettings": json.RawMessage(`[{"category":"HARM_CATEGORY_HARASSMENT","threshold":"BLOCK_NONE"}]`),
			},
		},
	})
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	body, err := google.MarshalRequest(genReq)
	if err != nil {
		t.Fatalf("MarshalRequest: %v", err)
	}

	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	settings, ok := raw["safetySettings"].([]any)
	if !ok || len(settings) != 1 {
		t.Fatalf("safetySettings = %v, want 1 setting", raw["safetySettings"])
	}

	if settings[0].(map[string]any)["threshold"] != "BLOCK_NONE" {
		t.Errorf("threshold = %v, want BLOCK_NONE", settings[0])
	}
}

func TestDecodeResponseToolCall(t *testing.T) {
	const body = `{
		"candidates": [
			{
				"content": {
					"role": "model",
					"parts": [
						{"functionCall": {"name": "get_weather", "args": {"city":"NYC"}}},
						{"functionCall": {"name": "get_time", "args": {"tz":"EST"}}}
					]
				},
				"finishReason": "STOP"
			}
		],
		"usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 6, "totalTokenCount": 18}
	}`

	var genResp google.GenerateContentResponse
	if err := json.Unmarshal([]byte(body), &genResp); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}

	resp := google.DecodeResponse(&genResp)

	if resp.FinishReason != model.FinishReasonToolCalls {
		t.Errorf("finish reason = %v, want tool_calls", resp.FinishReason)
	}

	if len(resp.Content) != 2 {
		t.Fatalf("content length = %d, want 2", len(resp.Content))
	}

	first, ok := resp.Content[0].(model.ToolCallContent)
	if !ok {
		t.Fatalf("content[0] type = %T, want ToolCallContent", resp.Content[0])
	}

	if first.ToolCallID != "call_0" {
		t.Errorf("tool call id = %q, want call_0", first.ToolCallID)
	}

	if first.ToolName != "get_weather" {
		t.Errorf("tool name = %q, want get_weather", first.ToolName)
	}

	if first.Args != `{"city":"NYC"}` {
		t.Errorf("args = %q, want JSON args", first.Args)
	}

	second, ok := resp.Content[1].(model.ToolCallContent)
	if !ok {
		t.Fatalf("content[1] type = %T, want ToolCallContent", resp.Content[1])
	}

	if second.ToolCallID != "call_1" {
		t.Errorf("second tool call id = %q, want call_1", second.ToolCallID)
	}
}

func TestDecodeResponseFinishReasons(t *testing.T) {
	tests := []struct {
		reason string
		want   model.FinishReason
	}{
		{reason: "STOP", want: model.FinishReasonStop},
		{reason: "MAX_TOKENS", want: model.FinishReasonLength},
		{reason: "SAFETY", want: model.FinishReasonContentFilter},
		{reason: "RECITATION", want: model.FinishReasonContentFilter},
		{reason: "BLOCKLIST", want: model.FinishReasonContentFilter},
		{reason: "PROHIBITED_CONTENT", want: model.FinishReasonContentFilter},
		{reason: "SPII", want: model.FinishReasonContentFilter},
		{reason: "MALFORMED_FUNCTION_CALL", want: model.FinishReasonError},
		{reason: "OTHER", want: model.FinishReasonUnknown},
	}

	for _, tt := range tests {
		t.Run(tt.reason, func(t *testing.T) {
			body := `{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"` + tt.reason + `"}]}`

			var genResp google.GenerateContentResponse
			if err := json.Unmarshal([]byte(body), &genResp); err != nil {
				t.Fatalf("unmarshal response: %v", err)
			}

			resp := google.DecodeResponse(&genResp)
			if resp.FinishReason != tt.want {
				t.Errorf("finish reason = %v, want %v", resp.FinishReason, tt.want)
			}

			if resp.RawFinishReason != tt.reason {
				t.Errorf("raw finish reason = %q, want %q", resp.RawFinishReason, tt.reason)
			}
		})
	}
}

func TestDecodeResponseUsageAndReasoning(t *testing.T) {
	const body = `{
		"candidates": [
			{
				"content": {
					"role": "model",
					"parts": [
						{"text": "planning the answer", "thought": true, "thoughtSignature": "sig_1"},
						{"text": "The answer is 4."}
					]
				},
				"finishReason": "STOP"
			}
		],
		"usageMetadata": {
			"promptTokenCount": 10,
			"candidatesTokenCount": 5,
			"totalTokenCount": 20,
			"thoughtsTokenCount": 3,
			"cachedContentTokenCount": 2
		},
		"modelVersion": "gemini-3-flash-preview",
		"responseId": "resp_9"
	}`

	var genResp google.GenerateContentResponse
	if err := json.Unmarshal([]byte(body), &genResp); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}

	resp := google.DecodeResponse(&genResp)

	if resp.Usage.InputTokens != 10 {
		t.Errorf("input tokens = %d, want 10", resp.Usage.InputTokens)
	}

	if resp.Usage.OutputTokens != 5 {
		t.Errorf("output tokens = %d, want 5", resp.Usage.OutputTokens)
	}

	if resp.Usage.ReasoningTokens != 3 {
		t.Errorf("reasoning tokens = %d, want 3", resp.Usage.ReasoningTokens)
	}

	if resp.Usage.CachedInputTokens != 2 {
		t.Errorf("cached input tokens = %d, want 2", resp.Usage.CachedInputTokens)
	}

	if len(resp.Content) != 2 {
		t.Fatalf("content length = %d, want 2", len(resp.Content))
	}

	reasoning, ok := resp.Content[0].(model.ReasoningContent)
	if !ok {
		t.Fatalf("content[0] type = %T, want ReasoningContent", resp.Content[0])
	}

	if reasoning.Text != "planning the answer" {
		t.Errorf("reasoning text = %q, want planning the answer", reasoning.Text)
	}

	if reasoning.Signature != "sig_1" {
		t.Errorf("reasoning signature = %q, want sig_1", reasoning.Signature)
	}

	if resp.Metadata.RequestID != "resp_9" {
		t.Errorf("request id = %q, want resp_9", resp.Metadata.RequestID)
	}

	if resp.Metadata.ModelID != "gemini-3-flash-preview" {
		t.Errorf("model id = %q, want gemini-3-flash-preview", resp.Metadata.ModelID)
	}
}

func TestDecodeStreamChunks(t *testing.T) {
	data := strings.Join([]string{
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]}}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1,"totalTokenCount":6},"modelVersion":"gemini-3-flash-preview","responseId":"resp_1"}`,
		"",
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":" world"}]}}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2,"totalTokenCount":7},"responseId":"resp_1"}`,
		"",
		`data: {"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},"responseId":"resp_1"}`,
		"",
	}, "\n")

	var chunks []model.Chunk
	for chunk := range google.DecodeStream(strings.NewReader(data)) {
		chunks = append(chunks, chunk)
	}

	if len(chunks) != 5 {
		t.Fatalf("chunks count = %d, want 5", len(chunks))
	}

	metadata, ok := chunks[0].(model.ProviderMetadataChunk)
	if !ok {
		t.Fatalf("chunk[0] type = %T, want ProviderMetadataChunk", chunks[0])
	}

	if metadata.Metadata.RequestID != "resp_1" {
		t.Errorf("request id = %q, want resp_1", metadata.Metadata.RequestID)
	}

	if metadata.Metadata.ModelID != "gemini-3-flash-preview" {
		t.Errorf("model id = %q, want gemini-3-flash-preview", metadata.Metadata.ModelID)
	}

	text1, ok := chunks[1].(model.TextDelta)
	if !ok {
		t.Fatalf("chunk[1] type = %T, want TextDelta", chunks[1])
	}

	if text1.Text != "Hello" {
		t.Errorf("chunk[1].Text = %q, want Hello", text1.Text)
	}

	text2, ok := chunks[2].(model.TextDelta)
	if !ok {
		t.Fatalf("chunk[2] type = %T, want TextDelta", chunks[2])
	}

	if text2.Text != " world" {
		t.Errorf("chunk[2].Text = %q, want ' world'", text2.Text)
	}

	usage, ok := chunks[3].(model.UsageChunk)
	if !ok {
		t.Fatalf("chunk[3] type = %T, want UsageChunk", chunks[3])
	}

	if usage.Usage.OutputTokens != 3 {
		t.Errorf("output tokens = %d, want final cumulative 3", usage.Usage.OutputTokens)
	}

	finish, ok := chunks[4].(model.FinishChunk)
	if !ok {
		t.Fatalf("chunk[4] type = %T, want FinishChunk", chunks[4])
	}

	if finish.FinishReason != model.FinishReasonStop {
		t.Errorf("finish reason = %v, want stop", finish.FinishReason)
	}
}

func TestDecodeStreamToolCallAndReasoning(t *testing.T) {
	data := strings.Join([]string{
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"let me check","thought":true,"thoughtSignature":"sig_1"}]}}],"responseId":"resp_2"}`,
		"",
		`data: {"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"city":"NYC"}}}]},"finishReason":"STOP"}],"responseId":"resp_2"}`,
		"",
	}, "\n")

	var chunks []model.Chunk
	for chunk := range google.DecodeStream(strings.NewReader(data)) {
		chunks = append(chunks, chunk)
	}

	if len(chunks) != 4 {
		t.Fatalf("chunks count = %d, want 4", len(chunks))
	}

	reasoning, ok := chunks[1].(model.ReasoningDelta)
	if !ok {
		t.Fatalf("chunk[1] type = %T, want ReasoningDelta", chunks[1])
	}

	if reasoning.Text != "let me check" {
		t.Errorf("reasoning text = %q, want let me check", reasoning.Text)
	}

	if reasoning.Signature != "sig_1" {
		t.Errorf("reasoning signature = %q, want sig_1", reasoning.Signature)
	}

	toolCall, ok := chunks[2].(model.ToolCallDelta)
	if !ok {
		t.Fatalf("chunk[2] type = %T, want ToolCallDelta", chunks[2])
	}

	if toolCall.ToolCallID != "call_0" {
		t.Errorf("tool call id = %q, want call_0", toolCall.ToolCallID)
	}

	if toolCall.ToolName != "get_weather" {
		t.Errorf("tool name = %q, want get_weather", toolCall.ToolName)
	}

	if toolCall.ArgsJSON != `{"city":"NYC"}` {
		t.Errorf("args = %q, want full JSON args", toolCall.ArgsJSON)
	}

	finish, ok := chunks[3].(model.FinishChunk)
	if !ok {
		t.Fatalf("chunk[3] type = %T, want FinishChunk", chunks[3])
	}

	if finish.FinishReason != model.FinishReasonToolCalls {
		t.Errorf("finish reason = %v, want tool_calls", finish.FinishReason)
	}
}
