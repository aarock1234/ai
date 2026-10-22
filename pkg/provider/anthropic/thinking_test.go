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

// TestDecodeResponseThinkingBlocks verifies that Anthropic extended
// thinking blocks become ReasoningContent segments while regular text
// and tool_use blocks decode as before.
func TestDecodeResponseThinkingBlocks(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		resp := `{
			"id": "msg_1",
			"type": "message",
			"role": "assistant",
			"model": "claude-opus-4-5",
			"content": [
				{
					"type": "thinking",
					"thinking": "Let me think step by step.",
					"signature": "sig-abc"
				},
				{
					"type": "redacted_thinking",
					"data": "OPAQUE-ENCRYPTED-BLOB"
				},
				{
					"type": "text",
					"text": "The answer is 42."
				}
			],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 10, "output_tokens": 20}
		}`
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(resp))
	}))
	defer server.Close()

	provider := anthropic.New(anthropic.WithAPIKey("k"), anthropic.WithBaseURL(server.URL))
	m := provider.Chat("claude-opus-4-5")

	resp, err := m.DoGenerate(context.Background(), &model.Request{
		Messages: []model.Message{{
			Role:  model.RoleUser,
			Parts: []model.Part{model.TextPart{Text: "think about it"}},
		}},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}

	var (
		reasoningBlocks []model.ReasoningContent
		textBlocks      []model.TextContent
	)
	for _, segment := range resp.Content {
		switch c := segment.(type) {
		case model.ReasoningContent:
			reasoningBlocks = append(reasoningBlocks, c)
		case model.TextContent:
			textBlocks = append(textBlocks, c)
		}
	}

	if len(reasoningBlocks) != 2 {
		t.Fatalf("reasoning blocks = %d, want 2", len(reasoningBlocks))
	}
	if reasoningBlocks[0].Text != "Let me think step by step." {
		t.Errorf("block 0 text = %q, want thinking text", reasoningBlocks[0].Text)
	}
	if reasoningBlocks[0].Signature != "sig-abc" {
		t.Errorf("block 0 signature = %q, want sig-abc", reasoningBlocks[0].Signature)
	}
	if reasoningBlocks[0].Format != "anthropic-thinking" {
		t.Errorf("block 0 format = %q, want anthropic-thinking", reasoningBlocks[0].Format)
	}

	if !reasoningBlocks[1].Redacted {
		t.Error("block 1 should be redacted")
	}
	if reasoningBlocks[1].Text != "OPAQUE-ENCRYPTED-BLOB" {
		t.Errorf("redacted data = %q, want encrypted payload", reasoningBlocks[1].Text)
	}

	if len(textBlocks) != 1 || textBlocks[0].Text != "The answer is 42." {
		t.Errorf("text blocks = %+v, want single answer", textBlocks)
	}
}

// TestEncodeRequestReasoningPartToThinkingBlock verifies that a prior
// assistant ReasoningPart becomes a `thinking` (or `redacted_thinking`)
// content block on the outgoing wire payload.
func TestEncodeRequestReasoningPartToThinkingBlock(t *testing.T) {
	var captured string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := make([]byte, r.ContentLength)
		_, _ = r.Body.Read(body)
		captured = string(body)

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-4-5","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer server.Close()

	provider := anthropic.New(anthropic.WithAPIKey("k"), anthropic.WithBaseURL(server.URL))
	m := provider.Chat("claude-opus-4-5")

	_, err := m.DoGenerate(context.Background(), &model.Request{
		Messages: []model.Message{
			{
				Role:  model.RoleUser,
				Parts: []model.Part{model.TextPart{Text: "continue"}},
			},
			{
				Role: model.RoleAssistant,
				Parts: []model.Part{
					model.ReasoningPart{
						Text:      "prior reasoning",
						Signature: "sig-x",
						Format:    "anthropic-thinking",
					},
					model.ReasoningPart{
						Text:     "ENCRYPTED",
						Redacted: true,
					},
					model.TextPart{Text: "ok"},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}

	var parsed struct {
		Messages []struct {
			Role    string            `json:"role"`
			Content []json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal([]byte(captured), &parsed); err != nil {
		t.Fatalf("unmarshal captured body: %v", err)
	}

	if len(parsed.Messages) != 2 {
		t.Fatalf("messages = %d, want 2", len(parsed.Messages))
	}

	assistant := parsed.Messages[1]
	if assistant.Role != "assistant" {
		t.Fatalf("role = %q, want assistant", assistant.Role)
	}

	if len(assistant.Content) != 3 {
		t.Fatalf("content blocks = %d, want 3", len(assistant.Content))
	}

	var thinking map[string]any
	if err := json.Unmarshal(assistant.Content[0], &thinking); err != nil {
		t.Fatalf("unmarshal thinking block: %v", err)
	}
	if thinking["type"] != "thinking" {
		t.Errorf("block 0 type = %v, want thinking", thinking["type"])
	}
	if thinking["thinking"] != "prior reasoning" {
		t.Errorf("thinking text = %v, want 'prior reasoning'", thinking["thinking"])
	}
	if thinking["signature"] != "sig-x" {
		t.Errorf("signature = %v, want sig-x", thinking["signature"])
	}

	var redacted map[string]any
	if err := json.Unmarshal(assistant.Content[1], &redacted); err != nil {
		t.Fatalf("unmarshal redacted block: %v", err)
	}
	if redacted["type"] != "redacted_thinking" {
		t.Errorf("block 1 type = %v, want redacted_thinking", redacted["type"])
	}
	if redacted["data"] != "ENCRYPTED" {
		t.Errorf("redacted data = %v, want ENCRYPTED", redacted["data"])
	}
}

// TestEncodeRequestThinkingBudget verifies that reasoning options map
// to the thinking wire config with the expected budget and that
// max_tokens is bumped above the budget when necessary.
func TestEncodeRequestThinkingBudget(t *testing.T) {
	tests := []struct {
		name          string
		reasoning     *model.ReasoningOptions
		maxTokens     *int
		wantBudget    int
		wantMaxTokens int
	}{
		{
			name:          "low effort",
			reasoning:     &model.ReasoningOptions{Effort: "low"},
			wantBudget:    2048,
			wantMaxTokens: 3072,
		},
		{
			name:          "medium effort",
			reasoning:     &model.ReasoningOptions{Effort: "medium"},
			wantBudget:    8192,
			wantMaxTokens: 9216,
		},
		{
			name:          "high effort",
			reasoning:     &model.ReasoningOptions{Effort: "high"},
			wantBudget:    24576,
			wantMaxTokens: 25600,
		},
		{
			name:          "empty effort defaults to medium",
			reasoning:     &model.ReasoningOptions{},
			wantBudget:    8192,
			wantMaxTokens: 9216,
		},
		{
			name:          "max tokens override wins over effort",
			reasoning:     &model.ReasoningOptions{Effort: "high", MaxTokens: 4000},
			wantBudget:    4000,
			wantMaxTokens: 5024,
		},
		{
			name:          "sufficient max tokens preserved",
			reasoning:     &model.ReasoningOptions{Effort: "low"},
			maxTokens:     intPtr(10000),
			wantBudget:    2048,
			wantMaxTokens: 10000,
		},
		{
			name:          "equal max tokens bumped",
			reasoning:     &model.ReasoningOptions{Effort: "low"},
			maxTokens:     intPtr(2048),
			wantBudget:    2048,
			wantMaxTokens: 3072,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msgReq, err := anthropic.EncodeRequest("claude-sonnet-4-5", &model.Request{
				Messages: []model.Message{{
					Role:  model.RoleUser,
					Parts: []model.Part{model.TextPart{Text: "think"}},
				}},
				MaxTokens: tt.maxTokens,
				Reasoning: tt.reasoning,
			}, false)
			if err != nil {
				t.Fatalf("EncodeRequest: %v", err)
			}

			body, err := anthropic.MarshalRequest(msgReq)
			if err != nil {
				t.Fatalf("MarshalRequest: %v", err)
			}

			var raw struct {
				MaxTokens int `json:"max_tokens"`
				Thinking  *struct {
					Type         string `json:"type"`
					BudgetTokens int    `json:"budget_tokens"`
				} `json:"thinking"`
			}
			if err := json.Unmarshal(body, &raw); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}

			if raw.Thinking == nil {
				t.Fatal("thinking missing from request body")
			}
			if raw.Thinking.Type != "enabled" {
				t.Errorf("thinking.type = %q, want enabled", raw.Thinking.Type)
			}
			if raw.Thinking.BudgetTokens != tt.wantBudget {
				t.Errorf("budget_tokens = %d, want %d", raw.Thinking.BudgetTokens, tt.wantBudget)
			}
			if raw.MaxTokens != tt.wantMaxTokens {
				t.Errorf("max_tokens = %d, want %d", raw.MaxTokens, tt.wantMaxTokens)
			}
		})
	}
}

// TestEncodeRequestThinkingStripsSamplingParams verifies that
// temperature, top_p, and top_k are dropped with warnings when thinking
// is enabled.
func TestEncodeRequestThinkingStripsSamplingParams(t *testing.T) {
	temperature := 0.7
	topP := 0.9
	topK := 40

	msgReq, err := anthropic.EncodeRequest("claude-sonnet-4-5", &model.Request{
		Messages: []model.Message{{
			Role:  model.RoleUser,
			Parts: []model.Part{model.TextPart{Text: "think"}},
		}},
		Temperature: &temperature,
		TopP:        &topP,
		TopK:        &topK,
		Reasoning:   &model.ReasoningOptions{Effort: "low"},
	}, false)
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

	for _, key := range []string{"temperature", "top_p", "top_k"} {
		if _, present := raw[key]; present {
			t.Errorf("%s should be dropped when thinking is enabled", key)
		}
	}

	if len(msgReq.Warnings) != 3 {
		t.Fatalf("warnings = %d, want 3: %v", len(msgReq.Warnings), msgReq.Warnings)
	}

	for i, param := range []string{"temperature", "top_p", "top_k"} {
		if !strings.Contains(msgReq.Warnings[i], param) {
			t.Errorf("warning[%d] = %q, want mention of %s", i, msgReq.Warnings[i], param)
		}
	}
}

// TestDecodeStreamThinkingBlocks verifies streamed thinking deltas
// produce ReasoningDelta chunks with the anthropic format marker.
func TestDecodeStreamThinkingBlocks(t *testing.T) {
	data := "" +
		`event: message_start` + "\n" +
		`data: {"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","model":"claude-opus-4-5","content":[],"stop_reason":"","usage":{"input_tokens":1,"output_tokens":0}}}` + "\n\n" +
		`event: content_block_start` + "\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}` + "\n\n" +
		`event: content_block_delta` + "\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"let me "}}` + "\n\n" +
		`event: content_block_delta` + "\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"think"}}` + "\n\n" +
		`event: content_block_delta` + "\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-final"}}` + "\n\n" +
		`event: content_block_stop` + "\n" +
		`data: {"type":"content_block_stop","index":0}` + "\n\n" +
		`event: content_block_start` + "\n" +
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}` + "\n\n" +
		`event: content_block_delta` + "\n" +
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"answer"}}` + "\n\n" +
		`event: message_delta` + "\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":1,"output_tokens":2}}` + "\n\n" +
		`event: message_stop` + "\n" +
		`data: {"type":"message_stop"}` + "\n\n"

	var (
		reasoningDeltas []model.ReasoningDelta
		textDeltas      []model.TextDelta
	)
	for chunk := range anthropic.DecodeStream(strings.NewReader(data)) {
		switch c := chunk.(type) {
		case model.ReasoningDelta:
			reasoningDeltas = append(reasoningDeltas, c)
		case model.TextDelta:
			textDeltas = append(textDeltas, c)
		}
	}

	if len(reasoningDeltas) < 2 {
		t.Fatalf("reasoning deltas = %d, want >=2", len(reasoningDeltas))
	}

	var joined strings.Builder
	var signature string
	for _, rd := range reasoningDeltas {
		joined.WriteString(rd.Text)
		if rd.Signature != "" {
			signature = rd.Signature
		}
		if rd.Format != "anthropic-thinking" {
			t.Errorf("format = %q, want anthropic-thinking", rd.Format)
		}
	}

	if joined.String() != "let me think" {
		t.Errorf("joined reasoning = %q, want 'let me think'", joined.String())
	}
	if signature != "sig-final" {
		t.Errorf("signature = %q, want sig-final", signature)
	}

	if len(textDeltas) != 1 || textDeltas[0].Text != "answer" {
		t.Errorf("text deltas = %+v, want [answer]", textDeltas)
	}
}
