package anthropic_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/anthropic"
)

// cacheRequest builds a request with a system prompt and two user
// messages, the last of which carries two content blocks.
func cacheRequest(options model.ProviderOptions) *model.Request {
	return &model.Request{
		Messages: []model.Message{
			{
				Role:  model.RoleSystem,
				Parts: []model.Part{model.TextPart{Text: "Be concise."}},
			},
			{
				Role:  model.RoleUser,
				Parts: []model.Part{model.TextPart{Text: "First question"}},
			},
			{
				Role:  model.RoleAssistant,
				Parts: []model.Part{model.TextPart{Text: "First answer"}},
			},
			{
				Role: model.RoleUser,
				Parts: []model.Part{
					model.TextPart{Text: "Context"},
					model.TextPart{Text: "Second question"},
				},
			},
		},
		ProviderOptions: options,
	}
}

// marshalRaw encodes a request and unmarshals the wire body for
// structural assertions.
func marshalRaw(t *testing.T, req *model.Request) map[string]any {
	t.Helper()

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

	return raw
}

// cacheControlOf returns the cache_control object of a wire content
// block, or nil when absent.
func cacheControlOf(block any) map[string]any {
	control, _ := block.(map[string]any)["cache_control"].(map[string]any)

	return control
}

// TestEncodeRequestSystemMarshalsAsPlainString is a regression test for
// the systemContent union: without a cache option the system field must
// remain a plain JSON string.
func TestEncodeRequestSystemMarshalsAsPlainString(t *testing.T) {
	raw := marshalRaw(t, cacheRequest(nil))

	system, ok := raw["system"].(string)
	if !ok {
		t.Fatalf("system = %T (%v), want JSON string", raw["system"], raw["system"])
	}

	if system != "Be concise." {
		t.Errorf("system = %q, want Be concise.", system)
	}
}

func TestEncodeRequestCacheControl(t *testing.T) {
	tests := []struct {
		name            string
		option          string
		wantSystemBlock bool
		wantLastMessage bool
		wantTTL         string
	}{
		{
			name:            "system mode",
			option:          `{"mode":"system","ttl":"1h"}`,
			wantSystemBlock: true,
			wantTTL:         "1h",
		},
		{
			name:            "last-message mode",
			option:          `{"mode":"last-message"}`,
			wantLastMessage: true,
		},
		{
			name:            "auto mode",
			option:          `{"mode":"auto","ttl":"1h"}`,
			wantSystemBlock: true,
			wantLastMessage: true,
			wantTTL:         "1h",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw := marshalRaw(t, cacheRequest(model.ProviderOptions{
				"anthropic": model.ProviderOptionSet{
					"cache_control": json.RawMessage(tt.option),
				},
			}))

			if tt.wantSystemBlock {
				blocks, ok := raw["system"].([]any)
				if !ok || len(blocks) != 1 {
					t.Fatalf("system = %v, want single-block array", raw["system"])
				}

				block := blocks[0].(map[string]any)
				if block["type"] != "text" || block["text"] != "Be concise." {
					t.Errorf("system block = %v, want text block with prompt", block)
				}

				control := cacheControlOf(blocks[0])
				if control == nil {
					t.Fatal("system block missing cache_control")
				}
				if control["type"] != "ephemeral" {
					t.Errorf("cache_control.type = %v, want ephemeral", control["type"])
				}
				if tt.wantTTL != "" && control["ttl"] != tt.wantTTL {
					t.Errorf("cache_control.ttl = %v, want %s", control["ttl"], tt.wantTTL)
				}
			} else if _, ok := raw["system"].(string); !ok {
				t.Errorf("system = %T, want plain string when not cached", raw["system"])
			}

			messages := raw["messages"].([]any)
			lastUser := messages[len(messages)-1].(map[string]any)
			content := lastUser["content"].([]any)

			if control := cacheControlOf(content[0]); control != nil {
				t.Error("first block of last user message should not carry cache_control")
			}

			finalControl := cacheControlOf(content[len(content)-1])
			if tt.wantLastMessage {
				if finalControl == nil {
					t.Fatal("final block of last user message missing cache_control")
				}
				if finalControl["type"] != "ephemeral" {
					t.Errorf("cache_control.type = %v, want ephemeral", finalControl["type"])
				}
			} else if finalControl != nil {
				t.Error("final block of last user message should not carry cache_control")
			}
		})
	}
}

func TestEncodeRequestCacheControlRejectsUnknownMode(t *testing.T) {
	_, err := anthropic.EncodeRequest("claude-sonnet-4-5", cacheRequest(model.ProviderOptions{
		"anthropic": model.ProviderOptionSet{
			"cache_control": json.RawMessage(`{"mode":"everything"}`),
		},
	}), false)
	if err == nil {
		t.Fatal("expected cache_control mode error")
	}

	if !strings.Contains(err.Error(), "cache_control mode must be system, last-message, or auto") {
		t.Errorf("error = %q, want mode validation message", err)
	}
}

// TestDecodeResponseCacheUsage verifies cache token counters map onto
// the normalized usage for non-streaming responses.
func TestDecodeResponseCacheUsage(t *testing.T) {
	const body = `{
		"id": "msg_1",
		"type": "message",
		"role": "assistant",
		"content": [{"type": "text", "text": "ok"}],
		"model": "claude-sonnet-4-5",
		"stop_reason": "end_turn",
		"usage": {
			"input_tokens": 10,
			"output_tokens": 5,
			"cache_creation_input_tokens": 200,
			"cache_read_input_tokens": 300
		}
	}`

	var msgResp anthropic.MessageResponse
	if err := json.Unmarshal([]byte(body), &msgResp); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}

	resp := anthropic.DecodeResponse(&msgResp)

	if resp.Usage.CacheCreationInputTokens != 200 {
		t.Errorf("cache creation tokens = %d, want 200", resp.Usage.CacheCreationInputTokens)
	}

	if resp.Usage.CachedInputTokens != 300 {
		t.Errorf("cached input tokens = %d, want 300", resp.Usage.CachedInputTokens)
	}
}

// TestDecodeStreamCacheUsage verifies cache token counters map onto the
// usage chunk emitted from the message_start stream event.
func TestDecodeStreamCacheUsage(t *testing.T) {
	data := strings.Join([]string{
		"event: message_start",
		`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-5","usage":{"input_tokens":10,"output_tokens":0,"cache_creation_input_tokens":50,"cache_read_input_tokens":75}}}`,
		"",
		"event: message_stop",
		`data: {"type":"message_stop"}`,
		"",
	}, "\n")

	var usage *model.UsageChunk
	for chunk := range anthropic.DecodeStream(strings.NewReader(data)) {
		if u, ok := chunk.(model.UsageChunk); ok {
			usage = &u
		}
	}

	if usage == nil {
		t.Fatal("no usage chunk emitted")
	}

	if usage.Usage.CacheCreationInputTokens != 50 {
		t.Errorf("cache creation tokens = %d, want 50", usage.Usage.CacheCreationInputTokens)
	}

	if usage.Usage.CachedInputTokens != 75 {
		t.Errorf("cached input tokens = %d, want 75", usage.Usage.CachedInputTokens)
	}
}
