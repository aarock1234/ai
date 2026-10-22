package openai_test

import (
	"encoding/json"
	"testing"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

func marshalToMap(t *testing.T, chatReq *openai.ChatRequest) map[string]any {
	t.Helper()

	body, err := openai.MarshalRequest(chatReq)
	if err != nil {
		t.Fatalf("MarshalRequest: %v", err)
	}

	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	return raw
}

// TestEncodeReasoningOpenAIShape verifies the default dialect encodes
// reasoning as the flat reasoning_effort string OpenAI expects, with
// no OpenRouter-style reasoning object.
func TestEncodeReasoningOpenAIShape(t *testing.T) {
	req := &model.Request{
		Messages: []model.Message{
			{
				Role:  model.RoleUser,
				Parts: []model.Part{model.TextPart{Text: "hi"}},
			},
		},
		Reasoning: &model.ReasoningOptions{
			Effort:    "high",
			MaxTokens: 2000,
		},
	}

	chatReq, err := openai.EncodeRequest("gpt-5.4", req, false)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	raw := marshalToMap(t, chatReq)
	if raw["reasoning_effort"] != "high" {
		t.Errorf("reasoning_effort = %v, want high", raw["reasoning_effort"])
	}
	if _, ok := raw["reasoning"]; ok {
		t.Error("reasoning object present, want flat reasoning_effort only")
	}
}

// TestEncodeReasoningOpenRouterShape verifies the OpenRouter dialect
// keeps the full reasoning object.
func TestEncodeReasoningOpenRouterShape(t *testing.T) {
	req := &model.Request{
		Messages: []model.Message{
			{
				Role:  model.RoleUser,
				Parts: []model.Part{model.TextPart{Text: "hi"}},
			},
		},
		Reasoning: &model.ReasoningOptions{
			Effort:    "medium",
			MaxTokens: 1500,
			Exclude:   true,
		},
	}

	chatReq, err := openai.EncodeRequest("openai/gpt-5.4", req, false, openai.WithReasoningShape(openai.ReasoningShapeOpenRouter))
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	raw := marshalToMap(t, chatReq)
	reasoning, ok := raw["reasoning"].(map[string]any)
	if !ok {
		t.Fatal("reasoning object missing")
	}
	if reasoning["effort"] != "medium" || reasoning["max_tokens"] != float64(1500) || reasoning["exclude"] != true {
		t.Errorf("reasoning = %v, want medium/1500/exclude", reasoning)
	}
	if _, ok := raw["reasoning_effort"]; ok {
		t.Error("reasoning_effort present, want reasoning object only")
	}
}

// TestEncodeProviderOptions verifies service_tier, logprobs, and
// parallel_tool_calls provider options map onto typed request fields.
func TestEncodeProviderOptions(t *testing.T) {
	opts := model.ProviderOptions{
		"openai": model.ProviderOptionSet{
			"service_tier":        json.RawMessage(`"flex"`),
			"logprobs":            json.RawMessage(`true`),
			"top_logprobs":        json.RawMessage(`5`),
			"parallel_tool_calls": json.RawMessage(`false`),
		},
	}
	req := &model.Request{
		Messages: []model.Message{
			{
				Role:  model.RoleUser,
				Parts: []model.Part{model.TextPart{Text: "hi"}},
			},
		},
		ProviderOptions: opts,
	}

	chatReq, err := openai.EncodeRequest("gpt-5.4", req, false)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	raw := marshalToMap(t, chatReq)
	if raw["service_tier"] != "flex" {
		t.Errorf("service_tier = %v, want flex", raw["service_tier"])
	}
	if raw["logprobs"] != true {
		t.Errorf("logprobs = %v, want true", raw["logprobs"])
	}
	if raw["top_logprobs"] != float64(5) {
		t.Errorf("top_logprobs = %v, want 5", raw["top_logprobs"])
	}
	if raw["parallel_tool_calls"] != false {
		t.Errorf("parallel_tool_calls = %v, want false", raw["parallel_tool_calls"])
	}
}

// TestEncodeRejectsMultiChoice verifies the unsupported n option fails
// loudly instead of silently dropping extra choices.
func TestEncodeRejectsMultiChoice(t *testing.T) {
	req := &model.Request{
		Messages: []model.Message{
			{
				Role:  model.RoleUser,
				Parts: []model.Part{model.TextPart{Text: "hi"}},
			},
		},
		ProviderOptions: model.ProviderOptions{
			"openai": model.ProviderOptionSet{
				"n": json.RawMessage(`3`),
			},
		},
	}

	if _, err := openai.EncodeRequest("gpt-5.4", req, false); err == nil {
		t.Fatal("want error for n option, got nil")
	}
}
