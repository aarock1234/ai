package model_test

import (
	"encoding/json"
	"testing"

	"github.com/aarock1234/ai/pkg/model"
)

func TestUsageTotalTokens(t *testing.T) {
	u := model.Usage{
		InputTokens:  100,
		OutputTokens: 50,
	}

	if got := u.TotalTokens(); got != 150 {
		t.Errorf("TotalTokens() = %d, want 150", got)
	}
}

func TestUsageTotalTokensZero(t *testing.T) {
	var u model.Usage
	if got := u.TotalTokens(); got != 0 {
		t.Errorf("TotalTokens() = %d, want 0", got)
	}
}

func TestFinishReasonConstants(t *testing.T) {
	tests := []struct {
		reason model.FinishReason
		want   string
	}{
		{model.FinishReasonStop, "stop"},
		{model.FinishReasonLength, "length"},
		{model.FinishReasonToolCalls, "tool_calls"},
		{model.FinishReasonContentFilter, "content_filter"},
		{model.FinishReasonUnknown, "unknown"},
	}

	for _, tt := range tests {
		if string(tt.reason) != tt.want {
			t.Errorf("FinishReason = %q, want %q", tt.reason, tt.want)
		}
	}
}

func TestRoleConstants(t *testing.T) {
	tests := []struct {
		role model.Role
		want string
	}{
		{model.RoleSystem, "system"},
		{model.RoleUser, "user"},
		{model.RoleAssistant, "assistant"},
		{model.RoleTool, "tool"},
	}

	for _, tt := range tests {
		if string(tt.role) != tt.want {
			t.Errorf("Role = %q, want %q", tt.role, tt.want)
		}
	}
}

// TestPartImplementations verifies that all Part types implement the
// Part interface via their unexported marker method.
func TestPartImplementations(t *testing.T) {
	parts := []model.Part{
		model.TextPart{Text: "hello"},
		model.ImagePart{URL: "https://example.com/img.png"},
		model.ToolCallPart{ToolCallID: "1", ToolName: "test", Args: json.RawMessage(`{}`)},
		model.ToolResultPart{ToolCallID: "1", ToolName: "test", Content: "ok"},
	}

	if len(parts) != 4 {
		t.Errorf("expected 4 part types, got %d", len(parts))
	}
}

// TestContentImplementations verifies that all Content types implement
// the Content interface.
func TestContentImplementations(t *testing.T) {
	contents := []model.Content{
		model.TextContent{Text: "hello"},
		model.ToolCallContent{ToolCallID: "1", ToolName: "test", Args: "{}"},
	}

	if len(contents) != 2 {
		t.Errorf("expected 2 content types, got %d", len(contents))
	}
}

// TestChunkImplementations verifies that all Chunk types implement
// the Chunk interface.
func TestChunkImplementations(t *testing.T) {
	chunks := []model.Chunk{
		model.TextDelta{Text: "hi"},
		model.ToolCallDelta{ToolCallID: "1", ToolName: "test", ArgsJSON: "{}"},
		model.FinishChunk{FinishReason: model.FinishReasonStop},
		model.UsageChunk{Usage: model.Usage{InputTokens: 10, OutputTokens: 5}},
		model.ProviderMetadataChunk{Metadata: model.ProviderMetadata{RequestID: "req_1"}},
	}

	if len(chunks) != 5 {
		t.Errorf("expected 5 chunk types, got %d", len(chunks))
	}
}

func TestProviderOptionsClone(t *testing.T) {
	original := model.ProviderOptions{
		"anthropic": {
			"metadata": json.RawMessage(`{"trace_id":"abc"}`),
		},
	}

	cloned := original.Clone()
	cloned["anthropic"]["metadata"][0] = '['

	got, ok := original.Lookup("anthropic", "metadata")
	if !ok {
		t.Fatal("Lookup returned false")
	}

	if string(got) != `{"trace_id":"abc"}` {
		t.Errorf("original metadata = %s, want unchanged", got)
	}
}

func TestProviderMetadataClone(t *testing.T) {
	original := model.ProviderMetadata{
		RequestID: "req_1",
		ModelID:   "model_1",
		Provider: map[string]string{
			"stop_reason": "end_turn",
		},
	}

	cloned := original.Clone()
	cloned.Provider["stop_reason"] = "max_tokens"

	if original.Provider["stop_reason"] != "end_turn" {
		t.Errorf("original provider metadata mutated = %q, want end_turn", original.Provider["stop_reason"])
	}
}

func TestProviderOptionsLookupReturnsCopy(t *testing.T) {
	original := model.ProviderOptions{
		"openai": {
			"metadata": json.RawMessage(`{"trace_id":"trace-1"}`),
		},
	}

	got, ok := original.Lookup("openai", "metadata")
	if !ok {
		t.Fatal("Lookup returned false")
	}

	got[0] = '['

	again, ok := original.Lookup("openai", "metadata")
	if !ok {
		t.Fatal("Lookup returned false on second read")
	}

	if string(again) != `{"trace_id":"trace-1"}` {
		t.Errorf("Lookup mutated original value = %s, want unchanged", again)
	}
}

func TestProviderMetadataChunkImplementation(t *testing.T) {
	chunks := []model.Chunk{
		model.ProviderMetadataChunk{Metadata: model.ProviderMetadata{RequestID: "req_2"}},
	}

	if len(chunks) != 1 {
		t.Errorf("expected 1 chunk type, got %d", len(chunks))
	}
}

// TestMessageConstruction verifies that messages can be constructed
// with multiple parts in different combinations.
func TestMessageConstruction(t *testing.T) {
	msg := model.Message{
		Role: model.RoleUser,
		Parts: []model.Part{
			model.TextPart{Text: "Describe this image:"},
			model.ImagePart{
				URL:      "https://example.com/photo.jpg",
				MIMEType: "image/jpeg",
			},
		},
	}

	if msg.Role != model.RoleUser {
		t.Errorf("Role = %q, want %q", msg.Role, model.RoleUser)
	}

	if len(msg.Parts) != 2 {
		t.Errorf("Parts length = %d, want 2", len(msg.Parts))
	}
}

// TestResponseFormatJSONSchema verifies the response format can be
// constructed with a JSON Schema specification.
func TestResponseFormatJSONSchema(t *testing.T) {
	rf := &model.ResponseFormat{
		Type: "json_schema",
		JSONSchema: &model.JSONSchema{
			Name:   "recipe",
			Schema: json.RawMessage(`{"type":"object"}`),
			Strict: true,
		},
	}

	if rf.Type != "json_schema" {
		t.Errorf("Type = %q, want %q", rf.Type, "json_schema")
	}

	if rf.JSONSchema.Name != "recipe" {
		t.Errorf("JSONSchema.Name = %q, want %q", rf.JSONSchema.Name, "recipe")
	}

	if !rf.JSONSchema.Strict {
		t.Error("JSONSchema.Strict = false, want true")
	}
}
