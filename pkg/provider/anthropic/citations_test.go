package anthropic_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/anthropic"
)

// TestEncodeRequestCitationsEnabledOnDocuments verifies the citations
// provider option stamps {"enabled":true} on every document block.
func TestEncodeRequestCitationsEnabledOnDocuments(t *testing.T) {
	req := &model.Request{
		Messages: []model.Message{{
			Role: model.RoleUser,
			Parts: []model.Part{
				model.FilePart{
					Data:     []byte("%PDF-1.4"),
					MIMEType: "application/pdf",
				},
				model.TextPart{Text: "Summarize this document"},
			},
		}},
		ProviderOptions: model.ProviderOptions{
			"anthropic": model.ProviderOptionSet{
				"citations": json.RawMessage(`true`),
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

	var raw struct {
		Messages []struct {
			Content []struct {
				Type      string `json:"type"`
				Citations *struct {
					Enabled bool `json:"enabled"`
				} `json:"citations"`
			} `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if len(raw.Messages) != 1 || len(raw.Messages[0].Content) != 2 {
		t.Fatalf("messages = %+v, want 1 message with 2 blocks", raw.Messages)
	}

	document := raw.Messages[0].Content[0]
	if document.Type != "document" {
		t.Fatalf("block 0 type = %q, want document", document.Type)
	}
	if document.Citations == nil || !document.Citations.Enabled {
		t.Errorf("document citations = %+v, want enabled", document.Citations)
	}

	text := raw.Messages[0].Content[1]
	if text.Citations != nil {
		t.Errorf("text block should not carry citations config, got %+v", text.Citations)
	}
}

// TestDecodeResponseCitations verifies text block citations decode into
// SourceContent entries following the text.
func TestDecodeResponseCitations(t *testing.T) {
	const body = `{
		"id": "msg_1",
		"type": "message",
		"role": "assistant",
		"model": "claude-sonnet-4-5",
		"content": [
			{
				"type": "text",
				"text": "The sky is blue.",
				"citations": [
					{
						"type": "char_location",
						"cited_text": "the sky appears blue",
						"document_index": 0,
						"document_title": "Atmospheric Optics",
						"start_char_index": 12,
						"end_char_index": 32
					},
					{
						"type": "web_search_result_location",
						"cited_text": "rayleigh scattering",
						"url": "https://example.com/sky",
						"title": "Why is the sky blue?"
					}
				]
			}
		],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 10, "output_tokens": 5}
	}`

	var msgResp anthropic.MessageResponse
	if err := json.Unmarshal([]byte(body), &msgResp); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}

	resp := anthropic.DecodeResponse(&msgResp)

	if len(resp.Content) != 3 {
		t.Fatalf("content length = %d, want 3 (text + 2 sources)", len(resp.Content))
	}

	text, ok := resp.Content[0].(model.TextContent)
	if !ok || text.Text != "The sky is blue." {
		t.Fatalf("content[0] = %+v, want text segment", resp.Content[0])
	}

	docSource, ok := resp.Content[1].(model.SourceContent)
	if !ok {
		t.Fatalf("content[1] type = %T, want SourceContent", resp.Content[1])
	}
	if docSource.SourceType != model.SourceTypeDocument {
		t.Errorf("source type = %q, want document", docSource.SourceType)
	}
	if docSource.Title != "Atmospheric Optics" {
		t.Errorf("title = %q, want Atmospheric Optics", docSource.Title)
	}
	if docSource.StartIndex != 12 || docSource.EndIndex != 32 {
		t.Errorf("indices = %d..%d, want 12..32", docSource.StartIndex, docSource.EndIndex)
	}

	urlSource, ok := resp.Content[2].(model.SourceContent)
	if !ok {
		t.Fatalf("content[2] type = %T, want SourceContent", resp.Content[2])
	}
	if urlSource.SourceType != model.SourceTypeURL {
		t.Errorf("source type = %q, want url", urlSource.SourceType)
	}
	if urlSource.URL != "https://example.com/sky" {
		t.Errorf("url = %q, want citation url", urlSource.URL)
	}
	if urlSource.Title != "Why is the sky blue?" {
		t.Errorf("title = %q, want fallback to citation title", urlSource.Title)
	}
}

// TestDecodeStreamCitationsDelta verifies citations_delta events yield
// SourceChunk values.
func TestDecodeStreamCitationsDelta(t *testing.T) {
	data := strings.Join([]string{
		"event: content_block_start",
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		"",
		"event: content_block_delta",
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Blue."}}`,
		"",
		"event: content_block_delta",
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"char_location","cited_text":"the sky appears blue","document_index":0,"document_title":"Atmospheric Optics","start_char_index":3,"end_char_index":9}}}`,
		"",
		"event: message_delta",
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}`,
		"",
	}, "\n")

	var (
		sources []model.SourceChunk
		texts   []model.TextDelta
	)
	for chunk := range anthropic.DecodeStream(strings.NewReader(data)) {
		switch c := chunk.(type) {
		case model.SourceChunk:
			sources = append(sources, c)
		case model.TextDelta:
			texts = append(texts, c)
		}
	}

	if len(texts) != 1 || texts[0].Text != "Blue." {
		t.Errorf("text deltas = %+v, want [Blue.]", texts)
	}

	if len(sources) != 1 {
		t.Fatalf("source chunks = %d, want 1", len(sources))
	}

	source := sources[0]
	if source.SourceType != model.SourceTypeDocument {
		t.Errorf("source type = %q, want document", source.SourceType)
	}
	if source.Title != "Atmospheric Optics" {
		t.Errorf("title = %q, want Atmospheric Optics", source.Title)
	}
	if source.StartIndex != 3 || source.EndIndex != 9 {
		t.Errorf("indices = %d..%d, want 3..9", source.StartIndex, source.EndIndex)
	}
}
