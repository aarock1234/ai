package openai_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/openai"
)

const samplePNGBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

// TestDecodeResponseOpenRouterImageOutput verifies that OpenRouter's
// multimodal image response (message.images[].image_url.url) is decoded
// into FileContent segments.
func TestDecodeResponseOpenRouterImageOutput(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		resp := `{
			"id": "chatcmpl-image-1",
			"model": "google/gemini-2.5-flash-image-preview",
			"choices": [{
				"index": 0,
				"message": {
					"role": "assistant",
					"content": "Here is the image you asked for.",
					"images": [
						{
							"type": "image_url",
							"image_url": {
								"url": "data:image/png;base64,` + samplePNGBase64 + `"
							}
						}
					]
				},
				"finish_reason": "stop"
			}],
			"usage": {
				"prompt_tokens": 15,
				"completion_tokens": 7,
				"completion_tokens_details": {
					"image_tokens": 5
				}
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

	chatReq, err := openai.EncodeRequest("google/gemini-2.5-flash-image-preview", &model.Request{
		Messages: []model.Message{
			{
				Role:  model.RoleUser,
				Parts: []model.Part{model.TextPart{Text: "Generate an image."}},
			},
		},
		Modalities: []model.Modality{model.ModalityText, model.ModalityImage},
	}, false)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	chatResp, err := openai.DoRequest(context.Background(), cfg, chatReq)
	if err != nil {
		t.Fatalf("DoRequest: %v", err)
	}

	resp := openai.DecodeResponse(chatResp)

	var (
		textFound bool
		fileFound bool
	)
	for _, segment := range resp.Content {
		switch c := segment.(type) {
		case model.TextContent:
			textFound = true
			if c.Text != "Here is the image you asked for." {
				t.Errorf("text = %q, want expected copy", c.Text)
			}
		case model.FileContent:
			fileFound = true
			if c.MIMEType != "image/png" {
				t.Errorf("mime = %q, want image/png", c.MIMEType)
			}
			if len(c.Data) == 0 {
				t.Error("file data is empty")
			}
		}
	}

	if !textFound {
		t.Error("expected text content")
	}
	if !fileFound {
		t.Error("expected file content from images[] field")
	}

	if resp.Usage.ImageOutputTokens != 5 {
		t.Errorf("image output tokens = %d, want 5", resp.Usage.ImageOutputTokens)
	}
}

// TestDecodeResponseAudioOutput verifies that OpenAI's gpt-audio response
// (message.audio.data) is decoded into FileContent + TextContent
// (transcript).
func TestDecodeResponseAudioOutput(t *testing.T) {
	audio := base64.StdEncoding.EncodeToString([]byte("audio-bytes"))
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		resp := `{
			"id": "chatcmpl-audio-1",
			"model": "gpt-audio",
			"choices": [{
				"index": 0,
				"message": {
					"role": "assistant",
					"audio": {
						"id": "audio-xyz",
						"data": "` + audio + `",
						"transcript": "Hello there!",
						"format": "wav"
					}
				},
				"finish_reason": "stop"
			}],
			"usage": {
				"prompt_tokens": 5,
				"completion_tokens": 4,
				"completion_tokens_details": {
					"audio_tokens": 3
				}
			}
		}`
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(resp))
	}))
	defer server.Close()

	cfg := &openai.ClientConfig{
		BaseURL: server.URL,
		APIKey:  "k",
	}
	chatReq, err := openai.EncodeRequest("gpt-audio", &model.Request{
		Messages: []model.Message{{
			Role:  model.RoleUser,
			Parts: []model.Part{model.TextPart{Text: "say hi"}},
		}},
		Modalities: []model.Modality{model.ModalityText, model.ModalityAudio},
		Audio: &model.AudioOptions{
			Voice:  "alloy",
			Format: "wav",
		},
	}, false)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	chatResp, err := openai.DoRequest(context.Background(), cfg, chatReq)
	if err != nil {
		t.Fatalf("DoRequest: %v", err)
	}

	resp := openai.DecodeResponse(chatResp)

	var audioFile *model.FileContent
	var transcript string
	for _, segment := range resp.Content {
		switch c := segment.(type) {
		case model.FileContent:
			audioFile = &c
		case model.TextContent:
			transcript = c.Text
		}
	}

	if audioFile == nil {
		t.Fatal("expected audio file content")
	}
	if audioFile.MIMEType != "audio/wav" {
		t.Errorf("mime = %q, want audio/wav", audioFile.MIMEType)
	}
	if string(audioFile.Data) != "audio-bytes" {
		t.Errorf("data = %q, want audio-bytes", audioFile.Data)
	}
	if audioFile.Metadata.Provider["audio_id"] != "audio-xyz" {
		t.Errorf("audio_id = %q, want audio-xyz", audioFile.Metadata.Provider["audio_id"])
	}

	if transcript != "Hello there!" {
		t.Errorf("transcript = %q, want Hello there!", transcript)
	}

	if resp.Usage.AudioOutputTokens != 3 {
		t.Errorf("audio output tokens = %d, want 3", resp.Usage.AudioOutputTokens)
	}
}

// TestDecodeResponseReasoningDetails verifies that OpenRouter's
// reasoning_details[] array is decoded into ReasoningContent blocks,
// preserving signature and format so the block can round-trip back.
func TestDecodeResponseReasoningDetails(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		resp := `{
			"id": "cmpl-r1",
			"model": "anthropic/claude-sonnet-4",
			"choices": [{
				"index": 0,
				"message": {
					"role": "assistant",
					"content": "Final answer.",
					"reasoning_details": [
						{
							"type": "reasoning.text",
							"text": "step one",
							"signature": "sig-a",
							"id": "r-1",
							"format": "anthropic-claude-v1"
						},
						{
							"type": "reasoning.encrypted",
							"data": "opaque-blob",
							"id": "r-2"
						}
					]
				},
				"finish_reason": "stop"
			}],
			"usage": {
				"prompt_tokens": 10,
				"completion_tokens": 20,
				"completion_tokens_details": {
					"reasoning_tokens": 14
				}
			}
		}`
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(resp))
	}))
	defer server.Close()

	cfg := &openai.ClientConfig{
		BaseURL: server.URL,
		APIKey:  "k",
	}
	chatReq, _ := openai.EncodeRequest("anthropic/claude-sonnet-4", &model.Request{
		Messages: []model.Message{{
			Role:  model.RoleUser,
			Parts: []model.Part{model.TextPart{Text: "think"}},
		}},
	}, false)

	chatResp, err := openai.DoRequest(context.Background(), cfg, chatReq)
	if err != nil {
		t.Fatalf("DoRequest: %v", err)
	}

	resp := openai.DecodeResponse(chatResp)

	var reasoningBlocks []model.ReasoningContent
	var text string
	for _, segment := range resp.Content {
		switch c := segment.(type) {
		case model.ReasoningContent:
			reasoningBlocks = append(reasoningBlocks, c)
		case model.TextContent:
			text = c.Text
		}
	}

	if len(reasoningBlocks) != 2 {
		t.Fatalf("reasoning blocks = %d, want 2", len(reasoningBlocks))
	}

	if reasoningBlocks[0].Text != "step one" {
		t.Errorf("block 0 text = %q, want step one", reasoningBlocks[0].Text)
	}
	if reasoningBlocks[0].Signature != "sig-a" {
		t.Errorf("block 0 signature = %q, want sig-a", reasoningBlocks[0].Signature)
	}
	if reasoningBlocks[0].Format != "anthropic-claude-v1" {
		t.Errorf("block 0 format = %q, want anthropic-claude-v1", reasoningBlocks[0].Format)
	}

	if !reasoningBlocks[1].Redacted {
		t.Error("block 1 should be redacted")
	}

	if text != "Final answer." {
		t.Errorf("text = %q, want Final answer.", text)
	}

	if resp.Usage.ReasoningTokens != 14 {
		t.Errorf("reasoning tokens = %d, want 14", resp.Usage.ReasoningTokens)
	}
}

// TestDecodeResponseAnnotations verifies that URL citation annotations
// are decoded into SourceContent segments.
func TestDecodeResponseAnnotations(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		resp := `{
			"id": "cmpl-search",
			"choices": [{
				"index": 0,
				"message": {
					"role": "assistant",
					"content": "According to recent news...",
					"annotations": [
						{
							"type": "url_citation",
							"url_citation": {
								"url": "https://example.com/article",
								"title": "Breaking News",
								"start_index": 10,
								"end_index": 24
							}
						}
					]
				},
				"finish_reason": "stop"
			}],
			"usage": {"prompt_tokens": 5, "completion_tokens": 10}
		}`
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(resp))
	}))
	defer server.Close()

	cfg := &openai.ClientConfig{
		BaseURL: server.URL,
		APIKey:  "k",
	}
	chatReq, _ := openai.EncodeRequest("gpt-5-search", &model.Request{
		Messages: []model.Message{{
			Role:  model.RoleUser,
			Parts: []model.Part{model.TextPart{Text: "news"}},
		}},
	}, false)

	chatResp, _ := openai.DoRequest(context.Background(), cfg, chatReq)
	resp := openai.DecodeResponse(chatResp)

	var source *model.SourceContent
	for _, segment := range resp.Content {
		if s, ok := segment.(model.SourceContent); ok {
			source = &s
		}
	}

	if source == nil {
		t.Fatal("expected source content from annotations")
	}
	if source.SourceType != model.SourceTypeURL {
		t.Errorf("source type = %q, want url", source.SourceType)
	}
	if source.URL != "https://example.com/article" {
		t.Errorf("url = %q, want example article", source.URL)
	}
	if source.Title != "Breaking News" {
		t.Errorf("title = %q, want Breaking News", source.Title)
	}
	if source.StartIndex != 10 || source.EndIndex != 24 {
		t.Errorf("indices = (%d,%d), want (10,24)", source.StartIndex, source.EndIndex)
	}
}

// TestDecodeResponseRefusal verifies that message.refusal surfaces as
// a warning rather than being silently dropped.
func TestDecodeResponseRefusal(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		resp := `{
			"id": "cmpl-refuse",
			"choices": [{
				"index": 0,
				"message": {
					"role": "assistant",
					"refusal": "i can't help with that"
				},
				"finish_reason": "stop"
			}],
			"usage": {"prompt_tokens": 1, "completion_tokens": 0}
		}`
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(resp))
	}))
	defer server.Close()

	cfg := &openai.ClientConfig{
		BaseURL: server.URL,
		APIKey:  "k",
	}
	chatReq, _ := openai.EncodeRequest("gpt-5.4", &model.Request{
		Messages: []model.Message{{
			Role:  model.RoleUser,
			Parts: []model.Part{model.TextPart{Text: "do bad thing"}},
		}},
	}, false)

	chatResp, _ := openai.DoRequest(context.Background(), cfg, chatReq)
	resp := openai.DecodeResponse(chatResp)

	if len(resp.Warnings) == 0 {
		t.Fatal("expected refusal warning")
	}
	if !strings.Contains(resp.Warnings[0], "refusal") {
		t.Errorf("warning = %q, want to include 'refusal'", resp.Warnings[0])
	}
}

// TestDecodeResponseUsageBreakdown verifies that prompt/completion
// detail fields populate the per-modality usage counters.
func TestDecodeResponseUsageBreakdown(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		resp := `{
			"id": "cmpl-usage",
			"choices": [{
				"index": 0,
				"message": {"role": "assistant", "content": "hi"},
				"finish_reason": "stop"
			}],
			"usage": {
				"prompt_tokens": 100,
				"completion_tokens": 50,
				"prompt_tokens_details": {
					"cached_tokens": 20,
					"audio_tokens": 5,
					"image_tokens": 2
				},
				"completion_tokens_details": {
					"reasoning_tokens": 30,
					"audio_tokens": 4,
					"image_tokens": 3
				}
			}
		}`
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(resp))
	}))
	defer server.Close()

	cfg := &openai.ClientConfig{
		BaseURL: server.URL,
		APIKey:  "k",
	}
	chatReq, _ := openai.EncodeRequest("gpt-5.4", &model.Request{
		Messages: []model.Message{{
			Role:  model.RoleUser,
			Parts: []model.Part{model.TextPart{Text: "hi"}},
		}},
	}, false)

	chatResp, _ := openai.DoRequest(context.Background(), cfg, chatReq)
	resp := openai.DecodeResponse(chatResp)

	want := model.Usage{
		InputTokens:       100,
		OutputTokens:      50,
		CachedInputTokens: 20,
		ReasoningTokens:   30,
		ImageInputTokens:  2,
		ImageOutputTokens: 3,
		AudioInputTokens:  5,
		AudioOutputTokens: 4,
	}
	if resp.Usage != want {
		t.Errorf("usage = %+v, want %+v", resp.Usage, want)
	}
}

// TestEncodeRequestModalities verifies that the modalities slice is
// serialized as a plain string array on the wire.
func TestEncodeRequestModalities(t *testing.T) {
	req := &model.Request{
		Messages: []model.Message{{
			Role:  model.RoleUser,
			Parts: []model.Part{model.TextPart{Text: "hi"}},
		}},
		Modalities: []model.Modality{model.ModalityText, model.ModalityImage},
		Audio: &model.AudioOptions{
			Voice:  "alloy",
			Format: "wav",
		},
		Reasoning: &model.ReasoningOptions{
			Effort:    "high",
			MaxTokens: 2000,
		},
		Prediction: &model.PredictionOptions{
			Content: "predicted body",
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

	mods, ok := raw["modalities"].([]any)
	if !ok {
		t.Fatalf("modalities = %v, want []any", raw["modalities"])
	}
	if len(mods) != 2 || mods[0] != "text" || mods[1] != "image" {
		t.Errorf("modalities = %v, want [text image]", mods)
	}

	audio, ok := raw["audio"].(map[string]any)
	if !ok {
		t.Fatal("audio field missing")
	}
	if audio["voice"] != "alloy" || audio["format"] != "wav" {
		t.Errorf("audio = %v, want alloy/wav", audio)
	}

	reasoning, ok := raw["reasoning"].(map[string]any)
	if !ok {
		t.Fatal("reasoning field missing")
	}
	if reasoning["effort"] != "high" || reasoning["max_tokens"] != float64(2000) {
		t.Errorf("reasoning = %v, want high/2000", reasoning)
	}

	prediction, ok := raw["prediction"].(map[string]any)
	if !ok {
		t.Fatal("prediction field missing")
	}
	if prediction["type"] != "content" || prediction["content"] != "predicted body" {
		t.Errorf("prediction = %v, want content/predicted body", prediction)
	}
}

// TestEncodeRequestReasoningPartRoundTrip verifies that a prior
// reasoning turn is attached to the outgoing assistant message via
// reasoning_details, preserving signature + format.
func TestEncodeRequestReasoningPartRoundTrip(t *testing.T) {
	req := &model.Request{
		Messages: []model.Message{
			{
				Role:  model.RoleUser,
				Parts: []model.Part{model.TextPart{Text: "continue"}},
			},
			{
				Role: model.RoleAssistant,
				Parts: []model.Part{
					model.ReasoningPart{
						Text:      "prior thinking",
						Signature: "sig-xyz",
						ID:        "r-1",
						Format:    "reasoning.text",
					},
					model.TextPart{Text: "partial answer"},
				},
			},
		},
	}

	chatReq, err := openai.EncodeRequest("openrouter/deepseek-r1", req, false)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	body, _ := openai.MarshalRequest(chatReq)

	var raw struct {
		Messages []json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if len(raw.Messages) != 2 {
		t.Fatalf("messages = %d, want 2", len(raw.Messages))
	}

	var assistant map[string]any
	if err := json.Unmarshal(raw.Messages[1], &assistant); err != nil {
		t.Fatalf("unmarshal assistant: %v", err)
	}

	details, ok := assistant["reasoning_details"].([]any)
	if !ok {
		t.Fatalf("reasoning_details missing on assistant message: %v", assistant)
	}
	if len(details) != 1 {
		t.Fatalf("reasoning_details len = %d, want 1", len(details))
	}

	detail := details[0].(map[string]any)
	if detail["signature"] != "sig-xyz" {
		t.Errorf("signature = %v, want sig-xyz", detail["signature"])
	}
	if detail["id"] != "r-1" {
		t.Errorf("id = %v, want r-1", detail["id"])
	}
	if detail["text"] != "prior thinking" {
		t.Errorf("text = %v, want prior thinking", detail["text"])
	}
}

// TestDecodeStreamImageOutput verifies streaming image chunks are
// emitted as FileChunks.
func TestDecodeStreamImageOutput(t *testing.T) {
	data := `data: {"id":"cmpl-1","choices":[{"index":0,"delta":{"role":"assistant","images":[{"type":"image_url","image_url":{"url":"data:image/png;base64,` + samplePNGBase64 + `"}}]}}]}

data: {"id":"cmpl-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`
	var fileChunk *model.FileChunk
	for chunk := range openai.DecodeStream(strings.NewReader(data)) {
		if fc, ok := chunk.(model.FileChunk); ok {
			fileChunk = &fc
		}
	}

	if fileChunk == nil {
		t.Fatal("expected FileChunk")
	}
	if fileChunk.MIMEType != "image/png" {
		t.Errorf("mime = %q, want image/png", fileChunk.MIMEType)
	}
	if len(fileChunk.Data) == 0 {
		t.Error("file data is empty")
	}
}

// TestDecodeStreamReasoningDetails verifies streamed reasoning_details
// produce ReasoningDelta chunks with the provider's signature/format.
func TestDecodeStreamReasoningDetails(t *testing.T) {
	data := `data: {"id":"cmpl-r","choices":[{"index":0,"delta":{"reasoning_details":[{"type":"reasoning.text","text":"first ","id":"r-1","format":"anthropic-claude-v1"}]}}]}

data: {"id":"cmpl-r","choices":[{"index":0,"delta":{"reasoning_details":[{"type":"reasoning.text","text":"thought","id":"r-1","format":"anthropic-claude-v1","signature":"sig-1"}]}}]}

data: {"id":"cmpl-r","choices":[{"index":0,"delta":{"content":"answer"}}]}

data: {"id":"cmpl-r","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`
	var reasoningDeltas []model.ReasoningDelta
	for chunk := range openai.DecodeStream(strings.NewReader(data)) {
		if rd, ok := chunk.(model.ReasoningDelta); ok {
			reasoningDeltas = append(reasoningDeltas, rd)
		}
	}

	if len(reasoningDeltas) != 2 {
		t.Fatalf("reasoning deltas = %d, want 2", len(reasoningDeltas))
	}
	if reasoningDeltas[0].Text != "first " {
		t.Errorf("delta 0 text = %q, want 'first '", reasoningDeltas[0].Text)
	}
	if reasoningDeltas[0].ID != "r-1" {
		t.Errorf("delta 0 id = %q, want r-1", reasoningDeltas[0].ID)
	}
	if reasoningDeltas[1].Signature != "sig-1" {
		t.Errorf("delta 1 signature = %q, want sig-1", reasoningDeltas[1].Signature)
	}
}

// TestDecodeStreamAnnotations verifies streamed annotations produce
// SourceChunk values.
func TestDecodeStreamAnnotations(t *testing.T) {
	data := `data: {"id":"cmpl-1","choices":[{"index":0,"delta":{"content":"per source","annotations":[{"type":"url_citation","url_citation":{"url":"https://example.com","title":"Example"}}]}}]}

data: {"id":"cmpl-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`
	var source *model.SourceChunk
	for chunk := range openai.DecodeStream(strings.NewReader(data)) {
		if s, ok := chunk.(model.SourceChunk); ok {
			source = &s
		}
	}

	if source == nil {
		t.Fatal("expected source chunk")
	}
	if source.URL != "https://example.com" {
		t.Errorf("url = %q, want example.com", source.URL)
	}
}

// TestDecodeStreamAudioChunks verifies streamed audio deltas accumulate
// and produce a single FileChunk with the decoded bytes.
func TestDecodeStreamAudioChunks(t *testing.T) {
	first := base64.StdEncoding.EncodeToString([]byte("hello"))
	second := base64.StdEncoding.EncodeToString([]byte(" world"))

	data := `data: {"id":"c1","choices":[{"index":0,"delta":{"audio":{"id":"a1","format":"wav","data":"` + first + `"}}}]}

data: {"id":"c1","choices":[{"index":0,"delta":{"audio":{"data":"` + second + `","transcript":"hello world"}}}]}

data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`

	var fileChunk *model.FileChunk
	for chunk := range openai.DecodeStream(strings.NewReader(data)) {
		if fc, ok := chunk.(model.FileChunk); ok {
			fileChunk = &fc
		}
	}

	if fileChunk == nil {
		t.Fatal("expected audio FileChunk")
	}
	if fileChunk.MIMEType != "audio/wav" {
		t.Errorf("mime = %q, want audio/wav", fileChunk.MIMEType)
	}
	if fileChunk.Metadata.Provider["audio_id"] != "a1" {
		t.Errorf("audio_id = %q, want a1", fileChunk.Metadata.Provider["audio_id"])
	}
	// Confirm data was base64-decoded and concatenated.
	if string(fileChunk.Data) != "hello world" {
		t.Errorf("data = %q, want 'hello world'", fileChunk.Data)
	}
}
