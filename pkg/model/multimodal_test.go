package model_test

import (
	"errors"
	"testing"

	"github.com/aarock1234/ai/pkg/model"
)

// TestNewContentTypes verifies every content variant implements the
// Content interface and round-trips through the interface.
func TestNewContentTypes(t *testing.T) {
	segments := []model.Content{
		model.TextContent{Text: "hello"},
		model.ReasoningContent{
			Text:      "thinking",
			Signature: "sig",
		},
		model.FileContent{
			Data:     []byte{1, 2, 3},
			MIMEType: "image/png",
		},
		model.SourceContent{
			SourceType: model.SourceTypeURL,
			URL:        "https://example.com",
			Title:      "Example",
		},
		model.ToolCallContent{
			ToolCallID: "1",
			ToolName:   "t",
			Args:       "{}",
		},
		model.ToolResultContent{
			ToolCallID: "1",
			Content:    "result",
		},
	}

	if len(segments) != 6 {
		t.Errorf("content variants = %d, want 6", len(segments))
	}
}

// TestNewChunkTypes verifies every stream chunk variant implements
// the Chunk interface.
func TestNewChunkTypes(t *testing.T) {
	chunks := []model.Chunk{
		model.StreamStartChunk{Warnings: []string{"warn"}},
		model.TextDelta{Text: "hi"},
		model.ReasoningDelta{
			Text:      "step",
			Signature: "sig",
			Redacted:  false,
		},
		model.ToolCallDelta{
			ToolCallID: "1",
			ToolName:   "t",
			ArgsJSON:   "{}",
		},
		model.ToolResultChunk{
			ToolCallID: "1",
			Content:    "r",
		},
		model.FileChunk{
			Data:     []byte{1},
			MIMEType: "image/png",
		},
		model.SourceChunk{
			SourceType: model.SourceTypeURL,
			URL:        "https://example.com",
		},
		model.ProviderMetadataChunk{Metadata: model.ProviderMetadata{RequestID: "r"}},
		model.FinishChunk{FinishReason: model.FinishReasonStop},
		model.UsageChunk{Usage: model.Usage{InputTokens: 10}},
		model.ErrorChunk{Err: errors.New("boom")},
		model.RawChunk{Data: []byte(`{"x":1}`)},
	}

	if len(chunks) != 12 {
		t.Errorf("chunk variants = %d, want 12", len(chunks))
	}
}

// TestErrorChunkUnwrap verifies the error-chunk error interface.
func TestErrorChunkUnwrap(t *testing.T) {
	inner := errors.New("provider failure")
	chunk := model.ErrorChunk{Err: inner}

	if chunk.Error() != "provider failure" {
		t.Errorf("Error() = %q, want provider failure", chunk.Error())
	}

	if !errors.Is(chunk, inner) {
		t.Error("errors.Is should match the wrapped error")
	}
}

// TestErrorChunkZeroValue verifies the error chunk handles a nil
// inner error without panicking.
func TestErrorChunkZeroValue(t *testing.T) {
	chunk := model.ErrorChunk{}
	if chunk.Error() != "stream error" {
		t.Errorf("Error() = %q, want default message", chunk.Error())
	}
}

// TestFilePartClone verifies deep cloning of the new FilePart.
func TestFilePartClone(t *testing.T) {
	req := &model.Request{
		Messages: []model.Message{
			{
				Role: model.RoleUser,
				Parts: []model.Part{
					model.FilePart{
						Data:     []byte{0xde, 0xad},
						MIMEType: "application/pdf",
						Filename: "doc.pdf",
					},
				},
			},
		},
	}

	cloned := req.Clone()
	cloned.Messages[0].Parts[0].(model.FilePart).Data[0] = 0x00

	original := req.Messages[0].Parts[0].(model.FilePart)
	if original.Data[0] != 0xde {
		t.Errorf("original mutated: %v", original.Data)
	}
}

// TestReasoningPartClone verifies deep cloning of the new ReasoningPart.
func TestReasoningPartClone(t *testing.T) {
	req := &model.Request{
		Messages: []model.Message{
			{
				Role: model.RoleAssistant,
				Parts: []model.Part{
					model.ReasoningPart{
						Text:      "thinking",
						Signature: "sig",
						Format:    "anthropic-thinking",
					},
				},
			},
		},
	}

	cloned := req.Clone()
	part := cloned.Messages[0].Parts[0].(model.ReasoningPart)
	if part.Text != "thinking" {
		t.Errorf("reasoning text = %q, want thinking", part.Text)
	}
	if part.Signature != "sig" {
		t.Errorf("reasoning signature = %q, want sig", part.Signature)
	}
}

// TestRequestOptionsClone verifies Audio/Reasoning/Prediction clone
// behavior on the request.
func TestRequestOptionsClone(t *testing.T) {
	req := &model.Request{
		Modalities: []model.Modality{model.ModalityText, model.ModalityAudio},
		Audio: &model.AudioOptions{
			Voice:  "alloy",
			Format: "wav",
		},
		Reasoning: &model.ReasoningOptions{
			Effort:    "high",
			MaxTokens: 1000,
		},
		Prediction: &model.PredictionOptions{Content: "predicted"},
	}

	cloned := req.Clone()

	// Mutate clone, original must not change.
	cloned.Audio.Voice = "sage"
	cloned.Reasoning.Effort = "low"
	cloned.Prediction.Content = "other"
	cloned.Modalities[0] = model.ModalityImage

	if req.Audio.Voice != "alloy" {
		t.Errorf("audio voice mutated: %q", req.Audio.Voice)
	}
	if req.Reasoning.Effort != "high" {
		t.Errorf("reasoning effort mutated: %q", req.Reasoning.Effort)
	}
	if req.Prediction.Content != "predicted" {
		t.Errorf("prediction content mutated: %q", req.Prediction.Content)
	}
	if req.Modalities[0] != model.ModalityText {
		t.Errorf("modalities mutated: %v", req.Modalities)
	}
}
