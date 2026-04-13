package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strconv"

	"github.com/aarock1234/ai/internal/apierror"
	"github.com/aarock1234/ai/internal/provideropts"
	"github.com/aarock1234/ai/pkg/model"
)

// Compile-time checks that audio models implement their interfaces.
var (
	_ model.TranscriptionModel = (*TranscriptionModel)(nil)
	_ model.SpeechModel        = (*SpeechModel)(nil)
)

// TranscriptionModel is an OpenAI transcription model.
type TranscriptionModel struct {
	modelID      string
	clientConfig *ClientConfig
}

// Provider returns "openai".
func (m *TranscriptionModel) Provider() string { return "openai" }

// ModelID returns the model identifier.
func (m *TranscriptionModel) ModelID() string { return m.modelID }

// SpeechModel is an OpenAI speech model.
type SpeechModel struct {
	modelID      string
	clientConfig *ClientConfig
}

// Provider returns "openai".
func (m *SpeechModel) Provider() string { return "openai" }

// ModelID returns the model identifier.
func (m *SpeechModel) ModelID() string { return m.modelID }

type transcriptionResponse struct {
	Text     string                 `json:"text"`
	Language string                 `json:"language,omitempty"`
	Duration float64                `json:"duration,omitempty"`
	Segments []transcriptionSegment `json:"segments,omitempty"`
}

type transcriptionSegment struct {
	ID      json.RawMessage `json:"id"`
	Start   float64         `json:"start"`
	End     float64         `json:"end"`
	Text    string          `json:"text"`
	Speaker string          `json:"speaker,omitempty"`
}

type speechRequest struct {
	Model          string   `json:"model"`
	Input          string   `json:"input"`
	Voice          string   `json:"voice"`
	Instructions   string   `json:"instructions,omitempty"`
	ResponseFormat string   `json:"response_format,omitempty"`
	Speed          *float64 `json:"speed,omitempty"`
}

// DoTranscribe performs an OpenAI transcription request.
func (m *TranscriptionModel) DoTranscribe(ctx context.Context, req *model.TranscriptionRequest) (*model.TranscriptionResponse, error) {
	if req == nil {
		req = &model.TranscriptionRequest{}
	}

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)

	if err := writer.WriteField("model", m.modelID); err != nil {
		return nil, fmt.Errorf("writing model field: %w", err)
	}
	if req.Language != "" {
		if err := writer.WriteField("language", req.Language); err != nil {
			return nil, fmt.Errorf("writing language field: %w", err)
		}
	}
	if req.Prompt != "" {
		if err := writer.WriteField("prompt", req.Prompt); err != nil {
			return nil, fmt.Errorf("writing prompt field: %w", err)
		}
	}
	if req.Temperature != nil {
		if err := writer.WriteField("temperature", fmt.Sprintf("%f", *req.Temperature)); err != nil {
			return nil, fmt.Errorf("writing temperature field: %w", err)
		}
	}
	responseFormat := req.ResponseFormat
	if responseFormat == "" {
		responseFormat = "json"
	}
	if err := writer.WriteField("response_format", responseFormat); err != nil {
		return nil, fmt.Errorf("writing response_format field: %w", err)
	}

	filename := req.Audio.Filename
	if filename == "" {
		filename = "audio-input"
	}

	part, err := writer.CreateFormFile("file", filename)
	if err != nil {
		return nil, fmt.Errorf("creating audio form file: %w", err)
	}
	if _, err := part.Write(req.Audio.Data); err != nil {
		return nil, fmt.Errorf("writing audio data: %w", err)
	}

	if err := provideropts.WriteMultipartFields(writer, "openai", req.ProviderOptions,
		"model",
		"language",
		"prompt",
		"temperature",
		"response_format",
		"file",
	); err != nil {
		return nil, err
	}

	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("closing multipart writer: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, m.clientConfig.BaseURL+"/audio/transcriptions", bytes.NewReader(body.Bytes()))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	httpReq.Header.Set("Content-Type", writer.FormDataContentType())
	if m.clientConfig.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+m.clientConfig.APIKey)
	}
	for key, values := range m.clientConfig.ExtraHeaders {
		for _, value := range values {
			httpReq.Header.Set(key, value)
		}
	}
	for key, values := range req.Headers {
		for _, value := range values {
			httpReq.Header.Set(key, value)
		}
	}

	resp, err := m.clientConfig.httpClient().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("executing request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, apierror.New(resp.StatusCode, string(bodyBytes))
	}

	if isTextTranscriptionFormat(responseFormat) {
		return &model.TranscriptionResponse{
			Text: string(bodyBytes),
		}, nil
	}

	var result transcriptionResponse
	if err := json.Unmarshal(bodyBytes, &result); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}

	segments := make([]model.TranscriptionSegment, len(result.Segments))
	for i, segment := range result.Segments {
		segments[i] = model.TranscriptionSegment{
			ID:      transcriptionSegmentID(segment.ID),
			Start:   segment.Start,
			End:     segment.End,
			Text:    segment.Text,
			Speaker: segment.Speaker,
		}
	}

	return &model.TranscriptionResponse{
		Text:            result.Text,
		Language:        result.Language,
		DurationSeconds: result.Duration,
		Segments:        segments,
	}, nil
}

func isTextTranscriptionFormat(responseFormat string) bool {
	switch responseFormat {
	case "text", "srt", "vtt":
		return true
	default:
		return false
	}
}

func transcriptionSegmentID(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}

	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text
	}

	var id int
	if err := json.Unmarshal(raw, &id); err == nil {
		return strconv.Itoa(id)
	}

	var number float64
	if err := json.Unmarshal(raw, &number); err == nil {
		return strconv.FormatFloat(number, 'f', -1, 64)
	}

	return string(raw)
}

// DoGenerateSpeech performs an OpenAI text-to-speech request.
func (m *SpeechModel) DoGenerateSpeech(ctx context.Context, req *model.SpeechRequest) (*model.SpeechResponse, error) {
	if req == nil {
		req = &model.SpeechRequest{}
	}

	body, err := provideropts.JSONBody(speechRequest{
		Model:          m.modelID,
		Input:          req.Input,
		Voice:          req.Voice,
		Instructions:   req.Instructions,
		ResponseFormat: req.ResponseFormat,
		Speed:          req.Speed,
	}, "openai", req.ProviderOptions,
		"model",
		"input",
		"voice",
		"instructions",
		"response_format",
		"speed",
	)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, m.clientConfig.BaseURL+"/audio/speech", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	setHeaders(httpReq, m.clientConfig, req.Headers)

	resp, err := m.clientConfig.httpClient().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("executing request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, apierror.New(resp.StatusCode, string(bodyBytes))
	}

	return &model.SpeechResponse{
		Audio:    bodyBytes,
		MIMEType: resp.Header.Get("Content-Type"),
	}, nil
}
