package gateway

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// speechModelPath is the speech model endpoint.
const speechModelPath = "/speech-model"

// SpeechModel is an AI Gateway text-to-speech model. It is safe for
// concurrent use.
type SpeechModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "gateway" by default.
func (m *SpeechModel) Provider() string {
	return m.provider.name
}

// ModelID returns the model identifier.
func (m *SpeechModel) ModelID() string {
	return m.modelID
}

type speechRequest struct {
	Text            string                `json:"text"`
	Voice           string                `json:"voice,omitempty"`
	OutputFormat    string                `json:"outputFormat,omitempty"`
	Instructions    string                `json:"instructions,omitempty"`
	Speed           *float64              `json:"speed,omitempty"`
	Language        string                `json:"language,omitempty"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

type speechResponse struct {
	Audio            string          `json:"audio"`
	Warnings         []wireWarning   `json:"warnings"`
	Usage            json.RawMessage `json:"usage"`
	ProviderMetadata wireMetadata    `json:"providerMetadata"`
}

// Generate converts text into audio through the gateway.
func (m *SpeechModel) Generate(ctx context.Context, options model.SpeechOptions) (*model.SpeechResult, error) {
	request, method, err := m.provider.modelRequest(speechModelPath, options.Headers, modelConfigHeaders("speech", m.modelID))
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, speechRequest{
		Text:            options.Text,
		Voice:           options.Voice,
		OutputFormat:    options.OutputFormat,
		Instructions:    options.Instructions,
		Speed:           options.Speed,
		Language:        options.Language,
		ProviderOptions: options.ProviderOptions,
	})
	if err != nil {
		return nil, asGatewayError(err, method)
	}

	var decoded speechResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	audio, err := base64.StdEncoding.DecodeString(decoded.Audio)
	if err != nil {
		return nil, fmt.Errorf("%w: decode audio: %w", model.ErrInvalidResponse, err)
	}

	return &model.SpeechResult{
		Audio:            audio,
		Usage:            nullAsAbsent(decoded.Usage),
		ProviderMetadata: decoded.ProviderMetadata.providerMetadata(),
		Warnings:         decodeWarnings(decoded.Warnings),
		Request: model.RequestInfo{
			Body: response.RequestBody,
		},
		Response: model.ResponseInfo{
			ResponseMetadata: model.ResponseMetadata{
				Timestamp: time.Now(),
				ModelID:   m.modelID,
			},
			Headers: response.Headers,
			Body:    response.Body,
		},
	}, nil
}

// nullAsAbsent returns nil for absent or null JSON.
func nullAsAbsent(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}

	return raw
}

var _ model.SpeechModel = (*SpeechModel)(nil)
