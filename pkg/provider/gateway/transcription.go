package gateway

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"slices"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// transcriptionModelPath is the transcription model endpoint.
const transcriptionModelPath = "/transcription-model"

// xaiNamespaces are the provider option namespaces of xAI transcription
// models.
var xaiNamespaces = []string{
	"xai",
	"spacexai",
}

// TranscriptionModel is an AI Gateway transcription model. It is safe for
// concurrent use.
type TranscriptionModel struct {
	modelID  string
	provider *Provider
}

// Provider returns the provider identifier, "gateway" by default.
func (m *TranscriptionModel) Provider() string {
	return m.provider.name
}

// ModelID returns the model identifier.
func (m *TranscriptionModel) ModelID() string {
	return m.modelID
}

type transcriptionRequest struct {
	Audio           string                `json:"audio"`
	MediaType       string                `json:"mediaType"`
	ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
}

type transcriptionResponse struct {
	Text              string                 `json:"text"`
	Segments          []transcriptionSegment `json:"segments"`
	Language          *string                `json:"language"`
	DurationInSeconds *float64               `json:"durationInSeconds"`
	Warnings          []wireWarning          `json:"warnings"`
	Usage             json.RawMessage        `json:"usage"`
	ProviderMetadata  wireMetadata           `json:"providerMetadata"`
}

type transcriptionSegment struct {
	Text        string  `json:"text"`
	StartSecond float64 `json:"startSecond"`
	EndSecond   float64 `json:"endSecond"`
}

// diarizationOptions read the diarization flag of xAI options.
type diarizationOptions struct {
	Diarize *bool `json:"diarize"`
}

// Transcribe converts audio into text through the gateway.
func (m *TranscriptionModel) Transcribe(ctx context.Context, options model.TranscriptionOptions) (*model.TranscriptionResult, error) {
	request, method, err := m.provider.modelRequest(
		transcriptionModelPath,
		options.Headers,
		modelConfigHeaders("transcription", m.modelID),
	)
	if err != nil {
		return nil, err
	}

	response, err := apicall.PostJSON(ctx, request, transcriptionRequest{
		Audio:           base64.StdEncoding.EncodeToString(options.Audio),
		MediaType:       options.MediaType,
		ProviderOptions: options.ProviderOptions,
	})
	if err != nil {
		return nil, asGatewayError(err, method)
	}

	var decoded transcriptionResponse
	if err := apicall.DecodeJSON(response.Body, &decoded); err != nil {
		return nil, err
	}

	warnings := decodeWarnings(decoded.Warnings)
	if m.lacksRequestedDiarization(options.ProviderOptions, response.Body) {
		warnings = append(warnings, model.UnsupportedWarning(
			"providerOptions.xai.diarize",
			"AI Gateway does not currently expose xAI speaker diarization metadata.",
		))
	}

	return &model.TranscriptionResult{
		Text:             decoded.Text,
		Segments:         segments(decoded.Segments),
		Language:         stringValue(decoded.Language),
		Duration:         seconds(decoded.DurationInSeconds),
		Usage:            nullAsAbsent(decoded.Usage),
		ProviderMetadata: decoded.ProviderMetadata.providerMetadata(),
		Warnings:         warnings,
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

func segments(wire []transcriptionSegment) []model.TranscriptionSegment {
	converted := make([]model.TranscriptionSegment, 0, len(wire))

	for _, segment := range wire {
		converted = append(converted, model.TranscriptionSegment{
			Text:  segment.Text,
			Start: seconds(&segment.StartSecond),
			End:   seconds(&segment.EndSecond),
		})
	}

	return converted
}

func seconds(value *float64) time.Duration {
	if value == nil {
		return 0
	}

	return time.Duration(*value * float64(time.Second))
}

// lacksRequestedDiarization reports whether an xAI model was asked to
// diarize but the response has no speaker labels, which the gateway does
// not yet forward.
func (m *TranscriptionModel) lacksRequestedDiarization(options model.ProviderOptions, body []byte) bool {
	if !strings.HasPrefix(m.modelID, "xai/") && !strings.HasPrefix(m.modelID, "spacexai/") {
		return false
	}

	if !requestsDiarization(options) {
		return false
	}

	var document any
	if json.Unmarshal(body, &document) != nil {
		return true
	}

	return !containsSpeaker(document)
}

func requestsDiarization(options model.ProviderOptions) bool {
	for _, namespace := range xaiNamespaces {
		diarization, _, err := model.ProviderValue[diarizationOptions](options, namespace)
		if err == nil && diarization.Diarize != nil && *diarization.Diarize {
			return true
		}
	}

	return false
}

// containsSpeaker searches a JSON document of unknown shape for a non-null
// "speaker" member.
func containsSpeaker(value any) bool {
	switch value := value.(type) {
	case []any:
		if slices.ContainsFunc(value, containsSpeaker) {
			return true
		}

	case map[string]any:
		for key, nested := range value {
			if key == "speaker" && nested != nil {
				return true
			}

			if containsSpeaker(nested) {
				return true
			}
		}
	}

	return false
}

var _ model.TranscriptionModel = (*TranscriptionModel)(nil)
