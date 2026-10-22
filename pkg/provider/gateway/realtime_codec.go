package gateway

import (
	"encoding/json"
	"fmt"

	"github.com/aarock1234/ai/pkg/model"
)

// realtimeSessionConfig is the normalized session configuration on the
// wire.
type realtimeSessionConfig struct {
	Instructions             string                   `json:"instructions,omitempty"`
	Voice                    string                   `json:"voice,omitempty"`
	OutputModalities         []model.RealtimeModality `json:"outputModalities,omitempty"`
	InputAudioFormat         *realtimeAudioFormat     `json:"inputAudioFormat,omitempty"`
	InputAudioTranscription  *realtimeTranscription   `json:"inputAudioTranscription,omitempty"`
	OutputAudioTranscription *realtimeTranscription   `json:"outputAudioTranscription,omitempty"`
	OutputAudioFormat        *realtimeAudioFormat     `json:"outputAudioFormat,omitempty"`
	TurnDetection            *realtimeTurnDetection   `json:"turnDetection,omitempty"`
	Tools                    []realtimeToolDefinition `json:"tools,omitempty"`
	ProviderOptions          model.ProviderOptions    `json:"providerOptions,omitempty"`
}

type realtimeAudioFormat struct {
	Type string `json:"type"`
	Rate int    `json:"rate,omitempty"`
}

type realtimeTranscription struct {
	Model    string `json:"model,omitempty"`
	Language string `json:"language,omitempty"`
	Prompt   string `json:"prompt,omitempty"`
}

type realtimeTurnDetection struct {
	Type              model.RealtimeTurnDetectionType `json:"type"`
	Threshold         *float64                        `json:"threshold,omitempty"`
	SilenceDurationMs *int64                          `json:"silenceDurationMs,omitempty"`
	PrefixPaddingMs   *int64                          `json:"prefixPaddingMs,omitempty"`
}

type realtimeToolDefinition struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
}

func encodeRealtimeSessionConfig(config model.RealtimeSessionConfig) realtimeSessionConfig {
	return realtimeSessionConfig{
		Instructions:             config.Instructions,
		Voice:                    config.Voice,
		OutputModalities:         config.OutputModalities,
		InputAudioFormat:         realtimeAudioFormatOf(config.InputAudioFormat),
		InputAudioTranscription:  realtimeTranscriptionOf(config.InputAudioTranscription),
		OutputAudioTranscription: realtimeTranscriptionOf(config.OutputAudioTranscription),
		OutputAudioFormat:        realtimeAudioFormatOf(config.OutputAudioFormat),
		TurnDetection:            realtimeTurnDetectionOf(config.TurnDetection),
		Tools:                    realtimeToolsOf(config.Tools),
		ProviderOptions:          config.ProviderOptions,
	}
}

func realtimeAudioFormatOf(format *model.AudioFormat) *realtimeAudioFormat {
	if format == nil {
		return nil
	}

	return &realtimeAudioFormat{
		Type: format.Type,
		Rate: format.Rate,
	}
}

func realtimeTranscriptionOf(config *model.RealtimeTranscriptionConfig) *realtimeTranscription {
	if config == nil {
		return nil
	}

	return &realtimeTranscription{
		Model:    config.Model,
		Language: config.Language,
		Prompt:   config.Prompt,
	}
}

func realtimeTurnDetectionOf(detection *model.RealtimeTurnDetection) *realtimeTurnDetection {
	if detection == nil {
		return nil
	}

	wire := &realtimeTurnDetection{
		Type:      detection.Type,
		Threshold: detection.Threshold,
	}

	if detection.SilenceDuration != nil {
		wire.SilenceDurationMs = new(detection.SilenceDuration.Milliseconds())
	}

	if detection.PrefixPadding != nil {
		wire.PrefixPaddingMs = new(detection.PrefixPadding.Milliseconds())
	}

	return wire
}

func realtimeToolsOf(tools []model.RealtimeToolDefinition) []realtimeToolDefinition {
	if len(tools) == 0 {
		return nil
	}

	converted := make([]realtimeToolDefinition, len(tools))

	for index, tool := range tools {
		converted[index] = realtimeToolDefinition{
			Type:        "function",
			Name:        tool.Name,
			Description: tool.Description,
			Parameters:  tool.Parameters,
		}
	}

	return converted
}

// Normalized client events on the wire.
type (
	realtimeSessionEvent struct {
		Type    string                `json:"type"`
		Config  realtimeSessionConfig `json:"config"`
		EventID string                `json:"eventId,omitempty"`
	}

	realtimeEventIDEvent struct {
		Type    string `json:"type"`
		EventID string `json:"eventId,omitempty"`
	}

	realtimeContextAppendEvent struct {
		Type            string                `json:"type"`
		Content         string                `json:"content"`
		DelegationID    *string               `json:"delegationId"`
		EventID         string                `json:"eventId,omitempty"`
		ProviderOptions model.ProviderOptions `json:"providerOptions,omitempty"`
	}

	realtimeAudioAppendEvent struct {
		Type    string `json:"type"`
		Audio   string `json:"audio"`
		EventID string `json:"eventId,omitempty"`
	}

	realtimeTypeOnlyEvent struct {
		Type string `json:"type"`
	}

	realtimeItemCreateEvent struct {
		Type string           `json:"type"`
		Item realtimeWireItem `json:"item"`
	}

	realtimeWireItem struct {
		Type   string  `json:"type"`
		Role   string  `json:"role,omitempty"`
		Text   *string `json:"text,omitempty"`
		Audio  *string `json:"audio,omitempty"`
		CallID string  `json:"callId,omitempty"`
		Name   string  `json:"name,omitempty"`
		Output *string `json:"output,omitempty"`
	}

	realtimeTruncateEvent struct {
		Type         string `json:"type"`
		ItemID       string `json:"itemId"`
		ContentIndex int    `json:"contentIndex"`
		AudioEndMs   int64  `json:"audioEndMs"`
	}

	realtimeResponseCreateEvent struct {
		Type    string                   `json:"type"`
		Options *realtimeResponseOptions `json:"options,omitempty"`
	}

	realtimeResponseOptions struct {
		Modalities   []model.RealtimeModality `json:"modalities,omitempty"`
		Instructions string                   `json:"instructions,omitempty"`
		Metadata     map[string]any           `json:"metadata,omitempty"`
	}
)

// encodeRealtimeClientEvent encodes a normalized client event as the
// gateway accepts it.
func encodeRealtimeClientEvent(event model.RealtimeClientEvent) (json.RawMessage, error) {
	switch typed := event.(type) {
	case model.RealtimeSessionUpdate:
		return json.Marshal(realtimeSessionEvent{
			Type:    "session-update",
			Config:  encodeRealtimeSessionConfig(typed.Config),
			EventID: typed.EventID,
		})

	case model.RealtimeSessionStart:
		return json.Marshal(realtimeSessionEvent{
			Type:    "session-start",
			Config:  encodeRealtimeSessionConfig(typed.Config),
			EventID: typed.EventID,
		})

	case model.RealtimeSessionClose:
		return encodeEventIDEvent("session-close", typed.EventID)

	case model.RealtimeInputAudioMute:
		return encodeEventIDEvent("input-audio-mute", typed.EventID)

	case model.RealtimeInputAudioUnmute:
		return encodeEventIDEvent("input-audio-unmute", typed.EventID)

	case model.RealtimeContextAppend:
		return json.Marshal(realtimeContextAppendEvent{
			Type:            "context-append",
			Content:         typed.Content,
			DelegationID:    typed.DelegationID,
			EventID:         typed.EventID,
			ProviderOptions: typed.ProviderOptions,
		})

	case model.RealtimeInputAudioAppend:
		return json.Marshal(realtimeAudioAppendEvent{
			Type:    "input-audio-append",
			Audio:   typed.Audio,
			EventID: typed.EventID,
		})

	case model.RealtimeInputAudioCommit:
		return encodeTypeOnlyEvent("input-audio-commit")

	case model.RealtimeInputAudioClear:
		return encodeTypeOnlyEvent("input-audio-clear")

	case model.RealtimeConversationItemCreate:
		return encodeRealtimeItem(typed.Item)

	case model.RealtimeConversationItemTruncate:
		return json.Marshal(realtimeTruncateEvent{
			Type:         "conversation-item-truncate",
			ItemID:       typed.ItemID,
			ContentIndex: typed.ContentIndex,
			AudioEndMs:   typed.AudioEnd.Milliseconds(),
		})

	case model.RealtimeResponseCreate:
		return json.Marshal(realtimeResponseCreateEvent{
			Type:    "response-create",
			Options: realtimeResponseOptionsOf(typed.Options),
		})

	case model.RealtimeResponseCancel:
		return encodeTypeOnlyEvent("response-cancel")

	default:
		return nil, fmt.Errorf("%w: AI Gateway realtime client event %T", model.ErrUnsupported, event)
	}
}

func encodeEventIDEvent(eventType string, eventID string) (json.RawMessage, error) {
	return json.Marshal(realtimeEventIDEvent{
		Type:    eventType,
		EventID: eventID,
	})
}

func encodeTypeOnlyEvent(eventType string) (json.RawMessage, error) {
	return json.Marshal(realtimeTypeOnlyEvent{
		Type: eventType,
	})
}

func encodeRealtimeItem(item model.RealtimeConversationItem) (json.RawMessage, error) {
	var wireItem realtimeWireItem

	switch typed := item.(type) {
	case model.RealtimeTextMessage:
		wireItem = realtimeWireItem{
			Type: "text-message",
			Role: "user",
			Text: new(typed.Text),
		}

	case model.RealtimeAudioMessage:
		wireItem = realtimeWireItem{
			Type:  "audio-message",
			Role:  "user",
			Audio: new(typed.Audio),
		}

	case model.RealtimeFunctionCallOutput:
		wireItem = realtimeWireItem{
			Type:   "function-call-output",
			CallID: typed.CallID,
			Name:   typed.Name,
			Output: new(typed.Output),
		}

	default:
		return nil, fmt.Errorf("%w: AI Gateway realtime conversation item %T", model.ErrUnsupported, item)
	}

	return json.Marshal(realtimeItemCreateEvent{
		Type: "conversation-item-create",
		Item: wireItem,
	})
}

func realtimeResponseOptionsOf(options *model.RealtimeResponseOptions) *realtimeResponseOptions {
	if options == nil {
		return nil
	}

	return &realtimeResponseOptions{
		Modalities:   options.Modalities,
		Instructions: options.Instructions,
		Metadata:     options.Metadata,
	}
}
