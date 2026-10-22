package openai

import (
	"encoding/json"
	"fmt"
	"maps"
	"slices"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/pkg/model"
)

// realtimeDefaultTranscriptionModel transcribes input audio when the
// configuration names no model.
const realtimeDefaultTranscriptionModel = "gpt-realtime-whisper"

// realtimeWireEvent is a Realtime API server event. Its fields depend on
// the type.
type realtimeWireEvent struct {
	Type           string                `json:"type"`
	Session        *realtimeWireSession  `json:"session"`
	Item           json.RawMessage       `json:"item"`
	Response       *realtimeWireResponse `json:"response"`
	Error          *realtimeWireError    `json:"error"`
	ItemID         string                `json:"item_id"`
	PreviousItemID string                `json:"previous_item_id"`
	ResponseID     string                `json:"response_id"`
	Transcript     *string               `json:"transcript"`
	Delta          string                `json:"delta"`
	Text           *string               `json:"text"`
	CallID         string                `json:"call_id"`
	Name           string                `json:"name"`
	Arguments      string                `json:"arguments"`
	Message        string                `json:"message"`
	Code           string                `json:"code"`
}

type realtimeWireSession struct {
	ID string `json:"id"`
}

type realtimeWireResponse struct {
	ID     string `json:"id"`
	Status string `json:"status"`
}

type realtimeWireError struct {
	Message *string `json:"message"`
	Code    *string `json:"code"`
	EventID *string `json:"event_id"`
}

type realtimeWireItemHeader struct {
	ID string `json:"id"`
}

// itemID returns the ID of the event's item, or its item_id.
func (e realtimeWireEvent) itemID() string {
	var header realtimeWireItemHeader
	if len(e.Item) > 0 && json.Unmarshal(e.Item, &header) == nil && header.ID != "" {
		return header.ID
	}

	return e.ItemID
}

// responseID returns the ID of the event's response, or its response_id.
func (e realtimeWireEvent) responseID() string {
	if e.Response != nil && e.Response.ID != "" {
		return e.Response.ID
	}

	return e.ResponseID
}

// parseRealtimeServerEvent maps a Realtime API event. Unknown events
// become custom events.
func parseRealtimeServerEvent(raw json.RawMessage) (model.RealtimeServerEvent, error) {
	var event realtimeWireEvent
	if err := json.Unmarshal(raw, &event); err != nil {
		return nil, fmt.Errorf("%w: decode realtime event: %w", model.ErrInvalidResponse, err)
	}

	switch event.Type {
	case "session.created", "session.updated",
		"input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped", "input_audio_buffer.committed",
		"conversation.item.added", "conversation.item.input_audio_transcription.completed":
		return parseRealtimeSessionEvent(event, raw), nil

	case "response.created", "response.done",
		"response.output_item.added", "response.output_item.done",
		"response.content_part.added", "response.content_part.done":
		return parseRealtimeResponseEvent(event, raw), nil

	case "response.output_audio.delta", "response.output_audio.done",
		"response.output_audio_transcript.delta", "response.output_audio_transcript.done",
		"response.output_text.delta", "response.output_text.done":
		return parseRealtimeOutputEvent(event, raw), nil

	case "response.function_call_arguments.delta", "response.function_call_arguments.done", "error":
		return parseRealtimeCallEvent(event, raw), nil

	default:
		return model.RealtimeCustom{
			RawType: event.Type,
			Raw:     raw,
		}, nil
	}
}

// parseRealtimeSessionEvent maps session, input audio and conversation
// events.
func parseRealtimeSessionEvent(event realtimeWireEvent, raw json.RawMessage) model.RealtimeServerEvent {
	switch event.Type {
	case "session.created":
		created := model.RealtimeSessionCreated{
			Raw: raw,
		}

		if event.Session != nil {
			created.SessionID = event.Session.ID
		}

		return created

	case "session.updated":
		return model.RealtimeSessionUpdated{
			Raw: raw,
		}

	case "input_audio_buffer.speech_started":
		return model.RealtimeSpeechStarted{
			ItemID: event.ItemID,
			Raw:    raw,
		}

	case "input_audio_buffer.speech_stopped":
		return model.RealtimeSpeechStopped{
			ItemID: event.ItemID,
			Raw:    raw,
		}

	case "input_audio_buffer.committed":
		return model.RealtimeAudioCommitted{
			ItemID:         event.ItemID,
			PreviousItemID: event.PreviousItemID,
			Raw:            raw,
		}

	case "conversation.item.added":
		return model.RealtimeConversationItemAdded{
			ItemID: event.itemID(),
			Item:   event.Item,
			Raw:    raw,
		}

	default:
		return model.RealtimeInputTranscriptionCompleted{
			ItemID:     event.ItemID,
			Transcript: valueOr(event.Transcript, ""),
			Raw:        raw,
		}
	}
}

// parseRealtimeResponseEvent maps response, output item and content part
// lifecycle events.
func parseRealtimeResponseEvent(event realtimeWireEvent, raw json.RawMessage) model.RealtimeServerEvent {
	switch event.Type {
	case "response.created":
		return model.RealtimeResponseCreated{
			ResponseID: event.responseID(),
			Raw:        raw,
		}

	case "response.done":
		status := "completed"
		if event.Response != nil && event.Response.Status != "" {
			status = event.Response.Status
		}

		return model.RealtimeResponseDone{
			ResponseID: event.responseID(),
			Status:     status,
			Raw:        raw,
		}

	case "response.output_item.added":
		return model.RealtimeOutputItemAdded{
			ResponseID: event.ResponseID,
			ItemID:     event.itemID(),
			Raw:        raw,
		}

	case "response.output_item.done":
		return model.RealtimeOutputItemDone{
			ResponseID: event.ResponseID,
			ItemID:     event.itemID(),
			Raw:        raw,
		}

	case "response.content_part.added":
		return model.RealtimeContentPartAdded{
			ResponseID: event.ResponseID,
			ItemID:     event.ItemID,
			Raw:        raw,
		}

	default:
		return model.RealtimeContentPartDone{
			ResponseID: event.ResponseID,
			ItemID:     event.ItemID,
			Raw:        raw,
		}
	}
}

// parseRealtimeOutputEvent maps audio, transcript and text output events.
func parseRealtimeOutputEvent(event realtimeWireEvent, raw json.RawMessage) model.RealtimeServerEvent {
	switch event.Type {
	case "response.output_audio.delta":
		return model.RealtimeAudioDelta{
			ResponseID: event.ResponseID,
			ItemID:     event.ItemID,
			Delta:      event.Delta,
			Raw:        raw,
		}

	case "response.output_audio.done":
		return model.RealtimeAudioDone{
			ResponseID: event.ResponseID,
			ItemID:     event.ItemID,
			Raw:        raw,
		}

	case "response.output_audio_transcript.delta":
		return model.RealtimeAudioTranscriptDelta{
			ResponseID: event.ResponseID,
			ItemID:     event.ItemID,
			Delta:      event.Delta,
			Raw:        raw,
		}

	case "response.output_audio_transcript.done":
		return model.RealtimeAudioTranscriptDone{
			ResponseID: event.ResponseID,
			ItemID:     event.ItemID,
			Transcript: event.Transcript,
			Raw:        raw,
		}

	case "response.output_text.delta":
		return model.RealtimeTextDelta{
			ResponseID: event.ResponseID,
			ItemID:     event.ItemID,
			Delta:      event.Delta,
			Raw:        raw,
		}

	default:
		return model.RealtimeTextDone{
			ResponseID: event.ResponseID,
			ItemID:     event.ItemID,
			Text:       event.Text,
			Raw:        raw,
		}
	}
}

// parseRealtimeCallEvent maps function call and error events.
func parseRealtimeCallEvent(event realtimeWireEvent, raw json.RawMessage) model.RealtimeServerEvent {
	switch event.Type {
	case "response.function_call_arguments.delta":
		return model.RealtimeFunctionCallArgumentsDelta{
			ResponseID: event.ResponseID,
			ItemID:     event.ItemID,
			CallID:     event.CallID,
			Delta:      event.Delta,
			Raw:        raw,
		}

	case "response.function_call_arguments.done":
		return model.RealtimeFunctionCallArgumentsDone{
			ResponseID: event.ResponseID,
			ItemID:     event.ItemID,
			CallID:     event.CallID,
			Name:       event.Name,
			Arguments:  event.Arguments,
			Raw:        raw,
		}

	default:
		return realtimeErrorFromWire(event, raw)
	}
}

// realtimeErrorFromWire prefers the nested error object over the top-level
// message and code.
func realtimeErrorFromWire(event realtimeWireEvent, raw json.RawMessage) model.RealtimeError {
	failure := model.RealtimeError{
		Message: event.Message,
		Code:    event.Code,
		Raw:     raw,
	}

	if wireError := event.Error; wireError != nil {
		failure.Message = valueOr(wireError.Message, failure.Message)
		failure.Code = valueOr(wireError.Code, failure.Code)
		failure.ClientEventID = valueOr(wireError.EventID, "")
	}

	if failure.Message == "" {
		failure.Message = "Unknown error"
	}

	return failure
}

// Realtime API client event payloads.
type (
	realtimeSessionUpdateEvent struct {
		Type    string          `json:"type"`
		Session json.RawMessage `json:"session"`
		EventID string          `json:"event_id,omitempty"`
	}

	realtimeInputAppendEvent struct {
		Type    string `json:"type"`
		Audio   string `json:"audio"`
		EventID string `json:"event_id,omitempty"`
	}

	realtimeTypeOnlyEvent struct {
		Type string `json:"type"`
	}

	realtimeItemCreateEvent struct {
		Type string           `json:"type"`
		Item realtimeWireItem `json:"item"`
	}

	realtimeWireItem struct {
		Type    string                `json:"type"`
		Role    string                `json:"role,omitempty"`
		Content []realtimeWireContent `json:"content,omitempty"`
		CallID  string                `json:"call_id,omitempty"`
		Output  *string               `json:"output,omitempty"`
	}

	realtimeWireContent struct {
		Type  string `json:"type"`
		Text  string `json:"text,omitempty"`
		Audio string `json:"audio,omitempty"`
	}

	realtimeTruncateEvent struct {
		Type         string `json:"type"`
		ItemID       string `json:"item_id"`
		ContentIndex int    `json:"content_index"`
		AudioEndMs   int64  `json:"audio_end_ms"`
	}

	realtimeResponseCreateEvent struct {
		Type     string                   `json:"type"`
		Response *realtimeResponseOptions `json:"response,omitempty"`
	}

	realtimeResponseOptions struct {
		OutputModalities []model.RealtimeModality `json:"output_modalities,omitempty"`
		Instructions     string                   `json:"instructions,omitempty"`
		Metadata         map[string]any           `json:"metadata,omitempty"`
	}
)

// serializeClientEvent maps a normalized client event to its Realtime API
// payload.
func (m *RealtimeModel) serializeClientEvent(event model.RealtimeClientEvent) (json.RawMessage, error) {
	switch typed := event.(type) {
	case model.RealtimeSessionUpdate:
		session, err := m.BuildSessionConfig(typed.Config)
		if err != nil {
			return nil, err
		}

		return json.Marshal(realtimeSessionUpdateEvent{
			Type:    "session.update",
			Session: session,
			EventID: typed.EventID,
		})

	case model.RealtimeInputAudioAppend:
		return json.Marshal(realtimeInputAppendEvent{
			Type:    "input_audio_buffer.append",
			Audio:   typed.Audio,
			EventID: typed.EventID,
		})

	case model.RealtimeInputAudioCommit:
		return json.Marshal(realtimeTypeOnlyEvent{
			Type: "input_audio_buffer.commit",
		})

	case model.RealtimeInputAudioClear:
		return json.Marshal(realtimeTypeOnlyEvent{
			Type: "input_audio_buffer.clear",
		})

	case model.RealtimeConversationItemCreate:
		return serializeRealtimeItem(typed.Item)

	case model.RealtimeConversationItemTruncate:
		return json.Marshal(realtimeTruncateEvent{
			Type:         "conversation.item.truncate",
			ItemID:       typed.ItemID,
			ContentIndex: typed.ContentIndex,
			AudioEndMs:   typed.AudioEnd.Milliseconds(),
		})

	case model.RealtimeResponseCreate:
		return json.Marshal(realtimeResponseCreateEvent{
			Type:     "response.create",
			Response: realtimeResponseOptionsOf(typed.Options),
		})

	case model.RealtimeResponseCancel:
		return json.Marshal(realtimeTypeOnlyEvent{
			Type: "response.cancel",
		})

	default:
		return nil, fmt.Errorf("%w: OpenAI Realtime API client event %T", model.ErrUnsupported, event)
	}
}

func serializeRealtimeItem(item model.RealtimeConversationItem) (json.RawMessage, error) {
	var wireItem realtimeWireItem

	switch typed := item.(type) {
	case model.RealtimeTextMessage:
		wireItem = realtimeWireItem{
			Type: "message",
			Role: "user",
			Content: []realtimeWireContent{
				{
					Type: "input_text",
					Text: typed.Text,
				},
			},
		}

	case model.RealtimeAudioMessage:
		wireItem = realtimeWireItem{
			Type: "message",
			Role: "user",
			Content: []realtimeWireContent{
				{
					Type:  "input_audio",
					Audio: typed.Audio,
				},
			},
		}

	case model.RealtimeFunctionCallOutput:
		wireItem = realtimeWireItem{
			Type:   "function_call_output",
			CallID: typed.CallID,
			Output: new(typed.Output),
		}

	default:
		return nil, fmt.Errorf("%w: OpenAI Realtime API conversation item %T", model.ErrUnsupported, item)
	}

	return json.Marshal(realtimeItemCreateEvent{
		Type: "conversation.item.create",
		Item: wireItem,
	})
}

func realtimeResponseOptionsOf(options *model.RealtimeResponseOptions) *realtimeResponseOptions {
	if options == nil {
		return nil
	}

	return &realtimeResponseOptions{
		OutputModalities: options.Modalities,
		Instructions:     options.Instructions,
		Metadata:         options.Metadata,
	}
}

// realtimeModelSession is the Realtime API session configuration.
type realtimeModelSession struct {
	Type             string                   `json:"type"`
	Model            string                   `json:"model"`
	Instructions     string                   `json:"instructions,omitempty"`
	OutputModalities []model.RealtimeModality `json:"output_modalities,omitempty"`
	Audio            *realtimeModelAudio      `json:"audio,omitempty"`
	Tools            []realtimeModelTool      `json:"tools,omitempty"`
	ToolChoice       string                   `json:"tool_choice,omitempty"`
}

type realtimeModelAudio struct {
	Input  *realtimeModelAudioInput  `json:"input,omitempty"`
	Output *realtimeModelAudioOutput `json:"output,omitempty"`
}

type realtimeModelAudioInput struct {
	Format *realtimeAudioFormat `json:"format,omitempty"`

	// TurnDetection is JSON null when detection is disabled, and absent
	// when unset.
	TurnDetection json.RawMessage `json:"turn_detection,omitempty"`

	Transcription *realtimeModelTranscription `json:"transcription,omitempty"`
}

type realtimeModelTurnDetection struct {
	Type              string   `json:"type"`
	Threshold         *float64 `json:"threshold,omitempty"`
	SilenceDurationMs *int64   `json:"silence_duration_ms,omitempty"`
	PrefixPaddingMs   *int64   `json:"prefix_padding_ms,omitempty"`
}

type realtimeModelTranscription struct {
	Model    string `json:"model"`
	Language string `json:"language,omitempty"`
	Prompt   string `json:"prompt,omitempty"`
}

type realtimeModelAudioOutput struct {
	Format *realtimeAudioFormat `json:"format,omitempty"`
	Voice  string               `json:"voice,omitempty"`
}

type realtimeModelTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// buildRealtimeSession converts a normalized configuration. Provider
// options are merged into the session object, as upstream does.
func buildRealtimeSession(config model.RealtimeSessionConfig, modelID string) (json.RawMessage, error) {
	turnDetection, err := buildRealtimeTurnDetection(config.TurnDetection)
	if err != nil {
		return nil, err
	}

	session := realtimeModelSession{
		Type:             "realtime",
		Model:            modelID,
		Instructions:     config.Instructions,
		OutputModalities: config.OutputModalities,
		Audio:            buildRealtimeAudio(config, turnDetection),
	}

	if len(config.Tools) > 0 {
		session.Tools = realtimeModelTools(config.Tools)
		session.ToolChoice = "auto"
	}

	encoded, err := json.Marshal(session)
	if err != nil {
		return nil, fmt.Errorf("encode realtime session: %w", err)
	}

	return mergeRealtimeProviderOptions(encoded, config.ProviderOptions)
}

func buildRealtimeAudio(config model.RealtimeSessionConfig, turnDetection json.RawMessage) *realtimeModelAudio {
	audio := &realtimeModelAudio{}

	if config.InputAudioFormat != nil || config.InputAudioTranscription != nil || turnDetection != nil {
		audio.Input = &realtimeModelAudioInput{
			Format:        buildRealtimeFormat(config.InputAudioFormat),
			TurnDetection: turnDetection,
			Transcription: buildRealtimeTranscription(config.InputAudioTranscription),
		}
	}

	if config.OutputAudioFormat != nil || config.Voice != "" {
		audio.Output = &realtimeModelAudioOutput{
			Format: buildRealtimeFormat(config.OutputAudioFormat),
			Voice:  config.Voice,
		}
	}

	if audio.Input == nil && audio.Output == nil {
		return nil
	}

	return audio
}

func buildRealtimeFormat(format *model.AudioFormat) *realtimeAudioFormat {
	if format == nil {
		return nil
	}

	return &realtimeAudioFormat{
		Type: format.Type,
		Rate: format.Rate,
	}
}

func buildRealtimeTranscription(config *model.RealtimeTranscriptionConfig) *realtimeModelTranscription {
	if config == nil {
		return nil
	}

	transcriptionModel := config.Model
	if transcriptionModel == "" {
		transcriptionModel = realtimeDefaultTranscriptionModel
	}

	return &realtimeModelTranscription{
		Model:    transcriptionModel,
		Language: config.Language,
		Prompt:   config.Prompt,
	}
}

// buildRealtimeTurnDetection returns the turn detection payload: nil when
// unset, JSON null when disabled, and the detection settings otherwise.
func buildRealtimeTurnDetection(detection *model.RealtimeTurnDetection) (json.RawMessage, error) {
	if detection == nil {
		return nil, nil
	}

	var detectionType string

	switch detection.Type {
	case model.RealtimeTurnDetectionDisabled:
		return json.RawMessage("null"), nil
	case model.RealtimeServerVAD:
		detectionType = "server_vad"
	case model.RealtimeSemanticVAD:
		detectionType = "semantic_vad"
	default:
		return nil, fmt.Errorf("%w: unknown turn detection type %q", model.ErrInvalidArgument, detection.Type)
	}

	wire := realtimeModelTurnDetection{
		Type:      detectionType,
		Threshold: detection.Threshold,
	}

	if detection.SilenceDuration != nil {
		wire.SilenceDurationMs = new(detection.SilenceDuration.Milliseconds())
	}

	if detection.PrefixPadding != nil {
		wire.PrefixPaddingMs = new(detection.PrefixPadding.Milliseconds())
	}

	return json.Marshal(wire)
}

func realtimeModelTools(tools []model.RealtimeToolDefinition) []realtimeModelTool {
	converted := make([]realtimeModelTool, len(tools))

	for index, tool := range tools {
		converted[index] = realtimeModelTool{
			Type:        "function",
			Name:        tool.Name,
			Description: tool.Description,
			Parameters:  tool.Parameters,
		}
	}

	return converted
}

// mergeRealtimeProviderOptions sets each provider option as a top-level
// member of the session, replacing members with the same name.
func mergeRealtimeProviderOptions(session json.RawMessage, providerOptions model.ProviderOptions) (json.RawMessage, error) {
	if len(providerOptions) == 0 {
		return session, nil
	}

	object, err := jsonobject.Parse(session)
	if err != nil {
		return nil, fmt.Errorf("parse realtime session: %w", err)
	}

	for _, key := range slices.Sorted(maps.Keys(providerOptions)) {
		value, err := json.Marshal(providerOptions[key])
		if err != nil {
			return nil, fmt.Errorf("%w: encode realtime provider option %q: %w", model.ErrInvalidArgument, key, err)
		}

		object = object.With(key, value)
	}

	return object.Encode()
}
