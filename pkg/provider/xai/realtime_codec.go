package xai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"slices"

	"github.com/aarock1234/ai/internal/jsonobject"
	"github.com/aarock1234/ai/internal/optional"
	"github.com/aarock1234/ai/pkg/model"
)

// realtimeWireEvent is an xAI realtime server event. Its fields depend on
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
	Message        *string               `json:"message"`
	Code           *string               `json:"code"`
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
}

type realtimeWireItemHeader struct {
	ID string `json:"id"`
}

// itemID returns the ID of the event's item, or its item_id.
func (e realtimeWireEvent) itemID() string {
	if len(e.Item) == 0 {
		return e.ItemID
	}

	var header realtimeWireItemHeader
	if err := json.Unmarshal(e.Item, &header); err != nil || header.ID == "" {
		return e.ItemID
	}

	return header.ID
}

// responseID returns the ID of the event's response, or its response_id.
func (e realtimeWireEvent) responseID() string {
	if e.Response != nil && e.Response.ID != "" {
		return e.Response.ID
	}

	return e.ResponseID
}

// parseRealtimeServerEvent maps an xAI realtime event. Unknown events,
// conversation creation and MCP events become custom events.
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
		"response.text.delta", "response.text.done":
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
			Transcript: optional.Or(event.Transcript, ""),
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

	case "response.text.delta":
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
	message := event.Message
	code := event.Code

	if event.Error != nil {
		if event.Error.Message != nil {
			message = event.Error.Message
		}

		if event.Error.Code != nil {
			code = event.Error.Code
		}
	}

	return model.RealtimeError{
		Message: optional.Or(message, "Unknown error"),
		Code:    optional.Or(code, ""),
		Raw:     raw,
	}
}

// xAI realtime client event payloads.
type (
	realtimeSessionUpdateEvent struct {
		Type    string          `json:"type"`
		Session json.RawMessage `json:"session"`
	}

	realtimeInputAppendEvent struct {
		Type  string `json:"type"`
		Audio string `json:"audio"`
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

	realtimeResponseCreateEvent struct {
		Type     string                   `json:"type"`
		Response *realtimeResponseOptions `json:"response,omitempty"`
	}

	realtimeResponseOptions struct {
		Modalities   []model.RealtimeModality `json:"modalities,omitempty"`
		Instructions string                   `json:"instructions,omitempty"`
	}
)

// serializeRealtimeClientEvent maps a normalized client event to its xAI
// payload.
func serializeRealtimeClientEvent(event model.RealtimeClientEvent) (json.RawMessage, error) {
	switch typed := event.(type) {
	case model.RealtimeSessionUpdate:
		session, err := buildRealtimeSession(typed.Config)
		if err != nil {
			return nil, err
		}

		return json.Marshal(realtimeSessionUpdateEvent{
			Type:    "session.update",
			Session: session,
		})

	case model.RealtimeInputAudioAppend:
		return json.Marshal(realtimeInputAppendEvent{
			Type:  "input_audio_buffer.append",
			Audio: typed.Audio,
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
		// xAI silently ignores truncation over WebSocket. Barge-in still
		// works because playback stops when speech starts.
		return nil, nil

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
		return nil, fmt.Errorf("%w: xAI realtime client event %T", model.ErrUnsupported, event)
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
		return nil, fmt.Errorf("%w: xAI realtime conversation item %T", model.ErrUnsupported, item)
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
		Modalities:   options.Modalities,
		Instructions: options.Instructions,
	}
}

// realtimeSession is the xAI session configuration.
type realtimeSession struct {
	Instructions string         `json:"instructions,omitempty"`
	Voice        string         `json:"voice,omitempty"`
	Audio        *realtimeAudio `json:"audio,omitempty"`

	// TurnDetection is JSON null when detection is disabled, and absent
	// when unset.
	TurnDetection json.RawMessage `json:"turn_detection,omitempty"`

	Tools []realtimeTool `json:"tools,omitempty"`
}

type realtimeAudio struct {
	Input  *realtimeAudioDirection `json:"input,omitempty"`
	Output *realtimeAudioDirection `json:"output,omitempty"`
}

type realtimeAudioDirection struct {
	Format realtimeAudioFormat `json:"format"`
}

type realtimeAudioFormat struct {
	Type string `json:"type"`
	Rate int    `json:"rate,omitempty"`
}

type realtimeTurnDetection struct {
	Type              string   `json:"type"`
	Threshold         *float64 `json:"threshold,omitempty"`
	SilenceDurationMs *int64   `json:"silence_duration_ms,omitempty"`
	PrefixPaddingMs   *int64   `json:"prefix_padding_ms,omitempty"`
}

type realtimeTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// buildRealtimeSession converts a normalized configuration. Provider
// options are merged into the session object, with a "tools" option
// appended to the configured tools, as upstream does.
func buildRealtimeSession(config model.RealtimeSessionConfig) (json.RawMessage, error) {
	turnDetection, err := buildRealtimeTurnDetection(config.TurnDetection)
	if err != nil {
		return nil, err
	}

	session := realtimeSession{
		Instructions:  config.Instructions,
		Voice:         config.Voice,
		Audio:         buildRealtimeAudio(config),
		TurnDetection: turnDetection,
		Tools:         realtimeTools(config.Tools),
	}

	encoded, err := json.Marshal(session)
	if err != nil {
		return nil, fmt.Errorf("encode realtime session: %w", err)
	}

	return mergeRealtimeProviderOptions(encoded, config.ProviderOptions)
}

func buildRealtimeAudio(config model.RealtimeSessionConfig) *realtimeAudio {
	if config.InputAudioFormat == nil && config.OutputAudioFormat == nil {
		return nil
	}

	return &realtimeAudio{
		Input:  realtimeAudioDirectionOf(config.InputAudioFormat),
		Output: realtimeAudioDirectionOf(config.OutputAudioFormat),
	}
}

func realtimeAudioDirectionOf(format *model.AudioFormat) *realtimeAudioDirection {
	if format == nil {
		return nil
	}

	return &realtimeAudioDirection{
		Format: realtimeAudioFormat{
			Type: format.Type,
			Rate: format.Rate,
		},
	}
}

// buildRealtimeTurnDetection returns the turn detection payload: nil when
// unset, JSON null when disabled, and server voice activity detection
// otherwise, the only kind xAI offers.
func buildRealtimeTurnDetection(detection *model.RealtimeTurnDetection) (json.RawMessage, error) {
	if detection == nil {
		return nil, nil
	}

	if detection.Type == model.RealtimeTurnDetectionDisabled {
		return json.RawMessage("null"), nil
	}

	wire := realtimeTurnDetection{
		Type:      "server_vad",
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

func realtimeTools(tools []model.RealtimeToolDefinition) []realtimeTool {
	if len(tools) == 0 {
		return nil
	}

	converted := make([]realtimeTool, len(tools))

	for index, tool := range tools {
		converted[index] = realtimeTool{
			Type:        "function",
			Name:        tool.Name,
			Description: tool.Description,
			Parameters:  tool.Parameters,
		}
	}

	return converted
}

// mergeRealtimeProviderOptions sets each provider option as a top-level
// member of the session, replacing members with the same name. A "tools"
// array is appended to the configured tools instead.
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

		if key == "tools" {
			object, err = appendRealtimeTools(object, value)
			if err != nil {
				return nil, err
			}

			continue
		}

		object = object.With(key, value)
	}

	return object.Encode()
}

// appendRealtimeTools appends the tools of a "tools" provider option. A
// value that is not an array is ignored, as upstream does.
func appendRealtimeTools(session jsonobject.Object, value json.RawMessage) (jsonobject.Object, error) {
	if !isJSONArray(value) {
		return session, nil
	}

	var extraTools []json.RawMessage
	if err := json.Unmarshal(value, &extraTools); err != nil {
		return nil, fmt.Errorf("%w: realtime provider option tools: %w", model.ErrInvalidArgument, err)
	}

	var tools []json.RawMessage
	if existing, ok := session.Get("tools"); ok {
		if err := json.Unmarshal(existing, &tools); err != nil {
			return nil, fmt.Errorf("parse realtime session tools: %w", err)
		}
	}

	merged, err := json.Marshal(append(tools, extraTools...))
	if err != nil {
		return nil, fmt.Errorf("encode realtime session tools: %w", err)
	}

	return session.With("tools", merged), nil
}

func isJSONArray(value json.RawMessage) bool {
	trimmed := bytes.TrimSpace(value)

	return len(trimmed) > 0 && trimmed[0] == '['
}
