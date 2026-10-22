package openai

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

// liveInvalidEventCode marks Live API events that failed validation.
const liveInvalidEventCode = "invalid_server_event"

// liveContextChannels are the channels context can be appended to.
var liveContextChannels = []string{
	"instructions",
	"thinking",
	"commentary",
}

// liveEvent is a Live API server event. Its fields depend on the type;
// pointers mark fields whose presence is validated.
type liveEvent struct {
	Type    string          `json:"type"`
	Session *liveSessionRef `json:"session"`

	Usage         *liveUsage         `json:"usage"`
	Reason        *string            `json:"reason"`
	ContextWindow *liveContextWindow `json:"context_window"`

	Delta   *string  `json:"delta"`
	StartMs *float64 `json:"start_ms"`
	EndMs   *float64 `json:"end_ms"`

	OffsetMs   *float64            `json:"offset_ms"`
	Delegation *liveDelegationWire `json:"delegation"`

	ClientEventID *string        `json:"client_event_id"`
	Error         *liveErrorWire `json:"error"`
}

type liveSessionRef struct {
	ID         string                    `json:"id"`
	Delegation *liveSessionDelegationRef `json:"delegation"`
}

type liveSessionDelegationRef struct {
	Type string `json:"type"`
}

type liveUsage struct {
	Seconds *float64 `json:"seconds"`
}

type liveContextWindow struct {
	UsageRatio *float64 `json:"usage_ratio"`
}

type liveDelegationWire struct {
	ID         string  `json:"id"`
	Target     *string `json:"target"`
	ResponseID *string `json:"response_id"`
}

type liveErrorWire struct {
	Message       *string `json:"message"`
	Code          *string `json:"code"`
	ClientEventID *string `json:"client_event_id"`
}

// liveKnownEvents are the event types with a normalized form.
var liveKnownEvents = []string{
	"session.started",
	"session.closed",
	"session.usage.updated",
	"session.output_audio.delta",
	"session.input_transcript.delta",
	"session.output_transcript.delta",
	"session.delegation.created",
	"session.updated",
	"session.input_audio.muted",
	"session.input_audio.unmuted",
	"session.instructions.appended",
	"session.thinking.appended",
	"session.commentary.appended",
	"error",
}

// parseLiveServerEvent maps one Live API event. Unknown types become custom
// events; malformed known events become error events.
func parseLiveServerEvent(raw json.RawMessage) model.RealtimeServerEvent {
	var event liveEvent
	if err := json.Unmarshal(raw, &event); err != nil {
		return invalidLiveEvent(raw, "Invalid OpenAI Live server event.")
	}

	if event.Type == "" {
		return invalidLiveEvent(raw, "Invalid OpenAI Live server event.")
	}

	if !slices.Contains(liveKnownEvents, event.Type) {
		return model.RealtimeCustom{
			RawType: event.Type,
			Raw:     raw,
		}
	}

	parsed, ok := event.normalize(raw)
	if !ok {
		return invalidLiveEvent(raw, "Invalid OpenAI Live server event.")
	}

	return parsed
}

func invalidLiveEvent(raw json.RawMessage, message string) model.RealtimeServerEvent {
	return model.RealtimeError{
		Message: message,
		Code:    liveInvalidEventCode,
		Raw:     raw,
	}
}

// normalize maps a known event and reports whether its fields are valid.
func (e liveEvent) normalize(raw json.RawMessage) (model.RealtimeServerEvent, bool) {
	switch e.Type {
	case "session.started":
		return e.sessionStarted(raw)

	case "session.closed":
		return e.sessionClosed(raw)

	case "session.usage.updated":
		return e.sessionUsage(raw)

	case "session.output_audio.delta":
		if e.Delta == nil {
			return nil, false
		}

		return model.RealtimeAudioChunk{
			Delta: *e.Delta,
			Raw:   raw,
		}, true

	case "session.input_transcript.delta":
		return e.transcriptFragment(model.RealtimeUserSpeaker, raw)

	case "session.output_transcript.delta":
		return e.transcriptFragment(model.RealtimeAssistantSpeaker, raw)

	case "session.delegation.created":
		return e.delegationCreated(raw)

	case "error":
		return e.liveError(raw)

	default:
		return e.acknowledgement(raw)
	}
}

func (e liveEvent) sessionStarted(raw json.RawMessage) (model.RealtimeServerEvent, bool) {
	if e.Session == nil || e.Session.ID == "" {
		return nil, false
	}

	mode := model.RealtimeDelegateToClient
	if delegation := e.Session.Delegation; delegation != nil {
		switch delegation.Type {
		case "responses":
			mode = model.RealtimeDelegateToProvider
		case "client":
		default:
			return nil, false
		}
	}

	return model.RealtimeSessionStarted{
		SessionID:      e.Session.ID,
		DelegationMode: mode,
		Raw:            raw,
	}, true
}

func (e liveEvent) sessionClosed(raw json.RawMessage) (model.RealtimeServerEvent, bool) {
	duration, ok := e.usageDuration()
	if !ok || e.Reason == nil {
		return nil, false
	}

	closed := model.RealtimeSessionClosed{
		Duration: duration,
		Reason:   *e.Reason,
		Raw:      raw,
	}

	if e.Session != nil {
		if e.Session.ID == "" {
			return nil, false
		}

		closed.SessionID = e.Session.ID
	}

	return closed, true
}

func (e liveEvent) sessionUsage(raw json.RawMessage) (model.RealtimeServerEvent, bool) {
	duration, ok := e.usageDuration()
	if !ok {
		return nil, false
	}

	usage := model.RealtimeSessionUsage{
		Duration: duration,
		Raw:      raw,
	}

	if window := e.ContextWindow; window != nil {
		if window.UsageRatio == nil || *window.UsageRatio < 0 || *window.UsageRatio > 1 {
			return nil, false
		}

		usage.ContextWindowUsageRatio = window.UsageRatio
	}

	return usage, true
}

// usageDuration returns the nonnegative usage seconds as a duration.
func (e liveEvent) usageDuration() (time.Duration, bool) {
	if e.Usage == nil || e.Usage.Seconds == nil || *e.Usage.Seconds < 0 {
		return 0, false
	}

	return time.Duration(*e.Usage.Seconds * float64(time.Second)), true
}

// transcriptFragment maps a transcript delta whose interval must be
// nonnegative and ordered.
func (e liveEvent) transcriptFragment(speaker model.RealtimeSpeaker, raw json.RawMessage) (model.RealtimeServerEvent, bool) {
	if e.Delta == nil || !isNonNegative(e.StartMs) || !isNonNegative(e.EndMs) {
		return nil, false
	}

	if *e.EndMs < *e.StartMs {
		return invalidLiveEvent(raw, "Invalid OpenAI Live event time interval."), true
	}

	return model.RealtimeTranscriptFragment{
		Speaker: speaker,
		Delta:   *e.Delta,
		Start:   milliseconds(*e.StartMs),
		End:     milliseconds(*e.EndMs),
		Raw:     raw,
	}, true
}

func (e liveEvent) delegationCreated(raw json.RawMessage) (model.RealtimeServerEvent, bool) {
	delegation := e.Delegation
	if delegation == nil || delegation.ID == "" {
		return nil, false
	}

	if e.OffsetMs != nil && *e.OffsetMs < 0 {
		return nil, false
	}

	created := model.RealtimeDelegationCreated{
		DelegationID: delegation.ID,
		ResponseID:   valueOr(delegation.ResponseID, ""),
		Raw:          raw,
	}

	if e.OffsetMs != nil {
		created.Offset = new(milliseconds(*e.OffsetMs))
	}

	if delegation.Target != nil {
		switch *delegation.Target {
		case "client":
			created.Target = model.RealtimeDelegateToClient
		case "responses":
			created.Target = model.RealtimeDelegateToProvider
		default:
			return nil, false
		}
	}

	return created, true
}

func (e liveEvent) liveError(raw json.RawMessage) (model.RealtimeServerEvent, bool) {
	if e.Error == nil || e.Error.Message == nil {
		return nil, false
	}

	return model.RealtimeError{
		Message:       *e.Error.Message,
		Code:          valueOr(e.Error.Code, ""),
		ClientEventID: valueOr(e.Error.ClientEventID, ""),
		Raw:           raw,
	}, true
}

// acknowledgement maps the events that acknowledge a client command to the
// command's name.
func (e liveEvent) acknowledgement(raw json.RawMessage) (model.RealtimeServerEvent, bool) {
	var command string

	switch e.Type {
	case "session.updated":
		if e.Session == nil || e.Session.ID == "" {
			return nil, false
		}

		command = "session.update"

	case "session.input_audio.muted", "session.input_audio.unmuted":
		command = strings.TrimSuffix(e.Type, "d")

	default:
		command = strings.TrimSuffix(e.Type, "ed")

		if !isNonNegative(e.StartMs) || !isNonNegative(e.EndMs) {
			return nil, false
		}

		if *e.EndMs < *e.StartMs {
			return invalidLiveEvent(raw, "Invalid OpenAI Live event time interval."), true
		}
	}

	return model.RealtimeCommandAcknowledged{
		Command:       command,
		ClientEventID: valueOr(e.ClientEventID, ""),
		Raw:           raw,
	}, true
}

func isNonNegative(value *float64) bool {
	return value != nil && *value >= 0
}

func milliseconds(value float64) time.Duration {
	return time.Duration(value * float64(time.Millisecond))
}

// Live API client event payloads.
type (
	liveSessionStartEvent struct {
		Type    string          `json:"type"`
		Session json.RawMessage `json:"session"`
		EventID string          `json:"event_id,omitempty"`
	}

	liveCommandEvent struct {
		Type    string `json:"type"`
		EventID string `json:"event_id,omitempty"`
	}

	liveAudioAppendEvent struct {
		Type    string `json:"type"`
		Audio   string `json:"audio"`
		EventID string `json:"event_id,omitempty"`
	}

	liveContextAppendEvent struct {
		Type    string `json:"type"`
		Content string `json:"content"`

		// DelegationID is null for context that answers no delegation.
		DelegationID *string `json:"delegation_id"`

		EventID string `json:"event_id,omitempty"`
	}
)

// LiveContextOptions are the "openai" provider options of a context append
// event. Unknown fields are rejected.
type LiveContextOptions struct {
	// Channel is "instructions", "thinking" or "commentary". The default is
	// "thinking".
	Channel string `json:"channel,omitempty"`
}

// serializeLiveClientEvent maps the commands of continuous conversations.
func serializeLiveClientEvent(event model.RealtimeClientEvent, modelID string) (json.RawMessage, error) {
	switch typed := event.(type) {
	case model.RealtimeSessionStart:
		session, err := buildLiveSession(typed.Config, modelID, liveWebSocket)
		if err != nil {
			return nil, err
		}

		return json.Marshal(liveSessionStartEvent{
			Type:    "session.start",
			Session: session,
			EventID: typed.EventID,
		})

	case model.RealtimeSessionUpdate:
		return nil, fmt.Errorf(
			"%w: OpenAI Live session-update; startup settings are immutable; use context-append or input-audio-mute/input-audio-unmute",
			model.ErrUnsupported,
		)

	case model.RealtimeSessionClose:
		return liveCommand("session.close", typed.EventID)

	case model.RealtimeInputAudioAppend:
		return json.Marshal(liveAudioAppendEvent{
			Type:    "session.input_audio.append",
			Audio:   typed.Audio,
			EventID: typed.EventID,
		})

	case model.RealtimeInputAudioMute:
		return liveCommand("session.input_audio.mute", typed.EventID)

	case model.RealtimeInputAudioUnmute:
		return liveCommand("session.input_audio.unmute", typed.EventID)

	case model.RealtimeContextAppend:
		return serializeLiveContext(typed)

	default:
		return nil, fmt.Errorf(
			"%w: OpenAI Live command %T; use continuous audio and context-append instead of voice-turn commands",
			model.ErrUnsupported,
			event,
		)
	}
}

func liveCommand(eventType, eventID string) (json.RawMessage, error) {
	return json.Marshal(liveCommandEvent{
		Type:    eventType,
		EventID: eventID,
	})
}

// serializeLiveContext appends context to the channel the options choose.
func serializeLiveContext(event model.RealtimeContextAppend) (json.RawMessage, error) {
	if event.DelegationID != nil && *event.DelegationID == "" {
		return nil, fmt.Errorf("%w: context delegation ID must not be empty", model.ErrInvalidArgument)
	}

	options, err := readLiveContextOptions(event.ProviderOptions)
	if err != nil {
		return nil, err
	}

	channel := options.Channel
	if channel == "" {
		channel = "thinking"
	}

	return json.Marshal(liveContextAppendEvent{
		Type:         "session." + channel + ".append",
		Content:      event.Content,
		DelegationID: event.DelegationID,
		EventID:      event.EventID,
	})
}

func readLiveContextOptions(providerOptions model.ProviderOptions) (LiveContextOptions, error) {
	raw, found, err := model.ProviderValue[json.RawMessage](providerOptions, Namespace)
	if err != nil || !found {
		return LiveContextOptions{}, err
	}

	var options LiveContextOptions
	if err := strictUnmarshal(raw, &options); err != nil {
		return LiveContextOptions{}, liveDecodeError("context providerOptions.openai", err)
	}

	if options.Channel != "" && !slices.Contains(liveContextChannels, options.Channel) {
		return LiveContextOptions{}, liveOptionsError("unknown context channel %q", options.Channel)
	}

	return options, nil
}
