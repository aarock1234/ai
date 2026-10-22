package gateway

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

// realtimeServerEvent is a normalized server event on the wire. Its fields
// depend on the type.
type realtimeServerEvent struct {
	Type string `json:"type"`

	SessionID      string                         `json:"sessionId"`
	DelegationMode model.RealtimeDelegationTarget `json:"delegationMode"`
	Usage          *realtimeUsage                 `json:"usage"`
	Reason         string                         `json:"reason"`

	ContextWindowUsageRatio *float64 `json:"contextWindowUsageRatio"`

	Speaker model.RealtimeSpeaker `json:"speaker"`
	StartMs float64               `json:"startMs"`
	EndMs   float64               `json:"endMs"`

	DelegationID string                         `json:"delegationId"`
	Target       model.RealtimeDelegationTarget `json:"target"`
	OffsetMs     *float64                       `json:"offsetMs"`

	Command       string `json:"command"`
	ClientEventID string `json:"clientEventId"`

	ResponseID     string          `json:"responseId"`
	ItemID         string          `json:"itemId"`
	PreviousItemID string          `json:"previousItemId"`
	Item           json.RawMessage `json:"item"`
	Status         string          `json:"status"`

	Delta      string  `json:"delta"`
	Transcript *string `json:"transcript"`
	Text       *string `json:"text"`

	CallID    string `json:"callId"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`

	Message string `json:"message"`
	Code    string `json:"code"`
	RawType string `json:"rawType"`

	Raw json.RawMessage `json:"raw"`
}

type realtimeUsage struct {
	Seconds float64 `json:"seconds"`
}

// decodeRealtimeServerEvent decodes a normalized event. Raw is the
// provider's message the gateway forwarded, or the whole event when it has
// none.
func decodeRealtimeServerEvent(message json.RawMessage) (model.RealtimeServerEvent, error) {
	var event realtimeServerEvent
	if err := json.Unmarshal(message, &event); err != nil {
		return nil, fmt.Errorf("%w: decode realtime event: %w", model.ErrInvalidResponse, err)
	}

	raw := event.Raw
	if len(raw) == 0 {
		raw = message
	}

	if decoded, ok := decodeContinuousEvent(event, raw); ok {
		return decoded, nil
	}

	if decoded, ok := decodeLifecycleEvent(event, raw); ok {
		return decoded, nil
	}

	if decoded, ok := decodeOutputEvent(event, raw); ok {
		return decoded, nil
	}

	return model.RealtimeCustom{
		RawType: customRawType(event),
		Raw:     raw,
	}, nil
}

// customRawType returns the provider type of a custom event, or the type
// of an event this package does not know.
func customRawType(event realtimeServerEvent) string {
	if event.Type == "custom" {
		return event.RawType
	}

	return event.Type
}

// decodeContinuousEvent decodes the events of continuous conversations.
func decodeContinuousEvent(event realtimeServerEvent, raw json.RawMessage) (model.RealtimeServerEvent, bool) {
	switch event.Type {
	case "session-started":
		return model.RealtimeSessionStarted{
			SessionID:      event.SessionID,
			DelegationMode: event.DelegationMode,
			Raw:            raw,
		}, true

	case "session-closed":
		return model.RealtimeSessionClosed{
			SessionID: event.SessionID,
			Duration:  event.Usage.duration(),
			Reason:    event.Reason,
			Raw:       raw,
		}, true

	case "session-usage":
		return model.RealtimeSessionUsage{
			Duration:                event.Usage.duration(),
			ContextWindowUsageRatio: event.ContextWindowUsageRatio,
			Raw:                     raw,
		}, true

	case "audio-chunk":
		return model.RealtimeAudioChunk{
			Delta: event.Delta,
			Raw:   raw,
		}, true

	case "transcript-fragment":
		return model.RealtimeTranscriptFragment{
			Speaker: event.Speaker,
			Delta:   event.Delta,
			Start:   milliseconds(event.StartMs),
			End:     milliseconds(event.EndMs),
			Raw:     raw,
		}, true

	case "delegation-created":
		created := model.RealtimeDelegationCreated{
			DelegationID: event.DelegationID,
			Target:       event.Target,
			ResponseID:   event.ResponseID,
			Raw:          raw,
		}

		if event.OffsetMs != nil {
			created.Offset = new(milliseconds(*event.OffsetMs))
		}

		return created, true

	case "command-acknowledged":
		return model.RealtimeCommandAcknowledged{
			Command:       event.Command,
			ClientEventID: event.ClientEventID,
			Raw:           raw,
		}, true

	default:
		return nil, false
	}
}

// decodeLifecycleEvent decodes session, input, conversation, response and
// error events.
func decodeLifecycleEvent(event realtimeServerEvent, raw json.RawMessage) (model.RealtimeServerEvent, bool) {
	switch event.Type {
	case "session-created":
		return model.RealtimeSessionCreated{
			SessionID: event.SessionID,
			Raw:       raw,
		}, true

	case "session-updated":
		return model.RealtimeSessionUpdated{
			Raw: raw,
		}, true

	case "speech-started":
		return model.RealtimeSpeechStarted{
			ItemID: event.ItemID,
			Raw:    raw,
		}, true

	case "speech-stopped":
		return model.RealtimeSpeechStopped{
			ItemID: event.ItemID,
			Raw:    raw,
		}, true

	case "audio-committed":
		return model.RealtimeAudioCommitted{
			ItemID:         event.ItemID,
			PreviousItemID: event.PreviousItemID,
			Raw:            raw,
		}, true

	case "conversation-item-added":
		return model.RealtimeConversationItemAdded{
			ItemID: event.ItemID,
			Item:   event.Item,
			Raw:    raw,
		}, true

	case "input-transcription-completed":
		return model.RealtimeInputTranscriptionCompleted{
			ItemID:     event.ItemID,
			Transcript: stringValue(event.Transcript),
			Raw:        raw,
		}, true

	case "response-created":
		return model.RealtimeResponseCreated{
			ResponseID: event.ResponseID,
			Raw:        raw,
		}, true

	case "response-done":
		return model.RealtimeResponseDone{
			ResponseID: event.ResponseID,
			Status:     event.Status,
			Raw:        raw,
		}, true

	case "error":
		return model.RealtimeError{
			Message:       event.Message,
			Code:          event.Code,
			ClientEventID: event.ClientEventID,
			Raw:           raw,
		}, true

	default:
		return nil, false
	}
}

// decodeOutputEvent decodes output item, audio, text and function call
// events.
func decodeOutputEvent(event realtimeServerEvent, raw json.RawMessage) (model.RealtimeServerEvent, bool) {
	switch event.Type {
	case "output-item-added":
		return model.RealtimeOutputItemAdded{
			ResponseID: event.ResponseID,
			ItemID:     event.ItemID,
			Raw:        raw,
		}, true

	case "output-item-done":
		return model.RealtimeOutputItemDone{
			ResponseID: event.ResponseID,
			ItemID:     event.ItemID,
			Raw:        raw,
		}, true

	case "content-part-added":
		return model.RealtimeContentPartAdded{
			ResponseID: event.ResponseID,
			ItemID:     event.ItemID,
			Raw:        raw,
		}, true

	case "content-part-done":
		return model.RealtimeContentPartDone{
			ResponseID: event.ResponseID,
			ItemID:     event.ItemID,
			Raw:        raw,
		}, true

	case "audio-delta":
		return model.RealtimeAudioDelta{
			ResponseID: event.ResponseID,
			ItemID:     event.ItemID,
			Delta:      event.Delta,
			Raw:        raw,
		}, true

	case "audio-done":
		return model.RealtimeAudioDone{
			ResponseID: event.ResponseID,
			ItemID:     event.ItemID,
			Raw:        raw,
		}, true

	case "audio-transcript-delta":
		return model.RealtimeAudioTranscriptDelta{
			ResponseID: event.ResponseID,
			ItemID:     event.ItemID,
			Delta:      event.Delta,
			Raw:        raw,
		}, true

	case "audio-transcript-done":
		return model.RealtimeAudioTranscriptDone{
			ResponseID: event.ResponseID,
			ItemID:     event.ItemID,
			Transcript: event.Transcript,
			Raw:        raw,
		}, true

	case "text-delta":
		return model.RealtimeTextDelta{
			ResponseID: event.ResponseID,
			ItemID:     event.ItemID,
			Delta:      event.Delta,
			Raw:        raw,
		}, true

	case "text-done":
		return model.RealtimeTextDone{
			ResponseID: event.ResponseID,
			ItemID:     event.ItemID,
			Text:       event.Text,
			Raw:        raw,
		}, true

	case "function-call-arguments-delta":
		return model.RealtimeFunctionCallArgumentsDelta{
			ResponseID: event.ResponseID,
			ItemID:     event.ItemID,
			CallID:     event.CallID,
			Delta:      event.Delta,
			Raw:        raw,
		}, true

	case "function-call-arguments-done":
		return model.RealtimeFunctionCallArgumentsDone{
			ResponseID: event.ResponseID,
			ItemID:     event.ItemID,
			CallID:     event.CallID,
			Name:       event.Name,
			Arguments:  event.Arguments,
			Raw:        raw,
		}, true

	default:
		return nil, false
	}
}

// duration converts the usage seconds. Missing usage is zero.
func (u *realtimeUsage) duration() time.Duration {
	if u == nil {
		return 0
	}

	return time.Duration(u.Seconds * float64(time.Second))
}

func milliseconds(value float64) time.Duration {
	return time.Duration(value * float64(time.Millisecond))
}
