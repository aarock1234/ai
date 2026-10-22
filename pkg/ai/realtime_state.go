package ai

import (
	"encoding/json"
	"slices"
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

// RealtimeStatus is the connection status of a realtime session.
type RealtimeStatus string

// Realtime session statuses.
const (
	RealtimeDisconnected RealtimeStatus = "disconnected"
	RealtimeConnecting   RealtimeStatus = "connecting"
	RealtimeConnected    RealtimeStatus = "connected"
	RealtimeClosing      RealtimeStatus = "closing"
	RealtimeFailed       RealtimeStatus = "error"
)

// RealtimeState is a snapshot of a realtime session.
type RealtimeState struct {
	Status RealtimeStatus

	// Messages is the conversation rebuilt from the session's events.
	Messages []RealtimeMessage

	// Events are the most recent server events, oldest first.
	Events []model.RealtimeServerEvent

	// Session is the lifecycle state of models that start sessions with a
	// session start event or finalize them with a session close event. It
	// is nil for other models.
	Session *RealtimeSessionState
}

// RealtimeRole is the author of a realtime message.
type RealtimeRole string

// Realtime message roles.
const (
	RealtimeUser      RealtimeRole = "user"
	RealtimeAssistant RealtimeRole = "assistant"
)

// RealtimeMessage is one message of a realtime conversation.
type RealtimeMessage struct {
	ID    string
	Role  RealtimeRole
	Parts []RealtimeMessagePart
}

// RealtimeMessagePart is a RealtimeTextPart or a RealtimeToolPart.
type RealtimeMessagePart interface {
	isRealtimeMessagePart()
}

// RealtimeTextPart is the text or audio transcript of a message.
type RealtimeTextPart struct {
	Text string

	// Streaming reports that more text may follow.
	Streaming bool
}

// RealtimeToolState is the progress of a realtime tool call.
type RealtimeToolState string

// Realtime tool call states.
const (
	RealtimeToolInputStreaming RealtimeToolState = "input-streaming"
	RealtimeToolInputAvailable RealtimeToolState = "input-available"
	RealtimeToolOutputReady    RealtimeToolState = "output-available"
)

// RealtimeToolPart is a function call of the model.
type RealtimeToolPart struct {
	ToolCallID string

	// ToolName is empty while the input streams.
	ToolName string

	State RealtimeToolState

	// Input is the parsed arguments object once available.
	Input json.RawMessage

	// Output is the result submitted with AddToolOutput.
	Output any
}

func (RealtimeTextPart) isRealtimeMessagePart() {}
func (RealtimeToolPart) isRealtimeMessagePart() {}

// RealtimeFinalization is whether the provider confirmed the end of a
// session.
type RealtimeFinalization string

// Realtime finalization states.
const (
	RealtimeFinalizationPending     RealtimeFinalization = "pending"
	RealtimeFinalizationConfirmed   RealtimeFinalization = "confirmed"
	RealtimeFinalizationUnconfirmed RealtimeFinalization = "unconfirmed"
)

// RealtimeSessionState is the lifecycle state of a session.
type RealtimeSessionState struct {
	SessionID string

	// DelegationMode is empty until the session starts.
	DelegationMode model.RealtimeDelegationTarget

	// Transcripts and Delegations are the most recent fragments and
	// delegations, oldest first.
	Transcripts []model.RealtimeTranscriptFragment
	Delegations []model.RealtimeDelegationCreated

	// Duration is the latest reported session duration, or nil.
	Duration *time.Duration

	Finalization RealtimeFinalization

	// TerminationReason is the provider's reason for ending the session.
	TerminationReason string

	IsInputMuted bool
}

func newRealtimeSessionState() *RealtimeSessionState {
	return &RealtimeSessionState{
		Finalization: RealtimeFinalizationPending,
	}
}

// reduce applies one server event. Muted is the mute state an acknowledged
// command set, or nil.
func (s *RealtimeSessionState) reduce(event model.RealtimeServerEvent, limit int, muted *bool) {
	switch typed := event.(type) {
	case model.RealtimeSessionStarted:
		s.SessionID = typed.SessionID
		s.DelegationMode = typed.DelegationMode

	case model.RealtimeSessionUsage:
		s.Duration = new(typed.Duration)

	case model.RealtimeSessionClosed:
		if typed.SessionID != "" {
			s.SessionID = typed.SessionID
		}

		s.Duration = new(typed.Duration)
		s.TerminationReason = typed.Reason
		s.Finalization = RealtimeFinalizationConfirmed

	case model.RealtimeTranscriptFragment:
		s.Transcripts = lastItems(append(s.Transcripts, typed), limit)

	case model.RealtimeDelegationCreated:
		if s.hasDelegation(typed.DelegationID) {
			return
		}

		s.Delegations = lastItems(append(s.Delegations, typed), limit)

	case model.RealtimeCommandAcknowledged:
		if muted != nil {
			s.IsInputMuted = *muted
		}
	}
}

func (s *RealtimeSessionState) hasDelegation(delegationID string) bool {
	return slices.ContainsFunc(s.Delegations, func(delegation model.RealtimeDelegationCreated) bool {
		return delegation.DelegationID == delegationID
	})
}

// clone copies the session state so that a snapshot does not change with
// the session.
func (s *RealtimeSessionState) clone() *RealtimeSessionState {
	if s == nil {
		return nil
	}

	copied := *s
	copied.Transcripts = slices.Clone(s.Transcripts)
	copied.Delegations = slices.Clone(s.Delegations)

	return &copied
}

// lastItems keeps the last limit items.
func lastItems[Item any](items []Item, limit int) []Item {
	if len(items) <= limit {
		return items
	}

	return slices.Clone(items[len(items)-limit:])
}
