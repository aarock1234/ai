package ai

import (
	"errors"
	"slices"

	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/pkg/model"
)

// Command tracking limits.
const (
	maxPendingRealtimeCommands = 512
	maxRecentRealtimeCommands  = 4096
)

// Command tracking errors.
var (
	errRealtimeCommandIDUsed = errors.New("realtime command ID was already used; retry with a fresh event ID")
	errRealtimeCommandsFull  = errors.New("too many pending realtime commands")
)

// realtimeCommand is a sent command awaiting its outcome.
type realtimeCommand struct {
	// muted is the mute state the command sets, or nil.
	muted *bool
}

// realtimeCommandTracker correlates commands with their acknowledgements
// within one connection. Unresolved commands are never evicted. The
// session guards it with its lock.
type realtimeCommandTracker struct {
	pending map[string]realtimeCommand

	// recent holds completed IDs, oldest first, so that they are not
	// reused.
	recent []string
}

func newRealtimeCommandTracker() *realtimeCommandTracker {
	return &realtimeCommandTracker{
		pending: make(map[string]realtimeCommand),
	}
}

// validateID rejects reused IDs and a full queue.
func (t *realtimeCommandTracker) validateID(eventID string) error {
	if _, pending := t.pending[eventID]; pending || slices.Contains(t.recent, eventID) {
		return errRealtimeCommandIDUsed
	}

	if len(t.pending) >= maxPendingRealtimeCommands {
		return errRealtimeCommandsFull
	}

	return nil
}

// trackedCommand is a command prepared for sending.
type trackedCommand struct {
	event   model.RealtimeClientEvent
	eventID string

	// awaitsAcknowledgement keeps the command pending after it was sent,
	// until the provider acknowledges it.
	awaitsAcknowledgement bool
}

// prepare gives mute commands an ID, registers commands with an ID and
// returns the event to send.
func (t *realtimeCommandTracker) prepare(event model.RealtimeClientEvent) (trackedCommand, error) {
	muted := realtimeMuteState(event)
	event = withMuteEventID(event)

	command := trackedCommand{
		event:   event,
		eventID: realtimeClientEventID(event),
	}

	switch event.(type) {
	case model.RealtimeContextAppend, model.RealtimeSessionUpdate:
		command.awaitsAcknowledgement = true
	default:
		command.awaitsAcknowledgement = muted != nil
	}

	if command.eventID == "" {
		return command, nil
	}

	if err := t.validateID(command.eventID); err != nil {
		return trackedCommand{}, err
	}

	t.pending[command.eventID] = realtimeCommand{
		muted: muted,
	}

	return command, nil
}

// sent records the outcome of sending a command.
func (t *realtimeCommandTracker) sent(command trackedCommand, err error) {
	if command.eventID == "" {
		return
	}

	if err != nil || !command.awaitsAcknowledgement {
		t.complete(command.eventID)
	}
}

// receive completes the command a server event acknowledges or rejects,
// and returns the mute state an acknowledged command set.
func (t *realtimeCommandTracker) receive(event model.RealtimeServerEvent) *bool {
	switch typed := event.(type) {
	case model.RealtimeCommandAcknowledged:
		if typed.ClientEventID == "" {
			return nil
		}

		command, ok := t.complete(typed.ClientEventID)
		if !ok {
			return nil
		}

		return command.muted

	case model.RealtimeError:
		if typed.ClientEventID != "" {
			t.complete(typed.ClientEventID)
		}
	}

	return nil
}

func (t *realtimeCommandTracker) complete(eventID string) (realtimeCommand, bool) {
	command, ok := t.pending[eventID]
	if !ok {
		return realtimeCommand{}, false
	}

	delete(t.pending, eventID)

	t.recent = append(t.recent, eventID)
	if len(t.recent) > maxRecentRealtimeCommands {
		t.recent = slices.Delete(t.recent, 0, 1)
	}

	return command, true
}

// realtimeMuteState returns the mute state a command sets, or nil.
func realtimeMuteState(event model.RealtimeClientEvent) *bool {
	switch event.(type) {
	case model.RealtimeInputAudioMute:
		return new(true)
	case model.RealtimeInputAudioUnmute:
		return new(false)
	default:
		return nil
	}
}

// withMuteEventID gives mute commands without an ID a generated one, so
// that their acknowledgement can update the mute state.
func withMuteEventID(event model.RealtimeClientEvent) model.RealtimeClientEvent {
	switch typed := event.(type) {
	case model.RealtimeInputAudioMute:
		if typed.EventID == "" {
			typed.EventID = id.New("")
		}

		return typed

	case model.RealtimeInputAudioUnmute:
		if typed.EventID == "" {
			typed.EventID = id.New("")
		}

		return typed

	default:
		return event
	}
}

// realtimeClientEventID returns the client event ID of the commands that
// carry one.
func realtimeClientEventID(event model.RealtimeClientEvent) string {
	switch typed := event.(type) {
	case model.RealtimeSessionUpdate:
		return typed.EventID
	case model.RealtimeSessionStart:
		return typed.EventID
	case model.RealtimeSessionClose:
		return typed.EventID
	case model.RealtimeInputAudioMute:
		return typed.EventID
	case model.RealtimeInputAudioUnmute:
		return typed.EventID
	case model.RealtimeContextAppend:
		return typed.EventID
	case model.RealtimeInputAudioAppend:
		return typed.EventID
	default:
		return ""
	}
}
