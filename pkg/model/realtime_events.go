package model

import (
	"encoding/json"
	"time"
)

// RealtimeClientEvent is a normalized event sent to a realtime model. Each
// model serializes it into its own message format.
type RealtimeClientEvent interface {
	isRealtimeClientEvent()
}

// RealtimeSessionUpdate changes the session configuration.
type RealtimeSessionUpdate struct {
	Config  RealtimeSessionConfig
	EventID string
}

// RealtimeSessionStart starts a session on models whose Startup is
// RealtimeStartupSessionStart.
type RealtimeSessionStart struct {
	Config  RealtimeSessionConfig
	EventID string
}

// RealtimeSessionClose asks the provider to end the session and report the
// final usage.
type RealtimeSessionClose struct {
	EventID string
}

// RealtimeInputAudioMute stops the model from hearing input audio.
type RealtimeInputAudioMute struct {
	EventID string
}

// RealtimeInputAudioUnmute resumes input audio.
type RealtimeInputAudioUnmute struct {
	EventID string
}

// RealtimeContextAppend adds context to a continuous conversation, such as
// the result of a delegated task.
type RealtimeContextAppend struct {
	Content string

	// DelegationID answers the delegation it names. Nil appends context
	// that answers no delegation.
	DelegationID *string

	EventID         string
	ProviderOptions ProviderOptions
}

// RealtimeInputAudioAppend appends audio to the input buffer.
type RealtimeInputAudioAppend struct {
	// Audio is base64-encoded audio in the input format.
	Audio   string
	EventID string
}

// RealtimeInputAudioCommit commits the input buffer as a user turn.
type RealtimeInputAudioCommit struct{}

// RealtimeInputAudioClear discards the input buffer.
type RealtimeInputAudioClear struct{}

// RealtimeConversationItemCreate adds an item to the conversation.
type RealtimeConversationItemCreate struct {
	Item RealtimeConversationItem
}

// RealtimeConversationItemTruncate cuts off assistant audio that the user
// did not hear, such as after an interruption.
type RealtimeConversationItemTruncate struct {
	// ItemID is the assistant message to truncate.
	ItemID string

	// ContentIndex is the content part to truncate.
	ContentIndex int

	// AudioEnd is the audio played before the truncation.
	AudioEnd time.Duration
}

// RealtimeResponseCreate asks the model to respond.
type RealtimeResponseCreate struct {
	// Options is nil for the session defaults.
	Options *RealtimeResponseOptions
}

// RealtimeResponseOptions override the session defaults for one response.
// Empty fields keep the defaults.
type RealtimeResponseOptions struct {
	Modalities   []RealtimeModality
	Instructions string
	Metadata     map[string]any
}

// RealtimeResponseCancel cancels the response in progress.
type RealtimeResponseCancel struct{}

func (RealtimeSessionUpdate) isRealtimeClientEvent()            {}
func (RealtimeSessionStart) isRealtimeClientEvent()             {}
func (RealtimeSessionClose) isRealtimeClientEvent()             {}
func (RealtimeInputAudioMute) isRealtimeClientEvent()           {}
func (RealtimeInputAudioUnmute) isRealtimeClientEvent()         {}
func (RealtimeContextAppend) isRealtimeClientEvent()            {}
func (RealtimeInputAudioAppend) isRealtimeClientEvent()         {}
func (RealtimeInputAudioCommit) isRealtimeClientEvent()         {}
func (RealtimeInputAudioClear) isRealtimeClientEvent()          {}
func (RealtimeConversationItemCreate) isRealtimeClientEvent()   {}
func (RealtimeConversationItemTruncate) isRealtimeClientEvent() {}
func (RealtimeResponseCreate) isRealtimeClientEvent()           {}
func (RealtimeResponseCancel) isRealtimeClientEvent()           {}

// RealtimeConversationItem is an item the client adds to the conversation:
// RealtimeTextMessage, RealtimeAudioMessage or RealtimeFunctionCallOutput.
type RealtimeConversationItem interface {
	isRealtimeConversationItem()
}

// RealtimeTextMessage is a text message from the user.
type RealtimeTextMessage struct {
	Text string
}

// RealtimeAudioMessage is a complete audio message from the user.
type RealtimeAudioMessage struct {
	// Audio is base64-encoded audio in the input format.
	Audio string
}

// RealtimeFunctionCallOutput is the result of a function call, sent back so
// that the model can continue its response.
type RealtimeFunctionCallOutput struct {
	// CallID is the call ID of the function call arguments done event.
	CallID string

	// Name is the function name. Some providers route results by it.
	Name string

	// Output is the JSON-encoded result.
	Output string
}

func (RealtimeTextMessage) isRealtimeConversationItem()        {}
func (RealtimeAudioMessage) isRealtimeConversationItem()       {}
func (RealtimeFunctionCallOutput) isRealtimeConversationItem() {}

// RealtimeServerEvent is a normalized event received from a realtime model.
// Every event carries the provider's message in Raw.
type RealtimeServerEvent interface {
	isRealtimeServerEvent()
}

// RealtimeDelegationTarget is who handles delegated work.
type RealtimeDelegationTarget string

// Delegation targets.
const (
	RealtimeDelegateToClient   RealtimeDelegationTarget = "client"
	RealtimeDelegateToProvider RealtimeDelegationTarget = "provider"
)

// RealtimeSpeaker is the speaker of a transcript fragment.
type RealtimeSpeaker string

// Transcript speakers.
const (
	RealtimeUserSpeaker      RealtimeSpeaker = "user"
	RealtimeAssistantSpeaker RealtimeSpeaker = "assistant"
)

// RealtimeSessionStarted reports a session started with a session start
// event.
type RealtimeSessionStarted struct {
	SessionID string

	// DelegationMode is empty when the provider did not report it.
	DelegationMode RealtimeDelegationTarget

	Raw json.RawMessage
}

// RealtimeSessionClosed reports the end of a session with its final usage.
type RealtimeSessionClosed struct {
	SessionID string

	// Duration is the total session duration.
	Duration time.Duration

	Reason string
	Raw    json.RawMessage
}

// RealtimeSessionUsage reports the usage of a session so far.
type RealtimeSessionUsage struct {
	// Duration is the cumulative session duration, not an increment.
	Duration time.Duration

	// ContextWindowUsageRatio is nil when the provider did not report it.
	ContextWindowUsageRatio *float64

	Raw json.RawMessage
}

// RealtimeAudioChunk is audio output of a continuous conversation.
type RealtimeAudioChunk struct {
	// Delta is base64-encoded audio in the output format.
	Delta string
	Raw   json.RawMessage
}

// RealtimeTranscriptFragment is a timed piece of the transcript of a
// continuous conversation.
type RealtimeTranscriptFragment struct {
	Speaker RealtimeSpeaker
	Delta   string

	Start time.Duration
	End   time.Duration

	Raw json.RawMessage
}

// RealtimeDelegationCreated reports work the model delegated.
type RealtimeDelegationCreated struct {
	DelegationID string

	// Target is empty when the provider did not report it.
	Target RealtimeDelegationTarget

	// Offset is nil when the provider did not report it.
	Offset *time.Duration

	ResponseID string
	Raw        json.RawMessage
}

// RealtimeCommandAcknowledged reports that the provider applied a client
// command.
type RealtimeCommandAcknowledged struct {
	// Command is the provider's name of the command.
	Command string

	// ClientEventID is the event ID the client sent, if any.
	ClientEventID string

	Raw json.RawMessage
}

// RealtimeSessionCreated reports a new session.
type RealtimeSessionCreated struct {
	SessionID string
	Raw       json.RawMessage
}

// RealtimeSessionUpdated reports an applied session update.
type RealtimeSessionUpdated struct {
	Raw json.RawMessage
}

// RealtimeSpeechStarted reports that the user started speaking.
type RealtimeSpeechStarted struct {
	ItemID string
	Raw    json.RawMessage
}

// RealtimeSpeechStopped reports that the user stopped speaking.
type RealtimeSpeechStopped struct {
	ItemID string
	Raw    json.RawMessage
}

// RealtimeAudioCommitted reports that the input buffer became an item.
type RealtimeAudioCommitted struct {
	ItemID         string
	PreviousItemID string
	Raw            json.RawMessage
}

// RealtimeConversationItemAdded reports an item added to the conversation.
type RealtimeConversationItemAdded struct {
	ItemID string

	// Item is the provider's item.
	Item json.RawMessage

	Raw json.RawMessage
}

// RealtimeInputTranscriptionCompleted is the transcript of the user's
// audio.
type RealtimeInputTranscriptionCompleted struct {
	ItemID     string
	Transcript string
	Raw        json.RawMessage
}

// RealtimeResponseCreated reports the start of a response.
type RealtimeResponseCreated struct {
	ResponseID string
	Raw        json.RawMessage
}

// RealtimeResponseDone reports the end of a response.
type RealtimeResponseDone struct {
	ResponseID string

	// Status is the provider's response status, such as "completed".
	Status string

	Raw json.RawMessage
}

// RealtimeOutputItemAdded reports a new output item of a response.
type RealtimeOutputItemAdded struct {
	ResponseID string
	ItemID     string
	Raw        json.RawMessage
}

// RealtimeOutputItemDone reports a finished output item.
type RealtimeOutputItemDone struct {
	ResponseID string
	ItemID     string
	Raw        json.RawMessage
}

// RealtimeContentPartAdded reports a new content part of an output item.
type RealtimeContentPartAdded struct {
	ResponseID string
	ItemID     string
	Raw        json.RawMessage
}

// RealtimeContentPartDone reports a finished content part.
type RealtimeContentPartDone struct {
	ResponseID string
	ItemID     string
	Raw        json.RawMessage
}

// RealtimeAudioDelta is audio output of a turn-based response.
type RealtimeAudioDelta struct {
	ResponseID string
	ItemID     string

	// Delta is base64-encoded audio in the output format.
	Delta string

	Raw json.RawMessage
}

// RealtimeAudioDone reports the end of an item's audio output.
type RealtimeAudioDone struct {
	ResponseID string
	ItemID     string
	Raw        json.RawMessage
}

// RealtimeAudioTranscriptDelta appends to the transcript of the model's
// audio.
type RealtimeAudioTranscriptDelta struct {
	ResponseID string
	ItemID     string
	Delta      string
	Raw        json.RawMessage
}

// RealtimeAudioTranscriptDone is the complete transcript of an item's
// audio.
type RealtimeAudioTranscriptDone struct {
	ResponseID string
	ItemID     string

	// Transcript is nil when the provider did not repeat it.
	Transcript *string

	Raw json.RawMessage
}

// RealtimeTextDelta appends to the model's text output.
type RealtimeTextDelta struct {
	ResponseID string
	ItemID     string
	Delta      string
	Raw        json.RawMessage
}

// RealtimeTextDone is the complete text of an item.
type RealtimeTextDone struct {
	ResponseID string
	ItemID     string

	// Text is nil when the provider did not repeat it.
	Text *string

	Raw json.RawMessage
}

// RealtimeFunctionCallArgumentsDelta appends to the JSON arguments of a
// function call.
type RealtimeFunctionCallArgumentsDelta struct {
	ResponseID string
	ItemID     string
	CallID     string
	Delta      string
	Raw        json.RawMessage
}

// RealtimeFunctionCallArgumentsDone is a complete function call.
type RealtimeFunctionCallArgumentsDone struct {
	ResponseID string
	ItemID     string
	CallID     string
	Name       string

	// Arguments is the JSON-encoded arguments.
	Arguments string

	Raw json.RawMessage
}

// RealtimeError is an error reported by the provider.
type RealtimeError struct {
	Message string
	Code    string

	// ClientEventID is the client event that failed, if any.
	ClientEventID string

	Raw json.RawMessage
}

// RealtimeCustom is a provider event without a normalized form.
type RealtimeCustom struct {
	// RawType is the provider's event type.
	RawType string
	Raw     json.RawMessage
}

func (RealtimeSessionStarted) isRealtimeServerEvent()              {}
func (RealtimeSessionClosed) isRealtimeServerEvent()               {}
func (RealtimeSessionUsage) isRealtimeServerEvent()                {}
func (RealtimeAudioChunk) isRealtimeServerEvent()                  {}
func (RealtimeTranscriptFragment) isRealtimeServerEvent()          {}
func (RealtimeDelegationCreated) isRealtimeServerEvent()           {}
func (RealtimeCommandAcknowledged) isRealtimeServerEvent()         {}
func (RealtimeSessionCreated) isRealtimeServerEvent()              {}
func (RealtimeSessionUpdated) isRealtimeServerEvent()              {}
func (RealtimeSpeechStarted) isRealtimeServerEvent()               {}
func (RealtimeSpeechStopped) isRealtimeServerEvent()               {}
func (RealtimeAudioCommitted) isRealtimeServerEvent()              {}
func (RealtimeConversationItemAdded) isRealtimeServerEvent()       {}
func (RealtimeInputTranscriptionCompleted) isRealtimeServerEvent() {}
func (RealtimeResponseCreated) isRealtimeServerEvent()             {}
func (RealtimeResponseDone) isRealtimeServerEvent()                {}
func (RealtimeOutputItemAdded) isRealtimeServerEvent()             {}
func (RealtimeOutputItemDone) isRealtimeServerEvent()              {}
func (RealtimeContentPartAdded) isRealtimeServerEvent()            {}
func (RealtimeContentPartDone) isRealtimeServerEvent()             {}
func (RealtimeAudioDelta) isRealtimeServerEvent()                  {}
func (RealtimeAudioDone) isRealtimeServerEvent()                   {}
func (RealtimeAudioTranscriptDelta) isRealtimeServerEvent()        {}
func (RealtimeAudioTranscriptDone) isRealtimeServerEvent()         {}
func (RealtimeTextDelta) isRealtimeServerEvent()                   {}
func (RealtimeTextDone) isRealtimeServerEvent()                    {}
func (RealtimeFunctionCallArgumentsDelta) isRealtimeServerEvent()  {}
func (RealtimeFunctionCallArgumentsDone) isRealtimeServerEvent()   {}
func (RealtimeError) isRealtimeServerEvent()                       {}
func (RealtimeCustom) isRealtimeServerEvent()                      {}
