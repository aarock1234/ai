package elevenlabs

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/livetranscription"
	"github.com/aarock1234/ai/internal/websocket"
	"github.com/aarock1234/ai/pkg/model"
)

// finalCommitGracePeriod is how long the stream waits after the final
// commit for trailing events before it finishes.
const finalCommitGracePeriod = 250 * time.Millisecond

// realtimeDefaultErrorMessage describes error events without a message.
const realtimeDefaultErrorMessage = "ElevenLabs realtime transcription error"

// Realtime message types.
const (
	realtimeSessionStarted                = "session_started"
	realtimePartialTranscript             = "partial_transcript"
	realtimeFinalTranscript               = "final_transcript"
	realtimeCommittedTranscript           = "committed_transcript"
	realtimeFinalTranscriptTimestamps     = "final_transcript_with_timestamps"
	realtimeCommittedTranscriptTimestamps = "committed_transcript_with_timestamps"
	realtimeAudioChunkMessage             = "input_audio_chunk"
	realtimeDefaultSessionID              = "session"
	realtimeTranscriptSeparator           = " "
	realtimeSegmentIDSeparator            = ":"
)

// realtimeErrorTypes are the message types that report an error.
var realtimeErrorTypes = []string{
	"auth_error",
	"chunk_size_exceeded",
	"commit_throttled",
	"error",
	"input_error",
	"insufficient_audio_activity",
	"queue_overflow",
	"quota_exceeded",
	"rate_limited",
	"resource_exhausted",
	"session_time_limit_exceeded",
	"transcriber_error",
	"unaccepted_terms",
}

// realtimeLateFinalizationErrorTypes are errors that the final commit may
// cause once transcripts were committed; they finish the stream instead.
var realtimeLateFinalizationErrorTypes = []string{
	"commit_throttled",
	"input_error",
	"insufficient_audio_activity",
}

// realtimeAudioChunk sends audio, or commits it when Commit is set.
type realtimeAudioChunk struct {
	MessageType  string  `json:"message_type"`
	AudioBase64  string  `json:"audio_base_64"`
	Commit       bool    `json:"commit"`
	SampleRate   int     `json:"sample_rate"`
	PreviousText *string `json:"previous_text,omitempty"`
}

// realtimeEvent is a server message; fields beyond the type depend on it.
type realtimeEvent struct {
	MessageType  string         `json:"message_type"`
	SessionID    string         `json:"session_id"`
	Text         string         `json:"text"`
	LanguageCode *string        `json:"language_code"`
	Words        []realtimeWord `json:"words"`
	Error        *string        `json:"error"`
}

type realtimeWord struct {
	Text  string   `json:"text"`
	Start *float64 `json:"start"`
	End   *float64 `json:"end"`
	Type  string   `json:"type"`
}

// realtimeProtocolSetup configures a realtime transcription.
type realtimeProtocolSetup struct {
	warnings                 []model.Warning
	sampleRate               int
	previousText             *string
	language                 string
	includeTimestamps        bool
	includeLanguageDetection bool
}

// realtimeProtocol is the ElevenLabs realtime transcription protocol. The
// audio starts once the session started; after the input ends, the stream
// finishes shortly after the commit of the remaining audio.
type realtimeProtocol struct {
	realtimeProtocolSetup

	sessionID    string
	segmentIndex int
	language     string

	endOfInput              bool
	receivedPostInputCommit bool

	committedEvents             int
	committedEventsAtEndOfInput int
	finalCommitEvents           *int
	timestampedCommits          int

	finalTexts    []string
	finalSegments []model.TranscriptionSegment
}

func newRealtimeProtocol(setup realtimeProtocolSetup) *realtimeProtocol {
	return &realtimeProtocol{
		realtimeProtocolSetup: setup,
		language:              setup.language,
	}
}

// expectsDetailedCommit reports whether each commit is followed by a
// timestamped commit event.
func (p *realtimeProtocol) expectsDetailedCommit() bool {
	return p.includeTimestamps || p.includeLanguageDetection
}

// Open waits for the session to start.
func (p *realtimeProtocol) Open(*livetranscription.TranscriptionSession) {}

// HandleMessage handles one server event.
func (p *realtimeProtocol) HandleMessage(session *livetranscription.TranscriptionSession, message []byte) {
	var event realtimeEvent
	if err := json.Unmarshal(message, &event); err != nil {
		session.Fail(fmt.Errorf("%w: decode elevenlabs realtime event: %w", model.ErrInvalidResponse, err))

		return
	}

	if slices.Contains(realtimeErrorTypes, event.MessageType) {
		p.handleError(session, event, message)

		return
	}

	switch event.MessageType {
	case realtimeSessionStarted:
		p.sessionID = event.SessionID

		session.Emit(model.StreamStart{
			Warnings: p.warnings,
		})
		session.StartAudio()

	case realtimePartialTranscript:
		session.Emit(model.TranscriptPartial{
			ID:   p.segmentID(),
			Text: event.Text,
		})

	case realtimeFinalTranscript, realtimeCommittedTranscript:
		p.handleCommit(session, event)

	case realtimeFinalTranscriptTimestamps, realtimeCommittedTranscriptTimestamps:
		p.handleTimestampedCommit(session, event)
	}
}

// handleError finishes the stream when the final commit was rejected after
// earlier transcripts were committed, and fails it otherwise.
func (p *realtimeProtocol) handleError(session *livetranscription.TranscriptionSession, event realtimeEvent, message []byte) {
	committed := p.committedEvents > 0 || p.timestampedCommits > 0
	if p.endOfInput && committed && slices.Contains(realtimeLateFinalizationErrorTypes, event.MessageType) {
		p.receivedPostInputCommit = true
		p.finish(session)

		return
	}

	errorMessage := realtimeDefaultErrorMessage
	if event.Error != nil {
		errorMessage = *event.Error
	}

	session.Fail(&model.APICallError{
		Message:      errorMessage,
		ResponseBody: string(message),
	})
}

// handleCommit emits a committed transcript. After the input ended, the
// commit of the remaining audio schedules the finish, unless a timestamped
// commit event follows it.
func (p *realtimeProtocol) handleCommit(session *livetranscription.TranscriptionSession, event realtimeEvent) {
	p.committedEvents++

	text := strings.TrimSpace(event.Text)
	if text != "" {
		final := model.TranscriptFinal{
			ID:   p.segmentID(),
			Text: text,
		}

		p.finalTexts = append(p.finalTexts, text)
		p.segmentIndex++

		session.Emit(final)
	}

	if !p.endOfInput || p.committedEvents <= p.committedEventsAtEndOfInput {
		return
	}

	p.receivedPostInputCommit = true
	p.finalCommitEvents = new(p.committedEvents)

	if !p.expectsDetailedCommit() || event.MessageType == realtimeFinalTranscript {
		session.SetTimer(finalCommitGracePeriod)
	}
}

// handleTimestampedCommit records the language and word timestamps of a
// commit. It normally follows the commit's transcript event; a timestamped
// event without one emits the transcript itself.
func (p *realtimeProtocol) handleTimestampedCommit(session *livetranscription.TranscriptionSession, event realtimeEvent) {
	text := strings.TrimSpace(event.Text)
	words := timedWords(event.Words)

	if event.LanguageCode != nil {
		p.language = *event.LanguageCode
	}

	if p.timestampedCommits >= len(p.finalTexts) && text != "" {
		session.Emit(p.timestampedFinal(text, words))
		p.finalTexts = append(p.finalTexts, text)
	}

	p.timestampedCommits++

	if p.includeTimestamps {
		p.finalSegments = append(p.finalSegments, words...)
	}

	isFinalCommit := p.finalCommitEvents == nil || p.timestampedCommits >= *p.finalCommitEvents
	if p.endOfInput && p.timestampedCommits > p.committedEventsAtEndOfInput && isFinalCommit {
		p.receivedPostInputCommit = true
		session.SetTimer(finalCommitGracePeriod)
	}
}

// timestampedFinal returns the final transcript of a timestamped commit,
// timed by its words when timestamps were requested.
func (p *realtimeProtocol) timestampedFinal(text string, words []model.TranscriptionSegment) model.TranscriptFinal {
	final := model.TranscriptFinal{
		ID:   p.segmentID(),
		Text: text,
	}

	p.segmentIndex++

	if p.includeTimestamps && len(words) > 0 {
		final.Start = new(words[0].Start)
		final.End = new(words[len(words)-1].End)
	}

	return final
}

// EncodeAudio returns one audio chunk. The previous text, if any, rides
// the first chunk.
func (p *realtimeProtocol) EncodeAudio(chunk []byte, first bool) (livetranscription.Frame, error) {
	message := realtimeAudioChunk{
		MessageType: realtimeAudioChunkMessage,
		AudioBase64: base64.StdEncoding.EncodeToString(chunk),
		Commit:      false,
		SampleRate:  p.sampleRate,
	}

	if first {
		message.PreviousText = p.previousText
	}

	return livetranscription.JSONFrame(message)
}

// AudioEnded commits the remaining audio.
func (p *realtimeProtocol) AudioEnded(session *livetranscription.TranscriptionSession, _ int64) {
	p.committedEventsAtEndOfInput = p.committedEvents
	p.endOfInput = true

	session.SendJSON(realtimeAudioChunk{
		MessageType: realtimeAudioChunkMessage,
		AudioBase64: "",
		Commit:      true,
		SampleRate:  p.sampleRate,
	})
}

// TimerFired finishes the stream once the final commit's grace period
// passed.
func (p *realtimeProtocol) TimerFired(session *livetranscription.TranscriptionSession) {
	p.finish(session)
}

// Closed finishes the stream when the remaining audio was committed. Any
// other close fails it.
func (p *realtimeProtocol) Closed(session *livetranscription.TranscriptionSession, _ *websocket.CloseError) {
	if p.endOfInput && p.receivedPostInputCommit {
		p.finish(session)
	}
}

func (p *realtimeProtocol) finish(session *livetranscription.TranscriptionSession) {
	finish := model.TranscriptFinish{
		Text:     strings.TrimSpace(strings.Join(p.finalTexts, realtimeTranscriptSeparator)),
		Segments: p.finalSegments,
		Language: p.language,
	}

	if len(p.finalSegments) > 0 {
		finish.Duration = p.finalSegments[len(p.finalSegments)-1].End
	}

	session.Finish(finish)
}

// segmentID identifies the current segment within the session.
func (p *realtimeProtocol) segmentID() string {
	sessionID := p.sessionID
	if sessionID == "" {
		sessionID = realtimeDefaultSessionID
	}

	return sessionID + realtimeSegmentIDSeparator + strconv.Itoa(p.segmentIndex)
}

// timedWords returns the words that have both a start and an end time.
func timedWords(words []realtimeWord) []model.TranscriptionSegment {
	var segments []model.TranscriptionSegment

	for _, word := range words {
		if word.Start == nil || word.End == nil {
			continue
		}

		segments = append(segments, model.TranscriptionSegment{
			Text:  word.Text,
			Start: seconds(*word.Start),
			End:   seconds(*word.End),
		})
	}

	return segments
}
