package geminilive

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"github.com/aarock1234/ai/internal/livetranscription"
	"github.com/aarock1234/ai/internal/websocket"
	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/model"
)

// segmentIDPrefix starts the synthetic segment IDs: Live messages carry no
// response or item IDs.
const segmentIDPrefix = "google-segment-"

// Interaction statuses that report that all processing is complete. The
// early access builds report REQUIRES_ACTION.
const (
	interactionIdle           = "IDLE"
	interactionRequiresAction = "REQUIRES_ACTION"
)

// serverMessage is one Live API server message.
type serverMessage struct {
	SetupComplete      json.RawMessage `json:"setupComplete"`
	ServerContent      *serverContent  `json:"serverContent"`
	InputTranscription *transcription  `json:"inputTranscription"`
	UsageMetadata      json.RawMessage `json:"usageMetadata"`
	Error              *serverError    `json:"error"`
}

type serverContent struct {
	InputTranscription        *transcription `json:"inputTranscription"`
	InterimInputTranscription *transcription `json:"interimInputTranscription"`
	TurnComplete              *bool          `json:"turnComplete"`
	InteractionStatus         *string        `json:"interactionStatus"`
}

type transcription struct {
	Text         string  `json:"text"`
	Finished     *bool   `json:"finished"`
	LanguageCode *string `json:"languageCode"`
}

type serverError struct {
	Message *string `json:"message"`
}

// realtimeInputMessage sends audio or ends it.
type realtimeInputMessage struct {
	RealtimeInput realtimeInput `json:"realtimeInput"`
}

type realtimeInput struct {
	Audio          *realtimeAudio `json:"audio,omitempty"`
	AudioStreamEnd bool           `json:"audioStreamEnd,omitempty"`
}

type realtimeAudio struct {
	Data     string `json:"data"`
	MimeType string `json:"mimeType"`
}

// ProtocolSetup configures a Live transcription.
type ProtocolSetup struct {
	Warnings []model.Warning

	// SetupMessage is the first client message.
	SetupMessage []byte

	// FinishGrace is the quiet window after the input audio ended. Zero
	// selects DefaultFinishGrace.
	FinishGrace time.Duration

	// DefaultErrorMessage describes error messages without one, such as
	// "Google Live API error".
	DefaultErrorMessage string

	// Metadata returns the provider metadata of the reported usage. Nil
	// reports no provider metadata.
	Metadata func(usageMetadata json.RawMessage) model.ProviderMetadata
}

// Protocol is the Live transcription protocol. Audio may only be sent once
// the server acknowledged the setup. Transcription fragments accumulate per
// segment; a finished transcription or a complete turn finalizes it.
type Protocol struct {
	ProtocolSetup

	segmentCounter int
	segmentBuffer  string
	fullText       string

	// latestInterim is the latest revisable text: the final text of a
	// segment that never receives a finished transcription.
	latestInterim string

	language      string
	audioEnded    bool
	usageMetadata json.RawMessage
}

// NewProtocol returns the protocol of one Live transcription.
func NewProtocol(setup ProtocolSetup) *Protocol {
	if setup.FinishGrace <= 0 {
		setup.FinishGrace = DefaultFinishGrace
	}

	return &Protocol{
		ProtocolSetup: setup,
	}
}

// Open starts the stream and sends the setup.
func (p *Protocol) Open(session *livetranscription.TranscriptionSession) {
	session.Emit(model.StreamStart{
		Warnings: p.Warnings,
	})
	session.Send(livetranscription.TextFrame(p.SetupMessage))
}

// HandleMessage handles one server message.
func (p *Protocol) HandleMessage(session *livetranscription.TranscriptionSession, message []byte) {
	var decoded serverMessage
	if err := json.Unmarshal(message, &decoded); err != nil {
		session.Fail(fmt.Errorf("%w: decode gemini live message: %w", model.ErrInvalidResponse, err))

		return
	}

	if wirejson.HasValue(decoded.SetupComplete) {
		session.StartAudio()
	}

	if wirejson.HasValue(decoded.UsageMetadata) {
		p.usageMetadata = decoded.UsageMetadata
	}

	if decoded.Error != nil {
		p.fail(session, decoded.Error, message)

		return
	}

	content := decoded.ServerContent
	if content == nil {
		content = &serverContent{}
	}

	p.handleInterim(session, content.InterimInputTranscription)

	input := content.InputTranscription
	if input == nil {
		input = decoded.InputTranscription
	}

	p.handleTranscription(session, input)

	turnComplete := content.TurnComplete != nil && *content.TurnComplete
	if turnComplete {
		p.completeSegment(session)
	}

	if p.audioEnded && isComplete(content.InteractionStatus, turnComplete) {
		p.finish(session)
	}
}

func (p *Protocol) fail(session *livetranscription.TranscriptionSession, detail *serverError, message []byte) {
	errorMessage := p.DefaultErrorMessage
	if detail.Message != nil {
		errorMessage = *detail.Message
	}

	session.Fail(&model.APICallError{
		Message:      errorMessage,
		ResponseBody: string(message),
	})
}

// handleInterim emits low-latency revisable text while the user speaks.
func (p *Protocol) handleInterim(session *livetranscription.TranscriptionSession, interim *transcription) {
	if interim == nil || interim.Text == "" {
		return
	}

	p.schedulePendingFinish(session)
	p.latestInterim = interim.Text

	session.Emit(model.TranscriptPartial{
		ID:   p.segmentID(),
		Text: interim.Text,
	})
}

// handleTranscription accumulates a transcription fragment, which
// supersedes interim text, and finalizes the segment once it is finished.
func (p *Protocol) handleTranscription(session *livetranscription.TranscriptionSession, input *transcription) {
	if input == nil {
		return
	}

	if input.LanguageCode != nil {
		p.language = *input.LanguageCode
	}

	if input.Text != "" {
		p.schedulePendingFinish(session)
		p.latestInterim = ""
		p.segmentBuffer += input.Text

		session.Emit(model.TranscriptDelta{
			ID:    p.segmentID(),
			Delta: input.Text,
		})
	}

	if input.Finished != nil && *input.Finished {
		p.completeSegment(session)
	}
}

// isComplete reports the definitive all-processing-complete signal: an idle
// interaction, or a complete turn without an interaction status.
func isComplete(status *string, turnComplete bool) bool {
	if status == nil {
		return turnComplete
	}

	return *status == interactionIdle || *status == interactionRequiresAction
}

// completeSegment emits the final text of the current segment, falling
// back to the latest interim text.
func (p *Protocol) completeSegment(session *livetranscription.TranscriptionSession) {
	if p.segmentBuffer == "" {
		if p.latestInterim == "" {
			return
		}

		p.segmentBuffer = p.latestInterim
	}

	p.latestInterim = ""

	session.Emit(model.TranscriptFinal{
		ID:   p.segmentID(),
		Text: p.segmentBuffer,
	})

	if p.fullText == "" {
		p.fullText = p.segmentBuffer
	} else {
		p.fullText += " " + p.segmentBuffer
	}

	p.segmentBuffer = ""
	p.segmentCounter++
}

// schedulePendingFinish restarts the quiet window once the input audio
// ended: trailing transcripts may still arrive.
func (p *Protocol) schedulePendingFinish(session *livetranscription.TranscriptionSession) {
	if !p.audioEnded {
		return
	}

	session.SetTimer(p.FinishGrace)
}

// EncodeAudio sends one chunk of base64 audio.
func (p *Protocol) EncodeAudio(chunk []byte, _ bool) (livetranscription.Frame, error) {
	return livetranscription.JSONFrame(realtimeInputMessage{
		RealtimeInput: realtimeInput{
			Audio: &realtimeAudio{
				Data:     base64.StdEncoding.EncodeToString(chunk),
				MimeType: audioMimeType,
			},
		},
	})
}

// AudioEnded ends the audio stream and starts the quiet window.
func (p *Protocol) AudioEnded(session *livetranscription.TranscriptionSession, _ int64) {
	session.SendJSON(realtimeInputMessage{
		RealtimeInput: realtimeInput{
			AudioStreamEnd: true,
		},
	})

	p.audioEnded = true
	p.schedulePendingFinish(session)
}

// TimerFired finishes the stream after the quiet window.
func (p *Protocol) TimerFired(session *livetranscription.TranscriptionSession) {
	p.finish(session)
}

// Closed finishes the stream once the input audio ended: the server
// delivered everything it will. An earlier close fails it.
func (p *Protocol) Closed(session *livetranscription.TranscriptionSession, _ *websocket.CloseError) {
	if p.audioEnded {
		p.finish(session)
	}
}

func (p *Protocol) finish(session *livetranscription.TranscriptionSession) {
	p.completeSegment(session)

	finish := model.TranscriptFinish{
		Text:     p.fullText,
		Language: p.language,
	}

	if p.usageMetadata != nil {
		finish.Usage = p.usageMetadata

		if p.Metadata != nil {
			finish.ProviderMetadata = p.Metadata(p.usageMetadata)
		}
	}

	session.Finish(finish)
}

func (p *Protocol) segmentID() string {
	return segmentIDPrefix + strconv.Itoa(p.segmentCounter)
}
