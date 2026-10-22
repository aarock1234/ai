package xai

import (
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/internal/livetranscription"
	"github.com/aarock1234/ai/internal/websocket"
	"github.com/aarock1234/ai/pkg/model"
)

// Streaming transcription message types.
const (
	transcriptCreated = "transcript.created"
	transcriptPartial = "transcript.partial"
	transcriptDone    = "transcript.done"
	streamingError    = "error"
	audioDone         = "audio.done"
)

// defaultStreamingErrorMessage describes error events without a message.
const defaultStreamingErrorMessage = "xAI STT error"

// Separators of the finish text: utterances within a channel, and
// channels.
const (
	utteranceSeparator = " "
	channelSeparator   = "\n"
)

// streamingEvent is a server message; fields beyond the type depend on it.
type streamingEvent struct {
	Type         string   `json:"type"`
	Text         string   `json:"text"`
	IsFinal      bool     `json:"is_final"`
	SpeechFinal  bool     `json:"speech_final"`
	Start        *float64 `json:"start"`
	Duration     *float64 `json:"duration"`
	ChannelIndex *int     `json:"channel_index"`
	Message      *string  `json:"message"`
}

// audioDoneMessage ends the audio input.
type audioDoneMessage struct {
	Type string `json:"type"`
}

// streamingProtocolSetup configures a streaming transcription.
type streamingProtocolSetup struct {
	warnings []model.Warning
	language string

	// expectedChannels is the number of done events that finish the
	// stream, one per channel.
	expectedChannels int
}

// streamingProtocol is the xAI streaming transcription protocol. Partial
// events carry revisable text until speech_final completes an utterance;
// done events, one per channel, carry no text, so the finish text is
// rebuilt from the utterances.
type streamingProtocol struct {
	streamingProtocolSetup

	finalized map[int][]string
	pending   map[int]string
	done      map[int]string
	duration  *float64
}

func newStreamingProtocol(setup streamingProtocolSetup) *streamingProtocol {
	return &streamingProtocol{
		streamingProtocolSetup: setup,
		finalized:              make(map[int][]string),
		pending:                make(map[int]string),
		done:                   make(map[int]string),
	}
}

// Open waits for the transcript to be created.
func (p *streamingProtocol) Open(*livetranscription.TranscriptionSession) {}

// HandleMessage handles one server event.
func (p *streamingProtocol) HandleMessage(session *livetranscription.TranscriptionSession, message []byte) {
	var event streamingEvent
	if err := json.Unmarshal(message, &event); err != nil {
		session.Fail(fmt.Errorf("%w: decode xai streaming transcription event: %w", model.ErrInvalidResponse, err))

		return
	}

	switch event.Type {
	case transcriptCreated:
		session.Emit(model.StreamStart{
			Warnings: p.warnings,
		})
		session.StartAudio()

	case transcriptPartial:
		p.handlePartial(session, event)

	case transcriptDone:
		p.handleDone(session, event)

	case streamingError:
		errorMessage := defaultStreamingErrorMessage
		if event.Message != nil {
			errorMessage = *event.Message
		}

		session.Fail(&model.APICallError{
			Message:      errorMessage,
			ResponseBody: string(message),
		})
	}
}

// handlePartial emits a final transcript when speech_final completes an
// utterance. Other partials, including is_final fragments that are revised
// later, stay partial.
func (p *streamingProtocol) handlePartial(session *livetranscription.TranscriptionSession, event streamingEvent) {
	channel := channelOf(event)

	if !event.IsFinal || !event.SpeechFinal {
		p.pending[channel] = event.Text

		session.Emit(model.TranscriptPartial{
			ID:           channelID(event.ChannelIndex),
			Text:         event.Text,
			Start:        secondsOf(event.Start),
			Duration:     secondsOf(event.Duration),
			ChannelIndex: event.ChannelIndex,
		})

		return
	}

	if event.Text != "" {
		p.finalized[channel] = append(p.finalized[channel], event.Text)
	}

	delete(p.pending, channel)

	final := model.TranscriptFinal{
		ID:           channelID(event.ChannelIndex),
		Text:         event.Text,
		Start:        secondsOf(event.Start),
		ChannelIndex: event.ChannelIndex,
	}

	if event.Start != nil && event.Duration != nil {
		final.End = new(seconds(*event.Start + *event.Duration))
	}

	session.Emit(final)
}

// handleDone records the text of a finished channel, falling back to its
// utterances and trailing partial text, and finishes the stream once every
// channel is done.
func (p *streamingProtocol) handleDone(session *livetranscription.TranscriptionSession, event streamingEvent) {
	channel := channelOf(event)

	text := event.Text
	if text == "" {
		utterances := slices.Clone(p.finalized[channel])
		if pending := p.pending[channel]; pending != "" {
			utterances = append(utterances, pending)
		}

		text = strings.Join(utterances, utteranceSeparator)
	}

	p.done[channel] = text

	if event.Duration != nil {
		p.duration = event.Duration
	}

	if len(p.done) < p.expectedChannels {
		return
	}

	p.finish(session)
}

func (p *streamingProtocol) finish(session *livetranscription.TranscriptionSession) {
	channels := slices.Sorted(maps.Keys(p.done))

	texts := make([]string, 0, len(channels))
	for _, channel := range channels {
		texts = append(texts, p.done[channel])
	}

	finish := model.TranscriptFinish{
		Text:     strings.Join(texts, channelSeparator),
		Language: p.language,
	}

	if p.duration != nil {
		finish.Duration = seconds(*p.duration)
	}

	session.Finish(finish)
}

// EncodeAudio sends the raw audio as a BINARY frame.
func (p *streamingProtocol) EncodeAudio(chunk []byte, _ bool) (livetranscription.Frame, error) {
	return livetranscription.BinaryFrame(chunk), nil
}

// AudioEnded tells the server that the audio is complete.
func (p *streamingProtocol) AudioEnded(session *livetranscription.TranscriptionSession, _ int64) {
	session.SendJSON(audioDoneMessage{
		Type: audioDone,
	})
}

// TimerFired does nothing: the protocol sets no timer.
func (p *streamingProtocol) TimerFired(*livetranscription.TranscriptionSession) {}

// Closed lets a close before every channel finished fail the stream.
func (p *streamingProtocol) Closed(*livetranscription.TranscriptionSession, *websocket.CloseError) {}

func channelOf(event streamingEvent) int {
	if event.ChannelIndex == nil {
		return 0
	}

	return *event.ChannelIndex
}

// channelID identifies the transcripts of a channel, or is empty for
// single-channel audio.
func channelID(channelIndex *int) string {
	if channelIndex == nil {
		return ""
	}

	return "channel-" + strconv.Itoa(*channelIndex)
}

func secondsOf(value *float64) *time.Duration {
	if value == nil {
		return nil
	}

	return new(seconds(*value))
}
