package azure

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"slices"
	"time"

	"github.com/aarock1234/ai/internal/livetranscription"
	"github.com/aarock1234/ai/internal/websocket"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// maiRealtimePath is the MAI realtime WebSocket with the transcription
	// intent.
	maiRealtimePath = "/realtime?intent=transcription"

	// maiDefaultSampleRate is the PCM rate when the input format has none.
	maiDefaultSampleRate = 24000

	// maiBytesPerSample is the size of one 16-bit mono PCM sample.
	maiBytesPerSample = 2
)

// maiSampleRates are the PCM rates MAI streaming transcription accepts.
var maiSampleRates = []int{
	16000,
	24000,
}

// MAI realtime event types.
const (
	maiSessionCreated          = "session.created"
	maiSessionUpdated          = "session.updated"
	maiTranscriptDelta         = "conversation.item.input_audio_transcription.delta"
	maiTranscriptIntermediate  = "conversation.item.input_audio_transcription.intermediate"
	maiTranscriptCompleted     = "conversation.item.input_audio_transcription.completed"
	maiTranscriptFailed        = "conversation.item.input_audio_transcription.failed"
	maiErrorEvent              = "error"
	maiAudioAppendEvent        = "input_audio_buffer.append"
	maiAudioCommitEvent        = "input_audio_buffer.commit"
	maiDefaultErrorDescription = "unknown error"
)

// maiSessionUpdate configures a MAI transcription session.
type maiSessionUpdate struct {
	Type    string     `json:"type"`
	Session maiSession `json:"session"`
}

type maiSession struct {
	Type  string          `json:"type"`
	Audio maiSessionAudio `json:"audio"`
}

type maiSessionAudio struct {
	Input maiAudioInput `json:"input"`
}

type maiAudioInput struct {
	Format        maiAudioFormat           `json:"format"`
	Transcription maiTranscriptionSettings `json:"transcription"`

	// TurnDetection and NoiseReduction are always null: MAI supports
	// neither, and the audio is committed once it ends.
	TurnDetection  *maiUnsupportedSetting `json:"turn_detection"`
	NoiseReduction *maiUnsupportedSetting `json:"noise_reduction"`
}

type maiAudioFormat struct {
	Type string `json:"type"`
	Rate int    `json:"rate"`
}

type maiTranscriptionSettings struct {
	Model    string `json:"model"`
	Language string `json:"language,omitempty"`
}

// maiUnsupportedSetting is never sent; see maiAudioInput.
type maiUnsupportedSetting struct{}

// maiAudioAppend sends one chunk of base64 audio.
type maiAudioAppend struct {
	Type  string `json:"type"`
	Audio string `json:"audio"`
}

// maiAudioCommit ends the audio input.
type maiAudioCommit struct {
	Type string `json:"type"`
}

// maiEvent is a server event; fields beyond the type depend on it.
type maiEvent struct {
	Type         string         `json:"type"`
	ItemID       string         `json:"item_id"`
	Delta        string         `json:"delta"`
	Intermediate string         `json:"intermediate"`
	Transcript   *string        `json:"transcript"`
	Error        *maiEventError `json:"error"`
}

type maiEventError struct {
	Message *string `json:"message"`
	Code    *string `json:"code"`
}

// maiTranscriptionModel transcribes streams over the OpenAI
// Realtime-compatible MAI WebSocket. The model ID is the Foundry deployment
// name.
type maiTranscriptionModel struct {
	modelID  string
	settings *settings
}

// streamTranscribe opens a MAI transcription stream. The input audio must
// be 16-bit mono PCM at 16000 or 24000 Hz.
func (m *maiTranscriptionModel) streamTranscribe(ctx context.Context, options model.TranscriptionStreamOptions, azureOptions TranscriptionOptions, warnings []model.Warning) (*model.TranscriptionStreamResult, error) {
	timestamp := time.Now()

	rate, err := maiSampleRate(options.InputAudioFormat)
	if err != nil {
		return nil, err
	}

	sessionUpdate, err := json.Marshal(m.sessionUpdate(rate, azureOptions.Language))
	if err != nil {
		return nil, fmt.Errorf("encode mai session: %w", err)
	}

	conn, err := m.dial(ctx, options)
	if err != nil {
		return nil, err
	}

	protocol := &maiProtocol{
		warnings:       warnings,
		sessionUpdate:  sessionUpdate,
		bytesPerSecond: rate * maiBytesPerSample,
		language:       azureOptions.Language,
	}

	result := livetranscription.NewResult(ctx, livetranscription.TranscriptionSetup{
		Conn:             conn,
		Protocol:         protocol,
		Audio:            options.Audio,
		IncludeRawChunks: options.IncludeRawChunks,
		Name:             "azure mai transcription",
	})
	result.Request = model.RequestInfo{
		Body: sessionUpdate,
	}
	result.Response = model.ResponseInfo{
		Timestamp: timestamp,
		ModelID:   m.modelID,
		Headers:   conn.ResponseHeader(),
	}

	return result, nil
}

// maiSampleRate validates the input audio format and returns its rate.
func maiSampleRate(format model.AudioFormat) (int, error) {
	rate := format.Rate
	if rate == 0 {
		rate = maiDefaultSampleRate
	}

	if format.Type == "audio/pcm" && slices.Contains(maiSampleRates, rate) {
		return rate, nil
	}

	received := format.Type
	if format.Rate != 0 {
		received = fmt.Sprintf("%s at %d Hz", format.Type, format.Rate)
	}

	return 0, fmt.Errorf(
		"%w: MAI streaming transcription requires 16-bit mono PCM (audio/pcm) at 16000 or 24000 Hz, got %s",
		model.ErrInvalidArgument,
		received,
	)
}

func (m *maiTranscriptionModel) sessionUpdate(rate int, language string) maiSessionUpdate {
	return maiSessionUpdate{
		Type: "session.update",
		Session: maiSession{
			Type: "transcription",
			Audio: maiSessionAudio{
				Input: maiAudioInput{
					Format: maiAudioFormat{
						Type: "audio/pcm",
						Rate: rate,
					},
					Transcription: maiTranscriptionSettings{
						Model:    m.modelID,
						Language: language,
					},
				},
			},
		},
	}
}

// dial opens the MAI WebSocket with the API key header, or with a bearer
// token that the token source's transport adds to the handshake.
func (m *maiTranscriptionModel) dial(ctx context.Context, options model.TranscriptionStreamOptions) (*websocket.Conn, error) {
	endpoint, err := m.settings.maiURL(maiRealtimePath)
	if err != nil {
		return nil, err
	}

	request, err := m.settings.serviceRequest(apiMAI, endpoint, options.Headers)
	if err != nil {
		return nil, err
	}

	target, err := livetranscription.WebSocketURL(request.URL)
	if err != nil {
		return nil, err
	}

	return livetranscription.Dial(ctx, livetranscription.DialOptions{
		URL:        target.String(),
		Client:     request.Client,
		Header:     request.Headers,
		ParseError: request.ParseError,
	})
}

// maiProtocol is the MAI realtime transcription protocol. The session is
// configured once the server created it, and the audio starts once the
// server applied the configuration, which the first append locks.
type maiProtocol struct {
	warnings       []model.Warning
	sessionUpdate  []byte
	bytesPerSecond int
	language       string

	audioStarted bool
	committed    bool
	audioBytes   int64
	deltas       string
}

// Open starts the stream.
func (p *maiProtocol) Open(session *livetranscription.TranscriptionSession) {
	session.Emit(model.StreamStart{
		Warnings: p.warnings,
	})
}

// HandleMessage handles one server event.
func (p *maiProtocol) HandleMessage(session *livetranscription.TranscriptionSession, message []byte) {
	var event maiEvent
	if err := json.Unmarshal(message, &event); err != nil {
		session.Fail(fmt.Errorf("%w: decode azure mai event: %w", model.ErrInvalidResponse, err))

		return
	}

	switch event.Type {
	case maiSessionCreated:
		session.Send(livetranscription.TextFrame(p.sessionUpdate))

	case maiSessionUpdated:
		if !p.audioStarted {
			p.audioStarted = true
			session.StartAudio()
		}

	case maiTranscriptDelta:
		p.deltas += event.Delta

		session.Emit(model.TranscriptDelta{
			ID:    event.ItemID,
			Delta: event.Delta,
		})

	case maiTranscriptIntermediate:
		session.Emit(model.TranscriptPartial{
			ID:   event.ItemID,
			Text: event.Intermediate,
		})

	case maiTranscriptCompleted:
		p.complete(session, event)

	case maiTranscriptFailed, maiErrorEvent:
		session.Fail(&model.APICallError{
			Message:      maiErrorMessage(event.Error),
			ResponseBody: string(message),
		})
	}
}

// complete emits the final transcript, which falls back to the deltas, and
// finishes the stream once the audio was committed.
func (p *maiProtocol) complete(session *livetranscription.TranscriptionSession, event maiEvent) {
	transcript := p.deltas
	if event.Transcript != nil {
		transcript = *event.Transcript
	}

	session.Emit(model.TranscriptFinal{
		ID:   event.ItemID,
		Text: transcript,
	})

	if p.committed {
		p.finish(session, transcript)
	}
}

func maiErrorMessage(detail *maiEventError) string {
	code := ""
	description := maiDefaultErrorDescription

	if detail != nil && detail.Code != nil {
		code = " (" + *detail.Code + ")"
	}

	if detail != nil && detail.Message != nil {
		description = *detail.Message
	}

	return "Azure MAI transcription error" + code + ": " + description
}

// EncodeAudio sends one chunk of base64 audio.
func (p *maiProtocol) EncodeAudio(chunk []byte, _ bool) (livetranscription.Frame, error) {
	return livetranscription.JSONFrame(maiAudioAppend{
		Type:  maiAudioAppendEvent,
		Audio: base64.StdEncoding.EncodeToString(chunk),
	})
}

// AudioEnded commits the audio. Without audio, the stream finishes with an
// empty transcript.
func (p *maiProtocol) AudioEnded(session *livetranscription.TranscriptionSession, totalBytes int64) {
	p.audioBytes = totalBytes

	if totalBytes == 0 {
		p.finish(session, "")

		return
	}

	p.committed = true

	session.SendJSON(maiAudioCommit{
		Type: maiAudioCommitEvent,
	})
}

// TimerFired is unused: the completed transcript finishes the stream.
func (p *maiProtocol) TimerFired(*livetranscription.TranscriptionSession) {}

// Closed leaves the stream to fail: a close before the transcript completed
// is an error.
func (p *maiProtocol) Closed(*livetranscription.TranscriptionSession, *websocket.CloseError) {}

// finish ends the stream. The duration follows from the audio sent.
func (p *maiProtocol) finish(session *livetranscription.TranscriptionSession, text string) {
	finish := model.TranscriptFinish{
		Text:     text,
		Language: p.language,
	}

	if p.audioBytes > 0 {
		finish.Duration = time.Duration(p.audioBytes) * time.Second / time.Duration(p.bytesPerSecond)
	}

	session.Finish(finish)
}
