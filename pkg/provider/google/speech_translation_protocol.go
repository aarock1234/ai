package google

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"github.com/aarock1234/ai/internal/livetranscription"
	"github.com/aarock1234/ai/internal/websocket"
	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// translationAudioMimeType is the media type of the input audio chunks.
	translationAudioMimeType = "audio/pcm;rate=16000"

	// silenceAmplitudeThreshold is the largest 16-bit PCM amplitude that
	// counts as silence.
	silenceAmplitudeThreshold = 128

	translationDefaultErrorMessage = "Google Live API error"
)

// translationServerMessage is one Live API server message of a translation.
type translationServerMessage struct {
	SetupComplete      json.RawMessage           `json:"setupComplete"`
	ServerContent      *translationServerContent `json:"serverContent"`
	InputTranscription *realtimeTranscription    `json:"inputTranscription"`
	UsageMetadata      *translationUsageMetadata `json:"usageMetadata"`
	Error              *translationServerError   `json:"error"`
}

type translationServerContent struct {
	ModelTurn           *realtimeModelTurn     `json:"modelTurn"`
	OutputTranscription *realtimeTranscription `json:"outputTranscription"`
	InputTranscription  *realtimeTranscription `json:"inputTranscription"`
	TurnComplete        bool                   `json:"turnComplete"`
}

type translationUsageMetadata struct {
	PromptTokensDetails   []translationTokensDetail `json:"promptTokensDetails"`
	ResponseTokensDetails []translationTokensDetail `json:"responseTokensDetails"`
}

type translationTokensDetail struct {
	Modality   string `json:"modality"`
	TokenCount *int   `json:"tokenCount"`
}

type translationServerError struct {
	Message *string `json:"message"`
}

// translationProtocol is the Live API translation protocol. The audio is
// sent once the server acknowledged the setup. Messages carry no IDs, so a
// turn counter names the segments; transcripts accumulate per turn. Live
// translation keeps sending silence after the input ends and does not
// complete turns, so trailing silence finishes the stream; turn completions
// remain a fallback that starts the quiet window once.
type translationProtocol struct {
	warnings     []model.Warning
	setupMessage []byte
	finishGrace  time.Duration

	turnCounter int

	sourceText      string
	sourceTurn      string
	translationText string
	translationTurn string

	inputEnded      bool
	openTurn        bool
	sawTurnComplete bool
	trailingSilence time.Duration

	usage *model.SpeechTranslationUsage
}

// Open starts the stream and sends the setup.
func (p *translationProtocol) Open(session *livetranscription.TranslationSession) {
	session.Emit(model.StreamStart{
		Warnings: p.warnings,
	})
	session.Send(livetranscription.TextFrame(p.setupMessage))
}

// HandleMessage handles one server message.
func (p *translationProtocol) HandleMessage(session *livetranscription.TranslationSession, message []byte) {
	var decoded translationServerMessage
	if err := json.Unmarshal(message, &decoded); err != nil {
		session.Fail(fmt.Errorf("%w: decode google live translation message: %w", model.ErrInvalidResponse, err))

		return
	}

	if wirejson.HasValue(decoded.SetupComplete) {
		session.StartAudio()
	}

	if decoded.UsageMetadata != nil {
		p.addUsage(*decoded.UsageMetadata)
	}

	if decoded.Error != nil {
		session.Fail(translationError(decoded.Error, message))

		return
	}

	p.sourceTranscription(session, decoded)

	content := decoded.ServerContent
	if content == nil {
		return
	}

	if content.ModelTurn != nil && !p.translatedAudio(session, content.ModelTurn.Parts) {
		return
	}

	p.outputTranscription(session, content.OutputTranscription)

	if content.TurnComplete {
		p.completeTurn(session)
		p.openTurn = false
		p.sawTurnComplete = true

		if p.inputEnded {
			session.SetTimer(p.finishGrace)
		}
	}
}

// sourceTranscription accumulates the transcript of the source audio.
func (p *translationProtocol) sourceTranscription(session *livetranscription.TranslationSession, message translationServerMessage) {
	transcription := message.InputTranscription
	if message.ServerContent != nil && message.ServerContent.InputTranscription != nil {
		transcription = message.ServerContent.InputTranscription
	}

	if transcription == nil || transcription.Text == "" {
		return
	}

	p.turnActivity(session)
	p.sourceTurn += transcription.Text

	session.Emit(model.SourceTranscriptDelta{
		ID:    p.itemID(),
		Delta: transcription.Text,
	})
}

// outputTranscription accumulates the transcript of the translated audio.
func (p *translationProtocol) outputTranscription(session *livetranscription.TranslationSession, transcription *realtimeTranscription) {
	if transcription == nil || transcription.Text == "" {
		return
	}

	p.turnActivity(session)
	p.translationTurn += transcription.Text

	session.Emit(model.OutputTextDelta{
		ID:    p.itemID(),
		Delta: transcription.Text,
	})
}

// translatedAudio emits the audio of the model turn and reports whether to
// keep handling the message. After the input ended, enough trailing silence
// finishes the stream. Audio that is not base64 is reported without ending
// the stream and counts as activity, as upstream treats audio it cannot
// measure.
func (p *translationProtocol) translatedAudio(session *livetranscription.TranslationSession, parts []realtimeModelPart) bool {
	for _, part := range parts {
		if part.InlineData == nil || part.InlineData.Data == "" {
			continue
		}

		audio, err := base64.StdEncoding.DecodeString(part.InlineData.Data)
		if err != nil {
			session.Report(fmt.Errorf("%w: decode translated audio: %w", model.ErrInvalidResponse, err))
			p.turnActivity(session)

			continue
		}

		session.Emit(model.TranslationAudio{
			ID:    p.itemID(),
			Audio: audio,
		})

		silence, isSilent := silenceDuration(audio)
		if !p.inputEnded || !isSilent {
			p.turnActivity(session)

			continue
		}

		p.trailingSilence += silence
		if p.trailingSilence >= p.finishGrace {
			p.finish(session)

			return false
		}
	}

	return true
}

// turnActivity marks the turn open and cancels a pending finish.
func (p *translationProtocol) turnActivity(session *livetranscription.TranslationSession) {
	p.openTurn = true
	p.trailingSilence = 0
	session.StopTimer()
}

// EncodeAudio sends one chunk of base64 audio.
func (p *translationProtocol) EncodeAudio(chunk []byte, _ bool) (livetranscription.Frame, error) {
	return livetranscription.JSONFrame(realtimeInputMessage{
		RealtimeInput: realtimeClientInput{
			Audio: &realtimeAudioBlob{
				Data:     base64.StdEncoding.EncodeToString(chunk),
				MimeType: translationAudioMimeType,
			},
		},
	})
}

// AudioEnded ends the audio stream. A turn completion that already followed
// the last audio satisfies the finish condition.
func (p *translationProtocol) AudioEnded(session *livetranscription.TranslationSession, _ int64) {
	session.SendJSON(realtimeInputMessage{
		RealtimeInput: realtimeClientInput{
			AudioStreamEnd: true,
		},
	})

	p.inputEnded = true

	if p.sawTurnComplete && !p.openTurn {
		session.SetTimer(p.finishGrace)
	}
}

// TimerFired finishes the stream after the quiet window.
func (p *translationProtocol) TimerFired(session *livetranscription.TranslationSession) {
	p.finish(session)
}

// Closed finishes the stream when a finish is pending: the close confirms
// that no more output follows. Any other close is an abnormal end.
func (p *translationProtocol) Closed(session *livetranscription.TranslationSession, closeErr *websocket.CloseError) {
	if session.TimerPending() {
		p.finish(session)

		return
	}

	session.Fail(fmt.Errorf(
		"google live translation websocket closed unexpectedly before finishing (code %d, reason: %q): %w",
		closeErr.Code,
		closeErr.Reason,
		closeErr,
	))
}

// completeTurn emits the final texts of the current turn.
func (p *translationProtocol) completeTurn(session *livetranscription.TranslationSession) {
	if p.sourceTurn != "" {
		session.Emit(model.SourceTranscriptFinal{
			ID:   p.itemID(),
			Text: p.sourceTurn,
		})

		p.sourceText += p.sourceTurn
		p.sourceTurn = ""
	}

	if p.translationTurn != "" {
		session.Emit(model.OutputTextFinal{
			ID:   p.itemID(),
			Text: p.translationTurn,
		})

		p.translationText += p.translationTurn
		p.translationTurn = ""
	}

	p.turnCounter++
}

// finish completes the open turn and ends the stream with the full texts.
func (p *translationProtocol) finish(session *livetranscription.TranslationSession) {
	if session.Ended() {
		return
	}

	if p.sourceTurn != "" || p.translationTurn != "" {
		p.completeTurn(session)
	}

	session.Finish(model.TranslationFinish{
		SourceText: p.sourceText,
		OutputText: p.translationText,
		Usage:      p.usage,
	})
}

func (p *translationProtocol) itemID() string {
	return "google-item-" + strconv.Itoa(p.turnCounter)
}

// addUsage adds a periodic usage delta. Only the billable audio modalities
// count: the text prompt detail is internal translation context.
func (p *translationProtocol) addUsage(usage translationUsageMetadata) {
	inputAudio := audioTokens(usage.PromptTokensDetails)
	outputAudio := audioTokens(usage.ResponseTokensDetails)

	if inputAudio == nil && outputAudio == nil {
		return
	}

	if p.usage == nil {
		p.usage = &model.SpeechTranslationUsage{}
	}

	p.usage.InputAudioTokens = addCounts(p.usage.InputAudioTokens, inputAudio)
	p.usage.OutputAudioTokens = addCounts(p.usage.OutputAudioTokens, outputAudio)
}

// audioTokens sums the audio token counts, or returns nil when there are
// none.
func audioTokens(details []translationTokensDetail) *int {
	var total *int

	for _, detail := range details {
		if detail.Modality == "AUDIO" && detail.TokenCount != nil {
			total = new(valueOrZero(total) + *detail.TokenCount)
		}
	}

	return total
}

// silenceDuration returns the duration of 24 kHz 16-bit PCM audio whose
// every sample is quiet, and reports false for audio with sound.
func silenceDuration(audio []byte) (time.Duration, bool) {
	sampleCount := len(audio) / 2
	if sampleCount == 0 {
		return 0, false
	}

	for index := range sampleCount {
		sample := int16(binary.LittleEndian.Uint16(audio[index*2:]))
		if sample > silenceAmplitudeThreshold || sample < -silenceAmplitudeThreshold {
			return 0, false
		}
	}

	return time.Duration(sampleCount) * time.Second / translationOutputRate, true
}

func translationError(serverError *translationServerError, message []byte) error {
	errorMessage := translationDefaultErrorMessage
	if serverError.Message != nil {
		errorMessage = *serverError.Message
	}

	return &model.APICallError{
		Message:      errorMessage,
		ResponseBody: string(message),
	}
}
