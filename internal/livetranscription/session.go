package livetranscription

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"sync"
	"sync/atomic"
	"time"

	"github.com/aarock1234/ai/internal/websocket"
)

// defaultAudioChunkBytes bounds the audio of one frame when
// Setup.AudioChunkBytes is zero.
const defaultAudioChunkBytes = 32 << 10

// inbound is one result of reading the connection.
type inbound struct {
	message []byte
	err     error
}

// audioResult reports how the audio goroutine ended.
type audioResult struct {
	totalBytes int64
	stopped    bool

	// failure ends the stream: the audio could not be read or encoded.
	failure error

	// sendErr is a failed write, which a closing connection explains.
	sendErr error
}

// Session is one open live stream. Protocols drive it from the
// consumer's goroutine.
type Session[Part any] struct {
	conn             *websocket.Conn
	protocol         Protocol[Part]
	kinds            partKinds[Part]
	audio            io.Reader
	audioChunkBytes  int
	includeRawChunks bool
	keepPendingTimer bool
	name             string

	messages     chan inbound
	audioResults chan audioResult

	// done is closed when the stream is released; the reader and audio
	// goroutines stop handing over results.
	done     chan struct{}
	doneOnce sync.Once

	// audioStop is closed when the audio must stop; the audio goroutine
	// stops at its next chunk.
	audioStop     chan struct{}
	audioStopOnce sync.Once

	// completed is set once the stream finished with a finish part, so the
	// connection closes normally.
	completed atomic.Bool

	// The fields below belong to the consumer's goroutine.

	// ctx is the context of the iteration, which ended connections report.
	ctx          context.Context
	pending      []Part
	ended        bool
	audioStarted bool
	timer        *time.Timer
}

func newSession[Part any](setup Setup[Part], kinds partKinds[Part]) *Session[Part] {
	chunkBytes := setup.AudioChunkBytes
	if chunkBytes <= 0 {
		chunkBytes = defaultAudioChunkBytes
	}

	return &Session[Part]{
		conn:             setup.Conn,
		protocol:         setup.Protocol,
		kinds:            kinds,
		audio:            setup.Audio,
		audioChunkBytes:  chunkBytes,
		includeRawChunks: setup.IncludeRawChunks,
		keepPendingTimer: setup.KeepPendingTimer,
		name:             setup.Name,
		messages:         make(chan inbound),
		audioResults:     make(chan audioResult),
		done:             make(chan struct{}),
		audioStop:        make(chan struct{}),
	}
}

// Emit queues a part. Parts after the stream ended are dropped.
func (s *Session[Part]) Emit(part Part) {
	if s.ended {
		return
	}

	s.pending = append(s.pending, part)
}

// Send sends a frame. A failed write ends the stream, unless the server
// closed the connection: the close is then reported when it is read.
func (s *Session[Part]) Send(frame Frame) {
	if s.ended {
		return
	}

	if err := s.conn.WriteMessage(frame.Type, frame.Data); err != nil {
		s.sendFailed(err)
	}
}

// SendJSON encodes value and sends it as a TEXT frame.
func (s *Session[Part]) SendJSON(value any) {
	frame, err := JSONFrame(value)
	if err != nil {
		s.Fail(err)

		return
	}

	s.Send(frame)
}

// StartAudio starts sending the audio on its own goroutine. Later calls do
// nothing.
func (s *Session[Part]) StartAudio() {
	if s.ended || s.audioStarted {
		return
	}

	s.audioStarted = true

	go s.sendAudio()
}

// StopAudio stops sending the audio at its next chunk. Protocol.AudioEnded
// does not run afterwards.
func (s *Session[Part]) StopAudio() {
	s.audioStopOnce.Do(func() {
		close(s.audioStop)
	})
}

// SetTimer schedules Protocol.TimerFired after delay. A pending timer is
// replaced, or kept when Setup.KeepPendingTimer is set.
func (s *Session[Part]) SetTimer(delay time.Duration) {
	if s.keepPendingTimer && s.timer != nil {
		return
	}

	s.StopTimer()

	if s.ended {
		return
	}

	s.timer = time.NewTimer(delay)
}

// TimerPending reports whether a timer set with SetTimer has not fired or
// been stopped yet.
func (s *Session[Part]) TimerPending() bool {
	return s.timer != nil
}

// StopTimer cancels the pending timer, if any.
func (s *Session[Part]) StopTimer() {
	if s.timer == nil {
		return
	}

	s.timer.Stop()
	s.timer = nil
}

// Finish ends the stream with the finish part, and the connection closes
// normally.
func (s *Session[Part]) Finish(finish Part) {
	if s.ended {
		return
	}

	s.Emit(finish)
	s.completed.Store(true)
	s.end()
}

// Fail ends the stream with an error part.
func (s *Session[Part]) Fail(err error) {
	if s.ended {
		return
	}

	s.Emit(s.kinds.errorPart(err))
	s.end()
}

// Report emits an error part without ending the stream, for provider
// errors that upstream surfaces in the stream.
func (s *Session[Part]) Report(err error) {
	s.Emit(s.kinds.errorPart(err))
}

// End ends the stream without a finish or error part, for providers whose
// stream simply stops when the server closes the connection. The missing
// finish part tells the caller that the result is incomplete.
func (s *Session[Part]) End() {
	if s.ended {
		return
	}

	s.end()
}

// Ended reports whether the stream has ended.
func (s *Session[Part]) Ended() bool {
	return s.ended
}

func (s *Session[Part]) end() {
	s.ended = true
	s.StopTimer()
	s.StopAudio()
}

// parts reads server messages and runs the protocol, yielding its parts,
// until the stream ends or the consumer stops.
func (s *Session[Part]) parts(ctx context.Context) iter.Seq[Part] {
	return func(yield func(Part) bool) {
		s.ctx = ctx

		go s.readMessages()

		s.protocol.Open(s)

		for s.flush(yield) {
			s.awaitEvent()
		}
	}
}

// flush yields the queued parts and reports whether to keep running.
func (s *Session[Part]) flush(yield func(Part) bool) bool {
	for len(s.pending) > 0 {
		part := s.pending[0]
		s.pending = s.pending[1:]

		if !yield(part) {
			s.end()

			return false
		}
	}

	return !s.ended
}

// awaitEvent handles the next server message, audio result, timer or
// cancellation.
func (s *Session[Part]) awaitEvent() {
	select {
	case <-s.ctx.Done():
		s.Fail(context.Cause(s.ctx))

	case <-s.done:
		s.end()

	case received := <-s.messages:
		s.handleInbound(received)

	case result := <-s.audioResults:
		s.handleAudioResult(result)

	case <-s.timerChannel():
		s.timer = nil
		s.protocol.TimerFired(s)
	}
}

// timerChannel returns the pending timer's channel, or nil, which blocks.
func (s *Session[Part]) timerChannel() <-chan time.Time {
	if s.timer == nil {
		return nil
	}

	return s.timer.C
}

func (s *Session[Part]) handleInbound(received inbound) {
	if received.err != nil {
		s.readFailed(received.err)

		return
	}

	// Frames that are not JSON are ignored, as upstream.
	if !json.Valid(received.message) {
		return
	}

	if s.includeRawChunks {
		s.Emit(s.kinds.rawPart(json.RawMessage(received.message)))
	}

	s.protocol.HandleMessage(s, received.message)
}

// readFailed reports why reading stopped. A close by the server goes to the
// protocol, which may still finish the stream.
func (s *Session[Part]) readFailed(err error) {
	if s.ctx.Err() != nil {
		s.Fail(context.Cause(s.ctx))

		return
	}

	closeErr, closed := errors.AsType[*websocket.CloseError](err)
	if !closed {
		s.Fail(fmt.Errorf("%s: %w", s.name, err))

		return
	}

	s.protocol.Closed(s, closeErr)

	s.Fail(fmt.Errorf("%s: connection closed before the stream finished: %w", s.name, closeErr))
}

func (s *Session[Part]) handleAudioResult(result audioResult) {
	switch {
	case result.failure != nil:
		s.Fail(result.failure)

	case result.sendErr != nil:
		s.sendFailed(result.sendErr)

	case result.stopped:
		// The stream was released while sending; nothing is left to report.

	default:
		s.protocol.AudioEnded(s, result.totalBytes)
	}
}

// sendFailed ends the stream after a failed write. A write fails with the
// close error once the server closed the connection; the reader reports
// that close, so the protocol can still finish the stream.
func (s *Session[Part]) sendFailed(err error) {
	if s.ctx.Err() != nil {
		s.Fail(context.Cause(s.ctx))

		return
	}

	if _, closed := errors.AsType[*websocket.CloseError](err); closed {
		return
	}

	s.Fail(fmt.Errorf("%s: %w", s.name, err))
}

// readMessages hands each server message to the consumer until reading
// fails or the stream is released.
func (s *Session[Part]) readMessages() {
	for {
		_, message, err := s.conn.ReadMessage()

		select {
		case s.messages <- inbound{
			message: message,
			err:     err,
		}:
		case <-s.done:
			return
		}

		if err != nil {
			return
		}
	}
}

// sendAudio sends the audio and hands the outcome to the consumer. The
// goroutine is not joined: it ends at its next chunk once the stream is
// released, but a Read in progress is not interrupted.
func (s *Session[Part]) sendAudio() {
	result := s.streamAudio()

	select {
	case s.audioResults <- result:
		// The consumer handles the outcome.
	case <-s.done:
		// The stream was released, so nobody waits for the outcome.
	}
}

func (s *Session[Part]) streamAudio() audioResult {
	buffer := make([]byte, s.audioChunkBytes)
	first := true

	var totalBytes int64

	for {
		if s.audioStopped() {
			return audioResult{
				stopped: true,
			}
		}

		count, readErr := s.audio.Read(buffer)

		if count > 0 && !s.audioStopped() {
			if result, failed := s.sendAudioChunk(buffer[:count], first); failed {
				return result
			}

			first = false
			totalBytes += int64(count)
		}

		if errors.Is(readErr, io.EOF) {
			return audioResult{
				totalBytes: totalBytes,
				stopped:    s.audioStopped(),
			}
		}

		if readErr != nil {
			return audioResult{
				failure: fmt.Errorf("%s: read audio: %w", s.name, readErr),
			}
		}
	}
}

// sendAudioChunk encodes and sends one chunk. It reports the result that
// ends the audio when it fails.
func (s *Session[Part]) sendAudioChunk(chunk []byte, first bool) (audioResult, bool) {
	frame, err := s.protocol.EncodeAudio(chunk, first)
	if err != nil {
		return audioResult{
			failure: err,
		}, true
	}

	if err := s.conn.WriteMessage(frame.Type, frame.Data); err != nil {
		return audioResult{
			sendErr: err,
		}, true
	}

	return audioResult{}, false
}

func (s *Session[Part]) audioStopped() bool {
	select {
	case <-s.audioStop:
		return true
	default:
		return false
	}
}

// close releases the stream: the audio stops and the connection closes,
// normally once the transcript finished.
func (s *Session[Part]) close() error {
	s.doneOnce.Do(func() {
		close(s.done)
	})

	s.StopAudio()

	code := websocket.StatusNoStatus
	if s.completed.Load() {
		code = websocket.StatusNormalClosure
	}

	return s.conn.Close(code, "")
}
