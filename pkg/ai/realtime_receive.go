package ai

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/aarock1234/ai/internal/websocket"
	"github.com/aarock1234/ai/pkg/model"
)

// readLoop reads server messages until the connection ends. It runs on
// the session's reader goroutine.
func (s *RealtimeSession) readLoop() {
	defer close(s.done)

	for {
		_, message, err := s.conn.ReadMessage()
		if err != nil {
			s.readFailed(err)

			return
		}

		if err := s.receive(message); err != nil {
			s.fail(err)

			return
		}
	}
}

// readFailed decides how a closed connection ends the session: quietly
// after the session ended or the peer closed normally once ready, and as a
// failure otherwise.
func (s *RealtimeSession) readFailed(err error) {
	if s.hasEnded() {
		return
	}

	s.mu.Lock()
	ready := s.status == RealtimeConnected || s.status == RealtimeClosing
	finalized := s.session != nil && s.session.Finalization == RealtimeFinalizationConfirmed
	s.mu.Unlock()

	var closeErr *websocket.CloseError
	normalClose := errors.As(err, &closeErr) && closeErr.Code == websocket.StatusNormalClosure

	switch {
	case finalized, ready && normalClose:
		s.shutdown(nil)
	case !ready:
		s.fail(fmt.Errorf("%w: connection closed before becoming ready: %w", ErrRealtimeNotAccepting, err))
	default:
		s.fail(fmt.Errorf("realtime connection: %w", err))
	}
}

// receive answers health checks and handles the events of one message.
func (s *RealtimeSession) receive(message []byte) error {
	if !json.Valid(message) {
		return fmt.Errorf("%w: invalid JSON in realtime server message", model.ErrInvalidResponse)
	}

	if healthChecker, ok := s.model.(model.RealtimeHealthChecker); ok {
		if response, needed := healthChecker.HealthCheckResponse(message); needed {
			if err := s.conn.WriteMessage(websocket.TextMessage, response); err != nil {
				return fmt.Errorf("send realtime health check response: %w", err)
			}
		}
	}

	events, err := s.parse(message)
	if err != nil {
		return fmt.Errorf("parse realtime server event: %w", err)
	}

	for _, event := range events {
		if s.hasEnded() {
			return nil
		}

		if err := s.handleEvent(event); err != nil {
			return err
		}
	}

	return nil
}

// eventOutcome is what one server event changed.
type eventOutcome struct {
	effects   []realtimeEffect
	toolCalls []RealtimeToolCall

	becameReady      bool
	requestResponse  bool
	finalizedSession bool
}

// handleEvent updates the state with one server event, runs its effects
// and passes it to OnEvent.
func (s *RealtimeSession) handleEvent(event model.RealtimeServerEvent) error {
	if err := s.checkSupportedEvent(event); err != nil {
		return err
	}

	outcome := s.applyEvent(event)

	for _, effect := range outcome.effects {
		if failure, ok := effect.(realtimeErrorEffect); ok {
			s.reportError(failure.err)
		}
	}

	for _, call := range outcome.toolCalls {
		s.tasks.Go(func() {
			s.runTool(call)
		})
	}

	if outcome.requestResponse {
		if err := s.SendEvent(model.RealtimeResponseCreate{}); err != nil {
			s.reportError(err)
		}
	}

	if outcome.becameReady {
		s.readyOnce.Do(func() {
			close(s.ready)
		})
	}

	s.notify(event)

	if outcome.finalizedSession {
		s.finalizedOnce.Do(func() {
			close(s.finalized)
		})

		s.shutdown(nil)
	}

	return nil
}

// checkSupportedEvent rejects events this runtime cannot serve.
func (s *RealtimeSession) checkSupportedEvent(event model.RealtimeServerEvent) error {
	switch typed := event.(type) {
	case model.RealtimeSessionStarted:
		if typed.DelegationMode == model.RealtimeDelegateToProvider {
			return fmt.Errorf("%w: provider delegation mode; this realtime runtime supports client delegation only", model.ErrUnsupported)
		}

	case model.RealtimeAudioDelta:
		if s.isContinuous() {
			return fmt.Errorf("%w: turn-based audio delta in a continuous session; use audio chunks", model.ErrUnsupported)
		}
	}

	return nil
}

// applyEvent reduces one event under the lock and collects the work it
// asks for.
func (s *RealtimeSession) applyEvent(event model.RealtimeServerEvent) eventOutcome {
	s.mu.Lock()
	defer s.mu.Unlock()

	muted := s.commands.receive(event)
	outcome := eventOutcome{
		effects: s.reducer.reduce(event),
	}

	if _, started := event.(model.RealtimeSessionStarted); started && s.session == nil {
		s.session = newRealtimeSessionState()
	}

	if s.session != nil {
		s.session.reduce(event, s.options.MaxEvents, muted)
	}

	for _, effect := range outcome.effects {
		call, ok := effect.(realtimeToolCallEffect)
		if !ok || s.isContinuous() {
			continue
		}

		s.toolCallsInResponse[call.call.ToolCallID] = true
		outcome.toolCalls = append(outcome.toolCalls, call.call)
	}

	switch event.(type) {
	case model.RealtimeResponseDone:
		if len(s.toolCallsInResponse) > 0 {
			s.responseToolCallsClosed = true
			outcome.requestResponse = s.takeToolResponseLocked()
		}

	case model.RealtimeSessionClosed:
		outcome.finalizedSession = s.session != nil
	}

	outcome.becameReady = s.markReadyLocked(event)

	return outcome
}

// markReadyLocked moves a starting session to connected when the event
// confirms the startup: a session started event for models that start
// sessions explicitly, and a session created or updated event otherwise.
func (s *RealtimeSession) markReadyLocked(event model.RealtimeServerEvent) bool {
	if s.status != RealtimeConnecting || s.closing || s.ended {
		return false
	}

	_, isStarted := event.(model.RealtimeSessionStarted)
	_, isCreated := event.(model.RealtimeSessionCreated)
	_, isUpdated := event.(model.RealtimeSessionUpdated)

	ready := isCreated || isUpdated
	if s.capabilities.Startup == model.RealtimeStartupSessionStart {
		ready = isStarted
	}

	if ready {
		s.status = RealtimeConnected
	}

	return ready
}

// runTool runs one function call with OnToolCall and submits its output.
func (s *RealtimeSession) runTool(call RealtimeToolCall) {
	if s.options.OnToolCall == nil {
		s.reportError(fmt.Errorf("no handler provided for tool %q", call.ToolName))

		return
	}

	output, err := s.options.OnToolCall(s.ctx, call)
	if err != nil {
		if s.ctx.Err() == nil {
			s.reportError(err)
		}

		return
	}

	if output == nil || s.hasEnded() {
		return
	}

	if err := s.AddToolOutput(call.ToolCallID, output); err != nil {
		s.reportError(err)
	}
}

// notify passes an event to OnEvent.
func (s *RealtimeSession) notify(event model.RealtimeServerEvent) {
	if s.options.OnEvent != nil {
		s.options.OnEvent(s.ctx, event)
	}
}
