package ai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"sync"
	"time"

	"github.com/aarock1234/ai/internal/websocket"
	"github.com/aarock1234/ai/pkg/model"
)

// Realtime session defaults and limits.
const (
	defaultRealtimeMaxEvents      = 500
	defaultRealtimeStartupTimeout = 30 * time.Second
	defaultRealtimeCloseTimeout   = 15 * time.Second

	// maxRealtimeFrameBytes bounds one message of a continuous session.
	maxRealtimeFrameBytes = 128 << 10
)

// Realtime session errors.
var (
	// ErrRealtimeNotAccepting reports a command sent to a session that is
	// not connected, is closing or has failed.
	ErrRealtimeNotAccepting = errors.New("realtime session is not accepting submissions")

	// ErrRealtimeStartupTimeout reports a session that did not become ready
	// within the startup timeout.
	ErrRealtimeStartupTimeout = errors.New("realtime session startup timed out")

	// errRealtimeSessionEnded cancels the session context when the session
	// ends without a failure.
	errRealtimeSessionEnded = errors.New("realtime session ended")
)

// RealtimeSessionOptions configures ConnectRealtime.
type RealtimeSessionOptions struct {
	Model model.RealtimeModel

	// Relay connects through an application-owned WebSocket endpoint. Nil
	// connects to the provider: with the model's server WebSocket
	// configuration when it has one, and otherwise with a client secret.
	Relay *RealtimeRelay

	// SessionConfig configures the session when it starts.
	SessionConfig model.RealtimeSessionConfig

	// Tools are declared to the model after SessionConfig.Tools. Calls are
	// passed to OnToolCall; Execute functions are not called.
	Tools []Tool

	// HTTPClient performs the WebSocket handshake. Nil uses
	// http.DefaultClient.
	HTTPClient *http.Client

	// MaxEvents bounds the events, transcripts and delegations kept in the
	// state. Zero selects 500.
	MaxEvents int

	// StartupTimeout bounds connecting and receiving readiness. Zero
	// selects 30 seconds.
	StartupTimeout time.Duration

	// CloseTimeout bounds the wait for the final usage in Close. Zero
	// selects 15 seconds.
	CloseTimeout time.Duration

	// OnEvent receives every server event, in order.
	OnEvent func(ctx context.Context, event model.RealtimeServerEvent)

	// OnToolCall runs a function call of a turn-based session on its own
	// goroutine. A non-nil output is submitted with AddToolOutput, and the
	// model responds once every call of the response has an output. Return
	// nil to submit the output later.
	OnToolCall func(ctx context.Context, call RealtimeToolCall) (any, error)

	// OnError receives errors that do not end the session, such as provider
	// error events and failed tool calls, and the failure that ends it.
	OnError func(ctx context.Context, err error)
}

// RealtimeToolCall is a function call of a realtime model.
type RealtimeToolCall struct {
	ToolCallID string
	ToolName   string

	// Input is the JSON arguments.
	Input json.RawMessage
}

// RealtimeProviderError is an error event reported by a realtime model.
type RealtimeProviderError struct {
	Message string
	Code    string

	// ClientEventID is the client event that failed, if any.
	ClientEventID string
}

// Error returns the message with the code, if any.
func (e *RealtimeProviderError) Error() string {
	if e.Code == "" {
		return e.Message
	}

	return fmt.Sprintf("%s (%s)", e.Message, e.Code)
}

// RealtimeSession is a server-side realtime conversation over one
// WebSocket connection. A goroutine owned by the session reads server
// events; tool calls run on their own goroutines. Methods are safe for
// concurrent use. The realtime API is experimental and may change.
//
// Go adaptation: upstream's session runs in browsers with microphone
// capture, playback and WebRTC. This session sends and receives base64
// audio through its methods and events instead; WebRTC remains available
// to browsers through model.RealtimeWebRTCSessionCreator.
type RealtimeSession struct {
	model        model.RealtimeModel
	options      RealtimeSessionOptions
	capabilities model.RealtimeCapabilities
	conn         *websocket.Conn
	parse        model.RealtimeServerEventParser

	// ctx ends with the session; cancel ends it.
	ctx    context.Context
	cancel context.CancelCauseFunc

	// tasks are the reader and the tool call goroutines.
	tasks sync.WaitGroup

	ready     chan struct{}
	readyOnce sync.Once

	// finalized is closed when the provider confirms the end of the
	// session, and done when the reader stops.
	finalized     chan struct{}
	finalizedOnce sync.Once
	done          chan struct{}

	// sendMu orders client submissions, so that the events of one method,
	// such as a text message and its response request, are not interleaved
	// with those of another.
	sendMu sync.Mutex

	mu       sync.Mutex
	status   RealtimeStatus
	cause    error
	closing  bool
	ended    bool
	reducer  *realtimeReducer
	session  *RealtimeSessionState
	commands *realtimeCommandTracker

	// toolCallsInResponse and submittedOutputs track the calls of the
	// current response, which is answered once every call has an output.
	toolCallsInResponse     map[string]bool
	submittedOutputs        map[string]bool
	responseToolCallsClosed bool
}

// ConnectRealtime connects to a realtime model, starts the session and
// waits until it is ready. Ctx governs the whole session: cancelling it ends
// the session.
func ConnectRealtime(ctx context.Context, options RealtimeSessionOptions) (*RealtimeSession, error) {
	options, err := options.withDefaults()
	if err != nil {
		return nil, err
	}

	config, err := realtimeSessionConfig(options)
	if err != nil {
		return nil, err
	}

	sessionCtx, cancel := context.WithCancelCause(ctx)

	// Until the session is ready, the startup timeout ends the session.
	startupCtx, cancelStartup := context.WithTimeoutCause(sessionCtx, options.StartupTimeout, ErrRealtimeStartupTimeout)
	defer cancelStartup()

	stopStartupLimit := context.AfterFunc(startupCtx, func() {
		cancel(context.Cause(startupCtx))
	})

	conn, err := dialRealtime(sessionCtx, options, config)
	if err != nil {
		cancel(err)

		return nil, err
	}

	session := newRealtimeSession(sessionCtx, cancel, options, conn)
	session.tasks.Go(session.readLoop)

	err = session.start(config)
	if err == nil && !stopStartupLimit() {
		err = context.Cause(startupCtx)
	}

	if err != nil {
		session.shutdown(err)
		session.tasks.Wait()

		return nil, err
	}

	return session, nil
}

func (o RealtimeSessionOptions) withDefaults() (RealtimeSessionOptions, error) {
	if o.Model == nil {
		return o, &InvalidArgumentError{
			Parameter: "model",
			Message:   "model is required",
		}
	}

	if o.MaxEvents < 0 {
		return o, &InvalidArgumentError{
			Parameter: "maxEvents",
			Value:     o.MaxEvents,
			Message:   "maxEvents must not be negative",
		}
	}

	if o.StartupTimeout < 0 {
		return o, &InvalidArgumentError{
			Parameter: "startupTimeout",
			Value:     o.StartupTimeout,
			Message:   "startupTimeout must not be negative",
		}
	}

	if o.CloseTimeout < 0 {
		return o, &InvalidArgumentError{
			Parameter: "closeTimeout",
			Value:     o.CloseTimeout,
			Message:   "closeTimeout must not be negative",
		}
	}

	if o.MaxEvents == 0 {
		o.MaxEvents = defaultRealtimeMaxEvents
	}

	if o.StartupTimeout == 0 {
		o.StartupTimeout = defaultRealtimeStartupTimeout
	}

	if o.CloseTimeout == 0 {
		o.CloseTimeout = defaultRealtimeCloseTimeout
	}

	return o, nil
}

// realtimeSessionConfig declares the tools after the configured ones.
func realtimeSessionConfig(options RealtimeSessionOptions) (model.RealtimeSessionConfig, error) {
	config := options.SessionConfig

	definitions, err := RealtimeToolDefinitions(options.Tools)
	if err != nil {
		return model.RealtimeSessionConfig{}, err
	}

	if len(definitions) > 0 {
		config.Tools = slices.Concat(config.Tools, definitions)
	}

	return config, nil
}

func newRealtimeSession(
	ctx context.Context,
	cancel context.CancelCauseFunc,
	options RealtimeSessionOptions,
	conn *websocket.Conn,
) *RealtimeSession {
	capabilities := options.Model.Capabilities()

	session := &RealtimeSession{
		model:               options.Model,
		options:             options,
		capabilities:        capabilities,
		conn:                conn,
		parse:               realtimeEventParser(options.Model),
		ctx:                 ctx,
		cancel:              cancel,
		ready:               make(chan struct{}),
		finalized:           make(chan struct{}),
		done:                make(chan struct{}),
		status:              RealtimeConnecting,
		reducer:             newRealtimeReducer(options.MaxEvents),
		commands:            newRealtimeCommandTracker(),
		toolCallsInResponse: make(map[string]bool),
		submittedOutputs:    make(map[string]bool),
	}

	if session.hasSessionLifecycle() {
		session.session = newRealtimeSessionState()
	}

	return session
}

// realtimeEventParser returns a parser for one connection.
func realtimeEventParser(realtimeModel model.RealtimeModel) model.RealtimeServerEventParser {
	if factory, ok := realtimeModel.(model.RealtimeServerEventParserFactory); ok {
		return factory.NewServerEventParser()
	}

	return realtimeModel.ParseServerEvent
}

// hasSessionLifecycle reports whether the model starts sessions with a
// session start event or ends them with a session close event.
func (s *RealtimeSession) hasSessionLifecycle() bool {
	return s.capabilities.Startup == model.RealtimeStartupSessionStart ||
		s.capabilities.Finalization == model.RealtimeFinalizeSessionClose
}

func (s *RealtimeSession) isContinuous() bool {
	return s.capabilities.Conversation == model.RealtimeContinuous
}

// start sends the startup event and waits until the session is ready, the
// connection ends or the session context ends.
func (s *RealtimeSession) start(config model.RealtimeSessionConfig) error {
	if err := s.write(s.startupEvent(config)); err != nil {
		return err
	}

	select {
	case <-s.ready:
		return nil

	case <-s.done:
		if err := s.Err(); err != nil {
			return err
		}

		return fmt.Errorf("%w: connection closed before becoming ready", ErrRealtimeNotAccepting)

	case <-s.ctx.Done():
		return context.Cause(s.ctx)
	}
}

func (s *RealtimeSession) startupEvent(config model.RealtimeSessionConfig) model.RealtimeClientEvent {
	if s.capabilities.Startup == model.RealtimeStartupSessionStart {
		return model.RealtimeSessionStart{
			Config: config,
		}
	}

	return model.RealtimeSessionUpdate{
		Config: config,
	}
}

// State returns a snapshot of the session.
func (s *RealtimeSession) State() RealtimeState {
	s.mu.Lock()
	defer s.mu.Unlock()

	messages, events := s.reducer.snapshot()

	return RealtimeState{
		Status:   s.status,
		Messages: messages,
		Events:   events,
		Session:  s.session.clone(),
	}
}

// Done is closed when the connection has ended.
func (s *RealtimeSession) Done() <-chan struct{} {
	return s.done
}

// Err returns the failure that ended the session, or nil.
func (s *RealtimeSession) Err() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.cause
}

// SendEvent sends a client event. Use Close to end the session.
func (s *RealtimeSession) SendEvent(event model.RealtimeClientEvent) error {
	s.sendMu.Lock()
	defer s.sendMu.Unlock()

	return s.sendEvent(event)
}

// sendEvent sends a client event while the caller holds sendMu.
func (s *RealtimeSession) sendEvent(event model.RealtimeClientEvent) error {
	command, err := s.prepareCommand(event)
	if err != nil {
		return err
	}

	err = s.write(command.event)

	s.mu.Lock()
	s.commands.sent(command, err)
	s.mu.Unlock()

	return err
}

// prepareCommand checks that the session accepts the event and registers
// it with the command tracker.
func (s *RealtimeSession) prepareCommand(event model.RealtimeClientEvent) (trackedCommand, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.acceptingLocked() {
		return trackedCommand{}, ErrRealtimeNotAccepting
	}

	switch event.(type) {
	case model.RealtimeSessionClose:
		return trackedCommand{}, fmt.Errorf("%w: use Close to end the session", ErrRealtimeNotAccepting)

	case model.RealtimeSessionStart:
		if s.hasSessionLifecycle() {
			return trackedCommand{}, fmt.Errorf("%w: realtime session has already started", ErrRealtimeNotAccepting)
		}

	case model.RealtimeInputAudioCommit, model.RealtimeInputAudioClear:
		if s.isContinuous() {
			return trackedCommand{}, fmt.Errorf("%w: input audio commands in a continuous session", model.ErrUnsupported)
		}
	}

	return s.commands.prepare(event)
}

// acceptingLocked reports whether the session accepts submissions.
func (s *RealtimeSession) acceptingLocked() bool {
	return s.status == RealtimeConnected && !s.closing && !s.ended
}

// SendTextMessage sends a user text message and asks the model to respond.
// Continuous sessions take context through model.RealtimeContextAppend
// instead.
//
// Go adaptation: the message is recorded in the state before it is sent,
// rather than after as upstream does, because the response events arrive
// concurrently and must follow it. A message that cannot be sent is
// removed again.
func (s *RealtimeSession) SendTextMessage(text string) error {
	if s.isContinuous() {
		return fmt.Errorf("%w: text messages in a continuous session; use context append for context", model.ErrUnsupported)
	}

	s.sendMu.Lock()
	defer s.sendMu.Unlock()

	s.mu.Lock()
	messageID := s.reducer.addUserText(text)
	s.mu.Unlock()

	err := s.sendEvent(model.RealtimeConversationItemCreate{
		Item: model.RealtimeTextMessage{
			Text: text,
		},
	})
	if err != nil {
		s.mu.Lock()
		s.reducer.removeMessage(messageID)
		s.mu.Unlock()

		return err
	}

	return s.sendEvent(model.RealtimeResponseCreate{})
}

// SendAudio appends raw audio in the input format to the input buffer.
func (s *RealtimeSession) SendAudio(audio []byte) error {
	return s.SendEvent(model.RealtimeInputAudioAppend{
		Audio: base64.StdEncoding.EncodeToString(audio),
	})
}

// CommitAudio commits the input buffer as a user turn.
func (s *RealtimeSession) CommitAudio() error {
	return s.SendEvent(model.RealtimeInputAudioCommit{})
}

// ClearAudioBuffer discards the input buffer.
func (s *RealtimeSession) ClearAudioBuffer() error {
	return s.SendEvent(model.RealtimeInputAudioClear{})
}

// RequestResponse asks the model to respond. Nil options keep the session
// defaults.
func (s *RealtimeSession) RequestResponse(options *model.RealtimeResponseOptions) error {
	return s.SendEvent(model.RealtimeResponseCreate{
		Options: options,
	})
}

// CancelResponse cancels the response in progress.
func (s *RealtimeSession) CancelResponse() error {
	return s.SendEvent(model.RealtimeResponseCancel{})
}

// AddToolOutput submits the output of a function call. The model responds
// once every call of the response has an output.
func (s *RealtimeSession) AddToolOutput(callID string, output any) error {
	if s.isContinuous() {
		return fmt.Errorf("%w: tool outputs in a continuous session; delegation is application-owned", model.ErrUnsupported)
	}

	encoded, err := json.Marshal(output)
	if err != nil {
		return fmt.Errorf("encode tool output: %w", err)
	}

	s.sendMu.Lock()
	defer s.sendMu.Unlock()

	s.mu.Lock()
	toolName := s.reducer.toolName(callID)
	s.mu.Unlock()

	err = s.sendEvent(model.RealtimeConversationItemCreate{
		Item: model.RealtimeFunctionCallOutput{
			CallID: callID,
			Name:   toolName,
			Output: string(encoded),
		},
	})
	if err != nil {
		return err
	}

	s.mu.Lock()
	s.reducer.addToolOutput(callID, output)
	s.submittedOutputs[callID] = true
	respond := s.takeToolResponseLocked()
	s.mu.Unlock()

	if !respond {
		return nil
	}

	return s.sendEvent(model.RealtimeResponseCreate{})
}

// takeToolResponseLocked reports whether every call of a finished response
// has an output, and resets the tracking when it does.
func (s *RealtimeSession) takeToolResponseLocked() bool {
	if !s.acceptingLocked() || !s.responseToolCallsClosed || len(s.toolCallsInResponse) == 0 {
		return false
	}

	for callID := range s.toolCallsInResponse {
		if !s.submittedOutputs[callID] {
			return false
		}
	}

	clear(s.toolCallsInResponse)
	clear(s.submittedOutputs)
	s.responseToolCallsClosed = false

	return true
}

// Close ends the session and waits for its goroutines. Models that finalize
// sessions with a session close event are asked for the final usage first,
// waiting up to the close timeout or until ctx ends. Close must not be
// called from the session's callbacks, which it waits for; call Disconnect
// there.
func (s *RealtimeSession) Close(ctx context.Context) error {
	var err error

	if s.beginClose() {
		err = s.finalize(ctx)
	}

	s.Disconnect()
	s.tasks.Wait()

	return err
}

// beginClose marks the session as closing and reports whether to ask the
// provider for the final usage.
func (s *RealtimeSession) beginClose() bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	graceful := s.status == RealtimeConnected &&
		!s.closing &&
		!s.ended &&
		s.capabilities.Finalization == model.RealtimeFinalizeSessionClose

	s.closing = true

	if graceful {
		s.status = RealtimeClosing
	}

	return graceful
}

// finalize sends the session close event and waits for its
// acknowledgement. A failed send is reported and ends the wait.
func (s *RealtimeSession) finalize(ctx context.Context) error {
	if err := s.write(model.RealtimeSessionClose{}); err != nil {
		s.reportError(err)

		return nil
	}

	timer := time.NewTimer(s.options.CloseTimeout)
	defer timer.Stop()

	select {
	case <-s.finalized:
		return nil
	case <-s.done:
		return nil
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return context.Cause(ctx)
	}
}

// Disconnect ends the session without waiting for the final usage or for
// its goroutines. It is safe to call from callbacks.
func (s *RealtimeSession) Disconnect() {
	s.shutdown(nil)
}

// Wait blocks until the reader and every tool call goroutine have ended.
func (s *RealtimeSession) Wait() {
	s.tasks.Wait()
}

// shutdown ends the session once, recording cause as its failure when it
// is not nil, and closes the connection.
func (s *RealtimeSession) shutdown(cause error) {
	s.mu.Lock()

	if s.ended {
		s.mu.Unlock()

		return
	}

	s.ended = true

	if cause != nil {
		s.cause = cause
		s.status = RealtimeFailed
	} else {
		s.status = RealtimeDisconnected
	}

	if s.session != nil && s.session.Finalization != RealtimeFinalizationConfirmed {
		s.session.Finalization = RealtimeFinalizationUnconfirmed
	}

	s.mu.Unlock()

	code := websocket.StatusNormalClosure
	if cause != nil {
		code = websocket.StatusNoStatus
	}

	_ = s.conn.Close(code, "")

	if cause == nil {
		cause = errRealtimeSessionEnded
	}

	s.cancel(cause)
}

// fail ends the session with err and reports it.
func (s *RealtimeSession) fail(err error) {
	if s.hasEnded() {
		return
	}

	s.shutdown(err)
	s.reportError(err)
}

func (s *RealtimeSession) hasEnded() bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.ended
}

// write serializes one client event and sends it as a text or binary
// message, according to its frame type.
func (s *RealtimeSession) write(event model.RealtimeClientEvent) error {
	frame, err := s.model.SerializeClientEvent(event)
	if err != nil {
		return err
	}

	if frame == nil {
		return nil
	}

	messageType, err := realtimeMessageType(frame.Type)
	if err != nil {
		return err
	}

	if s.isContinuous() && len(frame.Data) > maxRealtimeFrameBytes {
		return fmt.Errorf(
			"%w: realtime frame exceeds the %d-byte limit; reduce the event payload",
			model.ErrInvalidArgument,
			maxRealtimeFrameBytes,
		)
	}

	if err := s.conn.WriteMessage(messageType, frame.Data); err != nil {
		if s.ctx.Err() != nil {
			return fmt.Errorf("%w: %w", ErrRealtimeNotAccepting, context.Cause(s.ctx))
		}

		return fmt.Errorf("send realtime event: %w", err)
	}

	return nil
}

// realtimeMessageType maps a frame type to its WebSocket message type.
func realtimeMessageType(frameType model.RealtimeFrameType) (websocket.MessageType, error) {
	switch frameType {
	case model.RealtimeTextFrame:
		return websocket.TextMessage, nil
	case model.RealtimeBinaryFrame:
		return websocket.BinaryMessage, nil
	default:
		return 0, fmt.Errorf("%w: realtime frame type %q", model.ErrInvalidArgument, frameType)
	}
}

// reportError passes a non-fatal error to OnError.
func (s *RealtimeSession) reportError(err error) {
	if s.options.OnError != nil {
		s.options.OnError(s.ctx, err)
	}
}
