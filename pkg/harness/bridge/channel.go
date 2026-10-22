package bridge

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sync"
	"time"

	"github.com/aarock1234/ai/pkg/harness"
)

// Close reasons reported by a Channel.
const (
	// CloseReasonClosed is the reason of a host-initiated close.
	CloseReasonClosed = "closed"

	// CloseReasonSuspended is the reason of Channel.Suspend. Adapters end
	// the active turn cleanly when they see it.
	CloseReasonSuspended = "suspended"

	// CloseReasonReconnectFailed is the reason when reconnecting gave up.
	CloseReasonReconnectFailed = "reconnect failed"
)

// Close codes reported by a Channel.
const (
	closeNormal   = 1000
	closeAbnormal = 1006
)

// Reconnect defaults, as upstream.
const (
	defaultReconnectMaxElapsed   = 30 * time.Second
	defaultReconnectInitialDelay = 50 * time.Millisecond
	defaultReconnectMaxDelay     = 2 * time.Second
)

// ErrChannelClosed is wrapped by the errors of a closed channel.
var ErrChannelClosed = errors.New("bridge channel closed")

// ClosedError reports a channel that ended, with the close code and reason.
type ClosedError struct {
	Code   int
	Reason string
}

// Error describes the close.
func (e *ClosedError) Error() string {
	return fmt.Sprintf("bridge channel closed with code %d: %s", e.Code, e.Reason)
}

// Unwrap returns ErrChannelClosed.
func (e *ClosedError) Unwrap() error {
	return ErrChannelClosed
}

// ChannelOptions configure NewChannel.
type ChannelOptions struct {
	// Connect opens a fresh connection to the bridge and returns once it can
	// carry frames, after any adapter handshake such as waiting for Hello.
	// It is called by Open and again on every reconnect. Ctx governs the
	// returned connection.
	Connect func(ctx context.Context) (Conn, error)

	Reconnect ReconnectOptions

	// OnDebug receives the reconnect events. Nil ignores them.
	OnDebug func(event ChannelDebugEvent)

	// OnDiagnostic receives SandboxLog and DebugEvent frames, which never
	// reach Next. Nil drops them.
	OnDiagnostic func(message Message)

	// OnBridgeError receives the error of each error frame, which also
	// reaches Next.
	OnBridgeError func(err error)

	// InitialLastSeenEventID seeds the cursor from a previous process, so
	// that Open with Resume replays only later events.
	InitialLastSeenEventID int64

	// CustomFrameTypes are adapter-specific frame types. Frames of these
	// types reach Next as CustomMessage values instead of failing to decode.
	CustomFrameTypes []string
}

// ReconnectOptions bound the reconnects after an unexpected disconnect.
// Zero values select the upstream defaults.
type ReconnectOptions struct {
	// MaxElapsed is the total reconnect budget. Default 30 seconds.
	MaxElapsed time.Duration

	// InitialDelay is the first backoff delay. Default 50 milliseconds.
	InitialDelay time.Duration

	// MaxDelay caps the backoff. Default 2 seconds.
	MaxDelay time.Duration
}

// ChannelDebugEventKind identifies a reconnect event.
type ChannelDebugEventKind string

// Channel debug event kinds.
const (
	ReconnectAttempt ChannelDebugEventKind = "reconnect-attempt"
	Reconnected      ChannelDebugEventKind = "reconnected"
	ReconnectFailed  ChannelDebugEventKind = "reconnect-failed"
)

// ChannelDebugEvent describes a reconnect.
type ChannelDebugEvent struct {
	Kind            ChannelDebugEventKind
	Attempt         int
	LastSeenEventID int64

	// Cause is the last failure of a failed reconnect.
	Cause error
}

// Channel is the host side of a bridge connection, the Go form of upstream
// SandboxChannel. It delivers frames in arrival order through Next and
// survives transient disconnects: the bridge keeps running and buffers its
// events by sequence number, so on an unexpected drop the channel
// reconnects and asks the bridge to replay everything after the last frame
// it received. Commands sent while disconnected are sent after the
// reconnect.
//
// A Channel owns one goroutine that reads the connection and reconnects;
// Close and Suspend stop it and wait for it.
type Channel struct {
	options ChannelOptions

	lifetime     context.Context
	stopLifetime context.CancelFunc
	reader       sync.WaitGroup

	// writeMu serializes the writes to the connection, so that a slow write
	// never holds mu, and so that commands sent while a connection is being
	// established wait until the resume and the queued commands went out.
	// It is taken before mu, never while mu is held.
	writeMu sync.Mutex

	mu   sync.Mutex
	conn Conn

	// connected reports a connection that carries commands: it is set only
	// after the resume and the queued commands were sent on it.
	connected     bool
	dialing       bool
	closing       bool
	suspended     bool
	terminal      bool
	closeCode     int
	closeReason   string
	lastSeen      int64
	pinned        *pinnedCursor
	pendingSends  [][]byte
	frames        []Frame
	ready         chan struct{}
	done          chan struct{}
	cancelConnect context.CancelFunc

	reconnectHandlers map[int]func()
	nextHandlerID     int
}

// pinnedCursor is a cursor pinned by an event checkpoint.
type pinnedCursor struct {
	seq   int64
	token *int
}

// NewChannel returns a channel that is not yet open.
func NewChannel(options ChannelOptions) *Channel {
	lifetime, stopLifetime := context.WithCancel(context.Background())

	return &Channel{
		options:           options,
		lifetime:          lifetime,
		stopLifetime:      stopLifetime,
		lastSeen:          options.InitialLastSeenEventID,
		ready:             make(chan struct{}, 1),
		done:              make(chan struct{}),
		reconnectHandlers: make(map[int]func()),
	}
}

// OpenOptions configure Channel.Open.
type OpenOptions struct {
	// Resume attaches to a bridge that is already mid-session: after the
	// connection opens, the channel asks the bridge to replay every event
	// after the seeded cursor.
	Resume bool
}

// Open establishes the first connection. It makes a single attempt;
// reconnects only apply to drops after a successful open. Ctx bounds the
// attempt; the connection itself lives until the channel ends.
func (c *Channel) Open(ctx context.Context, options OpenOptions) error {
	c.mu.Lock()
	if c.terminal {
		c.mu.Unlock()

		return errors.New("bridge channel: cannot open a closed channel")
	}
	c.mu.Unlock()

	conn, err := c.connect(ctx)
	if err != nil {
		return err
	}

	if err := c.establish(conn, options.Resume); err != nil {
		return err
	}

	c.reader.Go(func() {
		c.read(conn)
	})

	return nil
}

// Next returns the next frame. After the channel ended and every received
// frame was delivered, it returns a *ClosedError.
func (c *Channel) Next(ctx context.Context) (Frame, error) {
	for {
		c.mu.Lock()
		if len(c.frames) > 0 {
			frame := c.frames[0]
			c.frames[0] = Frame{}
			c.frames = c.frames[1:]
			c.mu.Unlock()

			return frame, nil
		}

		if c.terminal {
			closed := &ClosedError{
				Code:   c.closeCode,
				Reason: c.closeReason,
			}
			c.mu.Unlock()

			return Frame{}, closed
		}
		c.mu.Unlock()

		select {
		case <-c.ready:
		case <-c.done:
		case <-ctx.Done():
			return Frame{}, context.Cause(ctx)
		}
	}
}

// Send encodes a command and sends it, queueing it while disconnected. A
// command whose write fails is queued again and the connection is closed,
// so that the channel reconnects and sends it after the resume.
func (c *Channel) Send(command any) error {
	data, err := json.Marshal(command)
	if err != nil {
		return fmt.Errorf("encode bridge command: %w", err)
	}

	c.writeMu.Lock()
	defer c.writeMu.Unlock()

	c.mu.Lock()
	if c.terminal {
		c.mu.Unlock()

		return fmt.Errorf("bridge channel: cannot send %T: %w", command, ErrChannelClosed)
	}

	if !c.connected {
		c.pendingSends = append(c.pendingSends, data)
		c.mu.Unlock()

		return nil
	}

	conn := c.conn
	c.mu.Unlock()

	if err := conn.Write(data); err != nil {
		c.dropAfterWriteFailure(conn, [][]byte{
			data,
		})
	}

	return nil
}

// LastSeenEventID returns the highest event sequence number received.
// Persist it so that a later process can seed InitialLastSeenEventID.
func (c *Channel) LastSeenEventID() int64 {
	c.mu.Lock()
	defer c.mu.Unlock()

	return c.lastSeen
}

// OnReconnect registers a handler that runs after each reconnect, for
// example to resend unanswered requests. It returns a function that removes
// the handler.
func (c *Channel) OnReconnect(handler func()) func() {
	c.mu.Lock()
	defer c.mu.Unlock()

	handlerID := c.nextHandlerID
	c.nextHandlerID++
	c.reconnectHandlers[handlerID] = handler

	return func() {
		c.mu.Lock()
		defer c.mu.Unlock()

		delete(c.reconnectHandlers, handlerID)
	}
}

// BeginClose marks that the host is tearing the session down, so that the
// next disconnect ends the channel instead of reconnecting. Call it before
// sending a stop or destroy command whose completion closes the bridge
// connection.
func (c *Channel) BeginClose() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.closing = true
	c.abortConnectLocked()
}

// Close ends the channel and waits for its goroutine. Frames already
// received remain available through Next.
func (c *Channel) Close() error {
	c.shutdown(false)
	c.finalize(closeNormal, CloseReasonClosed)

	return nil
}

// Suspend ends the channel at a turn slice boundary. Frames that arrive
// afterwards are ignored, so the cursor freezes at the last received frame;
// the bridge keeps the turn running and replays the rest to the next
// process. Frames already received remain available through Next, which
// then reports CloseReasonSuspended. Suspend returns the cursor to persist:
// the pinned checkpoint when one is pinned, and the last received event
// otherwise.
func (c *Channel) Suspend() int64 {
	c.mu.Lock()
	cursor := c.cursorLocked()
	if c.terminal {
		c.mu.Unlock()

		return cursor
	}
	c.mu.Unlock()

	c.shutdown(true)
	c.finalize(closeNormal, CloseReasonSuspended)

	c.mu.Lock()
	defer c.mu.Unlock()

	return c.cursorLocked()
}

// Done is closed when the channel ended.
func (c *Channel) Done() <-chan struct{} {
	return c.done
}

// Closed reports whether the channel ended.
func (c *Channel) Closed() bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	return c.terminal
}

// shutdown stops reading, closes the connection and waits for the reader.
func (c *Channel) shutdown(suspend bool) {
	c.mu.Lock()
	c.closing = true
	if suspend {
		c.suspended = true
	}

	c.abortConnectLocked()
	conn := c.conn
	c.mu.Unlock()

	if conn != nil {
		_ = conn.Close()
	}

	c.stopLifetime()
	c.reader.Wait()
}

// connect opens a connection bounded by ctx while connecting. The
// connection lives until the channel ends or the next connection replaces
// it.
func (c *Channel) connect(ctx context.Context) (Conn, error) {
	connectCtx, cancel := context.WithCancel(c.lifetime)

	c.mu.Lock()
	releasePrevious := c.cancelConnect
	c.cancelConnect = cancel
	c.dialing = true
	c.mu.Unlock()

	if releasePrevious != nil {
		releasePrevious()
	}

	stopWatching := context.AfterFunc(ctx, cancel)
	defer stopWatching()

	conn, err := c.options.Connect(connectCtx)

	c.mu.Lock()
	c.dialing = false
	c.mu.Unlock()

	if err != nil {
		cancel()

		if ctx.Err() != nil {
			return nil, context.Cause(ctx)
		}

		return nil, fmt.Errorf("connect bridge channel: %w", err)
	}

	return conn, nil
}

// read reads frames from conn and reconnects after unexpected drops. It is
// the channel's goroutine.
func (c *Channel) read(conn Conn) {
	for {
		data, err := conn.Read()
		if err == nil {
			c.handleIncoming(data)

			continue
		}

		next, ok := c.handleDrop(conn)
		if !ok {
			return
		}

		conn = next
	}
}

// handleIncoming decodes and delivers one frame.
func (c *Channel) handleIncoming(data []byte) {
	c.mu.Lock()
	suspended := c.suspended
	c.mu.Unlock()

	if suspended {
		return
	}

	frame, err := c.decodeFrame(data)
	if err != nil {
		frame = Frame{
			Message: StreamMessage{
				Part: harness.ErrorPart{
					Err: err,
				},
			},
			Seq:    frameSeq(data),
			HasSeq: frameSeq(data) > 0,
		}
	}

	c.dispatch(frame)

	c.mu.Lock()
	defer c.mu.Unlock()

	if frame.HasSeq && frame.Seq > c.lastSeen {
		c.lastSeen = frame.Seq
	}
}

// decodeFrame decodes a frame, keeping the adapter-specific types whole.
func (c *Channel) decodeFrame(data []byte) (Frame, error) {
	if frame, ok := decodeCustomFrame(data, c.options.CustomFrameTypes); ok {
		return frame, nil
	}

	return DecodeFrame(data)
}

// dispatch routes diagnostics to their sink and queues every other frame.
func (c *Channel) dispatch(frame Frame) {
	switch typed := frame.Message.(type) {
	case SandboxLog, DebugEvent:
		if c.options.OnDiagnostic != nil {
			c.options.OnDiagnostic(typed)
		}

		return
	case StreamMessage:
		if failure, ok := typed.Part.(harness.ErrorPart); ok && c.options.OnBridgeError != nil {
			c.options.OnBridgeError(failure.Err)
		}

		if finishStep, ok := typed.Part.(harness.FinishStepPart); ok && frame.HasSeq {
			finishStep.Checkpoint = eventCheckpoint{
				channel: c,
				seq:     frame.Seq,
			}
			frame.Message = StreamMessage{
				Part: finishStep,
			}
		}
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	c.frames = append(c.frames, frame)

	select {
	case c.ready <- struct{}{}:
	default:
	}
}

// handleDrop handles a lost connection: it ends the channel during
// teardown and reconnects otherwise. It reports false when reading stops.
func (c *Channel) handleDrop(conn Conn) (Conn, bool) {
	c.mu.Lock()
	if conn != c.conn {
		c.mu.Unlock()

		return nil, false
	}

	c.connected = false
	closing := c.closing
	reason := CloseReasonClosed
	if c.suspended {
		reason = CloseReasonSuspended
	}
	c.mu.Unlock()

	if closing {
		c.finalize(closeNormal, reason)

		return nil, false
	}

	return c.reconnect()
}

// reconnect reconnects with exponential backoff within the reconnect
// budget, then replays the missed events and flushes queued commands.
func (c *Channel) reconnect() (Conn, bool) {
	maxElapsed := durationOr(c.options.Reconnect.MaxElapsed, defaultReconnectMaxElapsed)
	delay := durationOr(c.options.Reconnect.InitialDelay, defaultReconnectInitialDelay)
	maxDelay := durationOr(c.options.Reconnect.MaxDelay, defaultReconnectMaxDelay)
	deadline := time.Now().Add(maxElapsed)

	var lastErr error

	for attempt := 1; ; attempt++ {
		if c.isClosing() {
			return nil, false
		}

		remaining := time.Until(deadline)
		if remaining <= 0 {
			c.failReconnect(attempt-1, errors.Join(errors.New("reconnect deadline expired"), lastErr))

			return nil, false
		}

		c.debug(ChannelDebugEvent{
			Kind:            ReconnectAttempt,
			Attempt:         attempt,
			LastSeenEventID: c.LastSeenEventID(),
		})

		conn, err := c.connectWithin(remaining)
		if err == nil {
			err = c.adoptReconnected(conn)
		}

		if err == nil {
			c.afterReconnect(attempt)

			return conn, true
		}

		lastErr = err

		if c.isClosing() {
			return nil, false
		}

		if !c.sleep(min(delay, time.Until(deadline))) {
			return nil, false
		}

		delay = min(time.Duration(float64(delay)*1.5), maxDelay)
	}
}

// connectWithin makes one connection attempt bounded by the remaining
// budget.
func (c *Channel) connectWithin(remaining time.Duration) (Conn, error) {
	attemptCtx, cancel := context.WithTimeout(c.lifetime, remaining)
	defer cancel()

	return c.connect(attemptCtx)
}

// adoptReconnected establishes a new connection that asks for the missed
// events, unless the channel is ending. A failed connection is closed, so
// that the next attempt replaces it.
func (c *Channel) adoptReconnected(conn Conn) error {
	if c.isClosing() {
		_ = conn.Close()

		return fmt.Errorf("bridge channel: reconnected while closing: %w", ErrChannelClosed)
	}

	return c.establish(conn, true)
}

// afterReconnect runs the reconnect handlers and reports the reconnect.
func (c *Channel) afterReconnect(attempt int) {
	c.mu.Lock()
	handlers := make([]func(), 0, len(c.reconnectHandlers))
	for _, handler := range c.reconnectHandlers {
		handlers = append(handlers, handler)
	}
	c.mu.Unlock()

	for _, handler := range handlers {
		handler()
	}

	c.debug(ChannelDebugEvent{
		Kind:            Reconnected,
		Attempt:         attempt,
		LastSeenEventID: c.LastSeenEventID(),
	})
}

// failReconnect ends the channel after the reconnect budget ran out.
func (c *Channel) failReconnect(attempts int, cause error) {
	c.finalize(closeAbnormal, CloseReasonReconnectFailed)
	c.debug(ChannelDebugEvent{
		Kind:            ReconnectFailed,
		Attempt:         attempts,
		LastSeenEventID: c.LastSeenEventID(),
		Cause:           cause,
	})
}

// establish makes conn the channel's connection. It asks the bridge to
// replay the events after the cursor when resume is set, sends the commands
// queued while disconnected, and only then marks the channel connected.
// Sends wait on writeMu meanwhile, so commands keep their order. On failure
// the unsent commands stay queued and conn is closed.
func (c *Channel) establish(conn Conn, resume bool) error {
	c.writeMu.Lock()
	defer c.writeMu.Unlock()

	c.mu.Lock()
	if c.terminal {
		c.mu.Unlock()
		_ = conn.Close()

		return fmt.Errorf("bridge channel: cannot use a closed channel: %w", ErrChannelClosed)
	}

	c.conn = conn
	c.connected = false
	c.mu.Unlock()

	if resume {
		if err := c.sendResume(conn); err != nil {
			c.dropAfterWriteFailure(conn, nil)

			return err
		}
	}

	if err := c.flushPending(conn); err != nil {
		return err
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.conn == conn && !c.terminal {
		c.connected = true
	}

	return nil
}

// sendResume asks the bridge to replay the events after the cursor. The
// caller holds writeMu.
func (c *Channel) sendResume(conn Conn) error {
	c.mu.Lock()
	command := ResumeCommand{
		Type:            "resume",
		LastSeenEventID: c.lastSeen,
	}
	c.mu.Unlock()

	data, err := json.Marshal(command)
	if err != nil {
		return fmt.Errorf("encode bridge resume: %w", err)
	}

	if err := conn.Write(data); err != nil {
		return fmt.Errorf("send bridge resume: %w", err)
	}

	return nil
}

// flushPending sends the commands queued while disconnected. When a write
// fails, the command and those after it are queued again and conn is
// closed. The caller holds writeMu.
func (c *Channel) flushPending(conn Conn) error {
	c.mu.Lock()
	queued := c.pendingSends
	c.pendingSends = nil
	c.mu.Unlock()

	for index, data := range queued {
		if err := conn.Write(data); err != nil {
			c.dropAfterWriteFailure(conn, queued[index:])

			return fmt.Errorf("send queued bridge command: %w", err)
		}
	}

	return nil
}

// dropAfterWriteFailure queues the unsent commands ahead of those queued
// since, marks the channel disconnected and closes conn, which makes the
// reader reconnect.
func (c *Channel) dropAfterWriteFailure(conn Conn, unsent [][]byte) {
	c.mu.Lock()
	c.pendingSends = append(slices.Clone(unsent), c.pendingSends...)
	if c.conn == conn {
		c.connected = false
	}
	c.mu.Unlock()

	_ = conn.Close()
}

// finalize ends the channel once.
func (c *Channel) finalize(code int, reason string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.terminal {
		return
	}

	c.abortConnectLocked()
	c.terminal = true
	c.connected = false
	c.closeCode = code
	c.closeReason = reason
	close(c.done)
}

// sleep waits for d, reporting false when the channel is closing.
func (c *Channel) sleep(d time.Duration) bool {
	timer := time.NewTimer(d)
	defer timer.Stop()

	select {
	case <-timer.C:
		return !c.isClosing()
	case <-c.lifetime.Done():
		return false
	}
}

// isClosing reports teardown or a terminal channel.
func (c *Channel) isClosing() bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	return c.closing || c.terminal
}

// abortConnectLocked cancels the connection attempt in progress. The caller
// holds mu.
func (c *Channel) abortConnectLocked() {
	if c.cancelConnect != nil && c.dialing {
		c.cancelConnect()
	}
}

// cursorLocked returns the pinned cursor or the last seen event. The caller
// holds mu.
func (c *Channel) cursorLocked() int64 {
	if c.pinned != nil {
		return c.pinned.seq
	}

	return c.lastSeen
}

// debug reports a debug event.
func (c *Channel) debug(event ChannelDebugEvent) {
	if c.options.OnDebug != nil {
		c.options.OnDebug(event)
	}
}

// eventCheckpoint pins a channel's suspension cursor at one event.
type eventCheckpoint struct {
	channel *Channel
	seq     int64
}

// Pin pins the cursor and returns the function that releases the pin.
func (e eventCheckpoint) Pin() func() {
	token := new(int)

	e.channel.mu.Lock()
	e.channel.pinned = &pinnedCursor{
		seq:   e.seq,
		token: token,
	}
	e.channel.mu.Unlock()

	return func() {
		e.channel.mu.Lock()
		defer e.channel.mu.Unlock()

		if e.channel.pinned != nil && e.channel.pinned.token == token {
			e.channel.pinned = nil
		}
	}
}

// frameSeq returns the sequence number of an undecodable frame, or zero.
func frameSeq(data []byte) int64 {
	var envelope struct {
		Seq float64 `json:"seq"`
	}

	if err := json.Unmarshal(data, &envelope); err != nil {
		return 0
	}

	return int64(envelope.Seq)
}

// durationOr returns value, or fallback when value is not positive.
func durationOr(value time.Duration, fallback time.Duration) time.Duration {
	if value <= 0 {
		return fallback
	}

	return value
}
