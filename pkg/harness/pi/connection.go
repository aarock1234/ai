package pi

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/aarock1234/ai/internal/shellquote"
	"github.com/aarock1234/ai/pkg/sandbox"
)

// followerExitGrace bounds the wait for a killed follower to exit.
const followerExitGrace = 5 * time.Second

// errConnectionEnded reports that Pi's output ended, because Pi exited or
// the connection was frozen.
var errConnectionEnded = errors.New("pi rpc connection ended")

// recordHandler receives every record that is not a command response, in
// order, on the connection's reading goroutine. It must not wait for a
// command response.
type recordHandler func(recordType string, record []byte)

// connectionOptions configure newConnection.
type connectionOptions struct {
	sandbox  sandbox.Session
	layout   layout
	launchID string

	// inputSequence is the sequence number of the last record written to the
	// relay's inbox by an earlier host process.
	inputSequence int64

	// outboxOffset is where reading starts: the cursor of an earlier host
	// process, or zero for a new launch.
	outboxOffset int64

	handler recordHandler
}

// connection is the host side of one relay launch. Pi's standard input is
// the relay's inbox, where every record is a numbered file; its standard
// output is the relay's outbox, which a follower process streams from a byte
// offset. The connection correlates command responses and hands every other
// record to its handler.
//
// The connection owns the goroutine that reads the follower. Freeze and
// close end it and wait for it.
type connection struct {
	sandbox  sandbox.Session
	layout   layout
	launchID string
	handler  recordHandler

	writeMu       sync.Mutex
	inputSequence int64

	nextID atomic.Int64

	mu       sync.Mutex
	waiters  map[string]chan response
	offset   int64
	freezing bool
	started  bool
	endErr   error

	follower       sandbox.Process
	cancelFollower context.CancelFunc

	// ended is closed when the reading goroutine returned.
	ended chan struct{}
}

// newConnection returns a connection that does not read yet.
func newConnection(options connectionOptions) *connection {
	return &connection{
		sandbox:       options.sandbox,
		layout:        options.layout,
		launchID:      options.launchID,
		handler:       options.handler,
		inputSequence: options.inputSequence,
		offset:        options.outboxOffset,
		waiters:       make(map[string]chan response),
		ended:         make(chan struct{}),
	}
}

// stateDir returns the relay state directory.
func (c *connection) stateDir() string {
	return c.layout.relayStateDir(c.launchID)
}

// start starts following Pi's output at the cursor. It does nothing when the
// connection already reads.
func (c *connection) start(ctx context.Context) error {
	c.mu.Lock()
	if c.started {
		c.mu.Unlock()

		return nil
	}

	c.started = true
	c.mu.Unlock()

	followerCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))

	follower, err := c.sandbox.Spawn(followerCtx, sandbox.ProcessOptions{
		Command: followCommand(followOptions{
			layout:   c.layout,
			file:     path.Join(c.stateDir(), "outbox.jsonl"),
			exitFile: path.Join(c.stateDir(), "exit.json"),
			offset:   c.offset,
		}),
	})
	if err != nil {
		cancel()
		c.end(fmt.Errorf("follow pi output: %w", err))
		close(c.ended)

		return fmt.Errorf("follow pi output: %w", err)
	}

	c.mu.Lock()
	c.follower = follower
	c.cancelFollower = cancel
	c.mu.Unlock()

	go func() {
		defer close(c.ended)

		c.read(follower)
		c.reap(follower, cancel)
	}()

	return nil
}

// read splits the follower's output into records, strictly on line feeds,
// and dispatches them until the output ends or the connection freezes.
func (c *connection) read(follower sandbox.Process) {
	reader := bufio.NewReaderSize(follower.Stdout(), 64*1024)

	for {
		line, err := reader.ReadBytes('\n')
		if err != nil {
			c.end(errConnectionEnded)

			return
		}

		if c.isFreezing() {
			c.end(errConnectionEnded)

			return
		}

		c.dispatch(line)

		c.mu.Lock()
		c.offset += int64(len(line))
		c.mu.Unlock()
	}
}

// reap waits for the follower to exit after its output ended.
func (c *connection) reap(follower sandbox.Process, cancel context.CancelFunc) {
	waitCtx, stopWaiting := context.WithTimeout(context.Background(), followerExitGrace)
	defer stopWaiting()

	_, _ = follower.Wait(waitCtx)
	cancel()
}

// dispatch routes one record.
func (c *connection) dispatch(line []byte) {
	record := bytes.TrimSuffix(bytes.TrimSuffix(line, []byte("\n")), []byte("\r"))
	if len(bytes.TrimSpace(record)) == 0 {
		return
	}

	var header recordHeader
	if err := json.Unmarshal(record, &header); err != nil {
		return
	}

	if header.Type != recordResponse {
		c.handler(header.Type, record)

		return
	}

	var reply response
	if err := json.Unmarshal(record, &reply); err != nil {
		return
	}

	c.mu.Lock()
	waiter, ok := c.waiters[reply.ID]
	delete(c.waiters, reply.ID)
	c.mu.Unlock()

	if ok {
		waiter <- reply
	}
}

// end records why reading ended and fails the waiting commands.
func (c *connection) end(err error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.endErr == nil {
		c.endErr = err
	}

	for id, waiter := range c.waiters {
		close(waiter)
		delete(c.waiters, id)
	}
}

// isFreezing reports a requested freeze.
func (c *connection) isFreezing() bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	return c.freezing
}

// call sends a command and waits for its response. A rejected command is
// reported as *CommandError.
func (c *connection) call(ctx context.Context, command any) (response, error) {
	id := "go-" + strconv.FormatInt(c.nextID.Add(1), 10)
	waiter := make(chan response, 1)

	c.mu.Lock()
	if c.endErr != nil {
		err := c.endErr
		c.mu.Unlock()

		return response{}, err
	}

	c.waiters[id] = waiter
	c.mu.Unlock()

	encoded, err := encodeCommand(id, command)
	if err == nil {
		err = c.write(ctx, encoded)
	}

	if err != nil {
		c.forget(id)

		return response{}, err
	}

	select {
	case reply, ok := <-waiter:
		if !ok {
			return response{}, c.endError()
		}

		return reply, reply.err()
	case <-ctx.Done():
		c.forget(id)

		return response{}, context.Cause(ctx)
	}
}

// forget removes a waiter.
func (c *connection) forget(id string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	delete(c.waiters, id)
}

// endError returns why reading ended.
func (c *connection) endError() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.endErr == nil {
		return errConnectionEnded
	}

	return c.endErr
}

// send writes a record that has no response, such as an extension UI
// response.
func (c *connection) send(ctx context.Context, record any) error {
	encoded, err := json.Marshal(record)
	if err != nil {
		return fmt.Errorf("encode pi rpc record: %w", err)
	}

	return c.write(ctx, encoded)
}

// write writes one record to the relay's inbox as the next numbered file.
// Writes are serialized, so that a failed write does not leave a gap the
// relay would wait on.
func (c *connection) write(ctx context.Context, record []byte) error {
	c.writeMu.Lock()
	defer c.writeMu.Unlock()

	sequence := c.inputSequence + 1

	err := c.sandbox.WriteBinaryFile(ctx, sandbox.WriteBinaryFileOptions{
		Path:    c.inboxPath(sequence, "jsonl"),
		Content: append(record, '\n'),
	})
	if err != nil {
		return fmt.Errorf("write pi rpc record: %w", err)
	}

	c.inputSequence = sequence

	return nil
}

// closeInput closes Pi's standard input, which asks Pi to shut down.
func (c *connection) closeInput(ctx context.Context) error {
	c.writeMu.Lock()
	defer c.writeMu.Unlock()

	sequence := c.inputSequence + 1

	err := c.sandbox.WriteBinaryFile(ctx, sandbox.WriteBinaryFileOptions{
		Path:    c.inboxPath(sequence, "close"),
		Content: []byte{},
	})
	if err != nil {
		return fmt.Errorf("close pi rpc input: %w", err)
	}

	c.inputSequence = sequence

	return nil
}

// inboxPath returns the path of an inbox file.
func (c *connection) inboxPath(sequence int64, extension string) string {
	return path.Join(c.stateDir(), "inbox", fmt.Sprintf("%012d.%s", sequence, extension))
}

// cursor returns the output offset after the last dispatched record and the
// sequence of the last written record.
func (c *connection) cursor() (int64, int64) {
	c.mu.Lock()
	offset := c.offset
	c.mu.Unlock()

	c.writeMu.Lock()
	defer c.writeMu.Unlock()

	return offset, c.inputSequence
}

// freeze stops reading at a record boundary without stopping Pi, and
// returns the cursor of the last dispatched record.
func (c *connection) freeze() (int64, int64) {
	c.mu.Lock()
	c.freezing = true
	follower := c.follower
	started := c.started
	c.mu.Unlock()

	if follower != nil {
		_ = follower.Kill(context.Background())
	}

	if started {
		<-c.ended
	}

	return c.cursor()
}

// waitEnded waits until Pi's output ended or ctx is done, and reports
// whether it ended.
func (c *connection) waitEnded(ctx context.Context) bool {
	select {
	case <-c.ended:
		return true
	case <-ctx.Done():
		return false
	}
}

// stop kills the follower and waits for the reading goroutine.
func (c *connection) stop() {
	c.mu.Lock()
	follower := c.follower
	started := c.started
	c.mu.Unlock()

	if follower != nil {
		_ = follower.Kill(context.Background())
	}

	if started {
		<-c.ended
	}
}

// encodeCommand encodes a command with its correlation ID.
func encodeCommand(id string, command any) ([]byte, error) {
	encoded, err := json.Marshal(command)
	if err != nil {
		return nil, fmt.Errorf("encode pi rpc command: %w", err)
	}

	var fields map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &fields); err != nil {
		return nil, fmt.Errorf("encode pi rpc command: %w", err)
	}

	encodedID, err := json.Marshal(id)
	if err != nil {
		return nil, fmt.Errorf("encode pi rpc command: %w", err)
	}

	fields["id"] = encodedID

	return json.Marshal(fields)
}

// followOptions configure followCommand.
type followOptions struct {
	layout   layout
	file     string
	exitFile string
	offset   int64
}

// followCommand returns the command that streams a relay file from offset.
func followCommand(options followOptions) string {
	return fmt.Sprintf(
		"exec node %s %s %s %d 2>/dev/null",
		shellquote.Quote(options.layout.installFile(followScript)),
		shellquote.Quote(options.file),
		shellquote.Quote(options.exitFile),
		options.offset,
	)
}
