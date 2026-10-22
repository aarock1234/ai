package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"
)

// Shutdown of the server process: Close closes its standard input, waits
// stdioExitGrace for it to exit, terminates it, and kills it after another
// stdioExitGrace.
const stdioExitGrace = 2 * time.Second

// StdioTransportOptions configure NewStdioTransport.
type StdioTransportOptions struct {
	// Command is the server executable, resolved through PATH. It is
	// required.
	Command string

	Args []string

	// Env holds environment variables for the server. They are added to a
	// minimal inherited environment (PATH, HOME and similar variables)
	// rather than the full environment of this process.
	Env map[string]string

	// Dir is the working directory. Empty uses the current directory.
	Dir string

	// Stderr receives the server's standard error. Nil uses os.Stderr.
	Stderr io.Writer

	// Shell runs the server through a shell, upstream's shell option.
	// Command is then shell syntax, such as "npx -y server", and each of Args
	// is quoted for the shell so that arguments cannot inject commands.
	Shell bool

	// ShellPath selects the shell and implies Shell. Empty uses /bin/sh, or
	// COMSPEC (cmd.exe) on Windows.
	ShellPath string
}

// usesShell reports whether the server runs through a shell.
func (o StdioTransportOptions) usesShell() bool {
	return o.Shell || o.ShellPath != ""
}

// StdioTransport runs an MCP server as a subprocess and exchanges
// newline-delimited JSON-RPC messages over its standard input and output.
type StdioTransport struct {
	options StdioTransportOptions
	inbox   *inbox
	workers sync.WaitGroup

	// lifetime ends delivery to the inbox when Close begins.
	lifetime     context.Context
	stopDelivery context.CancelFunc

	writeMu sync.Mutex

	mu      sync.Mutex
	started bool
	closed  bool
	command *exec.Cmd
	stdin   io.WriteCloser
	exited  chan struct{}
}

// NewStdioTransport returns a transport that starts options.Command when the
// client starts it.
func NewStdioTransport(options StdioTransportOptions) (*StdioTransport, error) {
	if options.Command == "" {
		return nil, &ClientError{
			Message: "a stdio server command is required",
		}
	}

	if err := validateCommandLine(options.Command, options.Args); err != nil {
		return nil, err
	}

	lifetime, stopDelivery := context.WithCancel(context.Background())

	return &StdioTransport{
		options:      options,
		inbox:        newInbox(),
		lifetime:     lifetime,
		stopDelivery: stopDelivery,
		exited:       make(chan struct{}),
	}, nil
}

// SupportsProtocolVersionDiscovery reports that the client may probe with
// server/discover.
func (t *StdioTransport) SupportsProtocolVersionDiscovery() bool {
	return true
}

// Start launches the server process.
func (t *StdioTransport) Start(_ context.Context) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.started {
		return &ClientError{
			Message: "mcp stdio transport already started",
		}
	}

	t.started = true

	command, stdin, stdout, err := t.launch()
	if err != nil {
		return err
	}

	t.command = command
	t.stdin = stdin

	t.workers.Go(func() {
		t.readOutput(stdout)
	})

	t.workers.Go(func() {
		// The exit status is irrelevant: the transport reports the end of
		// the output instead.
		_ = command.Wait()
		close(t.exited)
	})

	return nil
}

// launch starts the process with pipes for its standard input and output.
func (t *StdioTransport) launch() (*exec.Cmd, io.WriteCloser, io.ReadCloser, error) {
	command, err := t.serverCommand()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("prepare mcp stdio server %q: %w", t.options.Command, err)
	}

	command.Env = serverEnvironment(t.options.Env)
	command.Dir = t.options.Dir
	command.Stderr = t.options.Stderr
	command.WaitDelay = stdioExitGrace

	if command.Stderr == nil {
		command.Stderr = os.Stderr
	}

	stdin, err := command.StdinPipe()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("open mcp stdio server input: %w", err)
	}

	stdout, err := command.StdoutPipe()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("open mcp stdio server output: %w", err)
	}

	if err := command.Start(); err != nil {
		return nil, nil, nil, fmt.Errorf("start mcp stdio server %q: %w", t.options.Command, err)
	}

	return command, stdin, stdout, nil
}

// readOutput delivers each line of the server's output as a message until
// the output ends, then closes the inbox.
func (t *StdioTransport) readOutput(stdout io.Reader) {
	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 0, 64<<10), maxMessageBytes)

	ctx := t.lifetime

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		message, err := ParseMessage([]byte(line))
		if err != nil {
			t.inbox.deliverError(ctx, &ClientError{
				Message: "mcp stdio transport error: failed to parse message",
				Cause:   err,
			})

			continue
		}

		t.inbox.deliverMessage(ctx, message)
	}

	t.inbox.close(outputEndCause(scanner.Err()))
}

func outputEndCause(err error) error {
	if err == nil || errors.Is(err, os.ErrClosed) {
		return errors.New("mcp stdio server closed its output")
	}

	return fmt.Errorf("read mcp stdio server output: %w", err)
}

// Receive returns the next message from the server.
func (t *StdioTransport) Receive(ctx context.Context) (Message, error) {
	return t.inbox.receive(ctx)
}

// Send writes message as one line to the server's standard input. ctx is
// checked before writing; a write blocked by a server that stopped reading
// ends when Close stops the process.
func (t *StdioTransport) Send(ctx context.Context, message Message, _ SendOptions) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	t.mu.Lock()
	stdin := t.stdin
	closed := t.closed
	t.mu.Unlock()

	if stdin == nil || closed {
		return &ClientError{
			Message: "mcp stdio transport is not connected",
			Cause:   ErrConnectionClosed,
		}
	}

	line, err := json.Marshal(message)
	if err != nil {
		return fmt.Errorf("encode json-rpc message: %w", err)
	}

	line = append(line, '\n')

	t.writeMu.Lock()
	defer t.writeMu.Unlock()

	if _, err := stdin.Write(line); err != nil {
		return &ClientError{
			Message: "mcp stdio transport error: write to server",
			Cause:   err,
		}
	}

	return nil
}

// Close shuts the server down: it closes the server's input and waits for
// the process to exit, terminating and then killing it when it does not.
// ctx ends the graceful waits early. Close waits for the transport's
// goroutines.
func (t *StdioTransport) Close(ctx context.Context) error {
	t.mu.Lock()
	if t.closed {
		t.mu.Unlock()

		return nil
	}

	t.closed = true
	command := t.command
	stdin := t.stdin
	t.mu.Unlock()

	t.stopDelivery()

	if command == nil {
		t.inbox.close(nil)

		return nil
	}

	// The server sees the end of its input as the shutdown request; a
	// failure to close means it is already gone.
	_ = stdin.Close()

	err := t.stopProcess(ctx, command)

	t.workers.Wait()
	t.inbox.close(nil)

	return err
}

// stopProcess waits for the process to exit, escalating to termination and
// then to killing it.
func (t *StdioTransport) stopProcess(ctx context.Context, command *exec.Cmd) error {
	if t.waitForExit(ctx, stdioExitGrace) {
		return nil
	}

	if err := terminateProcess(command.Process); err != nil && !errors.Is(err, os.ErrProcessDone) {
		return fmt.Errorf("terminate mcp stdio server: %w", err)
	}

	if t.waitForExit(ctx, stdioExitGrace) {
		return nil
	}

	if err := command.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
		return fmt.Errorf("kill mcp stdio server: %w", err)
	}

	<-t.exited

	return nil
}

// waitForExit reports whether the process exited within grace. It stops
// waiting early when ctx ends.
func (t *StdioTransport) waitForExit(ctx context.Context, grace time.Duration) bool {
	timer := time.NewTimer(grace)
	defer timer.Stop()

	select {
	case <-t.exited:
		return true
	case <-timer.C:
		return false
	case <-ctx.Done():
		return false
	}
}

// serverEnvironment returns the default inherited variables overridden and
// extended by custom. Inherited values that are shell function definitions
// are skipped.
func serverEnvironment(custom map[string]string) []string {
	environment := make([]string, 0, len(defaultInheritedEnvironment)+len(custom))
	customKeys := make(map[string]bool, len(custom))

	for key, value := range custom {
		environment = append(environment, key+"="+value)
		customKeys[environmentKey(key)] = true
	}

	for _, key := range defaultInheritedEnvironment {
		if customKeys[environmentKey(key)] {
			continue
		}

		value, ok := os.LookupEnv(key)
		if !ok || strings.HasPrefix(value, "()") {
			continue
		}

		environment = append(environment, key+"="+value)
	}

	return environment
}

var (
	_ Transport                  = (*StdioTransport)(nil)
	_ ProtocolDiscoveryTransport = (*StdioTransport)(nil)
)
