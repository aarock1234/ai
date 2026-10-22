package vercel

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"

	"github.com/aarock1234/ai/pkg/sandbox"
)

// killTimeout bounds the kill of a command whose caller went away.
const killTimeout = 10 * time.Second

// Run runs a command with bash and collects its output. Cancelling ctx
// kills the command.
func (c *core) Run(ctx context.Context, options sandbox.ProcessOptions) (sandbox.RunResult, error) {
	return c.runCommand(ctx, bashCommand(options))
}

// Spawn starts a command with bash without waiting for it. Ctx governs the
// command: cancelling it kills the command.
func (c *core) Spawn(ctx context.Context, options sandbox.ProcessOptions) (sandbox.Process, error) {
	if err := ctx.Err(); err != nil {
		return nil, context.Cause(ctx)
	}

	var started *process

	err := c.withResume(ctx, func(sessionID string) error {
		command, err := c.client.startCommand(ctx, sessionID, bashCommand(options))
		if err != nil {
			return err
		}

		started = newProcess(ctx, processSetup{
			client:    c.client,
			sessionID: sessionID,
			commandID: command.ID,
		})

		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("spawn vercel sandbox command: %w", err)
	}

	return started, nil
}

// runCommand runs a command, collecting the output of its log stream.
func (c *core) runCommand(ctx context.Context, request commandRequest) (sandbox.RunResult, error) {
	if err := ctx.Err(); err != nil {
		return sandbox.RunResult{}, context.Cause(ctx)
	}

	var result sandbox.RunResult

	err := c.withResume(ctx, func(sessionID string) error {
		collected, err := c.collectCommand(ctx, sessionID, request)
		if err != nil {
			return err
		}

		result = collected

		return nil
	})
	if err != nil {
		return sandbox.RunResult{}, fmt.Errorf("run vercel sandbox command: %w", err)
	}

	return result, nil
}

// collectCommand runs a command in one session and reads its log stream.
func (c *core) collectCommand(ctx context.Context, sessionID string, request commandRequest) (sandbox.RunResult, error) {
	stream, err := c.client.runCommandStream(ctx, sessionID, request)
	if err != nil {
		return sandbox.RunResult{}, err
	}

	defer func() {
		_ = stream.Close()
	}()

	var stdout strings.Builder
	var stderr strings.Builder
	var commandID string

	finished, err := readCommandStream(stream, commandStreamHandlers{
		onStart: func(command commandMetadata) {
			commandID = command.ID
		},
		onLog: func(line logLine) {
			if line.Stream == "stdout" {
				stdout.WriteString(logText(line))
			} else {
				stderr.WriteString(logText(line))
			}
		},
	})

	if ctx.Err() != nil {
		c.killAfterCancel(ctx, sessionID, commandID)

		return sandbox.RunResult{}, context.Cause(ctx)
	}

	if err != nil {
		return sandbox.RunResult{}, err
	}

	if finished == nil || finished.ExitCode == nil {
		return sandbox.RunResult{}, errNoCommand
	}

	return sandbox.RunResult{
		ExitCode: *finished.ExitCode,
		Stdout:   stdout.String(),
		Stderr:   stderr.String(),
	}, nil
}

// killAfterCancel kills a command whose caller cancelled.
func (c *core) killAfterCancel(ctx context.Context, sessionID string, commandID string) {
	if commandID == "" {
		return
	}

	killCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), killTimeout)
	defer cancel()

	_ = c.client.killCommand(killCtx, sessionID, commandID)
}

// commandStreamHandlers receive the events of a command's log stream.
type commandStreamHandlers struct {
	onStart func(command commandMetadata)
	onLog   func(line logLine)
}

// readCommandStream reads a running command's stream: the started command,
// its log lines and the finished command.
func readCommandStream(stream io.Reader, handlers commandStreamHandlers) (*commandMetadata, error) {
	started := false

	return readLogLines(stream, func(line logLine) {
		handlers.onLog(line)
	}, func(command commandMetadata) {
		if !started {
			started = true
			handlers.onStart(command)
		}
	})
}

// bashCommand runs a shell command with bash.
func bashCommand(options sandbox.ProcessOptions) commandRequest {
	env := options.Env
	if env == nil {
		env = map[string]string{}
	}

	return commandRequest{
		Command: "bash",
		Args: []string{
			"-c",
			options.Command,
		},
		Cwd: options.WorkingDirectory,
		Env: env,
	}
}

// processSetup describes a detached command.
type processSetup struct {
	client    *apiClient
	sessionID string
	commandID string
}

// process is a detached command. A goroutine started with it copies the
// command's log stream into the output streams until the command ends.
type process struct {
	setup  processSetup
	spawn  context.Context
	stdout *sandbox.OutputStream
	stderr *sandbox.OutputStream

	logsDone   chan struct{}
	stopLogs   context.CancelFunc
	stopKiller func() bool

	// killMu serializes kills; killed records a successful one.
	killMu sync.Mutex
	killed bool
}

// newProcess starts copying the command's logs and kills the command when
// the spawn context ends.
func newProcess(ctx context.Context, setup processSetup) *process {
	logsCtx, stopLogs := context.WithCancel(context.WithoutCancel(ctx))

	started := &process{
		setup:    setup,
		spawn:    ctx,
		stdout:   &sandbox.OutputStream{},
		stderr:   &sandbox.OutputStream{},
		logsDone: make(chan struct{}),
		stopLogs: stopLogs,
	}

	go started.copyLogs(logsCtx)

	started.stopKiller = context.AfterFunc(ctx, func() {
		killCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), killTimeout)
		defer cancel()

		_ = started.Kill(killCtx)
		stopLogs()
	})

	return started
}

// copyLogs copies the log stream into the output streams.
func (p *process) copyLogs(ctx context.Context) {
	defer close(p.logsDone)
	defer p.stopLogs()

	stream, err := p.setup.client.commandLogs(ctx, p.setup.sessionID, p.setup.commandID)
	if err != nil {
		p.closeOutput(err)

		return
	}

	defer func() {
		_ = stream.Close()
	}()

	_, err = readLogLines(stream, func(line logLine) {
		target := p.stderr
		if line.Stream == "stdout" {
			target = p.stdout
		}

		_, _ = io.WriteString(target, logText(line))
	}, nil)

	p.closeOutput(err)
}

// closeOutput ends both output streams.
func (p *process) closeOutput(err error) {
	if errors.Is(err, context.Canceled) {
		err = nil
	}

	_ = p.stdout.CloseWithError(err)
	_ = p.stderr.CloseWithError(err)
}

// PID reports that Vercel Sandbox does not expose process IDs.
func (p *process) PID() (int, bool) {
	return 0, false
}

// Stdout returns the command's standard output.
func (p *process) Stdout() io.Reader {
	return p.stdout
}

// Stderr returns the command's standard error.
func (p *process) Stderr() io.Reader {
	return p.stderr
}

// Wait waits for the command to finish and its output to be copied. A
// command killed because its spawn context ended reports the context's
// cause.
func (p *process) Wait(ctx context.Context) (int, error) {
	finished, err := p.setup.client.waitCommand(ctx, p.setup.sessionID, p.setup.commandID)

	select {
	case <-p.logsDone:
	case <-ctx.Done():
		return 0, context.Cause(ctx)
	}

	p.stopKiller()

	if p.spawn.Err() != nil {
		return 0, context.Cause(p.spawn)
	}

	if err != nil {
		return 0, fmt.Errorf("wait for vercel sandbox command: %w", err)
	}

	if finished.ExitCode == nil {
		return 0, errNoCommand
	}

	return *finished.ExitCode, nil
}

// Kill sends SIGTERM to the command. Once a kill succeeded, later calls do
// nothing; a failed kill can be retried.
func (p *process) Kill(ctx context.Context) error {
	p.killMu.Lock()
	defer p.killMu.Unlock()

	if p.killed {
		return nil
	}

	if err := p.setup.client.killCommand(ctx, p.setup.sessionID, p.setup.commandID); err != nil {
		return fmt.Errorf("kill vercel sandbox command: %w", err)
	}

	p.killed = true

	return nil
}
