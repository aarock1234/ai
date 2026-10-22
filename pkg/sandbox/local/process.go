package local

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"sync"
	"sync/atomic"
	"time"

	"github.com/aarock1234/ai/pkg/sandbox"
)

// processWaitDelay bounds the wait for output pipes that a killed process's
// descendants still hold open.
const processWaitDelay = 5 * time.Second

// Spawn starts a command with the shell. Ctx governs the process: cancelling
// it kills the process and its process group.
func (c *core) Spawn(ctx context.Context, options sandbox.ProcessOptions) (sandbox.Process, error) {
	return c.start(ctx, options)
}

// Run runs a command to completion and collects its output.
func (c *core) Run(ctx context.Context, options sandbox.ProcessOptions) (sandbox.RunResult, error) {
	started, err := c.start(ctx, options)
	if err != nil {
		return sandbox.RunResult{}, err
	}

	var stdout []byte
	var stderr []byte
	var readers sync.WaitGroup

	readers.Go(func() {
		stdout, _ = io.ReadAll(started.Stdout())
	})

	readers.Go(func() {
		stderr, _ = io.ReadAll(started.Stderr())
	})

	exitCode, waitErr := started.Wait(context.WithoutCancel(ctx))
	readers.Wait()

	if waitErr != nil {
		return sandbox.RunResult{}, waitErr
	}

	return sandbox.RunResult{
		ExitCode: exitCode,
		Stdout:   string(stdout),
		Stderr:   string(stderr),
	}, nil
}

// start starts a command and tracks it until it exits.
func (c *core) start(ctx context.Context, options sandbox.ProcessOptions) (*process, error) {
	if err := ctx.Err(); err != nil {
		return nil, context.Cause(ctx)
	}

	command := exec.CommandContext(ctx, c.shell, "-c", options.Command)
	command.Dir = c.directory
	if options.WorkingDirectory != "" {
		command.Dir = c.resolve(options.WorkingDirectory)
	}

	command.Env = c.environment(options.Env)
	command.WaitDelay = processWaitDelay
	configureProcessGroup(command)

	started := &process{
		ctx:     ctx,
		command: command,
		stdout:  &sandbox.OutputStream{},
		stderr:  &sandbox.OutputStream{},
		exited:  make(chan struct{}),
	}

	command.Stdout = started.stdout
	command.Stderr = started.stderr
	command.Cancel = started.killOnCancel

	if err := command.Start(); err != nil {
		return nil, fmt.Errorf("start %s: %w", c.shell, err)
	}

	c.track(started)

	go func() {
		defer c.untrack(started)

		started.wait()
	}()

	return started, nil
}

// track records a running process.
func (c *core) track(running *process) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.processes[running] = struct{}{}
}

// untrack forgets a process that exited.
func (c *core) untrack(running *process) {
	c.mu.Lock()
	defer c.mu.Unlock()

	delete(c.processes, running)
}

// killAll kills every running process and waits for them to exit.
func (c *core) killAll(ctx context.Context) error {
	c.mu.Lock()
	running := make([]*process, 0, len(c.processes))
	for tracked := range c.processes {
		running = append(running, tracked)
	}
	c.mu.Unlock()

	var errs []error

	for _, tracked := range running {
		if err := tracked.Kill(ctx); err != nil {
			errs = append(errs, err)
		}
	}

	for _, tracked := range running {
		select {
		case <-tracked.exited:
		case <-ctx.Done():
			errs = append(errs, context.Cause(ctx))

			return errors.Join(errs...)
		}
	}

	return errors.Join(errs...)
}

// process is a command started with Spawn. A goroutine started with it
// waits for the command and closes exited.
type process struct {
	ctx     context.Context
	command *exec.Cmd
	stdout  *sandbox.OutputStream
	stderr  *sandbox.OutputStream

	exited   chan struct{}
	exitCode int
	waitErr  error

	// cancelled records that the context ended and killed the process.
	cancelled atomic.Bool

	// mu guards reaped and killed, so that a kill never signals the process
	// group of a command that was already reaped, whose ID the system may
	// have reused.
	mu     sync.Mutex
	reaped bool
	killed bool
}

// wait waits for the command, records its outcome and ends the output
// streams.
func (p *process) wait() {
	err := p.command.Wait()
	p.markReaped()

	if p.command.ProcessState != nil {
		p.exitCode = p.command.ProcessState.ExitCode()
	}

	var exitErr *exec.ExitError
	switch {
	case p.cancelled.Load():
		p.waitErr = context.Cause(p.ctx)
	case err != nil && !errors.As(err, &exitErr):
		p.waitErr = err
	}

	_ = p.stdout.Close()
	_ = p.stderr.Close()
	close(p.exited)
}

// markReaped records that the command was reaped.
func (p *process) markReaped() {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.reaped = true
}

// killOnCancel is the command's Cancel function: it kills the process and
// records that the context killed it.
func (p *process) killOnCancel() error {
	killedNow, err := p.kill()
	if killedNow {
		p.cancelled.Store(true)
	}

	return err
}

// kill kills the process and its process group unless it was reaped or
// already killed, and reports whether this call killed it. Only a
// successful kill is recorded, so a failed one can be retried.
func (p *process) kill() (bool, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.reaped || p.killed {
		return false, nil
	}

	if err := killProcessGroup(p.command); err != nil {
		return false, err
	}

	p.killed = true

	return true, nil
}

// PID returns the process ID.
func (p *process) PID() (int, bool) {
	return p.command.Process.Pid, true
}

// Stdout returns the standard output of the process.
func (p *process) Stdout() io.Reader {
	return p.stdout
}

// Stderr returns the standard error of the process.
func (p *process) Stderr() io.Reader {
	return p.stderr
}

// Wait waits for the process to exit and returns its exit code. A process
// killed because its context ended reports the context's cause.
func (p *process) Wait(ctx context.Context) (int, error) {
	select {
	case <-p.exited:
		return p.exitCode, p.waitErr
	case <-ctx.Done():
		return 0, context.Cause(ctx)
	}
}

// Kill kills the process and its process group. Once a kill succeeded or
// the process exited, later calls do nothing.
func (p *process) Kill(context.Context) error {
	_, err := p.kill()

	return err
}
