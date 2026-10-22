package sandboxshell

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"

	"github.com/aarock1234/ai/pkg/sandbox"
)

// errProcessKilled is the cancellation cause of Process.Kill.
var errProcessKilled = errors.New("sandboxshell: process killed")

// Spawn starts a command in a fresh shell and returns immediately. Ctx
// governs the process: cancelling it stops the script, and Wait then
// reports ctx's cause.
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

// start starts a script and tracks it until it exits.
func (c *core) start(ctx context.Context, options sandbox.ProcessOptions) (*process, error) {
	if err := c.checkUsable(ctx); err != nil {
		return nil, err
	}

	directory, err := c.processDirectory(options.WorkingDirectory)
	if err != nil {
		return nil, err
	}

	runCtx, cancel := context.WithCancelCause(ctx)

	started := &process{
		parent: ctx,
		runCtx: runCtx,
		cancel: cancel,
		stdout: &sandbox.OutputStream{},
		stderr: &sandbox.OutputStream{},
		exited: make(chan struct{}),
	}

	if err := c.track(started); err != nil {
		cancel(err)

		return nil, err
	}

	run := script{
		name:   shellName,
		source: options.Command,
		dir:    directory,
		env:    newShellEnviron(mergeVariables(defaultVariables(directory), c.env, options.Env)),
		stdout: started.stdout,
		stderr: started.stderr,
	}

	go func() {
		defer c.untrack(started)

		started.run(c.shell, run)
	}()

	return started, nil
}

// processDirectory resolves and checks the working directory of a command.
func (c *core) processDirectory(workingDirectory string) (string, error) {
	directory := c.workingDirectory
	if workingDirectory != "" {
		directory = c.resolve(workingDirectory)
	}

	info, err := c.shell.fs.Stat(directory)
	if err != nil {
		return "", fmt.Errorf("working directory: %w", err)
	}

	if !info.IsDir() {
		return "", fmt.Errorf("working directory: %w", pathError("chdir", directory, errNotDirectory))
	}

	return directory, nil
}

// track records a running process, unless the session was destroyed.
func (c *core) track(running *process) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.destroyed {
		return ErrDestroyed
	}

	c.processes[running] = struct{}{}

	return nil
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

	for _, tracked := range running {
		_ = tracked.Kill(ctx)
	}

	for _, tracked := range running {
		select {
		case <-tracked.exited:
		case <-ctx.Done():
			return context.Cause(ctx)
		}
	}

	return nil
}

// process is a script started with Spawn. The goroutine that runs the
// script records its outcome and closes exited.
type process struct {
	parent context.Context
	runCtx context.Context
	cancel context.CancelCauseFunc

	stdout *sandbox.OutputStream
	stderr *sandbox.OutputStream

	exited   chan struct{}
	exitCode int
	waitErr  error
}

// run runs the script, records how it ended and ends the output streams.
// Background jobs of the script stop when it returns.
func (p *process) run(sessionShell *shell, run script) {
	code := sessionShell.run(p.runCtx, run)

	switch {
	case p.runCtx.Err() == nil:
		p.exitCode = code
	case errors.Is(context.Cause(p.runCtx), errProcessKilled):
		p.exitCode = exitKilledBySignal
	default:
		p.waitErr = context.Cause(p.parent)
	}

	p.cancel(nil)

	_ = p.stdout.Close()
	_ = p.stderr.Close()
	close(p.exited)
}

// PID reports that sandbox processes have no process ID.
func (p *process) PID() (int, bool) {
	return 0, false
}

// Stdout returns the standard output of the process.
func (p *process) Stdout() io.Reader {
	return p.stdout
}

// Stderr returns the standard error of the process.
func (p *process) Stderr() io.Reader {
	return p.stderr
}

// Wait waits for the process to exit and returns its exit code: 137 after
// Kill. A process stopped because its context ended reports the context's
// cause.
func (p *process) Wait(ctx context.Context) (int, error) {
	select {
	case <-p.exited:
		return p.exitCode, p.waitErr
	case <-ctx.Done():
		return 0, context.Cause(ctx)
	}
}

// Kill stops the process at its next command or cancellation check. It is
// idempotent and does nothing after the process exited.
func (p *process) Kill(context.Context) error {
	p.cancel(errProcessKilled)

	return nil
}
