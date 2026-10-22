package bridgeadapter

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/aarock1234/ai/pkg/harness/bridge"
	"github.com/aarock1234/ai/pkg/sandbox"
)

// SpawnOptions configure SpawnBridge.
type SpawnOptions struct {
	Sandbox sandbox.Session
	Command string
	Env     map[string]string

	// StateDir is the bridge's state directory, which holds its metadata
	// file.
	StateDir string

	// BridgeType is the bridge type the metadata file must announce, and
	// the source of the forwarded standard error lines.
	BridgeType string

	// Name names the bridge in startup errors, such as "codex" or
	// "grok-build ACP".
	Name string

	// StartupTimeout bounds the wait for the bridge to become ready.
	StartupTimeout time.Duration
}

// BridgeProcess is a spawned bridge with the goroutines that read its
// output. The goroutines end when the process exits.
type BridgeProcess struct {
	process sandbox.Process
	port    int
	output  *processOutput

	// cancel releases the process context.
	cancel context.CancelFunc
}

// SpawnBridge starts a bridge process and waits until it listens. The
// process outlives ctx once the bridge is ready; cancelling ctx during the
// startup kills it. On failure the process is killed and its output readers
// have ended.
func SpawnBridge(ctx context.Context, options SpawnOptions) (*BridgeProcess, error) {
	processCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	stopWatching := context.AfterFunc(ctx, cancel)
	defer stopWatching()

	process, err := options.Sandbox.Spawn(processCtx, sandbox.ProcessOptions{
		Command: options.Command,
		Env:     options.Env,
	})
	if err != nil {
		cancel()

		return nil, fmt.Errorf("spawn %s bridge: %w", options.Name, err)
	}

	spawned := &BridgeProcess{
		process: process,
		output:  newProcessOutput(process, options.BridgeType),
		cancel:  cancel,
	}

	ready, err := bridge.WaitForReady(ctx, bridge.ReadyOptions{
		Process:    process,
		Readers:    &spawned.output.readers,
		Sandbox:    options.Sandbox,
		StateDir:   options.StateDir,
		BridgeType: options.BridgeType,
		Timeout:    options.StartupTimeout,
	})
	if err != nil {
		startupErr := spawned.startupError(ctx, options.Name, err)
		spawned.Kill(ctx)
		spawned.WaitOutput()

		return nil, startupErr
	}

	spawned.output.drain(ready.Stdout)
	spawned.port = ready.Port

	return spawned, nil
}

// Process returns the bridge process.
func (p *BridgeProcess) Process() sandbox.Process {
	return p.process
}

// Port returns the port the bridge listens on.
func (p *BridgeProcess) Port() int {
	return p.port
}

// Kill kills the bridge process without waiting for its output readers.
func (p *BridgeProcess) Kill(ctx context.Context) {
	_ = p.process.Kill(context.WithoutCancel(ctx))
	p.cancel()
}

// WaitOutput waits until the output readers ended, which happens when the
// process exited.
func (p *BridgeProcess) WaitOutput() {
	p.output.wait()
}

// startupError describes a bridge that did not become ready, with the last
// lines of its output.
func (p *BridgeProcess) startupError(ctx context.Context, name string, err error) error {
	readyErr, ok := errors.AsType[*bridge.ReadyError](err)
	if !ok {
		return err
	}

	message := fmt.Sprintf("%s bridge did not become ready in time.", name)
	if readyErr.Exited {
		message = fmt.Sprintf("%s bridge exited before becoming ready.", name)
	}

	return bridge.StartupError(ctx, bridge.StartupErrorOptions{
		Message:    message,
		Process:    p.process,
		StdoutTail: readyErr.StdoutTail,
		StderrTail: p.output.stderrTail.Lines(),
	})
}

// processOutput consumes the output streams of a spawned bridge.
type processOutput struct {
	stderrTail bridge.Tail
	readers    sync.WaitGroup
}

// newProcessOutput starts forwarding the process's standard error.
func newProcessOutput(process sandbox.Process, source string) *processOutput {
	output := &processOutput{}

	output.readers.Go(func() {
		bridge.ForwardProcessStream(process.Stderr(), bridge.ForwardOptions{
			StreamName: "stderr",
			Source:     source,
			Tail:       &output.stderrTail,
		})
	})

	return output
}

// drain discards the rest of the process's standard output.
func (o *processOutput) drain(stdout io.Reader) {
	o.readers.Go(func() {
		bridge.DrainProcessStream(stdout)
	})
}

// wait waits until the process output ended.
func (o *processOutput) wait() {
	o.readers.Wait()
}

// CheckedCommand is a sandbox command whose failure is reported.
type CheckedCommand struct {
	Command string

	// Failure describes the command in its errors, such as "create bridge
	// directories".
	Failure string
}

// RunChecked runs a sandbox command and reports a failed run or a non-zero
// exit with the command's standard error.
func RunChecked(ctx context.Context, session sandbox.Session, command CheckedCommand) error {
	result, err := session.Run(ctx, sandbox.ProcessOptions{
		Command: command.Command,
	})
	if err != nil {
		return fmt.Errorf("%s: %w", command.Failure, err)
	}

	if result.ExitCode == 0 {
		return nil
	}

	if result.Stderr == "" {
		return fmt.Errorf("%s (exit %d)", command.Failure, result.ExitCode)
	}

	return fmt.Errorf("%s (exit %d): %s", command.Failure, result.ExitCode, result.Stderr)
}

// AbandonChannel tells a bridge whose session could not start to exit, and
// closes the channel.
func AbandonChannel(channel *bridge.Channel) {
	channel.BeginClose()
	sendDestroy(channel)

	_ = channel.Close()
}

// sendDestroy asks the bridge to exit unless the channel already ended.
func sendDestroy(channel *bridge.Channel) {
	if channel.Closed() {
		return
	}

	_ = channel.Send(bridge.SimpleCommand{
		Type: "destroy",
	})
}
