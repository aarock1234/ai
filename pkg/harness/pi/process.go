package pi

import (
	"context"
	"fmt"
	"path"
	"strings"

	"github.com/aarock1234/ai/internal/shellquote"
	"github.com/aarock1234/ai/pkg/harness/internal/bridgeadapter"
	"github.com/aarock1234/ai/pkg/sandbox"
)

// stderrTailLines is the number of Pi standard error lines startup and exit
// errors include.
const stderrTailLines = 40

// relayStatus is the state of a relay launch.
type relayStatus string

// Relay states.
const (
	relayRunning relayStatus = "running"
	relayExited  relayStatus = "exited"
	relayMissing relayStatus = "missing"
)

// relayOptions configure spawnRelay.
type relayOptions struct {
	sandbox  sandbox.Session
	layout   layout
	launchID string
	args     []string
	env      map[string]string
}

// spawnRelay starts the relay, which starts Pi, in the session working
// directory. The relay outlives ctx and the host process; it ends when Pi
// exits.
func spawnRelay(ctx context.Context, options relayOptions) (sandbox.Process, error) {
	stateDir := options.layout.relayStateDir(options.launchID)

	err := bridgeadapter.RunChecked(ctx, options.sandbox, bridgeadapter.CheckedCommand{
		Command: "mkdir -p -- " + shellquote.Quote(path.Join(stateDir, "inbox")),
		Failure: "create pi relay directory",
	})
	if err != nil {
		return nil, err
	}

	words := []string{
		"exec node",
		shellquote.Quote(options.layout.installFile(relayScript)),
		shellquote.Quote(stateDir),
		"node",
		shellquote.Quote(options.layout.installFile(piCLI)),
	}

	for _, argument := range options.args {
		words = append(words, shellquote.Quote(argument))
	}

	command := strings.Join(words, " ") + " >/dev/null 2>>" + shellquote.Quote(path.Join(stateDir, "relay.log"))

	process, err := options.sandbox.Spawn(context.WithoutCancel(ctx), sandbox.ProcessOptions{
		Command:          command,
		WorkingDirectory: options.layout.workDir,
		Env:              options.env,
	})
	if err != nil {
		return nil, fmt.Errorf("spawn pi relay: %w", err)
	}

	return process, nil
}

// relayState reports whether a relay launch is still running.
func relayState(ctx context.Context, session sandbox.Session, stateDir string) (relayStatus, error) {
	command := fmt.Sprintf(
		`if [ -e %[1]s/exit.json ]; then echo exited; elif [ -f %[1]s/relay.pid ] && kill -0 "$(cat %[1]s/relay.pid)" 2>/dev/null; then echo running; else echo missing; fi`,
		shellquote.Quote(stateDir),
	)

	result, err := session.Run(ctx, sandbox.ProcessOptions{
		Command: command,
	})
	if err != nil {
		return "", fmt.Errorf("check pi relay: %w", err)
	}

	switch status := relayStatus(strings.TrimSpace(result.Stdout)); status {
	case relayRunning, relayExited:
		return status, nil
	default:
		return relayMissing, nil
	}
}

// signalRelay sends SIGTERM to a running relay, which forwards it to Pi.
func signalRelay(ctx context.Context, session sandbox.Session, stateDir string) {
	command := fmt.Sprintf(
		`[ -f %[1]s/relay.pid ] && kill -TERM "$(cat %[1]s/relay.pid)" 2>/dev/null; true`,
		shellquote.Quote(stateDir),
	)

	_, _ = session.Run(ctx, sandbox.ProcessOptions{
		Command: command,
	})
}

// removeRelayState removes the state directory of an ended launch.
func removeRelayState(ctx context.Context, session sandbox.Session, stateDir string) {
	_, _ = session.Run(ctx, sandbox.ProcessOptions{
		Command: "rm -rf -- " + shellquote.Quote(stateDir),
	})
}

// stderrTail returns the last lines Pi and the relay wrote to standard
// error.
func stderrTail(ctx context.Context, session sandbox.Session, stateDir string) string {
	command := fmt.Sprintf(
		"tail -n %d %s %s 2>/dev/null",
		stderrTailLines,
		shellquote.Quote(path.Join(stateDir, "relay.log")),
		shellquote.Quote(path.Join(stateDir, "stderr.log")),
	)

	result, err := session.Run(context.WithoutCancel(ctx), sandbox.ProcessOptions{
		Command: command,
	})
	if err != nil {
		return ""
	}

	return strings.TrimSpace(result.Stdout)
}

// withStderr appends the standard error tail to an error.
func withStderr(err error, tail string) error {
	if tail == "" {
		return err
	}

	return fmt.Errorf("%w\n\nstderr:\n%s", err, tail)
}
