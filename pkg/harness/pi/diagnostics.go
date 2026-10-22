package pi

import (
	"bufio"
	"context"
	"log/slog"
	"path"
	"strings"
	"sync"
	"time"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/bridge"
	"github.com/aarock1234/ai/pkg/sandbox"
)

// diagnosticsSource labels Pi's captured standard error lines.
const diagnosticsSource = "pi"

// diagnosticsFollower forwards Pi's standard error, which the relay appends
// to its log, to the observability reporter. It owns the goroutine that
// reads the follower process; stop ends it and waits for it.
type diagnosticsFollower struct {
	process sandbox.Process
	cancel  context.CancelFunc
	done    chan struct{}
	once    sync.Once
}

// followDiagnostics starts forwarding the launch's standard error lines
// from offset zero, or returns nil when diagnostics are disabled or the
// follower cannot start.
func followDiagnostics(ctx context.Context, owner *session, observability *harness.Observability) *diagnosticsFollower {
	if observability == nil || observability.Report == nil {
		return nil
	}

	stateDir := owner.conn.stateDir()
	followerCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))

	process, err := owner.sandbox.Spawn(followerCtx, sandbox.ProcessOptions{
		Command: followCommand(followOptions{
			layout:   owner.layout,
			file:     path.Join(stateDir, "stderr.log"),
			exitFile: path.Join(stateDir, "exit.json"),
		}),
	})
	if err != nil {
		cancel()
		slog.Default().WarnContext(ctx, "pi diagnostics unavailable", slog.String("session", owner.sessionID), slog.Any("error", err))

		return nil
	}

	follower := &diagnosticsFollower{
		process: process,
		cancel:  cancel,
		done:    make(chan struct{}),
	}

	go func() {
		defer close(follower.done)

		scanner := bufio.NewScanner(process.Stdout())
		scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)

		for scanner.Scan() {
			line := strings.TrimRight(scanner.Text(), "\r")
			if strings.TrimSpace(line) == "" {
				continue
			}

			diagnostic, ok := bridge.DiagnosticFromFrame(bridge.SandboxLog{
				Source: diagnosticsSource,
				Stream: "stderr",
				Line:   line,
			}, owner.sessionID, time.Now())
			if ok {
				observability.Report(diagnostic)
			}
		}

		// A read error, such as a line longer than the buffer, stops the
		// scanner. The rest of the stream is drained so the follower
		// process never blocks on a full pipe.
		if err := scanner.Err(); err != nil {
			slog.Default().WarnContext(followerCtx, "pi diagnostics stopped", slog.String("session", owner.sessionID), slog.Any("error", err))

			bridge.DrainProcessStream(process.Stdout())
		}
	}()

	return follower
}

// stop kills the follower and waits for its goroutine. It does nothing for a
// nil follower.
func (f *diagnosticsFollower) stop() {
	if f == nil {
		return
	}

	f.once.Do(func() {
		_ = f.process.Kill(context.Background())
		<-f.done
		f.cancel()
	})
}

// reportExtensionError reports an extension handler failure as a structured
// diagnostic, and logs it when diagnostics are disabled.
func (s *session) reportExtensionError(event extensionErrorEvent) {
	observability := s.observability
	if observability == nil || observability.Report == nil {
		slog.Default().Warn(
			"pi extension reported an error",
			slog.String("session", s.sessionID),
			slog.String("extension", event.ExtensionPath),
			slog.String("event", event.Event),
			slog.String("error", event.Error),
		)

		return
	}

	observability.Report(harness.Diagnostic{
		Level:     harness.DebugWarn,
		Message:   "pi extension reported an error",
		Subsystem: "pi.extension",
		Kind:      harness.DiagnosticEvent,
		Attrs: map[string]any{
			"extensionPath": event.ExtensionPath,
			"event":         event.Event,
		},
		Error: &harness.DiagnosticError{
			Message: event.Error,
		},
		SessionID: s.sessionID,
		Timestamp: time.Now(),
	})
}
