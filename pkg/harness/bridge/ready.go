package bridge

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"path"
	"strings"
	"sync"
	"time"

	"github.com/aarock1234/ai/pkg/sandbox"
)

// Readiness defaults.
const (
	defaultReadyPollInterval = 100 * time.Millisecond
	readyTailLimit           = 20
)

// ReadySource is where the bridge's readiness was observed.
type ReadySource string

// Ready sources.
const (
	// ReadyFromStdout is the bridge-ready line on standard output.
	ReadyFromStdout ReadySource = "stdout"

	// ReadyFromMetadata is the bridge's metadata file, which a bridge that
	// already started (for example after a reconnect) keeps current.
	ReadyFromMetadata ReadySource = "metadata"
)

// metadataFileName is the bridge's metadata file in its state directory.
const metadataFileName = "bridge-meta.json"

// MarkStartingOptions configure MarkStarting.
type MarkStartingOptions struct {
	Sandbox    sandbox.Session
	StateDir   string
	BridgeType string
}

// MarkStarting writes a starting marker to the bridge's metadata file, so
// that WaitForReady does not mistake the metadata of a previous bridge for
// readiness. The write is best-effort; standard output stays the primary
// readiness signal.
func MarkStarting(ctx context.Context, options MarkStartingOptions) {
	content, err := json.Marshal(map[string]string{
		"type":  options.BridgeType,
		"state": "starting",
	})
	if err != nil {
		return
	}

	_ = options.Sandbox.WriteTextFile(ctx, sandbox.WriteTextFileOptions{
		Path:    path.Join(options.StateDir, metadataFileName),
		Content: string(content),
	})
}

// ReadyOptions configure WaitForReady.
type ReadyOptions struct {
	// Process is the spawned bridge process.
	Process sandbox.Process

	// Readers owns the goroutine that reads the process's standard output.
	// It is required. The goroutine ends when the output ends, so wait for
	// Readers after the process exited or was killed.
	Readers *sync.WaitGroup

	Sandbox    sandbox.Session
	StateDir   string
	BridgeType string

	// Timeout bounds the wait. It is required.
	Timeout time.Duration

	// PollInterval is the metadata polling interval. Zero selects 100
	// milliseconds.
	PollInterval time.Duration
}

// ReadyResult is the outcome of WaitForReady.
type ReadyResult struct {
	// Port is the bridge's WebSocket port.
	Port   int
	Source ReadySource

	// StdoutTail holds the last lines of standard output.
	StdoutTail []string

	// Stdout yields the process output after the lines WaitForReady read.
	// Read it, for example with ForwardProcessStream, or ignore it; it ends
	// when the process exits.
	Stdout io.Reader
}

// ReadyError reports a bridge that did not become ready, with the last
// lines of its output.
type ReadyError struct {
	// Exited reports that the process exited; otherwise the wait timed out.
	Exited     bool
	StdoutTail []string
}

// Error describes the failure.
func (e *ReadyError) Error() string {
	if e.Exited {
		return "bridge exited before becoming ready"
	}

	return "bridge did not become ready in time"
}

// WaitForReady waits until the bridge announces its port on standard output
// or in its metadata file. On timeout or cancellation it kills the process.
// The process's standard output is consumed from a goroutine of
// ReadyOptions.Readers that runs until the output ends; the rest of the
// output is available as ReadyResult.Stdout.
func WaitForReady(ctx context.Context, options ReadyOptions) (ReadyResult, error) {
	if options.Timeout <= 0 {
		return ReadyResult{}, errors.New("wait for bridge ready: timeout must be positive")
	}

	if options.Readers == nil {
		return ReadyResult{}, errors.New("wait for bridge ready: readers are required")
	}

	pollInterval := options.PollInterval
	if pollInterval <= 0 {
		pollInterval = defaultReadyPollInterval
	}

	finished := make(chan struct{})
	defer close(finished)

	lines, remaining := scanLines(scanSetup{
		reader:   options.Process.Stdout(),
		finished: finished,
		readers:  options.Readers,
	})
	waiter := readyWaiter{
		options:   options,
		lines:     lines,
		remaining: remaining,
	}

	return waiter.wait(ctx, pollInterval)
}

// readyWaiter tracks one readiness wait.
type readyWaiter struct {
	options   ReadyOptions
	lines     <-chan string
	remaining *sandbox.OutputStream
	tail      []string
}

// wait races stdout lines, metadata polls, the deadline and ctx.
func (w *readyWaiter) wait(ctx context.Context, pollInterval time.Duration) (ReadyResult, error) {
	deadline := time.NewTimer(w.options.Timeout)
	defer deadline.Stop()

	poll := time.NewTicker(pollInterval)
	defer poll.Stop()

	for {
		select {
		case <-ctx.Done():
			_ = w.options.Process.Kill(context.WithoutCancel(ctx))

			return ReadyResult{}, context.Cause(ctx)
		case <-deadline.C:
			_ = w.options.Process.Kill(context.WithoutCancel(ctx))

			return ReadyResult{}, &ReadyError{
				StdoutTail: w.tail,
			}
		case line, ok := <-w.lines:
			if !ok {
				return ReadyResult{}, &ReadyError{
					Exited:     true,
					StdoutTail: w.tail,
				}
			}

			if result, ready := w.readyLine(line); ready {
				return result, nil
			}
		case <-poll.C:
			if port, ok := readMetadataPort(ctx, w.options); ok {
				return w.result(port, ReadyFromMetadata), nil
			}
		}
	}
}

// readyLine records a stdout line and reports a bridge-ready line.
func (w *readyWaiter) readyLine(line string) (ReadyResult, bool) {
	w.tail = append(w.tail, line)
	if len(w.tail) > readyTailLimit {
		w.tail = w.tail[1:]
	}

	ready, ok := ParseReady(line)
	if !ok {
		return ReadyResult{}, false
	}

	return w.result(ready.Port, ReadyFromStdout), true
}

// result returns a successful result.
func (w *readyWaiter) result(port int, source ReadySource) ReadyResult {
	return ReadyResult{
		Port:       port,
		Source:     source,
		StdoutTail: append([]string(nil), w.tail...),
		Stdout:     w.remaining,
	}
}

// scanSetup configures scanLines.
type scanSetup struct {
	reader   io.Reader
	finished <-chan struct{}

	// readers owns the scanning goroutine.
	readers *sync.WaitGroup
}

// scanLines sends the trimmed, non-empty lines of the reader until finished
// is closed, then copies the rest of the output into the returned stream.
// The goroutine, owned by setup.readers, ends when the reader ends.
func scanLines(setup scanSetup) (<-chan string, *sandbox.OutputStream) {
	lines := make(chan string)
	remaining := &sandbox.OutputStream{}
	finished := setup.finished

	setup.readers.Go(func() {
		defer close(lines)

		buffered := bufio.NewReader(setup.reader)

		for {
			raw, err := buffered.ReadString('\n')
			line := trimmedLine(strings.TrimSuffix(raw, "\n"))

			if line != "" {
				select {
				case lines <- line:
				case <-finished:
					_, _ = io.WriteString(remaining, raw)
					_, copyErr := io.Copy(remaining, buffered)
					_ = remaining.CloseWithError(copyErr)

					return
				}
			}

			if err != nil {
				_ = remaining.CloseWithError(ignoreEOF(err))

				return
			}
		}
	})

	return lines, remaining
}

// readMetadataPort returns the port of a waiting bridge of the expected
// type from its metadata file.
func readMetadataPort(ctx context.Context, options ReadyOptions) (int, bool) {
	content, err := options.Sandbox.ReadTextFile(ctx, sandbox.ReadTextFileOptions{
		Path: path.Join(options.StateDir, metadataFileName),
	})
	if err != nil {
		return 0, false
	}

	var metadata struct {
		Type  string `json:"type"`
		Port  int    `json:"port"`
		State string `json:"state"`
	}

	if err := json.Unmarshal([]byte(content), &metadata); err != nil {
		return 0, false
	}

	if metadata.Type != options.BridgeType || metadata.State != "waiting" || metadata.Port <= 0 {
		return 0, false
	}

	return metadata.Port, true
}
