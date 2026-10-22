package bridge

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/aarock1234/ai/pkg/sandbox"
)

// Diagnostic defaults.
const (
	defaultTailLimit    = 20
	startupGracePeriod  = 250 * time.Millisecond
	processStreamStdout = "stdout"
	processStreamStderr = "stderr"
)

// Tail keeps the last lines of a process stream. It is safe for concurrent
// use; the zero value keeps 20 lines.
type Tail struct {
	// Limit is the number of lines kept. Zero keeps 20.
	Limit int

	mu    sync.Mutex
	lines []string
}

// Add records a line.
func (t *Tail) Add(line string) {
	t.mu.Lock()
	defer t.mu.Unlock()

	limit := t.Limit
	if limit <= 0 {
		limit = defaultTailLimit
	}

	t.lines = append(t.lines, line)
	if len(t.lines) > limit {
		t.lines = t.lines[len(t.lines)-limit:]
	}
}

// Lines returns a copy of the kept lines.
func (t *Tail) Lines() []string {
	t.mu.Lock()
	defer t.mu.Unlock()

	return append([]string(nil), t.lines...)
}

// LogErrorOptions configure LogError.
type LogErrorOptions struct {
	HarnessID string
	SessionID string

	// Message describes what failed, in lowercase.
	Message string

	Err error

	// Logger receives the record. Nil uses slog.Default.
	Logger *slog.Logger
}

// LogError logs a bridge failure with the harness and session.
func LogError(ctx context.Context, options LogErrorOptions) {
	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}

	attributes := []any{
		slog.String("harness", options.HarnessID),
		slog.Any("error", options.Err),
	}

	if options.SessionID != "" {
		attributes = append(attributes, slog.String("session", options.SessionID))
	}

	logger.ErrorContext(ctx, options.Message, attributes...)
}

// StartupErrorOptions configure StartupError.
type StartupErrorOptions struct {
	// Message describes the failure.
	Message string

	Process sandbox.Process

	StdoutTail []string
	StderrTail []string
}

// StartupError builds a bridge startup failure with the process's exit
// code, when it exits within a short grace period, and its output tails.
func StartupError(ctx context.Context, options StartupErrorOptions) error {
	message := options.Message

	waitCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), startupGracePeriod)
	defer cancel()

	if exitCode, err := options.Process.Wait(waitCtx); err == nil {
		message += fmt.Sprintf(" Exit code: %d.", exitCode)
	}

	var details []string

	if len(options.StdoutTail) > 0 {
		details = append(details, "stdout:\n"+strings.Join(options.StdoutTail, "\n"))
	}

	if len(options.StderrTail) > 0 {
		details = append(details, "stderr:\n"+strings.Join(options.StderrTail, "\n"))
	}

	if len(details) > 0 {
		message += "\n\n" + strings.Join(details, "\n\n")
	}

	return errors.New(message)
}

// ForwardOptions configure ForwardProcessStream.
type ForwardOptions struct {
	// StreamName is "stdout" or "stderr".
	StreamName string

	// Source labels the lines. Empty selects "bridge".
	Source string

	// Tail records the forwarded lines. Nil records none.
	Tail *Tail

	// Writer receives the lines. Nil writes to standard error.
	Writer io.Writer
}

// ForwardProcessStream copies a bridge process stream to standard error,
// one prefixed line at a time, until the stream ends.
func ForwardProcessStream(reader io.Reader, options ForwardOptions) {
	source := options.Source
	if source == "" {
		source = "bridge"
	}

	streamName := options.StreamName
	if streamName != processStreamStderr {
		streamName = processStreamStdout
	}

	writer := options.Writer
	if writer == nil {
		writer = os.Stderr
	}

	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)

	for scanner.Scan() {
		line := trimmedLine(scanner.Text())
		if line == "" {
			continue
		}

		if options.Tail != nil {
			options.Tail.Add(line)
		}

		_, _ = fmt.Fprintf(writer, "[harness:%s:%s] %s\n", source, streamName, line)
	}

	// A read error, such as a line longer than the buffer, stops the
	// scanner. The rest of the stream is drained so the process never
	// blocks on a full pipe.
	if err := scanner.Err(); err != nil {
		_, _ = fmt.Fprintf(writer, "[harness:%s:%s] stopped forwarding: %v\n", source, streamName, err)

		DrainProcessStream(reader)
	}
}

// DrainProcessStream reads a process stream until it ends, discarding it.
func DrainProcessStream(reader io.Reader) {
	_, _ = io.Copy(io.Discard, reader)
}

// ignoreEOF returns nil for io.EOF.
func ignoreEOF(err error) error {
	if errors.Is(err, io.EOF) {
		return nil
	}

	return err
}
