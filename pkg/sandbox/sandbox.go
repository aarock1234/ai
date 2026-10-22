// Package sandbox defines the sandbox session contract shared by tools,
// harnesses and sandbox implementations: an environment that runs commands
// and reads and writes files. It is the Go port of the upstream
// Experimental_SandboxSession and Experimental_SandboxProcess types from
// provider-utils.
//
// Go adaptations: cancellation comes from context.Context instead of
// abortSignal options; files that do not exist are reported with an error
// wrapping fs.ErrNotExist instead of a null result; byte streams are
// io.Reader values instead of web streams; and zero values mean unset for
// optional numbers. Implementations live in subpackages such as
// sandbox/local and sandbox/vercel.
package sandbox

import (
	"context"
	"io"
)

// Session is a sandbox that can execute commands and read and write files.
// Implementations must be safe for concurrent use.
type Session interface {
	// Description describes the sandbox environment for the agent's
	// instructions, for example its root directory, exposed ports or public
	// hostname.
	Description() string

	// ReadFile opens one file as a stream of bytes. The caller must close the
	// returned reader. A missing file is reported with an error wrapping
	// fs.ErrNotExist. Relative path handling is implementation-defined.
	ReadFile(ctx context.Context, options ReadFileOptions) (io.ReadCloser, error)

	// ReadBinaryFile reads one file as raw bytes. A missing file is reported
	// with an error wrapping fs.ErrNotExist.
	ReadBinaryFile(ctx context.Context, options ReadFileOptions) ([]byte, error)

	// ReadTextFile reads one text file, decoded with the requested encoding
	// and optionally limited to a line range. A missing file is reported with
	// an error wrapping fs.ErrNotExist.
	ReadTextFile(ctx context.Context, options ReadTextFileOptions) (string, error)

	// WriteFile writes one file from a stream of bytes, creating parent
	// directories and overwriting an existing file.
	WriteFile(ctx context.Context, options WriteFileOptions) error

	// WriteBinaryFile writes one file from raw bytes, creating parent
	// directories and overwriting an existing file.
	WriteBinaryFile(ctx context.Context, options WriteBinaryFileOptions) error

	// WriteTextFile writes one file from a string encoded with the requested
	// encoding, creating parent directories and overwriting an existing file.
	WriteTextFile(ctx context.Context, options WriteTextFileOptions) error

	// Spawn starts a long-running process and returns immediately with a
	// handle that streams its output. Ctx governs the whole process:
	// cancelling it kills the process, and Wait then reports ctx's cause.
	Spawn(ctx context.Context, options ProcessOptions) (Process, error)

	// Run runs a command to completion and collects its output. A non-zero
	// exit code is not an error; it is reported in the result.
	Run(ctx context.Context, options ProcessOptions) (RunResult, error)
}

// Process is a process started with Session.Spawn.
//
// Stdout and Stderr buffer the process output without bounds, so reading
// one stream never blocks the other. Read both until they end, or ignore
// them; they end when the process exits.
type Process interface {
	// PID returns the process identifier, and false when the sandbox does not
	// expose one.
	PID() (int, bool)

	// Stdout returns the bytes the process writes to standard output.
	Stdout() io.Reader

	// Stderr returns the bytes the process writes to standard error.
	Stderr() io.Reader

	// Wait waits for the process to exit and returns its exit code. Ctx only
	// bounds the wait; it does not kill the process.
	Wait(ctx context.Context) (int, error)

	// Kill terminates the process. It is idempotent.
	Kill(ctx context.Context) error
}

// ProcessOptions configure Session.Run and Session.Spawn.
type ProcessOptions struct {
	// Command is the shell command to execute.
	Command string

	// WorkingDirectory is the directory the command runs in. Empty uses the
	// sandbox's default working directory.
	WorkingDirectory string

	// Env is merged with the sandbox's default environment; its values take
	// precedence. Passing secrets here rather than in Command keeps them out
	// of logs.
	Env map[string]string
}

// RunResult is the outcome of Session.Run.
type RunResult struct {
	ExitCode int
	Stdout   string
	Stderr   string
}

// ReadFileOptions configure Session.ReadFile and Session.ReadBinaryFile.
type ReadFileOptions struct {
	Path string
}

// ReadTextFileOptions configure Session.ReadTextFile.
type ReadTextFileOptions struct {
	Path string

	// Encoding decodes the file bytes. Empty selects UTF-8. See DecodeText
	// for the supported names.
	Encoding string

	// StartLine is the 1-based inclusive first line. Zero starts at the first
	// line.
	StartLine int

	// EndLine is the 1-based inclusive last line. Zero or a line past the end
	// of the file reads through the end of the file.
	EndLine int
}

// WriteFileOptions configure Session.WriteFile.
type WriteFileOptions struct {
	Path string

	// Content is read until it ends. The session does not close it.
	Content io.Reader
}

// WriteBinaryFileOptions configure Session.WriteBinaryFile.
type WriteBinaryFileOptions struct {
	Path    string
	Content []byte
}

// WriteTextFileOptions configure Session.WriteTextFile.
type WriteTextFileOptions struct {
	Path    string
	Content string

	// Encoding encodes the content. Empty selects UTF-8. See EncodeText for
	// the supported names.
	Encoding string
}
