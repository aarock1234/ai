package sandboxshell

import (
	"bytes"
	"context"
	"errors"
	"io"
	"maps"
	"path"
	"slices"
	"strings"
	"sync"

	"github.com/aarock1234/ai/pkg/sandbox"
)

// ErrDestroyed is returned by the operations of a destroyed session.
var ErrDestroyed = errors.New("sandboxshell: session destroyed")

// core implements sandbox.Session for one in-memory sandbox. It is shared
// by a Session and its restricted views.
type core struct {
	shell            *shell
	workingDirectory string
	env              map[string]string

	mu        sync.Mutex
	processes map[*process]struct{}
	destroyed bool
}

// newCore returns the shared state of a session.
func newCore(sessionShell *shell, workingDirectory string, env map[string]string) *core {
	return &core{
		shell:            sessionShell,
		workingDirectory: workingDirectory,
		env:              maps.Clone(env),
		processes:        make(map[*process]struct{}),
	}
}

// Description describes the sandbox for the agent's instructions.
func (c *core) Description() string {
	return strings.Join([]string{
		"In-memory bash environment with a virtual filesystem, interpreted in-process; only built-in commands are available and there is no network access.",
		"Filesystem changes and spawned processes share the same in-memory state.",
		"Current working directory: " + c.workingDirectory,
	}, "\n")
}

// ReadFile opens a file. Relative paths resolve against the working
// directory.
func (c *core) ReadFile(ctx context.Context, options sandbox.ReadFileOptions) (io.ReadCloser, error) {
	contents, err := c.readFile(ctx, options.Path)
	if err != nil {
		return nil, err
	}

	return io.NopCloser(bytes.NewReader(contents)), nil
}

// ReadBinaryFile reads a file.
func (c *core) ReadBinaryFile(ctx context.Context, options sandbox.ReadFileOptions) ([]byte, error) {
	contents, err := c.readFile(ctx, options.Path)
	if err != nil {
		return nil, err
	}

	return slices.Clone(contents), nil
}

// ReadTextFile reads and decodes a text file, optionally limited to a line
// range.
func (c *core) ReadTextFile(ctx context.Context, options sandbox.ReadTextFileOptions) (string, error) {
	contents, err := c.readFile(ctx, options.Path)
	if err != nil {
		return "", err
	}

	text, err := sandbox.DecodeText(contents, options.Encoding)
	if err != nil {
		return "", err
	}

	return sandbox.ExtractLines(text, options.StartLine, options.EndLine), nil
}

// WriteFile writes a file from a stream, creating parent directories.
func (c *core) WriteFile(ctx context.Context, options sandbox.WriteFileOptions) error {
	contents, err := io.ReadAll(contextReader{
		ctx:    ctx,
		reader: options.Content,
	})
	if err != nil {
		return err
	}

	return c.WriteBinaryFile(ctx, sandbox.WriteBinaryFileOptions{
		Path:    options.Path,
		Content: contents,
	})
}

// WriteBinaryFile writes a file, creating parent directories.
func (c *core) WriteBinaryFile(ctx context.Context, options sandbox.WriteBinaryFileOptions) error {
	if err := c.checkUsable(ctx); err != nil {
		return err
	}

	target := c.resolve(options.Path)
	if err := c.shell.fs.MkdirAll(path.Dir(target), 0o755); err != nil {
		return err
	}

	return c.shell.fs.WriteFile(target, options.Content, 0o644)
}

// WriteTextFile encodes and writes a text file.
func (c *core) WriteTextFile(ctx context.Context, options sandbox.WriteTextFileOptions) error {
	contents, err := sandbox.EncodeText(options.Content, options.Encoding)
	if err != nil {
		return err
	}

	return c.WriteBinaryFile(ctx, sandbox.WriteBinaryFileOptions{
		Path:    options.Path,
		Content: contents,
	})
}

// readFile returns the shared contents of a file. A missing file is an
// error wrapping fs.ErrNotExist.
func (c *core) readFile(ctx context.Context, name string) ([]byte, error) {
	if err := c.checkUsable(ctx); err != nil {
		return nil, err
	}

	return c.shell.fs.ReadFile(c.resolve(name))
}

// resolve returns the virtual path of a session path.
func (c *core) resolve(name string) string {
	return resolvePath(c.workingDirectory, name)
}

// checkUsable fails when the context ended or the session was destroyed.
func (c *core) checkUsable(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return context.Cause(ctx)
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.destroyed {
		return ErrDestroyed
	}

	return nil
}
