package local

import (
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/aarock1234/ai/pkg/sandbox"
)

// core implements sandbox.Session for one working directory. It is shared
// by a Session and its restricted views.
type core struct {
	directory string
	home      string
	shell     string
	env       map[string]string

	mu        sync.Mutex
	processes map[*process]struct{}
}

// newCore returns the shared state of a session.
func newCore(directory string, home string, shell string, env map[string]string) *core {
	return &core{
		directory: directory,
		home:      home,
		shell:     shell,
		env:       maps.Clone(env),
		processes: make(map[*process]struct{}),
	}
}

// Description describes the sandbox for the agent's instructions.
func (c *core) Description() string {
	return strings.Join([]string{
		"Local sandbox: commands run with " + c.shell + " on the host, without isolation.",
		"Filesystem changes affect the host filesystem.",
		"Current working directory: " + c.directory,
	}, "\n")
}

// ReadFile opens a file. Relative paths resolve against the working
// directory.
func (c *core) ReadFile(ctx context.Context, options sandbox.ReadFileOptions) (io.ReadCloser, error) {
	if err := ctx.Err(); err != nil {
		return nil, context.Cause(ctx)
	}

	file, err := os.Open(c.resolve(options.Path))
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", options.Path, err)
	}

	return file, nil
}

// ReadBinaryFile reads a file.
func (c *core) ReadBinaryFile(ctx context.Context, options sandbox.ReadFileOptions) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, context.Cause(ctx)
	}

	data, err := os.ReadFile(c.resolve(options.Path))
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", options.Path, err)
	}

	return data, nil
}

// ReadTextFile reads and decodes a text file, optionally limited to a line
// range.
func (c *core) ReadTextFile(ctx context.Context, options sandbox.ReadTextFileOptions) (string, error) {
	data, err := c.ReadBinaryFile(ctx, sandbox.ReadFileOptions{
		Path: options.Path,
	})
	if err != nil {
		return "", err
	}

	text, err := sandbox.DecodeText(data, options.Encoding)
	if err != nil {
		return "", err
	}

	return sandbox.ExtractLines(text, options.StartLine, options.EndLine), nil
}

// WriteFile writes a file from a stream, creating parent directories.
func (c *core) WriteFile(ctx context.Context, options sandbox.WriteFileOptions) (err error) {
	if err := ctx.Err(); err != nil {
		return context.Cause(ctx)
	}

	target := c.resolve(options.Path)
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return fmt.Errorf("create parent of %s: %w", options.Path, err)
	}

	file, err := os.Create(target)
	if err != nil {
		return fmt.Errorf("create %s: %w", options.Path, err)
	}

	defer func() {
		if closeErr := file.Close(); closeErr != nil {
			err = errors.Join(err, fmt.Errorf("close %s: %w", options.Path, closeErr))
		}
	}()

	if _, err := io.Copy(file, contextReader{
		ctx:    ctx,
		reader: options.Content,
	}); err != nil {
		return fmt.Errorf("write %s: %w", options.Path, err)
	}

	return nil
}

// WriteBinaryFile writes a file, creating parent directories.
func (c *core) WriteBinaryFile(ctx context.Context, options sandbox.WriteBinaryFileOptions) error {
	if err := ctx.Err(); err != nil {
		return context.Cause(ctx)
	}

	target := c.resolve(options.Path)
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return fmt.Errorf("create parent of %s: %w", options.Path, err)
	}

	if err := os.WriteFile(target, options.Content, 0o644); err != nil {
		return fmt.Errorf("write %s: %w", options.Path, err)
	}

	return nil
}

// WriteTextFile encodes and writes a text file.
func (c *core) WriteTextFile(ctx context.Context, options sandbox.WriteTextFileOptions) error {
	data, err := sandbox.EncodeText(options.Content, options.Encoding)
	if err != nil {
		return err
	}

	return c.WriteBinaryFile(ctx, sandbox.WriteBinaryFileOptions{
		Path:    options.Path,
		Content: data,
	})
}

// resolve returns the host path of a sandbox path.
func (c *core) resolve(path string) string {
	if filepath.IsAbs(path) {
		return path
	}

	return filepath.Join(c.directory, filepath.FromSlash(path))
}

// environment returns the environment of a command: the host environment,
// the sandbox HOME, the session's variables and the command's variables, in
// increasing precedence.
func (c *core) environment(extra map[string]string) []string {
	merged := make(map[string]string)

	for _, entry := range os.Environ() {
		name, value, _ := strings.Cut(entry, "=")
		merged[name] = value
	}

	merged["HOME"] = c.home
	maps.Copy(merged, c.env)
	maps.Copy(merged, extra)

	environment := make([]string, 0, len(merged))
	for name, value := range merged {
		environment = append(environment, name+"="+value)
	}

	return environment
}

// contextReader stops reading when its context ends.
type contextReader struct {
	ctx    context.Context
	reader io.Reader
}

// Read reads from the underlying reader unless the context ended.
func (r contextReader) Read(data []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, context.Cause(r.ctx)
	}

	return r.reader.Read(data)
}
