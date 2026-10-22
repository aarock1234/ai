package vercel

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"fmt"
	"io"
	"io/fs"
	"path"
	"strings"

	"github.com/aarock1234/ai/pkg/sandbox"
)

// archiveExtractDir is where written files are extracted; archive entries
// are paths relative to it.
const archiveExtractDir = "/"

// ReadFile opens a file. Relative paths resolve against the session's
// working directory.
func (c *core) ReadFile(ctx context.Context, options sandbox.ReadFileOptions) (io.ReadCloser, error) {
	var body io.ReadCloser

	err := c.withResume(ctx, func(sessionID string) error {
		content, found, err := c.client.readFile(ctx, sessionID, options.Path, c.workingDirectory())
		if err != nil {
			return err
		}

		if !found {
			return fmt.Errorf("read %s: %w", options.Path, fs.ErrNotExist)
		}

		body = content

		return nil
	})
	if err != nil {
		return nil, err
	}

	return body, nil
}

// ReadBinaryFile reads a file.
func (c *core) ReadBinaryFile(ctx context.Context, options sandbox.ReadFileOptions) ([]byte, error) {
	body, err := c.ReadFile(ctx, options)
	if err != nil {
		return nil, err
	}

	defer func() {
		_ = body.Close()
	}()

	data, err := io.ReadAll(body)
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

// WriteFile writes a file from a stream. The stream is buffered, as the
// API takes whole archives.
func (c *core) WriteFile(ctx context.Context, options sandbox.WriteFileOptions) error {
	data, err := io.ReadAll(options.Content)
	if err != nil {
		return fmt.Errorf("read content of %s: %w", options.Path, err)
	}

	return c.WriteBinaryFile(ctx, sandbox.WriteBinaryFileOptions{
		Path:    options.Path,
		Content: data,
	})
}

// WriteBinaryFile writes a file, creating parent directories.
func (c *core) WriteBinaryFile(ctx context.Context, options sandbox.WriteBinaryFileOptions) error {
	if err := ctx.Err(); err != nil {
		return context.Cause(ctx)
	}

	if err := c.ensureParent(ctx, options.Path); err != nil {
		return err
	}

	return c.withResume(ctx, func(sessionID string) error {
		archive, err := fileArchive(archiveEntryName(options.Path, c.workingDirectory()), options.Content)
		if err != nil {
			return err
		}

		if err := c.client.writeFiles(ctx, sessionID, archive, archiveExtractDir); err != nil {
			return fmt.Errorf("write %s: %w", options.Path, err)
		}

		return nil
	})
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

// ensureParent creates the parent directory of a file.
func (c *core) ensureParent(ctx context.Context, filePath string) error {
	parent := path.Dir(filePath)
	if parent == "" || parent == "." || parent == "/" {
		return nil
	}

	_, err := c.runCommand(ctx, commandRequest{
		Command: "mkdir",
		Args: []string{
			"-p",
			parent,
		},
	})
	if err != nil {
		return fmt.Errorf("create parent of %s: %w", filePath, err)
	}

	return nil
}

// archiveEntryName returns the archive path of a file: relative paths
// resolve against cwd, and the result is relative to the extract directory.
func archiveEntryName(filePath string, cwd string) string {
	resolved := path.Clean(filePath)
	if !path.IsAbs(resolved) {
		resolved = path.Join(cwd, filePath)
	}

	return strings.TrimPrefix(resolved, archiveExtractDir)
}

// fileArchive returns a gzipped tar archive with one file.
func fileArchive(name string, content []byte) ([]byte, error) {
	var archive bytes.Buffer

	compressor := gzip.NewWriter(&archive)
	writer := tar.NewWriter(compressor)

	header := &tar.Header{
		Name:     name,
		Mode:     0o644,
		Size:     int64(len(content)),
		Typeflag: tar.TypeReg,
	}

	if err := writer.WriteHeader(header); err != nil {
		return nil, fmt.Errorf("write archive header: %w", err)
	}

	if _, err := writer.Write(content); err != nil {
		return nil, fmt.Errorf("write archive content: %w", err)
	}

	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("finish archive: %w", err)
	}

	if err := compressor.Close(); err != nil {
		return nil, fmt.Errorf("compress archive: %w", err)
	}

	return archive.Bytes(), nil
}
