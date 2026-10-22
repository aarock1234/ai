package sandboxshell

import (
	"context"
	"io"
	"io/fs"
	"os"
	"path"
	"runtime"
	"strconv"
	"strings"

	"mvdan.cc/sh/v3/interp"
)

// sourceMarkerPrefix starts the placeholder path the source builtin opens
// for a validated file. The NUL byte makes the interpreter's host PATH
// search fail before any system call, and no script can produce it.
const sourceMarkerPrefix = "\x00sandboxshell-source-"

// openFile is the interpreter's open handler: redirections, here strings
// and the source builtin open virtual files and devices only.
func (s *shell) openFile(ctx context.Context, name string, flag int, perm os.FileMode) (io.ReadWriteCloser, error) {
	hc := interp.HandlerCtx(ctx)

	if source, ok := s.takeSource(name); ok {
		return deviceFile{
			reader: strings.NewReader(source),
		}, nil
	}

	resolved := resolvePath(hc.Dir, name)
	if device, ok := openDevice(hc, resolved); ok {
		return device, nil
	}

	file, err := s.fs.OpenFile(resolved, flag, perm&permissionBits)
	if err != nil {
		return nil, err
	}

	return file, nil
}

// stat is the interpreter's stat handler. There are no symbolic links, so
// following them makes no difference.
func (s *shell) stat(ctx context.Context, name string, _ bool) (fs.FileInfo, error) {
	hc := interp.HandlerCtx(ctx)

	return s.fs.Stat(resolvePath(hc.Dir, name))
}

// readDir is the interpreter's directory handler, used for globbing.
func (s *shell) readDir(ctx context.Context, name string) ([]fs.DirEntry, error) {
	hc := interp.HandlerCtx(ctx)

	return s.fs.ReadDir(resolvePath(hc.Dir, name))
}

// access is the interpreter's access handler. Commands run with full
// permissions, so only the execute bit of files is checked.
func (s *shell) access(ctx context.Context, name string, mode interp.AccessMode) error {
	hc := interp.HandlerCtx(ctx)
	resolved := resolvePath(hc.Dir, name)

	info, err := s.fs.Stat(resolved)
	if err != nil {
		return err
	}

	if mode&interp.AccessExec != 0 && !isExecutable(info) {
		return pathError("access", resolved, fsError{
			message: "permission denied",
			target:  fs.ErrPermission,
		})
	}

	return nil
}

// registerSource stores the validated contents of a sourced file and
// returns the marker path the interpreter should open instead, so that the
// file cannot change between validation and execution.
func (s *shell) registerSource(contents string) string {
	marker := sourceMarkerPrefix + strconv.FormatUint(s.sourceCounter.Add(1), 10)
	s.sources.Store(marker, contents)

	return marker
}

// takeSource returns and forgets the contents registered for a marker path.
func (s *shell) takeSource(name string) (string, bool) {
	if !strings.HasPrefix(name, sourceMarkerPrefix) {
		return "", false
	}

	contents, ok := s.sources.LoadAndDelete(name)
	if !ok {
		return "", false
	}

	return contents.(string), true
}

// isExecutable reports whether a node can be executed or entered.
func isExecutable(info fs.FileInfo) bool {
	return info.IsDir() || info.Mode().Perm()&0o111 != 0
}

// openDevice opens the device files the shell provides.
func openDevice(hc interp.HandlerContext, name string) (io.ReadWriteCloser, bool) {
	switch name {
	case "/dev/null":
		return deviceFile{}, true
	case "/dev/stdin":
		return deviceFile{
			reader: hc.Stdin,
		}, true
	case "/dev/stdout":
		return deviceFile{
			writer: hc.Stdout,
		}, true
	case "/dev/stderr":
		return deviceFile{
			writer: hc.Stderr,
		}, true
	default:
		return nil, false
	}
}

// deviceFile is an open device. Without a reader it reads as empty, and
// without a writer it discards writes.
type deviceFile struct {
	reader io.Reader
	writer io.Writer
}

// Read reads from the device.
func (d deviceFile) Read(data []byte) (int, error) {
	if d.reader == nil {
		return 0, io.EOF
	}

	return d.reader.Read(data)
}

// Write writes to the device.
func (d deviceFile) Write(data []byte) (int, error) {
	if d.writer == nil {
		return len(data), nil
	}

	return d.writer.Write(data)
}

// Close closes the device. The underlying streams belong to the
// interpreter.
func (d deviceFile) Close() error {
	return nil
}

// resolvePath returns the clean absolute virtual path of name, relative to
// the absolute directory dir.
func resolvePath(dir string, name string) string {
	name = toSlash(name)
	if !path.IsAbs(name) {
		name = path.Join(toSlash(dir), name)
	}

	return path.Clean(name)
}

// toSlash converts the backslashes the interpreter's host path functions
// insert on Windows. Elsewhere a backslash is an ordinary file name byte.
func toSlash(name string) string {
	if runtime.GOOS != "windows" {
		return name
	}

	return strings.ReplaceAll(name, `\`, "/")
}
