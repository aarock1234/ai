package sandboxshell

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"slices"
	"sync"

	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/sandbox"
)

// providerID identifies the sandbox in errors.
const providerID = "sandboxshell"

var _ harness.NetworkSession = (*Session)(nil)

// Options configure New.
type Options struct {
	harness.SandboxSessionCreateOptions

	// WorkingDirectory is the absolute virtual directory commands start in
	// and relative paths resolve against; upstream's cwd option. It is
	// created when missing. Empty selects /home/user, which is also HOME.
	WorkingDirectory string

	// Env is added to the default environment of every command.
	Env map[string]string
}

// Session is an isolated in-memory sandbox session. It implements
// harness.NetworkSession without ports or network policy.
type Session struct {
	*core

	id        string
	closeOnce sync.Once
	closeErr  error
}

// New creates a fresh in-memory sandbox and applies the options' template
// to it. Every call creates a separate filesystem; sessions never share
// state.
func New(ctx context.Context, options Options) (*Session, error) {
	if err := ctx.Err(); err != nil {
		return nil, context.Cause(ctx)
	}

	workingDirectory := defaultWorkingDirectory
	if options.WorkingDirectory != "" {
		workingDirectory = resolvePath("/", options.WorkingDirectory)
	}

	sessionShell := newShell(newMemFS())
	if err := createLayout(sessionShell, workingDirectory); err != nil {
		return nil, fmt.Errorf("create sandboxshell layout: %w", err)
	}

	sandboxID := options.SandboxID
	if sandboxID == "" {
		sandboxID = id.New(providerID)
	}

	session := &Session{
		core: newCore(sessionShell, workingDirectory, options.Env),
		id:   sandboxID,
	}

	if options.Template != nil {
		if err := options.Template.Prepare(ctx, session.Restricted()); err != nil {
			return nil, errors.Join(fmt.Errorf("prepare sandboxshell template: %w", err), session.Destroy(ctx))
		}
	}

	return session, nil
}

// Resume reports that sandboxshell sessions live in this process and cannot
// be resumed by sandbox ID.
func Resume(ctx context.Context, _ harness.SandboxSessionResumeOptions) (*Session, error) {
	if err := ctx.Err(); err != nil {
		return nil, context.Cause(ctx)
	}

	return nil, &harness.CapabilityUnsupportedError{
		Message: "sandboxshell sandboxes run in-process and cannot be resumed by sandbox id",
	}
}

// ID returns the sandbox identifier: Options.SandboxID or a generated one.
// It is only a label and does not enforce uniqueness.
func (s *Session) ID() string {
	return s.id
}

// DefaultWorkingDirectory returns the working directory.
func (s *Session) DefaultWorkingDirectory() string {
	return s.workingDirectory
}

// Ports returns no ports: the sandbox has no network.
func (s *Session) Ports() []int {
	return nil
}

// PortEndpoint reports that the sandbox cannot expose ports.
func (s *Session) PortEndpoint(context.Context, harness.PortEndpointOptions) (harness.PortEndpoint, error) {
	return harness.PortEndpoint{}, &harness.CapabilityUnsupportedError{
		Message:   "sandboxshell sandboxes run in-process and cannot expose a port url; use a hosted sandbox such as sandbox/vercel for bridge-backed harness adapters",
		HarnessID: providerID,
	}
}

// Stop stops every running command. The filesystem is kept. It is
// idempotent.
func (s *Session) Stop(ctx context.Context) error {
	return s.killAll(ctx)
}

// Destroy stops every running command and releases the filesystem. Later
// operations fail with ErrDestroyed.
func (s *Session) Destroy(ctx context.Context) error {
	s.closeOnce.Do(func() {
		s.mu.Lock()
		s.destroyed = true
		s.mu.Unlock()

		s.closeErr = s.Stop(ctx)
		s.shell.fs.Clear()
	})

	return s.closeErr
}

// Restricted returns the files-and-processes view of the session.
func (s *Session) Restricted() sandbox.Session {
	return &restricted{
		core: s.core,
	}
}

// restricted is the view returned by Session.Restricted.
type restricted struct {
	*core
}

// createLayout creates just-bash's default directories, the device files
// and the command stubs.
func createLayout(sessionShell *shell, workingDirectory string) error {
	filesystem := sessionShell.fs

	if err := filesystem.Mkdir(temporaryDirectory, 0o777|fs.ModeSticky); err != nil {
		return err
	}

	directories := []string{
		"/bin",
		"/usr/bin",
		"/usr/local/bin",
		"/dev",
		defaultWorkingDirectory,
		workingDirectory,
	}

	for _, directory := range directories {
		if err := filesystem.MkdirAll(directory, 0o755); err != nil {
			return err
		}
	}

	for _, device := range []string{"/dev/null", "/dev/stdin", "/dev/stdout", "/dev/stderr"} {
		if err := filesystem.CreateDevice(device); err != nil {
			return err
		}
	}

	stubs := slices.Concat(slices.Collect(maps.Keys(sessionShell.commands)), stubbedBuiltins)
	for _, name := range stubs {
		if err := filesystem.WriteFile(commandDirectories[0]+"/"+name, []byte(commandStub(name)), 0o755); err != nil {
			return err
		}
	}

	return nil
}
