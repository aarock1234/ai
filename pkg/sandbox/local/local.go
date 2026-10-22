// Package local is a sandbox session that runs commands with os/exec in a
// working directory on the host, with a real shell (bash by default). The
// isolated in-memory port of upstream @ai-sdk/sandbox-just-bash is the
// separate github.com/aarock1234/ai/sandboxshell module.
//
// The local sandbox is NOT isolated. Commands run with the permissions of
// the current process and can read and write anything it can, including
// files outside the working directory, and reach the network. Use it for
// development, tests of tool wiring and trusted workloads; use a hosted
// sandbox such as sandbox/vercel for untrusted code.
//
// Like the upstream just-bash sandbox, a local session exposes no ports,
// cannot enforce network policies and cannot be resumed from another
// process, so bridge-backed harness adapters cannot use it.
package local

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/aarock1234/ai/internal/id"
	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/sandbox"
)

// providerID identifies the local sandbox in errors.
const providerID = "local-sandbox"

var _ harness.NetworkSession = (*Session)(nil)

// defaultShell runs the commands.
const defaultShell = "bash"

// Options configure New.
type Options struct {
	harness.SandboxSessionCreateOptions

	// Directory is the default working directory. Relative file paths and
	// commands resolve against it. Empty creates a temporary directory that
	// Destroy removes.
	Directory string

	// Home is the HOME directory of commands, where harnesses keep their
	// state. Empty uses ".home" inside Directory, so that harness state
	// stays out of the user's home directory; set it to os.UserHomeDir() to
	// share the host's home.
	Home string

	// Shell is the shell that runs commands with "-c". Empty selects bash,
	// which must be on PATH.
	Shell string

	// Env is added to the process environment of every command.
	Env map[string]string
}

// Session is a local sandbox session. It implements harness.NetworkSession
// without ports or network policy.
type Session struct {
	*core

	id         string
	ownsDir    bool
	closeOnce  sync.Once
	closeError error
}

// New creates a local sandbox session and applies the options' template.
func New(ctx context.Context, options Options) (*Session, error) {
	if err := ctx.Err(); err != nil {
		return nil, context.Cause(ctx)
	}

	directory, ownsDir, err := prepareDirectory(options.Directory)
	if err != nil {
		return nil, err
	}

	home := options.Home
	if home == "" {
		home = filepath.Join(directory, ".home")
	}

	if err := os.MkdirAll(home, 0o755); err != nil {
		return nil, fmt.Errorf("create local sandbox home: %w", err)
	}

	shell := options.Shell
	if shell == "" {
		shell = defaultShell
	}

	sandboxID := options.SandboxID
	if sandboxID == "" {
		sandboxID = id.New("local")
	}

	session := &Session{
		core:    newCore(directory, home, shell, options.Env),
		id:      sandboxID,
		ownsDir: ownsDir,
	}

	if options.Template != nil {
		if err := options.Template.Prepare(ctx, session.Restricted()); err != nil {
			return nil, errors.Join(fmt.Errorf("prepare local sandbox template: %w", err), session.Destroy(ctx))
		}
	}

	return session, nil
}

// Resume reports that local sandboxes cannot be resumed across processes.
func Resume(ctx context.Context, options harness.SandboxSessionResumeOptions) (*Session, error) {
	if err := ctx.Err(); err != nil {
		return nil, context.Cause(ctx)
	}

	return nil, &harness.CapabilityUnsupportedError{
		Message:   fmt.Sprintf("local sandboxes run in this process and cannot be resumed by sandbox id %q", options.SandboxID),
		HarnessID: providerID,
	}
}

// ID returns the sandbox identifier: Options.SandboxID or a generated one.
// It is only a label.
func (s *Session) ID() string {
	return s.id
}

// DefaultWorkingDirectory returns the working directory.
func (s *Session) DefaultWorkingDirectory() string {
	return s.directory
}

// Ports returns no ports: local sandboxes expose none.
func (s *Session) Ports() []int {
	return nil
}

// PortEndpoint reports that local sandboxes cannot expose ports.
func (s *Session) PortEndpoint(context.Context, harness.PortEndpointOptions) (harness.PortEndpoint, error) {
	return harness.PortEndpoint{}, &harness.CapabilityUnsupportedError{
		Message:   "local sandboxes cannot expose a port endpoint; use a hosted sandbox such as sandbox/vercel for bridge-backed harness adapters",
		HarnessID: providerID,
	}
}

// Stop kills every process the session started. It is idempotent.
func (s *Session) Stop(ctx context.Context) error {
	return s.killAll(ctx)
}

// Destroy stops the session and removes the working directory when the
// session created it.
func (s *Session) Destroy(ctx context.Context) error {
	s.closeOnce.Do(func() {
		stopErr := s.Stop(ctx)

		var removeErr error
		if s.ownsDir {
			removeErr = os.RemoveAll(s.directory)
		}

		s.closeError = errors.Join(stopErr, removeErr)
	})

	return s.closeError
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

// prepareDirectory resolves the working directory, creating a temporary one
// when none is configured. It reports whether the session owns it.
func prepareDirectory(directory string) (string, bool, error) {
	if directory == "" {
		created, err := os.MkdirTemp("", "ai-sdk-local-sandbox-")
		if err != nil {
			return "", false, fmt.Errorf("create local sandbox directory: %w", err)
		}

		return created, true, nil
	}

	absolute, err := filepath.Abs(directory)
	if err != nil {
		return "", false, fmt.Errorf("resolve local sandbox directory: %w", err)
	}

	if err := os.MkdirAll(absolute, 0o755); err != nil {
		return "", false, fmt.Errorf("create local sandbox directory: %w", err)
	}

	return absolute, false, nil
}
