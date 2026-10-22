package sandboxshell

import (
	"context"
	"errors"
	"fmt"
	"io"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"

	"mvdan.cc/sh/v3/expand"
	"mvdan.cc/sh/v3/interp"
	"mvdan.cc/sh/v3/syntax"
)

// shellName prefixes the shell's own error messages.
const shellName = "bash"

// Exit codes with a conventional meaning.
const (
	exitFailure        = 1
	exitUsage          = 2
	exitTimedOut       = 124
	exitCannotExecute  = 126
	exitNotFound       = 127
	exitInterrupted    = 130
	exitKilledBySignal = 137
)

// Errors of scripts that use interpreter features which would reach the
// host.
var (
	errProcessSubstitution = errors.New("process substitution is not supported in the sandbox")
	errOwnershipTest       = errors.New("the -O and -G file tests are not supported in the sandbox")
)

// shell runs bash scripts with the mvdan.cc/sh interpreter against one
// session's filesystem and the sandbox commands. Every script gets a fresh
// interpreter, so shell state never carries over between commands.
type shell struct {
	fs       *memFS
	commands map[string]commandFunc

	// sources holds validated files of the source builtin by marker path,
	// until the interpreter opens them.
	sources       sync.Map
	sourceCounter atomic.Uint64
}

// newShell returns a shell over a filesystem.
func newShell(filesystem *memFS) *shell {
	return &shell{
		fs:       filesystem,
		commands: newCommandTable(),
	}
}

// script is one script to run.
type script struct {
	// name is $0 and names the script in syntax errors.
	name   string
	source string

	dir string
	env expand.Environ

	// options are shell options such as "-e" or "-o pipefail".
	options []string
	params  []string

	stdin  io.Reader
	stdout io.Writer
	stderr io.Writer
}

// run runs a script and returns its exit code. Syntax errors and rejected
// features are reported on the script's stderr with exit code 2.
func (s *shell) run(ctx context.Context, run script) int {
	file, err := parseScript(run.source, run.name)
	if err != nil {
		printMessage(run.stderr, "%s: %v\n", shellName, err)

		return exitUsage
	}

	runner, err := s.newRunner(run)
	if err != nil {
		printMessage(run.stderr, "%s: %v\n", shellName, err)

		return exitUsage
	}

	return scriptExitCode(ctx, runner.Run(ctx, file), run.stderr)
}

// runArgs runs one command line given as separate arguments, as programs
// such as xargs and env do.
func (s *shell) runArgs(ctx context.Context, call *commandCall, args []string) int {
	quoted := make([]string, 0, len(args))
	for _, arg := range args {
		word, err := syntax.Quote(arg, syntax.LangBash)
		if err != nil {
			return call.fail("cannot run %q: %v", arg, err)
		}

		quoted = append(quoted, word)
	}

	return s.run(ctx, script{
		name:   shellName,
		source: strings.Join(quoted, " "),
		dir:    call.dir,
		env:    childEnviron(call.env),
		stdin:  call.stdin,
		stdout: call.stdout,
		stderr: call.stderr,
	})
}

// newRunner returns an interpreter whose every file, directory and command
// handler stays inside the sandbox.
func (s *shell) newRunner(run script) (*interp.Runner, error) {
	options := []interp.RunnerOption{
		interp.Env(isolatedTemporaryDirectory{
			Environ: run.env,
		}),
		workingDirectory(run.dir),
		interp.StdIO(run.stdin, run.stdout, run.stderr),
		interp.CallHandler(s.guardCall),
		interp.ExecHandlers(s.execMiddleware),
		interp.OpenHandler(s.openFile),
		interp.ReadDirHandler2(s.readDir),
		interp.StatHandler(s.stat),
		interp.AccessHandler(s.access),
	}

	if params := shellParams(run); len(params) > 0 {
		options = append(options, interp.Params(params...))
	}

	runner, err := interp.New(options...)
	if err != nil {
		return nil, err
	}

	// The interpreter fixes its temporary directory at the first reset and
	// opens host paths inside it directly, for process substitution FIFOs.
	// Resetting first with an unreachable TMPDIR and then with the script's
	// environment keeps that directory out of every script's reach.
	runner.Reset()
	runner.Env = run.env
	runner.Reset()

	return runner, nil
}

// workingDirectory sets the interpreter's directory without the host
// filesystem checks of interp.Dir.
func workingDirectory(dir string) interp.RunnerOption {
	return func(runner *interp.Runner) error {
		runner.Dir = dir

		return nil
	}
}

// shellParams returns the arguments of interp.Params for a script.
func shellParams(run script) []string {
	if len(run.options) == 0 && run.params == nil {
		return nil
	}

	params := append([]string{}, run.options...)
	params = append(params, "--")

	return append(params, run.params...)
}

// scriptExitCode converts the result of an interpreter run into an exit
// code, reporting unexpected errors on stderr.
func scriptExitCode(ctx context.Context, err error, stderr io.Writer) int {
	if err == nil {
		return 0
	}

	if status, ok := errors.AsType[interp.ExitStatus](err); ok {
		return int(status)
	}

	if ctx.Err() != nil {
		return exitInterrupted
	}

	printMessage(stderr, "%s: %v\n", shellName, err)

	return exitFailure
}

// printMessage writes a diagnostic or a builtin's output. Write failures,
// such as to a closed pipe, are ignored, as a shell does.
func printMessage(writer io.Writer, format string, values ...any) {
	_, _ = fmt.Fprintf(writer, format, values...)
}

// parseScript parses and validates bash source.
func parseScript(source string, name string) (*syntax.File, error) {
	parser := syntax.NewParser(syntax.Variant(syntax.LangBash))

	file, err := parser.Parse(strings.NewReader(source), name)
	if err != nil {
		return nil, err
	}

	if err := validateScript(file); err != nil {
		return nil, err
	}

	return file, nil
}

// validateScript rejects interpreter features that would reach the host:
// process substitution creates host FIFOs, and the -O and -G file tests
// consult the host's user database.
func validateScript(node syntax.Node) error {
	var rejected error

	syntax.Walk(node, func(node syntax.Node) bool {
		switch node := node.(type) {
		case *syntax.ProcSubst:
			rejected = errProcessSubstitution
		case *syntax.UnaryTest:
			if node.Op == syntax.TsUsrOwn || node.Op == syntax.TsGrpOwn {
				rejected = errOwnershipTest
			}
		}

		return rejected == nil
	})

	return rejected
}

// isolatedTemporaryDirectory overrides TMPDIR with a path that no script
// can name and that the host cannot open, because it contains a NUL byte.
type isolatedTemporaryDirectory struct {
	expand.Environ
}

// Get returns a variable, replacing TMPDIR.
func (e isolatedTemporaryDirectory) Get(name string) expand.Variable {
	if name != "TMPDIR" {
		return e.Environ.Get(name)
	}

	return expand.Variable{
		Set:  true,
		Kind: expand.String,
		Str:  unreachableHostDirectory(),
	}
}

// unreachableHostDirectory returns an absolute host path containing a NUL
// byte, which every host filesystem call rejects before reaching the
// system.
func unreachableHostDirectory() string {
	if runtime.GOOS == "windows" {
		return `C:\` + "\x00sandboxshell"
	}

	return "/\x00sandboxshell"
}
