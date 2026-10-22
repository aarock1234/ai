package sandboxshell

import (
	"context"
	"path"
	"slices"
	"strings"

	"mvdan.cc/sh/v3/expand"
	"mvdan.cc/sh/v3/interp"
)

// commandDirectories hold the stub files that stand for the sandbox
// commands, so that they can be listed and run by path.
var commandDirectories = []string{
	"/bin",
	"/usr/bin",
}

// stubbedBuiltins are the shell builtins that also have stubs, because
// scripts commonly run them by path.
var stubbedBuiltins = []string{
	"echo",
	"false",
	"printf",
	"pwd",
	"test",
	"true",
}

// commandStub returns the contents of the stub file of a command.
func commandStub(name string) string {
	return "#!/bin/bash\n# sandboxshell built-in command: " + name + "\n"
}

// resolvedCommand is the target of a command name.
type resolvedCommand struct {
	// path is the virtual path the name resolves to.
	path string

	// name is the sandbox command or builtin a stub stands for, and empty
	// for a script.
	name string

	// command runs the sandbox command; nil for builtins and scripts.
	command commandFunc
}

// execMiddleware replaces the interpreter's exec handler. It never calls the
// next handler, which would run host programs.
func (s *shell) execMiddleware(interp.ExecHandlerFunc) interp.ExecHandlerFunc {
	return s.exec
}

// exec runs a command that is neither a shell function nor a builtin: a
// sandbox command or a script of the virtual filesystem. Builtins arrive
// here through exec and through stubs such as /bin/echo.
func (s *shell) exec(ctx context.Context, args []string) error {
	hc := interp.HandlerCtx(ctx)
	name := args[0]

	if interp.IsBuiltin(name) {
		return s.runBuiltin(ctx, args)
	}

	resolved, ok := s.resolveCommand(hc.Env, hc.Dir, name)
	if !ok {
		return s.reportUnresolved(hc, name)
	}

	if resolved.command == nil && resolved.name != "" {
		return s.runBuiltin(ctx, slices.Concat([]string{resolved.name}, args[1:]))
	}

	if resolved.command != nil {
		call := newCommandCall(s, hc, resolved.name, args[1:])

		return exitError(resolved.command(ctx, call))
	}

	return exitError(s.runScriptFile(ctx, hc, resolved.path, args))
}

// runBuiltin runs a builtin from the exec handler, guarded like a direct
// call.
func (s *shell) runBuiltin(ctx context.Context, args []string) error {
	hc := interp.HandlerCtx(ctx)

	guarded, err := s.guardCall(ctx, args)
	if err != nil {
		return err
	}

	return hc.Builtin(ctx, guarded)
}

// resolveCommand resolves a command name as bash would: names with a slash
// are paths, other names are searched in PATH. Sandbox commands remain
// available by name when PATH does not reach their stubs.
func (s *shell) resolveCommand(env expand.Environ, dir string, name string) (resolvedCommand, bool) {
	if strings.Contains(name, "/") {
		resolved := resolvePath(dir, name)
		if !s.isExecutableFile(resolved) {
			return resolvedCommand{}, false
		}

		return s.describeFile(resolved), true
	}

	for _, directory := range searchPath(env) {
		candidate := resolvePath(dir, directory+"/"+name)
		if s.isExecutableFile(candidate) {
			return s.describeFile(candidate), true
		}
	}

	command, ok := s.commands[name]
	if !ok {
		return resolvedCommand{}, false
	}

	return resolvedCommand{
		path:    commandDirectories[0] + "/" + name,
		name:    name,
		command: command,
	}, true
}

// describeFile returns the target of an executable file: the command its
// stub stands for, or a script.
func (s *shell) describeFile(name string) resolvedCommand {
	resolved := resolvedCommand{
		path: name,
	}

	commandName, ok := s.stubCommand(name)
	if !ok {
		return resolved
	}

	resolved.name = commandName
	resolved.command = s.commands[commandName]

	return resolved
}

// stubCommand returns the command a file stands for when it is an
// unmodified stub in a command directory.
func (s *shell) stubCommand(name string) (string, bool) {
	commandName := path.Base(name)
	if !slices.Contains(commandDirectories, path.Dir(name)) {
		return "", false
	}

	_, isCommand := s.commands[commandName]
	if !isCommand && !slices.Contains(stubbedBuiltins, commandName) {
		return "", false
	}

	contents, err := s.fs.ReadFile(name)
	if err != nil || string(contents) != commandStub(commandName) {
		return "", false
	}

	return commandName, true
}

// isExecutableFile reports whether name is a regular file with an execute
// bit.
func (s *shell) isExecutableFile(name string) bool {
	info, err := s.fs.Stat(name)

	return err == nil && !info.IsDir() && isExecutable(info)
}

// reportUnresolved reports a command that cannot be run, with bash's
// messages and exit codes.
func (s *shell) reportUnresolved(hc interp.HandlerContext, name string) error {
	if !strings.Contains(name, "/") {
		printMessage(hc.Stderr, "%s: %s: command not found\n", shellName, name)

		return exitError(exitNotFound)
	}

	info, err := s.fs.Stat(resolvePath(hc.Dir, name))

	switch {
	case err != nil:
		printMessage(hc.Stderr, "%s: %s: %s\n", shellName, name, describeError(err))

		return exitError(exitNotFound)
	case info.IsDir():
		printMessage(hc.Stderr, "%s: %s: Is a directory\n", shellName, name)
	default:
		printMessage(hc.Stderr, "%s: %s: Permission denied\n", shellName, name)
	}

	return exitError(exitCannotExecute)
}

// runScriptFile runs an executable file of the virtual filesystem as a
// shell script in a child shell. Files whose "#!" line names another
// interpreter cannot run.
func (s *shell) runScriptFile(ctx context.Context, hc interp.HandlerContext, name string, args []string) int {
	contents, err := s.fs.ReadFile(name)
	if err != nil {
		printMessage(hc.Stderr, "%s: %s: %s\n", shellName, args[0], describeError(err))

		return exitCannotExecute
	}

	if interpreter, ok := scriptInterpreter(string(contents)); ok && !isShellInterpreter(interpreter) {
		printMessage(hc.Stderr, "%s: %s: %s: bad interpreter: No such file or directory\n", shellName, args[0], interpreter)

		return exitCannotExecute
	}

	return s.run(ctx, script{
		name:   args[0],
		source: string(contents),
		dir:    hc.Dir,
		env:    childEnviron(hc.Env),
		params: args[1:],
		stdin:  hc.Stdin,
		stdout: hc.Stdout,
		stderr: hc.Stderr,
	})
}

// scriptInterpreter returns the program of a script's "#!" line, looking
// through "env". It reports false when the script has no such line.
func scriptInterpreter(contents string) (string, bool) {
	line, ok := strings.CutPrefix(contents, "#!")
	if !ok {
		return "", false
	}

	line, _, _ = strings.Cut(line, "\n")
	fields := strings.Fields(line)
	if len(fields) == 0 {
		return "", false
	}

	program := fields[0]
	arguments := fields[1:]

	if path.Base(program) == "env" {
		arguments = slices.DeleteFunc(arguments, func(argument string) bool {
			return strings.HasPrefix(argument, "-")
		})

		if len(arguments) > 0 {
			program = arguments[0]
		}
	}

	return program, true
}

// isShellInterpreter reports whether a "#!" program is a shell the sandbox
// interprets.
func isShellInterpreter(program string) bool {
	switch path.Base(program) {
	case "bash", "sh":
		return true
	default:
		return false
	}
}

// searchPath returns the directories of PATH; an empty entry is the
// working directory.
func searchPath(env expand.Environ) []string {
	directories := strings.Split(env.Get("PATH").String(), ":")

	for index, directory := range directories {
		if directory == "" {
			directories[index] = "."
		}
	}

	return directories
}

// exitError converts an exit code into the interpreter's handler result.
func exitError(code int) error {
	if code == 0 {
		return nil
	}

	return interp.ExitStatus(uint8(code & 0xff))
}
