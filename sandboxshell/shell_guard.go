package sandboxshell

import (
	"context"
	"slices"
	"strings"

	"mvdan.cc/sh/v3/interp"
	"mvdan.cc/sh/v3/syntax"
)

// Replacement calls that only set an exit status, for builtins the guard
// already carried out.
var (
	succeededCall = []string{
		"true",
	}
	failedCall = []string{
		"false",
	}
)

// guardCall is the interpreter's call handler. It runs before every simple
// command and rewrites the builtins that would otherwise reach the host:
// type and command -v search the host PATH, source searches the host PATH
// and parses files without validation, pwd -P resolves host symbolic links,
// the -O and -G tests read the host's user database, and eval and trap
// parse code without validation.
func (s *shell) guardCall(ctx context.Context, args []string) ([]string, error) {
	hc := interp.HandlerCtx(ctx)
	call := unwrapBuiltinPrefixes(args)

	switch call[0] {
	case "type":
		return s.typeBuiltin(hc, call[1:]), nil
	case "command":
		return s.commandLookup(hc, call[1:], args), nil
	case "source", ".":
		return s.guardSource(hc, call), nil
	case "eval":
		return guardCode(hc, call, strings.Join(call[1:], " ")), nil
	case "trap":
		return guardTrap(hc, call), nil
	case "pwd":
		return []string{
			"pwd",
		}, nil
	case "test", "[":
		return guardTest(hc, call), nil
	default:
		return args, nil
	}
}

// unwrapBuiltinPrefixes removes leading "builtin" and "command" words,
// which run a builtin directly without the call handler, so that the
// wrapped builtin is guarded as well. Lookups such as "command -v" are
// kept.
func unwrapBuiltinPrefixes(args []string) []string {
	for len(args) > 1 {
		wrapsBuiltin := args[0] == "builtin" || args[0] == "command" && !strings.HasPrefix(args[1], "-")
		if !wrapsBuiltin {
			break
		}

		args = args[1:]
	}

	return args
}

// typeBuiltin carries out "type [-t|-p|-P] name...". Shell functions are
// not reported, because the interpreter does not expose them to handlers.
func (s *shell) typeBuiltin(hc interp.HandlerContext, args []string) []string {
	mode := ""
	for len(args) > 0 && strings.HasPrefix(args[0], "-") && args[0] != "-" {
		if args[0] == "--" {
			args = args[1:]

			break
		}

		mode = args[0]
		args = args[1:]
	}

	notFound := false
	for _, name := range args {
		if !s.describeCommand(hc, name, mode) {
			notFound = true
		}
	}

	if notFound {
		return failedCall
	}

	return succeededCall
}

// describeCommand prints what a name refers to in the format of type's
// mode, and reports whether the name was found.
func (s *shell) describeCommand(hc interp.HandlerContext, name string, mode string) bool {
	isShellWord := syntax.IsKeyword(name) || interp.IsBuiltin(name)

	if mode == "-p" && isShellWord {
		return true
	}

	if mode != "-P" && isShellWord {
		describeShellWord(hc, name, mode)

		return true
	}

	resolved, ok := s.resolveCommand(hc.Env, hc.Dir, name)
	if !ok {
		if mode != "-t" {
			printMessage(hc.Stderr, "%s: type: %s: not found\n", shellName, name)
		}

		return false
	}

	switch mode {
	case "-t":
		printMessage(hc.Stdout, "%s\n", "file")
	case "-p", "-P":
		printMessage(hc.Stdout, "%s\n", resolved.path)
	default:
		printMessage(hc.Stdout, "%s is %s\n", name, resolved.path)
	}

	return true
}

// describeShellWord prints a keyword or builtin in the format of type's
// mode.
func describeShellWord(hc interp.HandlerContext, name string, mode string) {
	kind := "builtin"
	description := "a shell builtin"

	if syntax.IsKeyword(name) {
		kind = "keyword"
		description = "a shell keyword"
	}

	if mode == "-t" {
		printMessage(hc.Stdout, "%s\n", kind)

		return
	}

	printMessage(hc.Stdout, "%s is %s\n", name, description)
}

// commandLookup carries out "command -v name..." and "command -V name...".
// Other uses of command are returned unchanged.
func (s *shell) commandLookup(hc interp.HandlerContext, args []string, original []string) []string {
	if len(args) == 0 {
		return original
	}

	switch args[0] {
	case "-v":
		return s.printCommandPaths(hc, args[1:])
	case "-V":
		return s.typeBuiltin(hc, args[1:])
	default:
		return original
	}
}

// printCommandPaths prints how each name would be run, failing when the
// last name is not found, as command -v does.
func (s *shell) printCommandPaths(hc interp.HandlerContext, names []string) []string {
	found := false

	for _, name := range names {
		found = true

		if syntax.IsKeyword(name) || interp.IsBuiltin(name) {
			printMessage(hc.Stdout, "%s\n", name)

			continue
		}

		resolved, ok := s.resolveCommand(hc.Env, hc.Dir, name)
		if !ok {
			found = false

			continue
		}

		printMessage(hc.Stdout, "%s\n", resolved.path)
	}

	if !found {
		return failedCall
	}

	return succeededCall
}

// guardSource validates a sourced file and replaces its path with a marker
// that serves the validated contents.
func (s *shell) guardSource(hc interp.HandlerContext, call []string) []string {
	if len(call) < 2 {
		return call
	}

	name := call[1]
	resolved := s.resolveSourcePath(hc, name)

	contents, err := s.fs.ReadFile(resolved)
	if err != nil {
		printMessage(hc.Stderr, "%s: %s: %s\n", shellName, name, describeError(err))

		return failedCall
	}

	if _, err := parseScript(string(contents), name); err != nil {
		printMessage(hc.Stderr, "%s: %v\n", shellName, err)

		return failedCall
	}

	marker := s.registerSource(string(contents))

	return slices.Concat([]string{call[0], marker}, call[2:])
}

// resolveSourcePath resolves a sourced file name: names with a slash are
// paths, and other names are searched in PATH and then in the working
// directory, as bash does.
func (s *shell) resolveSourcePath(hc interp.HandlerContext, name string) string {
	if strings.Contains(name, "/") {
		return resolvePath(hc.Dir, name)
	}

	for _, directory := range searchPath(hc.Env) {
		candidate := resolvePath(hc.Dir, directory+"/"+name)

		info, err := s.fs.Stat(candidate)
		if err == nil && !info.IsDir() {
			return candidate
		}
	}

	return resolvePath(hc.Dir, name)
}

// guardTrap validates the code of "trap code signal...".
func guardTrap(hc interp.HandlerContext, call []string) []string {
	operands := call[1:]
	if len(operands) > 0 && operands[0] == "--" {
		operands = operands[1:]
	}

	if len(operands) < 2 || strings.HasPrefix(operands[0], "-") {
		return call
	}

	return guardCode(hc, call, operands[0])
}

// guardCode validates code a builtin is about to parse and run, failing
// the call when it is rejected. Syntax errors are left to the builtin.
func guardCode(hc interp.HandlerContext, call []string, code string) []string {
	parser := syntax.NewParser(syntax.Variant(syntax.LangBash))

	file, err := parser.Parse(strings.NewReader(code), "")
	if err != nil {
		return call
	}

	if err := validateScript(file); err != nil {
		printMessage(hc.Stderr, "%s: %s: %v\n", shellName, call[0], err)

		return failedCall
	}

	return call
}

// guardTest rejects the -O and -G operators of test and [.
func guardTest(hc interp.HandlerContext, call []string) []string {
	for _, arg := range call[1:] {
		if arg == "-O" || arg == "-G" {
			printMessage(hc.Stderr, "%s: %s: %v\n", shellName, call[0], errOwnershipTest)

			return failedCall
		}
	}

	return call
}
