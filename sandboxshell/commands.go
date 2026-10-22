package sandboxshell

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"

	"mvdan.cc/sh/v3/expand"
	"mvdan.cc/sh/v3/interp"
)

// commandFunc runs a sandbox command and returns its exit code.
type commandFunc func(ctx context.Context, call *commandCall) int

// newCommandTable returns the sandbox commands by name: the just-bash
// command set that a Go program can provide without host programs, network
// access or embedded language runtimes.
func newCommandTable() map[string]commandFunc {
	return map[string]commandFunc{
		"base64":    runBase64,
		"basename":  runBasename,
		"bash":      runBash,
		"cat":       runCat,
		"chmod":     runChmod,
		"clear":     runClear,
		"cp":        runCopy,
		"cut":       runCut,
		"date":      runDate,
		"dirname":   runDirname,
		"egrep":     runExtendedGrep,
		"env":       runEnv,
		"fgrep":     runFixedGrep,
		"find":      runFind,
		"grep":      runGrep,
		"head":      runHead,
		"hostname":  runHostname,
		"ls":        runList,
		"md5sum":    runMD5Sum,
		"mkdir":     runMkdir,
		"mktemp":    runMktemp,
		"mv":        runMove,
		"nl":        runNumberLines,
		"printenv":  runPrintenv,
		"realpath":  runRealpath,
		"rev":       runReverse,
		"rm":        runRemove,
		"rmdir":     runRmdir,
		"sed":       runSed,
		"seq":       runSeq,
		"sh":        runBash,
		"sha1sum":   runSHA1Sum,
		"sha256sum": runSHA256Sum,
		"sleep":     runSleep,
		"sort":      runSort,
		"stat":      runStat,
		"tac":       runTac,
		"tail":      runTail,
		"tee":       runTee,
		"timeout":   runTimeout,
		"touch":     runTouch,
		"tr":        runTranslate,
		"uniq":      runUniq,
		"wc":        runWordCount,
		"which":     runWhich,
		"whoami":    runWhoami,
		"xargs":     runXargs,
		"yes":       runYes,
	}
}

// commandCall is one invocation of a sandbox command.
type commandCall struct {
	shell *shell
	name  string
	args  []string

	dir string
	env expand.Environ

	stdin  io.Reader
	stdout io.Writer
	stderr io.Writer
}

// newCommandCall describes an invocation from the interpreter's exec
// handler.
func newCommandCall(s *shell, hc interp.HandlerContext, name string, args []string) *commandCall {
	return &commandCall{
		shell:  s,
		name:   name,
		args:   args,
		dir:    resolvePath("/", hc.Dir),
		env:    hc.Env,
		stdin:  hc.Stdin,
		stdout: hc.Stdout,
		stderr: hc.Stderr,
	}
}

// fs returns the session filesystem.
func (c *commandCall) fs() *memFS {
	return c.shell.fs
}

// resolve returns the absolute virtual path of an operand.
func (c *commandCall) resolve(name string) string {
	return resolvePath(c.dir, name)
}

// report prints a diagnostic prefixed with the command name.
func (c *commandCall) report(format string, values ...any) {
	printMessage(c.stderr, c.name+": "+format+"\n", values...)
}

// fail reports a diagnostic and returns exit code 1.
func (c *commandCall) fail(format string, values ...any) int {
	c.report(format, values...)

	return exitFailure
}

// reportFile reports a failed file operation in the coreutils format.
func (c *commandCall) reportFile(operand string, err error) {
	c.report("%s: %s", operand, describeError(err))
}

// input returns the standard input, which reads as empty when the command
// has none.
func (c *commandCall) input() io.Reader {
	if c.stdin == nil {
		return strings.NewReader("")
	}

	return c.stdin
}

// readOperand reads an input operand: "-" and /dev/stdin read the standard
// input, and other names read virtual files.
func (c *commandCall) readOperand(operand string) ([]byte, error) {
	switch c.resolve(operand) {
	case "/dev/stdin":
		return io.ReadAll(c.input())
	case "/dev/null":
		return nil, nil
	}

	if operand == "-" {
		return io.ReadAll(c.input())
	}

	return c.fs().ReadFile(c.resolve(operand))
}

// readInputs reads the input operands in order, or the standard input when
// there are none, reporting unreadable files. It returns the contents that
// could be read and whether every operand was readable.
func (c *commandCall) readInputs(operands []string) ([]namedInput, bool) {
	if len(operands) == 0 {
		operands = []string{
			"-",
		}
	}

	inputs := make([]namedInput, 0, len(operands))
	ok := true

	for _, operand := range operands {
		contents, err := c.readOperand(operand)
		if err != nil {
			c.reportFile(operand, err)
			ok = false

			continue
		}

		inputs = append(inputs, namedInput{
			name:     operand,
			contents: contents,
		})
	}

	return inputs, ok
}

// namedInput is the contents of one input operand.
type namedInput struct {
	name     string
	contents []byte
}

// writeOutput writes text to stdout. Failing writes, such as to a closed
// pipe, are ignored: the command's next write or exit ends it.
func (c *commandCall) writeOutput(text string) {
	_, _ = io.WriteString(c.stdout, text)
}

// writeLines writes lines to stdout, each terminated by a newline.
func (c *commandCall) writeLines(lines []string) error {
	for _, line := range lines {
		if _, err := io.WriteString(c.stdout, line+"\n"); err != nil {
			return err
		}
	}

	return nil
}

// statusOf returns exit code 0 when ok and 1 otherwise.
func statusOf(ok bool) int {
	if ok {
		return 0
	}

	return exitFailure
}

// describeError returns the description of a filesystem error as coreutils
// prints it, such as "No such file or directory".
func describeError(err error) string {
	if pathErr, ok := errors.AsType[*fs.PathError](err); ok {
		err = pathErr.Err
	}

	return capitalize(err.Error())
}

// capitalize upper-cases the first letter of text.
func capitalize(text string) string {
	first, size := utf8.DecodeRuneInString(text)
	if size == 0 {
		return text
	}

	return string(unicode.ToUpper(first)) + text[size:]
}

// splitLines splits text into lines without their terminating newlines. A
// final newline does not start another line.
func splitLines(contents []byte) []string {
	if len(contents) == 0 {
		return nil
	}

	text := strings.TrimSuffix(string(contents), "\n")

	return strings.Split(text, "\n")
}

// parsedArgs are the flags and operands of a command line.
type parsedArgs struct {
	flags    []parsedFlag
	operands []string
}

// parsedFlag is one flag and its value.
type parsedFlag struct {
	// name is the flag as written, such as "-n" or "--lines".
	name  string
	value string
}

// argSpec describes the flags of a command.
type argSpec struct {
	// valueFlags take a value: attached ("-n5", "--lines=5") or the next
	// argument.
	valueFlags []string

	// stopAtOperand ends flag parsing at the first operand, for commands
	// that pass the remaining arguments to another command.
	stopAtOperand bool
}

// parseArgs splits a command line into flags and operands, GNU style:
// short flags cluster ("-rf"), flags may follow operands unless the spec
// stops at the first operand, "--" ends the flags, and "-" is an operand.
func parseArgs(args []string, spec argSpec) (parsedArgs, error) {
	var parsed parsedArgs

	for index := 0; index < len(args); index++ {
		arg := args[index]

		switch {
		case arg == "--":
			parsed.operands = append(parsed.operands, args[index+1:]...)

			return parsed, nil
		case arg == "-" || !strings.HasPrefix(arg, "-"):
			if spec.stopAtOperand {
				parsed.operands = append(parsed.operands, args[index:]...)

				return parsed, nil
			}

			parsed.operands = append(parsed.operands, arg)
		case strings.HasPrefix(arg, "--"):
			consumed, err := parsed.addLongFlag(arg, args[index+1:], spec)
			if err != nil {
				return parsedArgs{}, err
			}

			index += consumed
		default:
			consumed, err := parsed.addShortFlags(arg, args[index+1:], spec)
			if err != nil {
				return parsedArgs{}, err
			}

			index += consumed
		}
	}

	return parsed, nil
}

// addLongFlag adds a "--name" or "--name=value" flag and returns how many
// following arguments it consumed.
func (p *parsedArgs) addLongFlag(arg string, rest []string, spec argSpec) (int, error) {
	name, value, hasValue := strings.Cut(arg, "=")
	takesValue := slices.Contains(spec.valueFlags, name)

	switch {
	case hasValue || !takesValue:
		p.flags = append(p.flags, parsedFlag{
			name:  name,
			value: value,
		})

		return 0, nil
	case len(rest) == 0:
		return 0, fmt.Errorf("option '%s' requires an argument", name)
	default:
		p.flags = append(p.flags, parsedFlag{
			name:  name,
			value: rest[0],
		})

		return 1, nil
	}
}

// addShortFlags adds a cluster of short flags and returns how many
// following arguments its last flag consumed as a value.
func (p *parsedArgs) addShortFlags(arg string, rest []string, spec argSpec) (int, error) {
	cluster := arg[1:]

	for position, letter := range cluster {
		name := "-" + string(letter)
		if !slices.Contains(spec.valueFlags, name) {
			p.flags = append(p.flags, parsedFlag{
				name: name,
			})

			continue
		}

		attached := cluster[position+utf8.RuneLen(letter):]
		if attached != "" {
			p.flags = append(p.flags, parsedFlag{
				name:  name,
				value: attached,
			})

			return 0, nil
		}

		if len(rest) == 0 {
			return 0, fmt.Errorf("option requires an argument -- '%c'", letter)
		}

		p.flags = append(p.flags, parsedFlag{
			name:  name,
			value: rest[0],
		})

		return 1, nil
	}

	return 0, nil
}

// parseCommandArgs parses the call's arguments, reporting a usage error.
func (c *commandCall) parseCommandArgs(spec argSpec) (parsedArgs, bool) {
	parsed, err := parseArgs(c.args, spec)
	if err != nil {
		c.report("%v", err)

		return parsedArgs{}, false
	}

	return parsed, true
}

// invalidFlag reports an unsupported flag and returns exit code 1.
func (c *commandCall) invalidFlag(flag parsedFlag) int {
	if strings.HasPrefix(flag.name, "--") {
		return c.fail("unrecognized option '%s'", flag.name)
	}

	return c.fail("invalid option -- '%s'", strings.TrimPrefix(flag.name, "-"))
}
