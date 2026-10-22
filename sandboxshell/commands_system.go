package sandboxshell

import (
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"slices"
	"strconv"
	"strings"
	"time"
)

// runBash carries out "bash [options] -c script [name [arg...]]", "bash
// file [arg...]" and "bash" reading the script from the standard input, in
// a child shell of the sandbox.
func runBash(ctx context.Context, call *commandCall) int {
	invocation, ok := parseBashArgs(call)
	if !ok {
		return exitUsage
	}

	run := script{
		name:    call.name,
		dir:     call.dir,
		env:     childEnviron(call.env),
		options: invocation.options,
		stdin:   call.stdin,
		stdout:  call.stdout,
		stderr:  call.stderr,
	}

	switch {
	case invocation.command:
		run.source = invocation.operands[0]
		if len(invocation.operands) > 1 {
			run.name = invocation.operands[1]
			run.params = invocation.operands[2:]
		}
	case len(invocation.operands) > 0:
		contents, err := call.fs().ReadFile(call.resolve(invocation.operands[0]))
		if err != nil {
			call.reportFile(invocation.operands[0], err)

			return exitNotFound
		}

		run.name = invocation.operands[0]
		run.source = string(contents)
		run.params = invocation.operands[1:]
	default:
		contents, err := io.ReadAll(call.input())
		if err != nil {
			return call.fail("%v", err)
		}

		run.source = string(contents)
		run.stdin = nil
	}

	return call.shell.run(ctx, run)
}

// bashInvocation is the parsed command line of bash.
type bashInvocation struct {
	command  bool
	options  []string
	operands []string
}

// parseBashArgs reads bash's options. Shell options such as -e, -u, -x and
// -o pipefail are passed to the interpreter; login and startup file
// options are ignored, because the sandbox has no startup files.
func parseBashArgs(call *commandCall) (bashInvocation, bool) {
	var invocation bashInvocation
	args := call.args

	for len(args) > 0 {
		arg := args[0]

		if arg == "--" {
			args = args[1:]

			break
		}

		if isIgnoredBashOption(arg) {
			args = args[1:]

			continue
		}

		if len(arg) < 2 || arg[0] != '-' && arg[0] != '+' || strings.HasPrefix(arg, "--") {
			break
		}

		args = args[1:]
		for _, letter := range arg[1:] {
			switch letter {
			case 'c':
				invocation.command = true
			case 'l', 'i', 's':
			case 'o':
				if len(args) == 0 {
					call.report("%c: option requires an argument", letter)

					return bashInvocation{}, false
				}

				invocation.options = append(invocation.options, arg[:1]+"o", args[0])
				args = args[1:]
			default:
				invocation.options = append(invocation.options, arg[:1]+string(letter))
			}
		}
	}

	if invocation.command && len(args) == 0 {
		call.report("-c: option requires an argument")

		return bashInvocation{}, false
	}

	invocation.operands = args

	return invocation, true
}

// isIgnoredBashOption reports whether a long option of bash has no effect
// in the sandbox.
func isIgnoredBashOption(arg string) bool {
	switch arg {
	case "--login", "--norc", "--noprofile", "--posix", "--noediting":
		return true
	default:
		return false
	}
}

// runEnv carries out "env [-i] [-u name] [name=value...] [command
// [arg...]]".
func runEnv(ctx context.Context, call *commandCall) int {
	parsed, ok := call.parseCommandArgs(argSpec{
		valueFlags: []string{
			"-u",
			"--unset",
		},
		stopAtOperand: true,
	})
	if !ok {
		return exitCannotExecute
	}

	variables := exportedVariables(call.env)
	for _, flag := range parsed.flags {
		switch flag.name {
		case "-i", "--ignore-environment":
			variables = make(map[string]string)
		case "-u", "--unset":
			delete(variables, flag.value)
		default:
			call.invalidFlag(flag)

			return exitCannotExecute
		}
	}

	operands := parsed.operands
	for len(operands) > 0 && strings.Contains(operands[0], "=") {
		name, value, _ := strings.Cut(operands[0], "=")
		variables[name] = value
		operands = operands[1:]
	}

	if len(operands) == 0 {
		printVariables(call, variables)

		return 0
	}

	child := *call
	child.env = newShellEnviron(variables)

	return call.shell.runArgs(ctx, &child, operands)
}

// runPrintenv carries out "printenv [name...]".
func runPrintenv(_ context.Context, call *commandCall) int {
	variables := exportedVariables(call.env)
	if len(call.args) == 0 {
		printVariables(call, variables)

		return 0
	}

	ok := true
	for _, name := range call.args {
		value, found := variables[name]
		if !found {
			ok = false

			continue
		}

		call.writeOutput(value + "\n")
	}

	return statusOf(ok)
}

// printVariables prints variables as name=value lines sorted by name.
func printVariables(call *commandCall, variables map[string]string) {
	for _, name := range slices.Sorted(maps.Keys(variables)) {
		call.writeOutput(name + "=" + variables[name] + "\n")
	}
}

// runDate carries out "date [-u] [+format]". Times are always reported in
// UTC: loading another time zone would read the host's zone database.
func runDate(_ context.Context, call *commandCall) int {
	format := "%a %b %e %H:%M:%S %Z %Y"
	moment := time.Now().UTC()

	args := call.args
	for len(args) > 0 {
		arg := args[0]
		args = args[1:]

		switch {
		case arg == "-u" || arg == "--utc" || arg == "--universal":
		case strings.HasPrefix(arg, "+"):
			format = arg[1:]
		case arg == "-d" || arg == "--date":
			if len(args) == 0 {
				return call.fail("option requires an argument -- 'd'")
			}

			parsed, err := parseDate(args[0])
			if err != nil {
				return call.fail("invalid date '%s'", args[0])
			}

			moment = parsed
			args = args[1:]
		case arg == "-I" || arg == "--iso-8601":
			format = "%Y-%m-%d"
		case arg == "-R" || arg == "--rfc-email":
			format = "%a, %d %b %Y %H:%M:%S %z"
		default:
			return call.fail("invalid option '%s'", arg)
		}
	}

	call.writeOutput(formatDate(moment, format) + "\n")

	return 0
}

// parseDate parses the date of date -d: "@seconds" or an RFC 3339 or ISO
// date.
func parseDate(text string) (time.Time, error) {
	if seconds, ok := strings.CutPrefix(text, "@"); ok {
		value, err := strconv.ParseInt(seconds, 10, 64)
		if err != nil {
			return time.Time{}, err
		}

		return time.Unix(value, 0).UTC(), nil
	}

	for _, layout := range []string{time.RFC3339, "2006-01-02 15:04:05", "2006-01-02T15:04:05", "2006-01-02"} {
		if parsed, err := time.Parse(layout, text); err == nil {
			return parsed.UTC(), nil
		}
	}

	return time.Time{}, errors.New("unrecognized date")
}

// formatDate expands the strftime directives date supports.
func formatDate(moment time.Time, format string) string {
	var output strings.Builder

	for index := 0; index < len(format); index++ {
		if format[index] != '%' || index+1 == len(format) {
			output.WriteByte(format[index])

			continue
		}

		index++
		output.WriteString(dateDirective(moment, format[index]))
	}

	return output.String()
}

// dateDirective expands one strftime directive.
func dateDirective(moment time.Time, directive byte) string {
	layouts := map[byte]string{
		'Y': "2006",
		'y': "06",
		'm': "01",
		'd': "02",
		'e': "_2",
		'H': "15",
		'I': "03",
		'M': "04",
		'S': "05",
		'p': "PM",
		'a': "Mon",
		'A': "Monday",
		'b': "Jan",
		'h': "Jan",
		'B': "January",
		'Z': "MST",
		'z': "-0700",
		'F': "2006-01-02",
		'T': "15:04:05",
		'D': "01/02/06",
		'R': "15:04",
	}

	if layout, ok := layouts[directive]; ok {
		return moment.Format(layout)
	}

	switch directive {
	case 's':
		return strconv.FormatInt(moment.Unix(), 10)
	case 'N':
		return fmt.Sprintf("%09d", moment.Nanosecond())
	case 'j':
		return fmt.Sprintf("%03d", moment.YearDay())
	case 'u':
		return strconv.Itoa((int(moment.Weekday())+6)%7 + 1)
	case 'w':
		return strconv.Itoa(int(moment.Weekday()))
	case 'n':
		return "\n"
	case 't':
		return "\t"
	case '%':
		return "%"
	default:
		return "%" + string(directive)
	}
}

// runSleep carries out "sleep duration...", where durations are seconds
// with an optional s, m, h or d suffix.
func runSleep(ctx context.Context, call *commandCall) int {
	if len(call.args) == 0 {
		return call.fail("missing operand")
	}

	var total time.Duration
	for _, arg := range call.args {
		duration, err := parseShellDuration(arg)
		if err != nil {
			return call.fail("invalid time interval '%s'", arg)
		}

		total += duration
	}

	timer := time.NewTimer(total)
	defer timer.Stop()

	select {
	case <-timer.C:
		return 0
	case <-ctx.Done():
		return exitInterrupted
	}
}

// parseShellDuration parses a sleep or timeout duration such as "1.5",
// "2m" or "1h".
func parseShellDuration(text string) (time.Duration, error) {
	units := map[byte]time.Duration{
		's': time.Second,
		'm': time.Minute,
		'h': time.Hour,
		'd': 24 * time.Hour,
	}

	unit := time.Second
	if text != "" {
		if suffixUnit, ok := units[text[len(text)-1]]; ok {
			unit = suffixUnit
			text = text[:len(text)-1]
		}
	}

	value, err := strconv.ParseFloat(text, 64)
	if err != nil || value < 0 {
		return 0, errors.New("invalid duration")
	}

	return time.Duration(value * float64(unit)), nil
}

// runSeq carries out "seq [-s separator] [-w] [first [increment]] last".
func runSeq(ctx context.Context, call *commandCall) int {
	separator := "\n"
	equalWidth := false
	args := call.args

	for len(args) > 0 && strings.HasPrefix(args[0], "-") && !isNumber(args[0]) {
		switch {
		case args[0] == "-w":
			equalWidth = true
		case args[0] == "-s" && len(args) > 1:
			separator = unescapeBackslashes(args[1])
			args = args[1:]
		case strings.HasPrefix(args[0], "-s"):
			separator = unescapeBackslashes(args[0][2:])
		default:
			return call.fail("invalid option '%s'", args[0])
		}

		args = args[1:]
	}

	sequence, err := parseSequence(args)
	if err != nil {
		return call.fail("%v", err)
	}

	return call.printSequence(ctx, sequence, separator, equalWidth)
}

// sequence is the range seq prints.
type sequence struct {
	first     float64
	increment float64
	last      float64
	decimals  int
}

// parseSequence parses seq's operands.
func parseSequence(args []string) (sequence, error) {
	if len(args) == 0 || len(args) > 3 {
		return sequence{}, errors.New("missing operand")
	}

	values := make([]float64, len(args))
	decimals := 0

	for index, arg := range args {
		value, err := strconv.ParseFloat(arg, 64)
		if err != nil {
			return sequence{}, fmt.Errorf("invalid floating point argument: '%s'", arg)
		}

		values[index] = value
		if _, fraction, found := strings.Cut(arg, "."); found {
			decimals = max(decimals, len(fraction))
		}
	}

	result := sequence{
		first:     1,
		increment: 1,
		last:      values[len(values)-1],
		decimals:  decimals,
	}

	if len(values) > 1 {
		result.first = values[0]
	}

	if len(values) == 3 {
		result.increment = values[1]
	}

	if result.increment == 0 {
		return sequence{}, errors.New("invalid Zero increment value: '0'")
	}

	return result, nil
}

// printSequence prints the numbers of a sequence.
func (c *commandCall) printSequence(ctx context.Context, numbers sequence, separator string, equalWidth bool) int {
	width := 0
	if equalWidth {
		width = max(len(formatSequenceNumber(numbers.first, numbers.decimals)), len(formatSequenceNumber(numbers.last, numbers.decimals)))
	}

	first := true
	for step := 0; ; step++ {
		value := numbers.first + float64(step)*numbers.increment
		if numbers.increment > 0 && value > numbers.last || numbers.increment < 0 && value < numbers.last {
			break
		}

		if ctx.Err() != nil {
			return exitInterrupted
		}

		if !first {
			c.writeOutput(separator)
		}

		text := formatSequenceNumber(value, numbers.decimals)
		if _, err := fmt.Fprintf(c.stdout, "%0*s", width, text); err != nil {
			return exitFailure
		}

		first = false
	}

	if !first {
		c.writeOutput("\n")
	}

	return 0
}

// formatSequenceNumber formats a number with the operands' precision.
func formatSequenceNumber(value float64, decimals int) string {
	return strconv.FormatFloat(value, 'f', decimals, 64)
}

// isNumber reports whether text parses as a number, such as "-1".
func isNumber(text string) bool {
	_, err := strconv.ParseFloat(text, 64)

	return err == nil
}

// runYes carries out "yes [text...]" until its output closes or the
// command is cancelled.
func runYes(ctx context.Context, call *commandCall) int {
	line := "y\n"
	if len(call.args) > 0 {
		line = strings.Join(call.args, " ") + "\n"
	}

	for ctx.Err() == nil {
		if _, err := io.WriteString(call.stdout, line); err != nil {
			return 0
		}
	}

	return exitInterrupted
}

// runWhoami carries out "whoami".
func runWhoami(_ context.Context, call *commandCall) int {
	call.writeOutput(virtualUser + "\n")

	return 0
}

// runHostname carries out "hostname".
func runHostname(_ context.Context, call *commandCall) int {
	call.writeOutput(virtualHostname + "\n")

	return 0
}

// runClear carries out "clear".
func runClear(_ context.Context, call *commandCall) int {
	call.writeOutput("\x1b[H\x1b[2J")

	return 0
}

// runWhich carries out "which [-a] name...".
func runWhich(_ context.Context, call *commandCall) int {
	ok := true

	for _, name := range call.args {
		if name == "-a" || name == "-s" {
			continue
		}

		resolved, found := call.shell.resolveCommand(call.env, call.dir, name)
		if !found {
			ok = false

			continue
		}

		call.writeOutput(resolved.path + "\n")
	}

	return statusOf(ok)
}

// runTimeout carries out "timeout duration command [arg...]", which exits
// with 124 when the command runs out of time.
func runTimeout(ctx context.Context, call *commandCall) int {
	parsed, ok := call.parseCommandArgs(argSpec{
		valueFlags: []string{
			"-s",
			"-k",
			"--signal",
			"--kill-after",
		},
		stopAtOperand: true,
	})
	if !ok {
		return exitTimedOut + 1
	}

	if len(parsed.operands) < 2 {
		call.report("missing operand")

		return exitTimedOut + 1
	}

	limit, err := parseShellDuration(parsed.operands[0])
	if err != nil {
		call.report("invalid time interval '%s'", parsed.operands[0])

		return exitTimedOut + 1
	}

	limited, cancel := context.WithTimeout(ctx, limit)
	defer cancel()

	code := call.shell.runArgs(limited, call, parsed.operands[1:])
	if errors.Is(limited.Err(), context.DeadlineExceeded) && ctx.Err() == nil {
		return exitTimedOut
	}

	return code
}

// runXargs carries out "xargs [-0] [-d delimiter] [-n count] [-I
// replace] [-r] [command [arg...]]".
func runXargs(ctx context.Context, call *commandCall) int {
	parsed, ok := call.parseCommandArgs(argSpec{
		valueFlags: []string{
			"-d",
			"-n",
			"-I",
			"-L",
			"-P",
			"--delimiter",
			"--max-args",
		},
		stopAtOperand: true,
	})
	if !ok {
		return exitUsage
	}

	options, ok := parseXargsOptions(call, parsed.flags)
	if !ok {
		return exitUsage
	}

	command := parsed.operands
	if len(command) == 0 {
		command = []string{
			"echo",
		}
	}

	input, err := io.ReadAll(call.input())
	if err != nil {
		return call.fail("%v", err)
	}

	items := splitXargsInput(string(input), options)
	if len(items) == 0 && options.noRunIfEmpty {
		return 0
	}

	return call.runXargsBatches(ctx, command, items, options)
}

// xargsOptions are the flags of xargs.
type xargsOptions struct {
	delimiter    string
	nulSeparated bool
	maxArgs      int
	replace      string
	noRunIfEmpty bool
}

// parseXargsOptions reads the flags of xargs.
func parseXargsOptions(call *commandCall, flags []parsedFlag) (xargsOptions, bool) {
	var options xargsOptions

	for _, flag := range flags {
		switch flag.name {
		case "-0", "--null":
			options.nulSeparated = true
		case "-d", "--delimiter":
			options.delimiter = unescapeBackslashes(flag.value)
		case "-n", "--max-args", "-L":
			count, err := strconv.Atoi(flag.value)
			if err != nil || count < 1 {
				call.report("invalid number for -n option: '%s'", flag.value)

				return xargsOptions{}, false
			}

			options.maxArgs = count
		case "-I":
			options.replace = flag.value
			options.maxArgs = 1
		case "-r", "--no-run-if-empty":
			options.noRunIfEmpty = true
		case "-P", "-t", "--verbose", "-x":
		default:
			call.invalidFlag(flag)

			return xargsOptions{}, false
		}
	}

	return options, true
}

// splitXargsInput splits the standard input into arguments: at NUL bytes,
// at a delimiter, at lines for -I, or at blanks.
func splitXargsInput(input string, options xargsOptions) []string {
	var items []string

	switch {
	case options.nulSeparated:
		items = strings.Split(input, "\x00")
	case options.delimiter != "":
		items = strings.Split(input, options.delimiter)
	case options.replace != "":
		items = splitLines([]byte(input))
	default:
		return strings.Fields(input)
	}

	return slices.DeleteFunc(items, func(item string) bool {
		return item == ""
	})
}

// runXargsBatches runs the command once per batch of arguments, and
// returns 123 when any run failed, as xargs does.
func (c *commandCall) runXargsBatches(ctx context.Context, command []string, items []string, options xargsOptions) int {
	batchSize := len(items)
	if options.maxArgs > 0 {
		batchSize = options.maxArgs
	}

	const exitSomeFailed = 123

	// The commands do not inherit the standard input xargs consumed.
	child := *c
	child.stdin = nil

	exitCode := 0
	for start := 0; start < len(items) || start == 0; start += max(batchSize, 1) {
		if ctx.Err() != nil {
			return exitInterrupted
		}

		batch := items[start:min(start+batchSize, len(items))]

		if c.shell.runArgs(ctx, &child, xargsCommand(command, batch, options.replace)) != 0 {
			exitCode = exitSomeFailed
		}

		if len(items) == 0 {
			break
		}
	}

	return exitCode
}

// xargsCommand builds one command line: arguments replace the -I string,
// or follow the command.
func xargsCommand(command []string, batch []string, replace string) []string {
	if replace == "" {
		return slices.Concat(command, batch)
	}

	replaced := make([]string, 0, len(command))
	for _, arg := range command {
		replaced = append(replaced, strings.ReplaceAll(arg, replace, strings.Join(batch, " ")))
	}

	return replaced
}
