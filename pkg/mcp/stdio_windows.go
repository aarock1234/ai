//go:build windows

package mcp

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"syscall"
)

// defaultInheritedEnvironment lists the variables the server inherits from
// this process.
var defaultInheritedEnvironment = []string{
	"APPDATA",
	"HOMEDRIVE",
	"HOMEPATH",
	"LOCALAPPDATA",
	"PATH",
	"PROCESSOR_ARCHITECTURE",
	"SYSTEMDRIVE",
	"SYSTEMROOT",
	"TEMP",
	"USERNAME",
	"USERPROFILE",
}

var (
	// executablePattern matches files that start without a shell.
	executablePattern = regexp.MustCompile(`(?i)\.(?:com|exe)$`)

	// commandShimPattern matches npm command shims, whose arguments cmd.exe
	// expands twice.
	commandShimPattern = regexp.MustCompile(`(?i)node_modules[\\/]\.bin[\\/][^\\/]+\.cmd$`)

	// cmdShellPattern matches cmd.exe, which takes /d /s /c instead of -c.
	cmdShellPattern = regexp.MustCompile(`(?i)^(?:.*[\\/])?cmd(?:\.exe)?$`)

	// shellMetaCharacters are the characters cmd.exe interprets.
	shellMetaCharacters = regexp.MustCompile("([()\\][%!^\"`<>&|;, *?])")
)

// environmentKey returns the key used to match environment variables, which
// are case-insensitive on Windows.
func environmentKey(key string) string {
	return strings.ToUpper(key)
}

// validateCommandLine rejects line breaks, which cmd.exe would treat as
// command separators.
func validateCommandLine(command string, args []string) error {
	for _, value := range append([]string{command}, args...) {
		if strings.ContainsAny(value, "\r\n") {
			return &ClientError{
				Message: "stdio mcp commands and arguments must not contain line breaks on windows",
			}
		}
	}

	return nil
}

// newServerCommand runs executables directly and other commands, such as
// the npx.cmd batch shim, through cmd.exe with every argument escaped, so
// that arguments cannot inject shell commands.
func newServerCommand(command string, args []string) (*exec.Cmd, error) {
	resolved, err := exec.LookPath(command)
	if err != nil {
		return nil, err
	}

	if executablePattern.MatchString(resolved) {
		return exec.Command(resolved, args...), nil
	}

	doubleEscape := commandShimPattern.MatchString(resolved)

	parts := make([]string, 0, len(args)+1)
	parts = append(parts, escapeShellCommand(filepath.Clean(command)))

	for _, arg := range args {
		parts = append(parts, escapeShellArgument(arg, doubleEscape))
	}

	return cmdShellCommand(defaultShell(), strings.Join(parts, " ")), nil
}

// defaultShell returns COMSPEC, or cmd.exe when it is unset.
func defaultShell() string {
	shell := os.Getenv("COMSPEC")
	if shell == "" {
		return "cmd.exe"
	}

	return shell
}

// newShellCommand runs command, which is shell syntax, through shell with
// every argument quoted. cmd.exe receives Node.js's /d /s /c command line
// with its meta characters escaped in the arguments; other shells, such as
// Git Bash, receive -c and POSIX quoting.
func newShellCommand(shell, command string, args []string) *exec.Cmd {
	if shell == "" {
		shell = defaultShell()
	}

	if !cmdShellPattern.MatchString(shell) {
		return posixShellCommand(shell, command, args)
	}

	parts := make([]string, 0, len(args)+1)
	parts = append(parts, command)

	for _, arg := range args {
		parts = append(parts, escapeShellArgument(arg, false))
	}

	return cmdShellCommand(shell, strings.Join(parts, " "))
}

// cmdShellCommand runs line with cmd.exe, passing the command line verbatim
// so that Go does not quote it a second time.
func cmdShellCommand(shell, line string) *exec.Cmd {
	command := exec.Command(shell)
	command.SysProcAttr = &syscall.SysProcAttr{
		CmdLine: syscall.EscapeArg(shell) + ` /d /s /c "` + line + `"`,
	}

	return command
}

func escapeShellCommand(command string) string {
	return shellMetaCharacters.ReplaceAllString(command, "^$1")
}

// escapeShellArgument quotes arg for the program's argument parser and then
// escapes cmd.exe's meta characters.
func escapeShellArgument(arg string, doubleEscape bool) string {
	escaped := shellMetaCharacters.ReplaceAllString(quoteArgument(arg), "^$1")

	if doubleEscape {
		escaped = shellMetaCharacters.ReplaceAllString(escaped, "^$1")
	}

	return escaped
}

// quoteArgument wraps arg in double quotes, doubling the backslashes that
// precede a quote or the end and escaping quotes.
func quoteArgument(arg string) string {
	var builder strings.Builder

	builder.WriteByte('"')

	backslashes := 0
	for _, character := range arg {
		switch character {
		case '\\':
			backslashes++

			continue
		case '"':
			builder.WriteString(strings.Repeat(`\`, 2*backslashes+1))
		default:
			builder.WriteString(strings.Repeat(`\`, backslashes))
		}

		backslashes = 0
		builder.WriteRune(character)
	}

	builder.WriteString(strings.Repeat(`\`, 2*backslashes))
	builder.WriteByte('"')

	return builder.String()
}

// terminateProcess stops the process. Windows has no termination signal, so
// it is killed.
func terminateProcess(process *os.Process) error {
	return process.Kill()
}
