package mcp

import (
	"os/exec"
	"strings"

	"github.com/aarock1234/ai/internal/shellquote"
)

// serverCommand returns the command that starts the server: through a shell
// when the options ask for one, and directly otherwise.
func (t *StdioTransport) serverCommand() (*exec.Cmd, error) {
	if !t.options.usesShell() {
		return newServerCommand(t.options.Command, t.options.Args)
	}

	return newShellCommand(t.options.ShellPath, t.options.Command, t.options.Args), nil
}

// posixShellCommand runs command with shell -c, appending each argument in
// single quotes.
func posixShellCommand(shell, command string, args []string) *exec.Cmd {
	parts := make([]string, 0, len(args)+1)
	parts = append(parts, command)

	for _, arg := range args {
		parts = append(parts, shellquote.Quote(arg))
	}

	return exec.Command(shell, "-c", strings.Join(parts, " "))
}
