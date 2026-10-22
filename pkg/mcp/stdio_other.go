//go:build !windows

package mcp

import (
	"os"
	"os/exec"
	"syscall"
)

// defaultInheritedEnvironment lists the variables the server inherits from
// this process.
var defaultInheritedEnvironment = []string{
	"HOME",
	"LOGNAME",
	"PATH",
	"SHELL",
	"TERM",
	"USER",
}

// environmentKey returns the key used to match environment variables, which
// are case-sensitive outside Windows.
func environmentKey(key string) string {
	return key
}

func validateCommandLine(string, []string) error {
	return nil
}

func newServerCommand(command string, args []string) (*exec.Cmd, error) {
	return exec.Command(command, args...), nil
}

// newShellCommand runs command, which is shell syntax, through shell with
// every argument quoted. An empty shell uses /bin/sh.
func newShellCommand(shell, command string, args []string) *exec.Cmd {
	if shell == "" {
		shell = "/bin/sh"
	}

	return posixShellCommand(shell, command, args)
}

// terminateProcess asks the process to exit with SIGTERM.
func terminateProcess(process *os.Process) error {
	return process.Signal(syscall.SIGTERM)
}
