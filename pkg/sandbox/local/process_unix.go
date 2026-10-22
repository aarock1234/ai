//go:build unix

package local

import (
	"errors"
	"os/exec"
	"syscall"
)

// configureProcessGroup starts the command in its own process group, so
// that a kill reaches the commands the shell started too.
func configureProcessGroup(command *exec.Cmd) {
	command.SysProcAttr = &syscall.SysProcAttr{
		Setpgid: true,
	}
}

// killProcessGroup kills the command's process group.
func killProcessGroup(command *exec.Cmd) error {
	if command.Process == nil {
		return nil
	}

	err := syscall.Kill(-command.Process.Pid, syscall.SIGKILL)
	if errors.Is(err, syscall.ESRCH) {
		return nil
	}

	return err
}
