//go:build !unix

package local

import (
	"errors"
	"os"
	"os/exec"
)

// configureProcessGroup does nothing: platforms without process groups
// cannot kill the commands the shell started, only the shell.
func configureProcessGroup(*exec.Cmd) {}

// killProcessGroup kills the command's process.
func killProcessGroup(command *exec.Cmd) error {
	if command.Process == nil {
		return nil
	}

	err := command.Process.Kill()
	if errors.Is(err, os.ErrProcessDone) {
		return nil
	}

	return err
}
