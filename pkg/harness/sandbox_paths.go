package harness

import (
	"context"
	"fmt"
	"path"
	"strings"

	"github.com/aarock1234/ai/pkg/sandbox"
)

// ResolveSandboxHomeDir returns the sandbox's HOME directory.
func ResolveSandboxHomeDir(ctx context.Context, session sandbox.Session) (string, error) {
	result, err := session.Run(ctx, sandbox.ProcessOptions{
		Command: `printf "%s" "$HOME"`,
	})
	if err != nil {
		return "", fmt.Errorf("resolve sandbox home directory: %w", err)
	}

	homeDir := strings.TrimSpace(result.Stdout)
	if result.ExitCode != 0 || homeDir == "" || !path.IsAbs(homeDir) {
		return "", fmt.Errorf("resolve sandbox home directory: %s", commandOutput(result))
	}

	return homeDir, nil
}

// ResolveSandboxDefaultWorkingDirectory returns the directory relative
// commands run in: NetworkSession.DefaultWorkingDirectory for network
// sessions, and the output of pwd otherwise.
func ResolveSandboxDefaultWorkingDirectory(ctx context.Context, session sandbox.Session) (string, error) {
	if network, ok := session.(NetworkSession); ok {
		return network.DefaultWorkingDirectory(), nil
	}

	result, err := session.Run(ctx, sandbox.ProcessOptions{
		Command: "pwd",
	})
	if err != nil {
		return "", fmt.Errorf("resolve sandbox default working directory: %w", err)
	}

	if result.ExitCode != 0 {
		return "", fmt.Errorf(
			"resolve sandbox default working directory (exit %d): %s",
			result.ExitCode,
			commandOutput(result),
		)
	}

	directory := strings.TrimSpace(result.Stdout)
	if !path.IsAbs(directory) {
		return "", fmt.Errorf("resolve sandbox default working directory: expected an absolute path, got %q", directory)
	}

	if directory == "/" {
		return directory, nil
	}

	return strings.TrimRight(directory, "/"), nil
}

// ensureSandboxDirectory creates a directory in the sandbox.
func ensureSandboxDirectory(ctx context.Context, session sandbox.Session, directory string) error {
	result, err := session.Run(ctx, sandbox.ProcessOptions{
		Command: `mkdir -p "$WORK_DIR"`,
		Env: map[string]string{
			"WORK_DIR": directory,
		},
	})
	if err != nil {
		return fmt.Errorf("create sandbox directory %s: %w", directory, err)
	}

	if result.ExitCode != 0 {
		return fmt.Errorf(
			"create sandbox directory %s (exit %d): %s",
			directory,
			result.ExitCode,
			commandOutput(result),
		)
	}

	return nil
}

// runSandboxCommand runs a command and reports a non-zero exit as an error.
func runSandboxCommand(ctx context.Context, session sandbox.Session, command string, failure string) error {
	result, err := session.Run(ctx, sandbox.ProcessOptions{
		Command: command,
	})
	if err != nil {
		return fmt.Errorf("%s: %w", failure, err)
	}

	if result.ExitCode == 0 {
		return nil
	}

	if result.Stderr == "" {
		return fmt.Errorf("%s (exit %d)", failure, result.ExitCode)
	}

	return fmt.Errorf("%s (exit %d): %s", failure, result.ExitCode, result.Stderr)
}

// commandOutput returns the stderr of a command, or its stdout when stderr
// is empty.
func commandOutput(result sandbox.RunResult) string {
	if result.Stderr != "" {
		return result.Stderr
	}

	return result.Stdout
}
