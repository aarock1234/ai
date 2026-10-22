package credential

import (
	"bytes"
	"context"
	_ "embed" // embeds the Windows Credential Manager source
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"strings"
)

// macOSSecurityTool is the macOS keychain command-line tool.
const macOSSecurityTool = "/usr/bin/security"

// ReadMacOSKeychainPassword reads a generic password from the macOS keychain
// with the security tool. It reports false when the item is missing or the
// tool fails.
func ReadMacOSKeychainPassword(ctx context.Context, service string, account string) (string, bool) {
	output, err := exec.CommandContext(ctx, macOSSecurityTool, "find-generic-password", "-s", service, "-a", account, "-w").Output()
	if err != nil {
		return "", false
	}

	password := strings.TrimSpace(string(output))

	return password, password != ""
}

// MacOSKeychainEntry is a generic password of the macOS keychain.
type MacOSKeychainEntry struct {
	Service  string
	Account  string
	Password string
}

// WriteMacOSKeychainPassword creates or replaces a generic password in the
// macOS keychain. The password is sent hex-encoded on the standard input of
// the security tool's interactive mode, which keeps it out of the process
// arguments.
func WriteMacOSKeychainPassword(ctx context.Context, entry MacOSKeychainEntry) error {
	command := fmt.Sprintf(
		"add-generic-password -U -s %s -a %s -X %s\n",
		quoteForSecurity(entry.Service),
		quoteForSecurity(entry.Account),
		hex.EncodeToString([]byte(entry.Password)),
	)

	return runWithInput(ctx, commandInput{
		name: macOSSecurityTool,
		args: []string{
			"-i",
		},
		input: command,
	})
}

// quoteForSecurity quotes a value for the security tool's interactive mode.
func quoteForSecurity(value string) string {
	escaped := strings.ReplaceAll(value, `\`, `\\`)
	escaped = strings.ReplaceAll(escaped, `"`, `\"`)

	return `"` + escaped + `"`
}

// SecretAttribute is one attribute of a Secret Service item, in order.
type SecretAttribute struct {
	Name  string
	Value string
}

// ReadLinuxSecretServicePassword looks a password up in the Secret Service
// with secret-tool, matching every attribute. It reports false when the
// item is missing or the tool fails.
func ReadLinuxSecretServicePassword(ctx context.Context, attributes []SecretAttribute) (string, bool) {
	arguments := []string{
		"lookup",
	}

	arguments = appendAttributes(arguments, attributes)

	output, err := exec.CommandContext(ctx, "secret-tool", arguments...).Output()
	if err != nil || len(output) == 0 {
		return "", false
	}

	return string(output), true
}

// SecretServiceEntry is a password of the Secret Service.
type SecretServiceEntry struct {
	Label      string
	Attributes []SecretAttribute
	Password   string
}

// WriteLinuxSecretServicePassword creates or replaces a password in the
// Secret Service with secret-tool. The password is sent on standard input.
func WriteLinuxSecretServicePassword(ctx context.Context, entry SecretServiceEntry) error {
	arguments := []string{
		"store",
		"--label",
		entry.Label,
	}

	arguments = appendAttributes(arguments, entry.Attributes)

	return runWithInput(ctx, commandInput{
		name:  "secret-tool",
		args:  arguments,
		input: entry.Password,
	})
}

// appendAttributes appends the name and value of each attribute.
func appendAttributes(arguments []string, attributes []SecretAttribute) []string {
	for _, attribute := range attributes {
		arguments = append(arguments, attribute.Name, attribute.Value)
	}

	return arguments
}

// windowsCredentialManagerSource reads and writes generic credentials
// through the Win32 CredRead and CredWrite APIs from PowerShell.
//
//go:embed windows_credential_manager.cs
var windowsCredentialManagerSource string

// Environment variables that pass the source and the credential to
// PowerShell, so that neither appears in the process arguments.
const (
	windowsSourceVariable   = "AI_SDK_WINDOWS_CREDENTIAL_MANAGER_SOURCE"
	windowsTargetVariable   = "AI_SDK_WINDOWS_CREDENTIAL_MANAGER_TARGET"
	windowsUserNameVariable = "AI_SDK_WINDOWS_CREDENTIAL_MANAGER_USER"
)

// windowsReadCommand loads the credential manager and prints the password.
const windowsReadCommand = `Add-Type -TypeDefinition $env:AI_SDK_WINDOWS_CREDENTIAL_MANAGER_SOURCE; ` +
	`$value = [AISDKCredentialManager]::Read($env:AI_SDK_WINDOWS_CREDENTIAL_MANAGER_TARGET); ` +
	`if ($null -ne $value) { [Console]::Out.Write($value) }`

// windowsWriteCommand loads the credential manager and stores standard
// input.
const windowsWriteCommand = `Add-Type -TypeDefinition $env:AI_SDK_WINDOWS_CREDENTIAL_MANAGER_SOURCE; ` +
	`$value = [Console]::In.ReadToEnd(); ` +
	`[AISDKCredentialManager]::Write($env:AI_SDK_WINDOWS_CREDENTIAL_MANAGER_TARGET, $env:AI_SDK_WINDOWS_CREDENTIAL_MANAGER_USER, $value)`

// ReadWindowsCredentialManagerPassword reads a generic credential from the
// Windows Credential Manager through PowerShell. It reports false when the
// credential is missing or PowerShell fails.
func ReadWindowsCredentialManagerPassword(ctx context.Context, targetName string) (string, bool) {
	command := exec.CommandContext(ctx, "powershell.exe", powerShellArguments(windowsReadCommand)...)
	command.Env = append(
		os.Environ(),
		windowsSourceVariable+"="+windowsCredentialManagerSource,
		windowsTargetVariable+"="+targetName,
	)

	output, err := command.Output()
	if err != nil || len(output) == 0 {
		return "", false
	}

	return string(output), true
}

// WindowsCredentialEntry is a generic credential of the Windows Credential
// Manager.
type WindowsCredentialEntry struct {
	TargetName string
	UserName   string
	Password   string
}

// WriteWindowsCredentialManagerPassword creates or replaces a generic
// credential in the Windows Credential Manager through PowerShell. The
// password is sent on standard input.
func WriteWindowsCredentialManagerPassword(ctx context.Context, entry WindowsCredentialEntry) error {
	return runWithInput(ctx, commandInput{
		name:  "powershell.exe",
		args:  powerShellArguments(windowsWriteCommand),
		input: entry.Password,
		env: append(
			os.Environ(),
			windowsSourceVariable+"="+windowsCredentialManagerSource,
			windowsTargetVariable+"="+entry.TargetName,
			windowsUserNameVariable+"="+entry.UserName,
		),
	})
}

// powerShellArguments returns the arguments that run a PowerShell command
// without a profile or prompts.
func powerShellArguments(command string) []string {
	return []string{
		"-NoProfile",
		"-NonInteractive",
		"-Command",
		command,
	}
}

// commandInput is a command that reads its input from standard input.
type commandInput struct {
	name  string
	args  []string
	input string

	// env replaces the process environment when set.
	env []string
}

// runWithInput runs a command with input on standard input. A failure
// wraps the command's error with its standard error.
func runWithInput(ctx context.Context, command commandInput) error {
	process := exec.CommandContext(ctx, command.name, command.args...)
	process.Stdin = strings.NewReader(command.input)
	process.Env = command.env

	var stderr bytes.Buffer
	process.Stderr = &stderr

	err := process.Run()
	if err == nil {
		return nil
	}

	message := strings.TrimSpace(stderr.String())
	if message == "" {
		return fmt.Errorf("%s failed: %w", command.name, err)
	}

	return fmt.Errorf("%s failed: %s: %w", command.name, message, err)
}
