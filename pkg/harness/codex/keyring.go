package codex

import (
	"context"
	"runtime"

	"github.com/aarock1234/ai/pkg/harness/credential"
)

// keyring reads and writes the operating system's credential store, where
// the Codex CLI's keyring mode keeps its auth file.
type keyring interface {
	read(ctx context.Context, service string, account string) (string, bool)
	write(ctx context.Context, service string, account string, value string) error
}

// platformKeyring returns the keyring of the current platform, or nil.
func platformKeyring() keyring {
	switch runtime.GOOS {
	case "darwin":
		return macOSKeyring{}
	case "linux":
		return linuxKeyring{}
	case "windows":
		return windowsKeyring{}
	default:
		return nil
	}
}

// macOSKeyring is the macOS keychain.
type macOSKeyring struct{}

func (macOSKeyring) read(ctx context.Context, service string, account string) (string, bool) {
	return credential.ReadMacOSKeychainPassword(ctx, service, account)
}

func (macOSKeyring) write(ctx context.Context, service string, account string, value string) error {
	return credential.WriteMacOSKeychainPassword(ctx, credential.MacOSKeychainEntry{
		Service:  service,
		Account:  account,
		Password: value,
	})
}

// linuxKeyring is the Secret Service, as the Rust keyring crate the Codex
// CLI uses stores it.
type linuxKeyring struct{}

func (linuxKeyring) read(ctx context.Context, service string, account string) (string, bool) {
	return credential.ReadLinuxSecretServicePassword(ctx, secretAttributes(service, account))
}

func (linuxKeyring) write(ctx context.Context, service string, account string, value string) error {
	attributes := append(secretAttributes(service, account), credential.SecretAttribute{
		Name:  "application",
		Value: "rust-keyring",
	})

	return credential.WriteLinuxSecretServicePassword(ctx, credential.SecretServiceEntry{
		Label:      account + "@" + service + ":default",
		Attributes: attributes,
		Password:   value,
	})
}

// secretAttributes are the Secret Service attributes the Rust keyring crate
// identifies an entry by.
func secretAttributes(service string, account string) []credential.SecretAttribute {
	return []credential.SecretAttribute{
		{
			Name:  "service",
			Value: service,
		},
		{
			Name:  "username",
			Value: account,
		},
		{
			Name:  "target",
			Value: "default",
		},
	}
}

// windowsKeyring is the Windows Credential Manager.
type windowsKeyring struct{}

func (windowsKeyring) read(ctx context.Context, service string, account string) (string, bool) {
	return credential.ReadWindowsCredentialManagerPassword(ctx, windowsTargetName(service, account))
}

func (windowsKeyring) write(ctx context.Context, service string, account string, value string) error {
	return credential.WriteWindowsCredentialManagerPassword(ctx, credential.WindowsCredentialEntry{
		TargetName: windowsTargetName(service, account),
		UserName:   account,
		Password:   value,
	})
}

// windowsTargetName is the Credential Manager target of a keyring entry.
func windowsTargetName(service string, account string) string {
	return account + "." + service
}
