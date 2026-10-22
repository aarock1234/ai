package acp

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"regexp"
	"slices"
	"strings"
)

var (
	exactSemverPattern  = regexp.MustCompile(`^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$`)
	packageNamePattern  = regexp.MustCompile(`^(?:@[a-z0-9][a-z0-9._-]*/)?[a-z0-9][a-z0-9._-]*$`)
	executableNameRegex = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*$`)
	environmentVariable = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)
)

// implementation is the launch description of the ACP implementation: what
// the bootstrap installs and the bridge starts.
type implementation struct {
	source        Source
	executable    string
	args          []string
	forwardEnv    []string
	credentialEnv []string
	env           map[string]string
}

// newImplementation selects the launch settings.
func newImplementation(settings Settings) implementation {
	return implementation{
		source:        settings.Source,
		executable:    settings.Executable,
		args:          settings.Args,
		forwardEnv:    settings.ForwardEnv,
		credentialEnv: settings.CredentialEnv,
		env:           settings.Env,
	}
}

// validate checks the source, the executable and the environment names.
func (i implementation) validate() error {
	if err := validateSource(i.source); err != nil {
		return err
	}

	if !executableNameRegex.MatchString(i.executable) {
		return fmt.Errorf("acp executable must be a bare command name without a path; received %q", i.executable)
	}

	if err := validateEnvironmentNames(i.forwardEnv); err != nil {
		return err
	}

	if err := validateEnvironmentNames(i.credentialEnv); err != nil {
		return err
	}

	if err := validateLiteralEnvironment(i.env); err != nil {
		return err
	}

	return i.validateEnvironmentOverlap()
}

// validateEnvironmentOverlap rejects a variable configured in more than one
// of the environment settings.
func (i implementation) validateEnvironmentOverlap() error {
	for _, name := range i.credentialEnv {
		if slices.Contains(i.forwardEnv, name) {
			return fmt.Errorf("acp runtime environment key %q cannot be configured in both forwardEnv and credentialEnv", name)
		}
	}

	for _, name := range slices.Sorted(maps.Keys(i.env)) {
		if slices.Contains(i.forwardEnv, name) {
			return fmt.Errorf("acp runtime environment key %q cannot be configured in both forwardEnv and env", name)
		}

		if slices.Contains(i.credentialEnv, name) {
			return fmt.Errorf("acp runtime environment key %q cannot be configured in both credentialEnv and env", name)
		}
	}

	return nil
}

// validateSource checks the installation source.
func validateSource(source Source) error {
	switch typed := source.(type) {
	case NPMLockedSource:
		if typed.PackageJSON == "" {
			return errors.New("acp source package.json must not be empty")
		}

		if typed.PNPMLockYAML == "" {
			return errors.New("acp source pnpm-lock.yaml must not be empty")
		}

		return nil
	case NPMSimpleSource:
		if !packageNamePattern.MatchString(typed.PackageName) {
			return fmt.Errorf("acp npm package name is invalid: %q", typed.PackageName)
		}

		if typed.PackageVersion != "" && !exactSemverPattern.MatchString(typed.PackageVersion) {
			return fmt.Errorf("acp npm package version must be an exact semantic version; received %q", typed.PackageVersion)
		}

		return nil
	case InstallCommandSource:
		if strings.TrimSpace(typed.Command) == "" {
			return errors.New("acp source command must not be empty")
		}

		return nil
	case nil:
		return errors.New("acp source is required")
	default:
		return fmt.Errorf("acp source of type %T is not supported", source)
	}
}

// validateEnvironmentNames checks variable names.
func validateEnvironmentNames(names []string) error {
	for _, name := range names {
		if !environmentVariable.MatchString(name) {
			return fmt.Errorf("acp environment variable name is invalid: %q", name)
		}
	}

	return nil
}

// validateLiteralEnvironment checks the literal environment.
func validateLiteralEnvironment(env map[string]string) error {
	for _, name := range slices.Sorted(maps.Keys(env)) {
		if !environmentVariable.MatchString(name) {
			return fmt.Errorf("acp environment variable name is invalid: %q", name)
		}

		if strings.ContainsRune(env[name], 0) {
			return fmt.Errorf("acp runtime environment value for %s contains NUL", name)
		}
	}

	return nil
}

// isInstallCommand reports an implementation installed by a command into a
// private home.
func (i implementation) isInstallCommand() bool {
	_, ok := i.source.(InstallCommandSource)

	return ok
}

// manifest returns the package.json of the implementation directory, and
// false for an install command.
func (i implementation) manifest() (string, bool, error) {
	switch typed := i.source.(type) {
	case NPMLockedSource:
		return typed.PackageJSON, true, nil
	case NPMSimpleSource:
		version := typed.PackageVersion
		if version == "" {
			version = "latest"
		}

		manifest, err := indentedJSON(implementationManifest{
			Name:    "harness-acp-implementation",
			Version: "0.0.0",
			Private: true,
			Type:    "module",
			Dependencies: map[string]string{
				typed.PackageName: version,
			},
		})

		return manifest, true, err
	default:
		return "", false, nil
	}
}

// implementationManifest is the package.json of an npm-simple source.
type implementationManifest struct {
	Name         string            `json:"name"`
	Version      string            `json:"version"`
	Private      bool              `json:"private"`
	Type         string            `json:"type"`
	Dependencies map[string]string `json:"dependencies"`
}

// lockedSource returns the source of a locked install.
func (i implementation) lockedSource() (NPMLockedSource, bool) {
	locked, ok := i.source.(NPMLockedSource)

	return locked, ok
}

// descriptor returns the implementation.json the bridge reads to launch the
// implementation.
func (i implementation) descriptor() (string, error) {
	executablePath := "node_modules/.bin/" + i.executable
	if i.isInstallCommand() {
		executablePath = "home/.local/bin/" + i.executable
	}

	args := i.args
	if args == nil {
		args = []string{}
	}

	return indentedJSON(struct {
		ExecutablePath string   `json:"executablePath"`
		PrivateHome    bool     `json:"privateHome"`
		Args           []string `json:"args"`
		EnvKeys        []string `json:"envKeys"`
	}{
		ExecutablePath: executablePath,
		PrivateHome:    i.isInstallCommand(),
		Args:           args,
		EnvKeys:        i.environmentKeys(),
	})
}

// environmentKeys returns the sorted names of every runtime variable.
func (i implementation) environmentKeys() []string {
	keys := slices.Concat(i.forwardEnv, i.credentialEnv, slices.Collect(maps.Keys(i.env)))
	slices.Sort(keys)

	return slices.Compact(keys)
}

// installCommand returns the bootstrap command that installs the
// implementation.
func (i implementation) installCommand(implementationDir string, storeDir string) string {
	switch i.source.(type) {
	case InstallCommandSource:
		return "bash implementation/install.sh"
	case NPMLockedSource:
		return fmt.Sprintf("pnpm --dir %s install --frozen-lockfile --prod --store-dir %s", implementationDir, storeDir)
	default:
		return fmt.Sprintf("pnpm --dir %s install --prod --store-dir %s", implementationDir, storeDir)
	}
}

// installScript returns the install.sh of an install command, and false for
// other sources.
func (i implementation) installScript() (string, bool) {
	source, ok := i.source.(InstallCommandSource)
	if !ok {
		return "", false
	}

	script := `#!/usr/bin/env bash
set -euo pipefail

ACP_IMPLEMENTATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACP_INSTALL_HOME="$ACP_IMPLEMENTATION_DIR/home"
export HOME="$ACP_INSTALL_HOME"
export PATH="$ACP_INSTALL_HOME/.local/bin:$PATH"

mkdir -p "$ACP_INSTALL_HOME/.local/bin"
cd "$ACP_IMPLEMENTATION_DIR"

` + source.Command + "\n"

	return script, true
}

// environmentInput describes the environment the implementation receives.
type environmentInput struct {
	// env provides the forwarded variables.
	env map[string]string

	// credentialEnv provides the credential variables.
	credentialEnv map[string]string
}

// environment resolves the runtime environment: the non-empty forwarded and
// credential variables, then the literal environment.
func (i implementation) environment(input environmentInput) map[string]string {
	resolved := make(map[string]string)

	for _, name := range i.forwardEnv {
		if value := input.env[name]; value != "" {
			resolved[name] = value
		}
	}

	for _, name := range i.credentialEnv {
		if value := input.credentialEnv[name]; value != "" {
			resolved[name] = value
		}
	}

	maps.Copy(resolved, i.env)

	return resolved
}

// identityInput is everything the implementation identity covers.
type identityInput struct {
	harnessID              string
	implementation         implementation
	clientApp              ClientApp
	clientCapabilities     map[string]any
	modelMapping           ModelMapping
	providerAuthentication *providerAuthenticationCompatibility
	permissionModeMapping  *PermissionModeMapping
	nonTextContentTypes    []NonTextContentType
}

// implementationIdentity returns the digest that lifecycle state must match:
// the source, launch settings and protocol-level configuration, without
// secret values.
func implementationIdentity(input identityInput) (string, error) {
	source := input.implementation

	args := source.args
	if args == nil {
		args = []string{}
	}

	literal := make(map[string]any, len(source.env))
	for name, value := range source.env {
		literal[name] = map[string]string{
			"value": value,
		}
	}

	payload := map[string]any{
		"harnessId":              input.harnessID,
		"acpVersion":             "v1",
		"source":                 sourceIdentity(source.source),
		"executable":             source.executable,
		"args":                   args,
		"clientApp":              input.clientApp,
		"modelMapping":           input.modelMapping,
		"clientCapabilities":     nullable(input.clientCapabilities),
		"providerAuthentication": nullablePointer(input.providerAuthentication),
		"permissionModeMapping":  nullablePointer(input.permissionModeMapping),
		"environment": map[string]any{
			"forwarded":  sortedUnique(source.forwardEnv),
			"credential": sortedUnique(source.credentialEnv),
			"literal":    literal,
		},
	}

	if len(input.nonTextContentTypes) > 0 {
		payload["nonTextContentTypes"] = sortedUnique(input.nonTextContentTypes)
	}

	return stableFingerprint(payload)
}

// sourceIdentity returns the identity of the installation source.
func sourceIdentity(source Source) map[string]any {
	switch typed := source.(type) {
	case NPMLockedSource:
		identity := map[string]any{
			"type":         "npm-locked",
			"packageJson":  typed.PackageJSON,
			"pnpmLockYaml": typed.PNPMLockYAML,
		}

		if typed.PNPMWorkspaceYAML != "" {
			identity["pnpmWorkspaceYaml"] = typed.PNPMWorkspaceYAML
		}

		return identity
	case NPMSimpleSource:
		identity := map[string]any{
			"type":        "npm-simple",
			"packageName": typed.PackageName,
		}

		if typed.PackageVersion != "" {
			identity["packageVersion"] = typed.PackageVersion
		}

		return identity
	case InstallCommandSource:
		return map[string]any{
			"type":    "install-command",
			"command": typed.Command,
		}
	default:
		return nil
	}
}

// sortedUnique returns the sorted distinct values.
func sortedUnique[T ~string](values []T) []T {
	sorted := slices.Clone(values)
	slices.Sort(sorted)

	sorted = slices.Compact(sorted)
	if sorted == nil {
		return []T{}
	}

	return sorted
}

// nullable returns nil for an empty map, which encodes as JSON null.
func nullable(value map[string]any) any {
	if value == nil {
		return nil
	}

	return value
}

// nullablePointer returns nil for a nil pointer, which encodes as JSON null.
func nullablePointer[T any](value *T) any {
	if value == nil {
		return nil
	}

	return value
}

// indentedJSON encodes value with two-space indentation and a trailing
// newline, as JSON.stringify(value, null, 2) followed by a newline.
func indentedJSON(value any) (string, error) {
	encoded, err := marshalJSON(value)
	if err != nil {
		return "", err
	}

	var indented bytes.Buffer
	if err := json.Indent(&indented, encoded, "", "  "); err != nil {
		return "", fmt.Errorf("indent json: %w", err)
	}

	return indented.String() + "\n", nil
}
