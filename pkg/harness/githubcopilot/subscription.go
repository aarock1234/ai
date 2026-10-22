package githubcopilot

import (
	"context"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"time"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/acp"
	"github.com/aarock1234/ai/pkg/harness/credential"
)

// keyringService is the credential store service of Copilot CLI logins.
const keyringService = "copilot-cli"

// githubCLITokenTimeout bounds `gh auth token`.
const githubCLITokenTimeout = 10 * time.Second

// defaultWindowsPathExtensions are the executable extensions without
// PATHEXT.
const defaultWindowsPathExtensions = ".COM;.EXE;.BAT;.CMD"

// subscription is a GitHub token of the host's Copilot CLI or GitHub CLI
// login, with the origin of its GitHub host.
type subscription struct {
	token string
	host  string
}

// resolveSubscriptionEnvironment adds the host's Copilot CLI or GitHub CLI
// login as COPILOT_GITHUB_TOKEN and COPILOT_GH_HOST when no GitHub token or
// AI Gateway credential is configured.
func resolveSubscriptionEnvironment(ctx context.Context, event acp.AuthenticationEnvironmentEvent) (map[string]string, error) {
	if event.Auth.Environment != nil {
		return event.Auth.Environment, nil
	}

	shouldResolve := harness.ShouldResolveNativeSubscription(harness.NativeSubscriptionCheck{
		Mode:                event.Auth.Mode,
		DirectMode:          acp.AuthenticationDirect,
		Env:                 event.Env,
		HasDirectCredential: hasAnyVariable(event.Env, tokenVariables),
	})
	if !shouldResolve {
		return event.Env, nil
	}

	login, ok := readSubscription(ctx, subscriptionLocation{
		env:           event.Env,
		homeDirectory: credential.HomeDirectory(),
		goos:          runtime.GOOS,
	})
	if !ok {
		return event.Env, nil
	}

	host, err := normalizeGitHubHost(login.host)
	if err != nil {
		return nil, err
	}

	environment := make(map[string]string, len(event.Env)+2)
	maps.Copy(environment, event.Env)
	environment[copilotTokenVariable] = login.token
	environment[hostVariable] = host

	return environment, nil
}

// subscriptionLocation is where the logins are looked up.
type subscriptionLocation struct {
	env           map[string]string
	homeDirectory string
	goos          string
}

// readSubscription reads the Copilot CLI login from the credential store or
// the plaintext fallback of its configuration, and then the GitHub CLI
// token of the login's host. It reports false when there is no login.
func readSubscription(ctx context.Context, location subscriptionLocation) (subscription, bool) {
	config := readConfig(filepath.Join(copilotHome(location), "config.json"))
	account, hasAccount := config.account()

	if hasAccount {
		if token, ok := readSecureCredential(ctx, location.goos, account.key()); ok && isNonBlank(token) {
			return subscription{
				token: token,
				host:  account.host,
			}, true
		}

		if token, ok := config.plaintextToken(account.key()); ok {
			return subscription{
				token: token,
				host:  account.host,
			}, true
		}
	}

	host := account.host
	if !hasAccount {
		origin, ok := normalizeGitHubOrigin(configuredGitHubHost(location.env))
		if !ok {
			return subscription{}, false
		}

		host = origin
	}

	return readGitHubCLISubscription(ctx, location, host)
}

// readGitHubCLISubscription reads the GitHub CLI token of a host when gh is
// on PATH.
func readGitHubCLISubscription(ctx context.Context, location subscriptionLocation, host string) (subscription, bool) {
	executable, ok := findGitHubCLI(location)
	if !ok {
		return subscription{}, false
	}

	hostname, err := normalizeGitHubHost(host)
	if err != nil {
		return subscription{}, false
	}

	token, ok := readGitHubCLIToken(ctx, githubCLITokenRequest{
		executable: executable,
		hostname:   hostname,
		env:        location.env,
	})
	if !ok {
		return subscription{}, false
	}

	return subscription{
		token: token,
		host:  host,
	}, true
}

// copilotHome returns Copilot CLI's home directory: COPILOT_HOME, or
// .copilot in the user's home directory.
func copilotHome(location subscriptionLocation) string {
	home, ok := location.env["COPILOT_HOME"]
	if !ok {
		home = filepath.Join(location.homeDirectory, ".copilot")
	}

	absolute, err := filepath.Abs(home)
	if err != nil {
		return home
	}

	return absolute
}

// readSecureCredential reads a Copilot CLI token from the platform's
// credential store.
func readSecureCredential(ctx context.Context, goos string, account string) (string, bool) {
	switch goos {
	case "darwin":
		return credential.ReadMacOSKeychainPassword(ctx, keyringService, account)
	case "linux":
		return credential.ReadLinuxSecretServicePassword(ctx, []credential.SecretAttribute{
			{
				Name:  "service",
				Value: keyringService,
			},
			{
				Name:  "username",
				Value: account,
			},
		})
	case "windows":
		return credential.ReadWindowsCredentialManagerPassword(ctx, account+"."+keyringService)
	default:
		return "", false
	}
}

// findGitHubCLI returns the first gh executable on PATH.
func findGitHubCLI(location subscriptionLocation) (string, bool) {
	searchPath, ok := firstVariable(location.env, "PATH", "Path", "path")
	if !ok {
		return "", false
	}

	isWindows := location.goos == "windows"

	separator := ":"
	names := []string{
		"gh",
	}

	if isWindows {
		separator = ";"
		names = windowsGitHubCLINames(location.env)
	}

	for directory := range strings.SplitSeq(searchPath, separator) {
		directory = strings.TrimSuffix(strings.TrimPrefix(directory, `"`), `"`)
		if directory == "" {
			continue
		}

		for _, name := range names {
			candidate, err := filepath.Abs(filepath.Join(directory, name))
			if err != nil {
				continue
			}

			if isExecutable(candidate, isWindows) {
				return candidate, true
			}
		}
	}

	return "", false
}

// windowsGitHubCLINames returns gh with each executable extension of
// PATHEXT.
func windowsGitHubCLINames(env map[string]string) []string {
	extensions, ok := env["PATHEXT"]
	if !ok {
		extensions = defaultWindowsPathExtensions
	}

	names := []string{
		"gh",
	}

	for extension := range strings.SplitSeq(extensions, ";") {
		if isNonBlank(extension) {
			names = append(names, "gh"+extension)
		}
	}

	return names
}

// isExecutable reports a file that exists and, outside Windows, has an
// execute permission bit.
func isExecutable(candidate string, isWindows bool) bool {
	info, err := os.Stat(candidate)
	if err != nil {
		return false
	}

	return isWindows || info.Mode().Perm()&0o111 != 0
}

// githubCLITokenRequest runs `gh auth token` for one host.
type githubCLITokenRequest struct {
	executable string
	hostname   string
	env        map[string]string
}

// readGitHubCLIToken returns the GitHub CLI token of a host, without
// modifying its credential store.
func readGitHubCLIToken(ctx context.Context, request githubCLITokenRequest) (string, bool) {
	ctx, cancel := context.WithTimeout(ctx, githubCLITokenTimeout)
	defer cancel()

	command := exec.CommandContext(ctx, request.executable, "auth", "token", "--hostname", request.hostname)
	command.Env = environmentList(request.env)

	output, err := command.Output()
	if err != nil {
		return "", false
	}

	token := strings.TrimSpace(string(output))

	return token, token != ""
}

// environmentList converts an environment to KEY=value entries.
func environmentList(env map[string]string) []string {
	entries := make([]string, 0, len(env))

	for _, name := range slices.Sorted(maps.Keys(env)) {
		entries = append(entries, name+"="+env[name])
	}

	return entries
}

// firstVariable returns the first of the named variables that is set.
func firstVariable(env map[string]string, names ...string) (string, bool) {
	for _, name := range names {
		if value, ok := env[name]; ok {
			return value, true
		}
	}

	return "", false
}

// hasAnyVariable reports whether any of the named variables is set.
func hasAnyVariable(env map[string]string, names []string) bool {
	_, ok := firstVariable(env, names...)

	return ok
}

// isNonBlank reports a string with non-whitespace content.
func isNonBlank(value string) bool {
	return strings.TrimSpace(value) != ""
}
