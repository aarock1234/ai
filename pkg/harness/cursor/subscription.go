package cursor

import (
	"context"
	"encoding/json"
	"errors"
	"maps"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/acp"
	"github.com/aarock1234/ai/pkg/harness/credential"
)

// credentialStoreVariable selects where Cursor CLI stores its login: "file",
// "memory", or the platform default.
const credentialStoreVariable = "AGENT_CLI_CREDENTIAL_STORE"

// errSubscriptionExpiring reports a Cursor login about to expire.
var errSubscriptionExpiring = errors.New("cursor subscription access token is expiring soon; run cursor login again")

// resolveSubscriptionEnvironment adds the host's Cursor CLI login as
// CURSOR_API_KEY when no API key or AI Gateway credential is configured.
func resolveSubscriptionEnvironment(ctx context.Context, event acp.AuthenticationEnvironmentEvent) (map[string]string, error) {
	if event.Auth.Environment != nil {
		return event.Auth.Environment, nil
	}

	_, hasAPIKey := event.Env[apiKeyVariable]

	shouldResolve := harness.ShouldResolveNativeSubscription(harness.NativeSubscriptionCheck{
		Mode:                event.Auth.Mode,
		DirectMode:          acp.AuthenticationDirect,
		Env:                 event.Env,
		HasDirectCredential: hasAPIKey,
	})
	if !shouldResolve {
		return event.Env, nil
	}

	accessToken, err := readSubscription(ctx, subscriptionLocation{
		env:           event.Env,
		homeDirectory: credential.HomeDirectory(),
		goos:          runtime.GOOS,
	})
	if err != nil {
		return nil, err
	}

	if accessToken == "" {
		return event.Env, nil
	}

	environment := make(map[string]string, len(event.Env)+1)
	maps.Copy(environment, event.Env)
	environment[apiKeyVariable] = accessToken

	return environment, nil
}

// subscriptionLocation is where the Cursor CLI login is looked up.
type subscriptionLocation struct {
	env           map[string]string
	homeDirectory string
	goos          string
}

// readSubscription reads the Cursor CLI access token from its auth file,
// falling back to the macOS keychain. It returns empty when there is no
// login, and an error when the login is about to expire.
func readSubscription(ctx context.Context, location subscriptionLocation) (string, error) {
	store := location.env[credentialStoreVariable]
	if store == "memory" {
		return "", nil
	}

	accessToken := readAuthFile(authFilePath(location))

	if accessToken == "" && location.goos == "darwin" && store != "file" {
		accessToken, _ = credential.ReadMacOSKeychainPassword(ctx, "cursor-access-token", "cursor-user")
	}

	if accessToken == "" {
		return "", nil
	}

	expiresAt, ok := credential.JWTExpiresAt(accessToken)
	if ok && credential.IsAccessTokenExpiringSoon(expiresAt, time.Now(), 0) {
		return "", errSubscriptionExpiring
	}

	return accessToken, nil
}

// authFilePath returns the Cursor CLI auth file of the platform.
func authFilePath(location subscriptionLocation) string {
	switch location.goos {
	case "windows":
		appData, ok := location.env["APPDATA"]
		if !ok {
			appData = filepath.Join(location.homeDirectory, "AppData", "Roaming")
		}

		return filepath.Join(appData, "Cursor", "auth.json")
	case "linux":
		configHome, ok := location.env["XDG_CONFIG_HOME"]
		if !ok {
			configHome = filepath.Join(location.homeDirectory, ".config")
		}

		return filepath.Join(configHome, "cursor", "auth.json")
	default:
		return filepath.Join(location.homeDirectory, ".cursor", "auth.json")
	}
}

// readAuthFile returns the access token of an auth file, or empty when the
// file is missing or malformed.
func readAuthFile(authPath string) string {
	content, err := os.ReadFile(authPath)
	if err != nil {
		return ""
	}

	var auth struct {
		AccessToken string `json:"accessToken"`
	}

	if err := json.Unmarshal(content, &auth); err != nil {
		return ""
	}

	return auth.AccessToken
}
