package claudecode

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"time"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/credential"
)

// Claude OAuth constants of the native subscription.
const (
	claudeTokenURL = "https://platform.claude.com/v1/oauth/token"
	claudeClientID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

	// keychainService is the macOS keychain item of the claude CLI.
	keychainService = "Claude Code-credentials"
)

// resolveAuthentication returns the environment Claude Code authenticates
// with. Without an explicit environment, direct credentials or an AI
// Gateway key, it uses the claude CLI's native subscription, falling back to
// the API key helper.
func resolveAuthentication(ctx context.Context, auth harness.Authentication, processEnv map[string]string, client *http.Client) (map[string]string, error) {
	withoutHelper, err := resolveEnv(ctx, auth, processEnv, noAPIKeyHelper)
	if err != nil {
		return nil, err
	}

	if auth.Environment != nil {
		return withoutHelper, nil
	}

	shouldResolve := harness.ShouldResolveNativeSubscription(harness.NativeSubscriptionCheck{
		Mode:                auth.Mode,
		DirectMode:          AuthenticationDirect,
		Env:                 withoutHelper,
		HasDirectCredential: hasDirectCredential(withoutHelper),
	})
	if !shouldResolve {
		return withoutHelper, nil
	}

	subscription, err := readSubscription(ctx, processEnv, client)
	if err != nil {
		return nil, err
	}

	if subscription != nil {
		return subscription, nil
	}

	return resolveEnv(ctx, auth, processEnv, readAPIKeyHelper)
}

// hasDirectCredential reports a direct Anthropic credential.
func hasDirectCredential(env map[string]string) bool {
	for _, name := range []string{"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"} {
		if _, ok := env[name]; ok {
			return true
		}
	}

	return false
}

// readSubscription returns the environment of the claude CLI's stored OAuth
// credential, refreshing an access token that expires soon. It returns nil
// when no subscription is stored.
func readSubscription(ctx context.Context, processEnv map[string]string, client *http.Client) (map[string]string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, nil
	}

	configDir := filepath.Join(home, ".claude")
	if override, ok := processEnv["CLAUDE_CONFIG_DIR"]; ok {
		configDir = override
	}

	configDir, err = filepath.Abs(configDir)
	if err != nil {
		return nil, nil
	}

	credentialPath := filepath.Join(configDir, ".credentials.json")

	unlock := credential.LockRefresh(credentialPath)
	defer unlock()

	store := findCredentialStore(ctx, credentialStoreLocation{
		credentialPath: credentialPath,
		configDir:      configDir,
		homeDir:        home,
		user:           processEnv["USER"],
	})
	if store == nil {
		return nil, nil
	}

	accessToken, err := freshAccessToken(ctx, store, client)
	if err != nil {
		return nil, err
	}

	return map[string]string{
		"CLAUDE_CODE_OAUTH_TOKEN": accessToken,
		"ANTHROPIC_BASE_URL":      defaultAnthropicBaseURL,
	}, nil
}

// freshAccessToken returns the stored access token, refreshing and storing
// it when it expires soon.
func freshAccessToken(ctx context.Context, store *credentialStore, client *http.Client) (string, error) {
	oauth := store.oauth

	expiresAt := time.UnixMilli(int64(oauth.ExpiresAt))
	if !credential.IsAccessTokenExpiringSoon(expiresAt, time.Now(), 0) {
		return oauth.AccessToken, nil
	}

	refreshed, err := credential.RefreshOAuthAccessToken(ctx, credential.RefreshOptions{
		TokenURL:      claudeTokenURL,
		ClientID:      claudeClientID,
		RefreshToken:  oauth.RefreshToken,
		RequestFormat: credential.RequestJSON,
		Client:        client,
	})
	if err != nil {
		return "", fmt.Errorf("refresh claude code subscription: %w", err)
	}

	refreshToken := oauth.RefreshToken
	if refreshed.RefreshToken != "" {
		refreshToken = refreshed.RefreshToken
	}

	updated := claudeOAuth{
		AccessToken:  refreshed.AccessToken,
		RefreshToken: refreshToken,
		ExpiresAt:    float64(refreshed.ExpiresAt.UnixMilli()),
	}

	if err := store.save(ctx, updated); err != nil {
		return "", fmt.Errorf("store refreshed claude code subscription: %w", err)
	}

	return refreshed.AccessToken, nil
}

// claudeOAuth is the claudeAiOauth credential of the claude CLI.
type claudeOAuth struct {
	AccessToken  string
	RefreshToken string

	// ExpiresAt is the expiry in Unix milliseconds.
	ExpiresAt float64
}

// claudeOAuthJSON validates the required fields of claudeAiOauth.
type claudeOAuthJSON struct {
	AccessToken  *string  `json:"accessToken"`
	RefreshToken *string  `json:"refreshToken"`
	ExpiresAt    *float64 `json:"expiresAt"`
}

// credentialStoreLocation is where the claude CLI may store its credential.
type credentialStoreLocation struct {
	credentialPath string
	configDir      string
	homeDir        string
	user           string
}

// credentialStore is a stored credential file, kept whole so that a refresh
// preserves the fields the adapter does not read.
type credentialStore struct {
	// document is the credential file, and oauthFields its claudeAiOauth
	// object. Their fields are opaque.
	document    map[string]json.RawMessage
	oauthFields map[string]json.RawMessage
	oauth       claudeOAuth

	write func(ctx context.Context, document map[string]json.RawMessage) error
}

// findCredentialStore reads the credential file, then, on macOS with the
// default configuration directory, the keychain.
func findCredentialStore(ctx context.Context, location credentialStoreLocation) *credentialStore {
	if content, err := os.ReadFile(location.credentialPath); err == nil {
		store, ok := parseCredentialStore(content)
		if ok {
			store.write = func(_ context.Context, document map[string]json.RawMessage) error {
				return writeCredentialFile(location.credentialPath, document)
			}

			return store
		}
	}

	if !usesKeychain(location) {
		return nil
	}

	password, ok := credential.ReadMacOSKeychainPassword(ctx, keychainService, location.user)
	if !ok {
		return nil
	}

	store, ok := parseCredentialStore([]byte(password))
	if !ok {
		return nil
	}

	store.write = func(ctx context.Context, document map[string]json.RawMessage) error {
		updated, err := json.Marshal(document)
		if err != nil {
			return err
		}

		return credential.WriteMacOSKeychainPassword(ctx, credential.MacOSKeychainEntry{
			Service:  keychainService,
			Account:  location.user,
			Password: string(updated),
		})
	}

	return store
}

// usesKeychain reports whether the claude CLI keeps its credential in the
// macOS keychain: on macOS, with the default configuration directory.
func usesKeychain(location credentialStoreLocation) bool {
	if runtime.GOOS != "darwin" {
		return false
	}

	defaultConfigDir, err := filepath.Abs(filepath.Join(location.homeDir, ".claude"))
	if err != nil {
		return false
	}

	return location.configDir == defaultConfigDir
}

// parseCredentialStore decodes a credential file with a usable OAuth
// credential.
func parseCredentialStore(content []byte) (*credentialStore, bool) {
	var document map[string]json.RawMessage
	if err := json.Unmarshal(content, &document); err != nil || document == nil {
		return nil, false
	}

	rawOAuth, ok := document["claudeAiOauth"]
	if !ok {
		return nil, false
	}

	var oauthFields map[string]json.RawMessage
	if err := json.Unmarshal(rawOAuth, &oauthFields); err != nil || oauthFields == nil {
		return nil, false
	}

	var oauth claudeOAuthJSON
	if err := json.Unmarshal(rawOAuth, &oauth); err != nil {
		return nil, false
	}

	if oauth.AccessToken == nil || oauth.RefreshToken == nil || oauth.ExpiresAt == nil {
		return nil, false
	}

	return &credentialStore{
		document:    document,
		oauthFields: oauthFields,
		oauth: claudeOAuth{
			AccessToken:  *oauth.AccessToken,
			RefreshToken: *oauth.RefreshToken,
			ExpiresAt:    *oauth.ExpiresAt,
		},
	}, true
}

// save stores the refreshed credential, keeping every other field.
func (s *credentialStore) save(ctx context.Context, updated claudeOAuth) error {
	s.oauthFields["accessToken"] = jsonString(updated.AccessToken)
	s.oauthFields["refreshToken"] = jsonString(updated.RefreshToken)
	s.oauthFields["expiresAt"] = json.RawMessage(strconv.FormatFloat(updated.ExpiresAt, 'f', -1, 64))

	oauth, err := json.Marshal(s.oauthFields)
	if err != nil {
		return err
	}

	s.document["claudeAiOauth"] = oauth

	return s.write(ctx, s.document)
}

// writeCredentialFile replaces the credential file atomically, readable only
// by its owner.
func writeCredentialFile(credentialPath string, document map[string]json.RawMessage) error {
	content, err := json.MarshalIndent(document, "", "  ")
	if err != nil {
		return err
	}

	return credential.WriteFileAtomically(credentialPath, append(content, '\n'))
}

// jsonString encodes a string as JSON.
func jsonString(value string) json.RawMessage {
	encoded, err := json.Marshal(value)
	if err != nil {
		return json.RawMessage(`""`)
	}

	return encoded
}
