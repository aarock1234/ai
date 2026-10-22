package codex

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/credential"
)

// OpenAI OAuth constants of the native ChatGPT subscription.
const (
	chatGPTCodexBaseURL = "https://chatgpt.com/backend-api/codex"
	openAITokenURL      = "https://auth.openai.com/oauth/token"
	openAIClientID      = "app_EMoamEEZ73f0CkXaXp7hrann"

	// keyringService is the keyring service of the Codex CLI.
	keyringService = "Codex Auth"
)

// credentialsStoreMode is where the Codex CLI stores its credentials, its
// cli_auth_credentials_store setting.
type credentialsStoreMode string

// Credentials store modes.
const (
	storeFile      credentialsStoreMode = "file"
	storeKeyring   credentialsStoreMode = "keyring"
	storeAuto      credentialsStoreMode = "auto"
	storeEphemeral credentialsStoreMode = "ephemeral"
)

// credentialsStoreSetting matches a top-level cli_auth_credentials_store
// setting of config.toml.
var credentialsStoreSetting = regexp.MustCompile(`^\s*cli_auth_credentials_store\s*=\s*["'](file|keyring|auto|ephemeral)["']\s*(?:#.*)?$`)

// resolvedAuthentication is the environment Codex authenticates with and the
// headers its requests need.
type resolvedAuthentication struct {
	environment map[string]string

	// requestHeaders are added to the request transformations, such as the
	// ChatGPT account of a native subscription. They are nil otherwise.
	requestHeaders map[string]string
}

// authenticationOptions configure resolveAuthentication.
type authenticationOptions struct {
	auth       harness.Authentication
	processEnv map[string]string

	// storeSetting is CodexConfig's cli_auth_credentials_store value.
	storeSetting any

	client *http.Client
}

// resolveAuthentication returns the credentials Codex authenticates with.
// Without an explicit environment, a direct credential or an AI Gateway key,
// it uses the Codex CLI's native ChatGPT subscription.
func resolveAuthentication(ctx context.Context, options authenticationOptions) (resolvedAuthentication, error) {
	environment := resolveEnv(options.auth, options.processEnv)
	resolved := resolvedAuthentication{
		environment: environment,
	}

	if options.auth.Environment != nil {
		return resolved, nil
	}

	_, hasDirectCredential := environment["CODEX_API_KEY"]

	shouldResolve := harness.ShouldResolveNativeSubscription(harness.NativeSubscriptionCheck{
		Mode:                options.auth.Mode,
		DirectMode:          AuthenticationDirect,
		Env:                 environment,
		HasDirectCredential: hasDirectCredential,
	})
	if !shouldResolve {
		return resolved, nil
	}

	subscription, err := readSubscription(ctx, options)
	if err != nil {
		return resolvedAuthentication{}, err
	}

	if subscription != nil {
		return *subscription, nil
	}

	return resolved, nil
}

// readSubscription returns the credentials of the Codex CLI's stored
// ChatGPT login, refreshing an access token that expires soon. It returns
// nil when no login is stored.
func readSubscription(ctx context.Context, options authenticationOptions) (*resolvedAuthentication, error) {
	codexHome, ok := resolveCodexHome(options.processEnv)
	if !ok {
		return nil, nil
	}

	storeMode, ok := toStoreMode(options.storeSetting)
	if !ok {
		storeMode, ok = readStoreMode(codexHome)
	}
	if !ok {
		storeMode = storeFile
	}

	authPath := filepath.Join(codexHome, "auth.json")

	unlock := credential.LockRefresh(authPath)
	defer unlock()

	store := findAuthStore(ctx, authStoreLocation{
		codexHome: codexHome,
		authPath:  authPath,
		mode:      storeMode,
		keyring:   platformKeyring(),
	})
	if store == nil {
		return nil, nil
	}

	login, ok := chatGPTLoginOf(store.document)
	if !ok {
		return nil, nil
	}

	accessToken, err := freshAccessToken(ctx, store, login, options.client)
	if err != nil {
		return nil, err
	}

	resolved := &resolvedAuthentication{
		environment: map[string]string{
			"CODEX_API_KEY":   accessToken,
			"OPENAI_BASE_URL": chatGPTCodexBaseURL,
		},
	}

	if login.accountID != "" {
		resolved.requestHeaders = map[string]string{
			"ChatGPT-Account-ID": login.accountID,
		}
	}

	return resolved, nil
}

// resolveCodexHome returns CODEX_HOME, defaulting to ~/.codex.
func resolveCodexHome(processEnv map[string]string) (string, bool) {
	codexHome, ok := processEnv["CODEX_HOME"]
	if !ok {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", false
		}

		codexHome = filepath.Join(home, ".codex")
	}

	absolute, err := filepath.Abs(codexHome)
	if err != nil {
		return "", false
	}

	return absolute, true
}

// freshAccessToken returns the stored access token, refreshing and storing
// it when it expires soon.
func freshAccessToken(ctx context.Context, store *authStore, login chatGPTLogin, client *http.Client) (string, error) {
	if !credential.IsAccessTokenExpiringSoon(login.expiresAt, time.Now(), 0) {
		return login.accessToken, nil
	}

	refreshed, err := credential.RefreshOAuthAccessToken(ctx, credential.RefreshOptions{
		TokenURL:      openAITokenURL,
		ClientID:      openAIClientID,
		RefreshToken:  login.refreshToken,
		RequestFormat: credential.RequestJSON,
		Client:        client,
	})
	if err != nil {
		return "", fmt.Errorf("refresh codex subscription: %w", err)
	}

	refreshToken := login.refreshToken
	if refreshed.RefreshToken != "" {
		refreshToken = refreshed.RefreshToken
	}

	if err := store.saveTokens(ctx, refreshed.AccessToken, refreshToken); err != nil {
		return "", fmt.Errorf("store refreshed codex subscription: %w", err)
	}

	return refreshed.AccessToken, nil
}

// chatGPTLogin is the ChatGPT login of a Codex auth file.
type chatGPTLogin struct {
	accessToken  string
	refreshToken string
	expiresAt    time.Time

	// accountID is empty when the file does not record it.
	accountID string
}

// authFileJSON is the part of a Codex auth file the adapter reads.
type authFileJSON struct {
	AuthMode string `json:"auth_mode"`
	Tokens   *struct {
		AccessToken  *string `json:"access_token"`
		RefreshToken *string `json:"refresh_token"`
		AccountID    *string `json:"account_id"`
	} `json:"tokens"`
}

// chatGPTLoginOf returns the ChatGPT login of an auth file whose access
// token carries an expiry.
func chatGPTLoginOf(document map[string]json.RawMessage) (chatGPTLogin, bool) {
	encoded, err := json.Marshal(document)
	if err != nil {
		return chatGPTLogin{}, false
	}

	var file authFileJSON
	if err := json.Unmarshal(encoded, &file); err != nil {
		return chatGPTLogin{}, false
	}

	if file.AuthMode != "chatgpt" || file.Tokens == nil {
		return chatGPTLogin{}, false
	}

	if file.Tokens.AccessToken == nil || file.Tokens.RefreshToken == nil {
		return chatGPTLogin{}, false
	}

	expiresAt, ok := credential.JWTExpiresAt(*file.Tokens.AccessToken)
	if !ok {
		return chatGPTLogin{}, false
	}

	login := chatGPTLogin{
		accessToken:  *file.Tokens.AccessToken,
		refreshToken: *file.Tokens.RefreshToken,
		expiresAt:    expiresAt,
	}

	if file.Tokens.AccountID != nil {
		login.accountID = *file.Tokens.AccountID
	}

	return login, true
}

// authStoreLocation is where the Codex CLI may store its credentials.
type authStoreLocation struct {
	codexHome string
	authPath  string
	mode      credentialsStoreMode

	// keyring is nil on platforms without a supported keyring.
	keyring keyring
}

// authStore is a stored auth file, kept whole so that a refresh preserves
// the fields the adapter does not read.
type authStore struct {
	// document is the auth file. Its fields are opaque.
	document map[string]json.RawMessage

	write func(ctx context.Context, document map[string]json.RawMessage) error
}

// findAuthStore reads the auth file from the store the mode selects. The
// auto mode prefers the keyring.
func findAuthStore(ctx context.Context, location authStoreLocation) *authStore {
	switch location.mode {
	case storeEphemeral:
		return nil
	case storeFile:
		return readFileStore(location.authPath)
	case storeKeyring:
		return readKeyringStore(ctx, location)
	default:
		if store := readKeyringStore(ctx, location); store != nil {
			return store
		}

		return readFileStore(location.authPath)
	}
}

// readFileStore reads auth.json.
func readFileStore(authPath string) *authStore {
	content, err := os.ReadFile(authPath)
	if err != nil {
		return nil
	}

	document, ok := parseAuthDocument(content)
	if !ok {
		return nil
	}

	return &authStore{
		document: document,
		write: func(_ context.Context, document map[string]json.RawMessage) error {
			return writeAuthFile(authPath, document)
		},
	}
}

// readKeyringStore reads the auth file from the keyring entry of the Codex
// home directory.
func readKeyringStore(ctx context.Context, location authStoreLocation) *authStore {
	if location.keyring == nil {
		return nil
	}

	account := keyringAccount(location.codexHome)

	text, ok := location.keyring.read(ctx, keyringService, account)
	if !ok {
		return nil
	}

	document, ok := parseAuthDocument([]byte(text))
	if !ok {
		return nil
	}

	return &authStore{
		document: document,
		write: func(ctx context.Context, document map[string]json.RawMessage) error {
			encoded, err := json.Marshal(document)
			if err != nil {
				return err
			}

			return location.keyring.write(ctx, keyringService, account, string(encoded))
		},
	}
}

// keyringAccount returns the keyring account of a Codex home directory: a
// hash of its canonical path.
func keyringAccount(codexHome string) string {
	canonical, err := filepath.EvalSymlinks(codexHome)
	if err != nil {
		canonical = codexHome
	}

	sum := sha256.Sum256([]byte(canonical))

	return "cli|" + hex.EncodeToString(sum[:])[:16]
}

// saveTokens stores refreshed tokens with the refresh time, keeping every
// other field.
func (s *authStore) saveTokens(ctx context.Context, accessToken string, refreshToken string) error {
	tokens := make(map[string]json.RawMessage)
	if raw, ok := s.document["tokens"]; ok {
		if err := json.Unmarshal(raw, &tokens); err != nil || tokens == nil {
			tokens = make(map[string]json.RawMessage)
		}
	}

	tokens["access_token"] = jsonString(accessToken)
	tokens["refresh_token"] = jsonString(refreshToken)

	encodedTokens, err := json.Marshal(tokens)
	if err != nil {
		return err
	}

	s.document["tokens"] = encodedTokens
	s.document["last_refresh"] = jsonString(time.Now().UTC().Format("2006-01-02T15:04:05.000Z"))

	return s.write(ctx, s.document)
}

// readStoreMode reads the top-level cli_auth_credentials_store setting of
// config.toml.
func readStoreMode(codexHome string) (credentialsStoreMode, bool) {
	content, err := os.ReadFile(filepath.Join(codexHome, "config.toml"))
	if err != nil {
		return "", false
	}

	scanner := bufio.NewScanner(bytes.NewReader(content))

	for scanner.Scan() {
		line := strings.TrimSuffix(scanner.Text(), "\r")

		if strings.HasPrefix(strings.TrimLeft(line, " \t"), "[") {
			return "", false
		}

		if match := credentialsStoreSetting.FindStringSubmatch(line); match != nil {
			return toStoreMode(match[1])
		}
	}

	return "", false
}

// toStoreMode validates a credentials store mode.
func toStoreMode(value any) (credentialsStoreMode, bool) {
	text, ok := value.(string)
	if !ok {
		return "", false
	}

	switch mode := credentialsStoreMode(text); mode {
	case storeFile, storeKeyring, storeAuto, storeEphemeral:
		return mode, true
	default:
		return "", false
	}
}

// parseAuthDocument decodes an auth file as a JSON object.
func parseAuthDocument(content []byte) (map[string]json.RawMessage, bool) {
	var document map[string]json.RawMessage
	if err := json.Unmarshal(content, &document); err != nil || document == nil {
		return nil, false
	}

	return document, true
}

// writeAuthFile replaces auth.json atomically, readable only by its owner.
func writeAuthFile(authPath string, document map[string]json.RawMessage) error {
	content, err := json.MarshalIndent(document, "", "  ")
	if err != nil {
		return err
	}

	return credential.WriteFileAtomically(authPath, append(content, '\n'))
}

// jsonString encodes a string as JSON.
func jsonString(value string) json.RawMessage {
	encoded, err := json.Marshal(value)
	if err != nil {
		return json.RawMessage(`""`)
	}

	return encoded
}
