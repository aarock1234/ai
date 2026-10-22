package opencode

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"time"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/credential"
)

// OAuth clients of the native logins whose tokens the adapter refreshes.
const (
	openAITokenURL = "https://auth.openai.com/oauth/token"
	openAIClientID = "app_EMoamEEZ73f0CkXaXp7hrann"

	xAITokenURL = "https://auth.x.ai/oauth2/token"
	xAIClientID = "b1a00492-073a-47ea-816f-4c329264a828"

	gitLabClientID = "1d89f9fdb23ee96d4e603201f6861dab6e143c5c3c00469a018a2d94bdc03d4e"
)

// Model endpoints of the native logins.
const (
	chatGPTCodexURL          = "https://chatgpt.com/backend-api/codex"
	xAIURL                   = "https://api.x.ai/v1"
	gitHubCopilotURL         = "https://api.githubcopilot.com"
	poeURL                   = "https://api.poe.com"
	openCodeGoURL            = "https://opencode.ai/zen/go/v1"
	defaultGitLabInstanceURL = "https://gitlab.com"
)

// hostManagedRefreshToken stands in for the refresh token in the auth
// content of the sandbox: the host refreshes the access token itself.
const hostManagedRefreshToken = "host-managed"

// neverExpires is the expiry of the login in the sandbox's auth content,
// JavaScript's Number.MAX_SAFE_INTEGER milliseconds.
const neverExpires = 1<<53 - 1

// subscriptionProviders are the providers with a native OpenCode login.
var subscriptionProviders = []resolvedMode{
	resolvedOpenAI,
	resolvedXAI,
	resolvedGitHubCopilot,
	resolvedPoe,
	resolvedOpenCodeGo,
	resolvedGitLab,
}

// subscription is the native OpenCode login of a provider.
type subscription struct {
	provider    resolvedMode
	accessToken string

	// accountID is the ChatGPT account of an OpenAI login, when known.
	accountID string

	// enterpriseURL is the enterprise instance of a GitHub Copilot or GitLab
	// login, when the login names one.
	enterpriseURL string
}

// resolvedAuthentication is the environment OpenCode authenticates with.
type resolvedAuthentication struct {
	environment map[string]string
	mode        resolvedMode

	// subscription is the native login the environment's access token
	// belongs to, and nil when OpenCode authenticates with credentials.
	subscription *subscription
}

// authenticationOptions configure resolveAuthentication.
type authenticationOptions struct {
	auth       harness.Authentication
	provider   string
	processEnv map[string]string
	client     *http.Client
}

// resolveAuthentication returns the credentials OpenCode authenticates with.
// Without an explicit environment, the AI Gateway or a credential of the
// selected provider, it uses OpenCode's native login of the provider and
// carries its access token in subscriptionAccessTokenVariable.
func resolveAuthentication(ctx context.Context, options authenticationOptions) (resolvedAuthentication, error) {
	resolved := resolvedAuthentication{
		environment: resolveEnv(options.auth, options.provider, options.processEnv),
		mode:        resolveMode(options.auth, options.provider, options.processEnv),
	}

	usesConfiguredCredentials := options.auth.Environment != nil ||
		options.auth.Mode == harness.AuthenticationAIGateway ||
		resolved.mode == resolvedAIGateway ||
		hasCredential(resolved.environment, resolved.mode)
	if usesConfiguredCredentials {
		return resolved, nil
	}

	login, err := readSubscription(ctx, subscriptionLookup{
		provider: resolved.mode,
		env:      options.processEnv,
		client:   options.client,
	})
	if err != nil {
		return resolvedAuthentication{}, err
	}

	if login == nil {
		return resolved, nil
	}

	resolved.environment[subscriptionAccessTokenVariable] = login.accessToken
	resolved.subscription = login

	return resolved, nil
}

// subscriptionLookup configures readSubscription.
type subscriptionLookup struct {
	provider resolvedMode
	env      map[string]string
	client   *http.Client
}

// readSubscription returns the native login of the provider from OpenCode's
// auth store, refreshing an OAuth access token that expires soon. It
// returns nil when no usable login is stored.
func readSubscription(ctx context.Context, lookup subscriptionLookup) (*subscription, error) {
	if !slices.Contains(subscriptionProviders, lookup.provider) {
		return nil, nil
	}

	if authPath, ok := authFilePath(lookup.env); ok {
		unlock := credential.LockRefresh(authPath)
		defer unlock()
	}

	store := readAuthStore(lookup.env)
	if store == nil {
		return nil, nil
	}

	entry, ok := store.entry(lookup.provider)
	if !ok {
		return nil, nil
	}

	switch entry.stringField("type") {
	case "api":
		return apiKeyLogin(lookup.provider, entry), nil
	case "oauth":
		return oauthLogin(ctx, oauthLoginInput{
			lookup: lookup,
			store:  store,
			entry:  entry,
		})
	default:
		return nil, nil
	}
}

// apiKeyLogin returns an API key login.
func apiKeyLogin(provider resolvedMode, entry storeEntry) *subscription {
	key := entry.stringField("key")
	if key == "" {
		return nil
	}

	return &subscription{
		provider:    provider,
		accessToken: key,
	}
}

// oauthLoginInput is the input of oauthLogin.
type oauthLoginInput struct {
	lookup subscriptionLookup
	store  *authStore
	entry  storeEntry
}

// oauthLogin returns an OAuth login. GitHub Copilot authenticates with its
// long-lived refresh token and Poe with an API key that cannot be refreshed;
// the other providers refresh an access token that expires soon.
func oauthLogin(ctx context.Context, input oauthLoginInput) (*subscription, error) {
	provider := input.lookup.provider
	entry := input.entry

	if provider == resolvedOpenCodeGo {
		return nil, nil
	}

	access, hasAccess := entry.optionalString("access")
	refresh, hasRefresh := entry.optionalString("refresh")
	expires, hasExpires := entry.number("expires")

	if !hasAccess || !hasRefresh || !hasExpires {
		return nil, nil
	}

	enterpriseURL := entry.stringField("enterpriseUrl")

	switch provider {
	case resolvedGitHubCopilot:
		return &subscription{
			provider:      provider,
			accessToken:   refresh,
			enterpriseURL: enterpriseURL,
		}, nil
	case resolvedPoe:
		if isExpiringSoon(expires) {
			return nil, errors.New("opencode poe subscription api key is expiring soon; run opencode login again")
		}

		return &subscription{
			provider:    provider,
			accessToken: access,
		}, nil
	}

	login := &subscription{
		provider:      provider,
		accessToken:   access,
		accountID:     entry.stringField("accountId"),
		enterpriseURL: enterpriseURL,
	}

	if login.accountID == "" && provider == resolvedOpenAI {
		login.accountID = openAIAccountID(access)
	}

	if !isExpiringSoon(expires) {
		return login, nil
	}

	if err := refreshLogin(ctx, login, input, refresh); err != nil {
		return nil, err
	}

	return login, nil
}

// refreshLogin refreshes the login's access token and stores the refreshed
// tokens, keeping every other field of the login.
func refreshLogin(ctx context.Context, login *subscription, input oauthLoginInput, refreshToken string) error {
	tokenURL, clientID := refreshClient(login, input.lookup.env)

	refreshed, err := credential.RefreshOAuthAccessToken(ctx, credential.RefreshOptions{
		TokenURL:     tokenURL,
		ClientID:     clientID,
		RefreshToken: refreshToken,
		Client:       input.lookup.client,
	})
	if err != nil {
		return fmt.Errorf("refresh opencode %s subscription: %w", login.provider, err)
	}

	login.accessToken = refreshed.AccessToken

	if login.provider == resolvedOpenAI {
		if accountID := openAIAccountID(refreshed.AccessToken); accountID != "" {
			login.accountID = accountID
		}
	}

	if refreshed.RefreshToken != "" {
		refreshToken = refreshed.RefreshToken
	}

	updates := refreshedTokens{
		Access:    refreshed.AccessToken,
		Refresh:   refreshToken,
		Expires:   refreshed.ExpiresAt.UnixMilli(),
		AccountID: login.accountID,
	}

	if err := input.store.update(login.provider, input.entry, updates); err != nil {
		return fmt.Errorf("store refreshed opencode %s subscription: %w", login.provider, err)
	}

	return nil
}

// refreshedTokens are the fields a refresh replaces in a stored login.
type refreshedTokens struct {
	Access    string `json:"access"`
	Refresh   string `json:"refresh"`
	Expires   int64  `json:"expires"`
	AccountID string `json:"accountId,omitempty"`
}

// refreshClient returns the token endpoint and OAuth client of a login.
func refreshClient(login *subscription, env map[string]string) (string, string) {
	switch login.provider {
	case resolvedOpenAI:
		return openAITokenURL, openAIClientID
	case resolvedXAI:
		return xAITokenURL, xAIClientID
	}

	instanceURL := login.enterpriseURL
	if instanceURL == "" {
		instanceURL = firstPresentOr(env, "GITLAB_INSTANCE_URL", defaultGitLabInstanceURL)
	}

	clientID := firstPresentOr(env, "GITLAB_OAUTH_CLIENT_ID", gitLabClientID)

	return strings.TrimRight(instanceURL, "/") + "/oauth/token", clientID
}

// isExpiringSoon reports whether a token whose expiry is in Unix
// milliseconds expires within the refresh window.
func isExpiringSoon(expiresMilliseconds float64) bool {
	expiresAt := time.UnixMilli(int64(expiresMilliseconds))

	return credential.IsAccessTokenExpiringSoon(expiresAt, time.Now(), 0)
}

// openAIAccountID returns the ChatGPT account of an OpenAI access token, or
// empty.
func openAIAccountID(accessToken string) string {
	claims, ok := credential.ParseJWTPayload(accessToken)
	if !ok {
		return ""
	}

	auth, ok := claims["https://api.openai.com/auth"].(map[string]any)
	if !ok {
		return ""
	}

	accountID, _ := auth["chatgpt_account_id"].(string)

	return accountID
}

// subscriptionAuthEntry is a login in OpenCode's auth content.
type subscriptionAuthEntry struct {
	Type string `json:"type"`

	// Key is the API key of an API key login.
	Key string `json:"key,omitempty"`

	Access        string `json:"access,omitempty"`
	Refresh       string `json:"refresh,omitempty"`
	Expires       int64  `json:"expires,omitempty"`
	AccountID     string `json:"accountId,omitempty"`
	EnterpriseURL string `json:"enterpriseUrl,omitempty"`
}

// subscriptionAuthContent returns the OPENCODE_AUTH_CONTENT of the sandbox:
// the login with the token the sandbox may see in place of the access token,
// and a refresh token the host manages. The login never expires inside the
// sandbox.
func subscriptionAuthContent(login *subscription, accessToken string) (string, error) {
	entry := subscriptionAuthEntry{
		Type:          "oauth",
		Access:        accessToken,
		Refresh:       hostManagedRefreshToken,
		Expires:       neverExpires,
		AccountID:     login.accountID,
		EnterpriseURL: login.enterpriseURL,
	}

	switch login.provider {
	case resolvedOpenCodeGo:
		entry = subscriptionAuthEntry{
			Type: "api",
			Key:  accessToken,
		}
	case resolvedGitHubCopilot:
		entry.Refresh = accessToken
	}

	content, err := json.Marshal(map[resolvedMode]subscriptionAuthEntry{
		login.provider: entry,
	})
	if err != nil {
		return "", fmt.Errorf("encode opencode auth content: %w", err)
	}

	return string(content), nil
}

// subscriptionTransformations return the transformations that inject a
// login's access token in place of its sandbox placeholder.
func subscriptionTransformations(login *subscription, placeholder string) ([]harness.RequestTransformation, error) {
	matchURL := subscriptionURL(login)
	realToken := login.accessToken

	bearer := bearerReplacement(matchURL, placeholder, realToken)

	if login.provider == resolvedOpenAI && login.accountID != "" {
		return accountTransformations(bearer, login.accountID)
	}

	replacements := []headerReplacement{
		bearer,
	}

	if login.provider == resolvedPoe || login.provider == resolvedOpenCodeGo {
		replacements = append(replacements, headerReplacement{
			matchURL:   matchURL,
			header:     "x-api-key",
			matchValue: placeholder,
			value:      realToken,
		})
	}

	return transformations(replacements...)
}

// accountTransformations replace the bearer token of a ChatGPT login and
// add its account header.
func accountTransformations(bearer headerReplacement, accountID string) ([]harness.RequestTransformation, error) {
	transformation, err := harness.NewCredentialRequestTransformation(harness.CredentialRequestTransformationOptions{
		MatchURL: bearer.matchURL,
		MatchHeaders: map[string]string{
			bearer.header: bearer.matchValue,
		},
		TransformHeaders: map[string]string{
			"Authorization":      bearer.value,
			"ChatGPT-Account-Id": accountID,
		},
	})
	if err != nil {
		return nil, err
	}

	return []harness.RequestTransformation{
		transformation,
	}, nil
}

// subscriptionURL returns the endpoint a login's requests are sent to.
func subscriptionURL(login *subscription) string {
	switch login.provider {
	case resolvedOpenAI:
		return chatGPTCodexURL
	case resolvedXAI:
		return xAIURL
	case resolvedGitHubCopilot:
		if login.enterpriseURL == "" {
			return gitHubCopilotURL
		}

		return "https://copilot-api." + normalizeDomain(login.enterpriseURL)
	case resolvedPoe:
		return poeURL
	case resolvedOpenCodeGo:
		return openCodeGoURL
	default:
		if login.enterpriseURL == "" {
			return defaultGitLabInstanceURL
		}

		return login.enterpriseURL
	}
}

// normalizeDomain returns the host of an enterprise URL, which may omit its
// scheme.
func normalizeDomain(value string) string {
	withScheme := value
	if !strings.Contains(value, "://") {
		withScheme = "https://" + value
	}

	parsed, err := url.Parse(withScheme)
	if err == nil && parsed.Host != "" {
		return parsed.Host
	}

	trimmed := strings.TrimPrefix(strings.TrimPrefix(value, "https://"), "http://")

	return strings.TrimRight(trimmed, "/")
}

// authStore is OpenCode's auth store, kept whole so that a refresh
// preserves the logins and fields the adapter does not read.
type authStore struct {
	// document maps each provider to its login. The logins are opaque.
	document map[string]json.RawMessage

	// write replaces the store. It is nil for the read-only
	// OPENCODE_AUTH_CONTENT.
	write func(document map[string]json.RawMessage) error
}

// readAuthStore reads OPENCODE_AUTH_CONTENT when it is set, and OpenCode's
// auth.json otherwise. It returns nil when there is no readable store.
func readAuthStore(env map[string]string) *authStore {
	if content, ok := env["OPENCODE_AUTH_CONTENT"]; ok {
		document, ok := decodeObject([]byte(content))
		if !ok {
			return nil
		}

		return &authStore{
			document: document,
		}
	}

	authPath, ok := authFilePath(env)
	if !ok {
		return nil
	}

	content, err := os.ReadFile(authPath)
	if err != nil {
		return nil
	}

	document, ok := decodeObject(content)
	if !ok {
		return nil
	}

	return &authStore{
		document: document,
		write: func(document map[string]json.RawMessage) error {
			return writeAuthFile(authPath, document)
		},
	}
}

// authFilePath returns the path of OpenCode's auth.json: under
// XDG_DATA_HOME when it is set, and under the home directory otherwise.
func authFilePath(env map[string]string) (string, bool) {
	if dataHome, ok := env["XDG_DATA_HOME"]; ok {
		return filepath.Join(dataHome, "opencode", "auth.json"), true
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return "", false
	}

	if runtime.GOOS == "windows" {
		return filepath.Join(home, ".opencode", "auth.json"), true
	}

	return filepath.Join(home, ".local", "share", "opencode", "auth.json"), true
}

// entry returns the login of a provider when it is a JSON object.
func (s *authStore) entry(provider resolvedMode) (storeEntry, bool) {
	raw, ok := s.document[string(provider)]
	if !ok {
		return nil, false
	}

	fields, ok := decodeObject(raw)
	if !ok {
		return nil, false
	}

	return fields, true
}

// update merges refreshed tokens into a provider's login and writes the
// store. A read-only store is left unchanged.
func (s *authStore) update(provider resolvedMode, entry storeEntry, updates refreshedTokens) error {
	if s.write == nil {
		return nil
	}

	encodedUpdates, err := json.Marshal(updates)
	if err != nil {
		return err
	}

	updatedFields, ok := decodeObject(encodedUpdates)
	if !ok {
		return errors.New("encode refreshed login")
	}

	merged := maps.Clone(entry)
	maps.Copy(merged, updatedFields)

	encodedEntry, err := json.Marshal(merged)
	if err != nil {
		return err
	}

	s.document[string(provider)] = encodedEntry

	return s.write(s.document)
}

// storeEntry is one login of the auth store. Its fields are opaque JSON,
// read with the typed accessors below.
type storeEntry map[string]json.RawMessage

// optionalString returns a string field, reporting false when the field is
// missing or not a string.
func (e storeEntry) optionalString(name string) (string, bool) {
	raw, ok := e.present(name)
	if !ok {
		return "", false
	}

	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return "", false
	}

	return value, true
}

// stringField returns a string field, or empty.
func (e storeEntry) stringField(name string) string {
	value, _ := e.optionalString(name)

	return value
}

// number returns a numeric field, reporting false when the field is missing
// or not a number.
func (e storeEntry) number(name string) (float64, bool) {
	raw, ok := e.present(name)
	if !ok {
		return 0, false
	}

	var value float64
	if err := json.Unmarshal(raw, &value); err != nil {
		return 0, false
	}

	return value, true
}

// present returns a field that is set to a value other than null.
func (e storeEntry) present(name string) (json.RawMessage, bool) {
	raw, ok := e[name]
	if !ok || string(raw) == "null" {
		return nil, false
	}

	return raw, true
}

// decodeObject decodes a JSON object.
func decodeObject(content []byte) (map[string]json.RawMessage, bool) {
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
