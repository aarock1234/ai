package vercel

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/pkg/harness/credential"
)

// The Vercel CLI's OAuth client, used to refresh its access token.
const (
	vercelIssuer      = "https://vercel.com"
	vercelCLIClientID = "cl_HYyOPBNtFMfHhaUn9L4QPfTZz6TP47bp"
)

// credentialStorage is where the Vercel CLI keeps its credentials.
type credentialStorage string

const (
	credentialStorageAuto    credentialStorage = "auto"
	credentialStorageFile    credentialStorage = "file"
	credentialStorageKeyring credentialStorage = "keyring"
)

// cliConfigDir returns the Vercel CLI's global configuration directory, a
// port of @vercel/cli-config's getGlobalPathConfig: the first existing
// directory among the current, the legacy home and the legacy XDG
// locations, or the current location.
func cliConfigDir() string {
	current := cliDataDir("com.vercel.cli")

	candidates := []string{
		current,
	}

	if home, err := os.UserHomeDir(); err == nil {
		candidates = append(candidates, filepath.Join(home, ".now"))
	}

	candidates = append(candidates, cliDataDir("now"))

	for _, candidate := range candidates {
		if isDirectory(candidate) {
			return candidate
		}
	}

	return current
}

// cliDataDir returns the XDG data directory of an application as the
// xdg-app-paths package resolves it.
func cliDataDir(application string) string {
	if dataHome := os.Getenv("XDG_DATA_HOME"); dataHome != "" {
		return filepath.Join(dataHome, application)
	}

	home, _ := os.UserHomeDir()

	switch runtime.GOOS {
	case "darwin":
		return filepath.Join(home, "Library", "Application Support", application)

	case "windows":
		appData := os.Getenv("APPDATA")
		if appData == "" {
			appData = filepath.Join(home, "AppData", "Roaming")
		}

		return filepath.Join(appData, "xdg.data", application)

	default:
		return filepath.Join(home, ".local", "share", application)
	}
}

func isDirectory(path string) bool {
	info, err := os.Lstat(path)

	return err == nil && info.IsDir()
}

// cliAuthConfig is the part of the Vercel CLI's auth.json the client
// reads.
type cliAuthConfig struct {
	Token        string   `json:"token"`
	RefreshToken string   `json:"refreshToken"`
	ExpiresAt    *float64 `json:"expiresAt"`
}

func (c cliAuthConfig) hasTokenData() bool {
	return c.Token != "" || c.RefreshToken != ""
}

// isValid reports whether the access token stays valid for buffer. Tokens
// without an expiration stay valid.
func (c cliAuthConfig) isValid(buffer time.Duration) bool {
	if c.Token == "" {
		return false
	}

	if c.ExpiresAt == nil {
		return true
	}

	return *c.ExpiresAt >= float64(time.Now().Add(buffer).Unix())
}

func authConfigPath(configDir string) string {
	return filepath.Join(configDir, "auth.json")
}

// readCLIAuthConfig reads auth.json, reporting a missing or unreadable file
// as no configuration.
func readCLIAuthConfig(configDir string) (cliAuthConfig, bool) {
	content, err := os.ReadFile(authConfigPath(configDir))
	if err != nil {
		return cliAuthConfig{}, false
	}

	var config cliAuthConfig
	if err := json.Unmarshal(trimByteOrderMark(content), &config); err != nil {
		return cliAuthConfig{}, false
	}

	return config, true
}

func trimByteOrderMark(content []byte) []byte {
	return []byte(strings.TrimPrefix(string(content), "\uFEFF"))
}

// cliCredentialStorage returns where the CLI likely keeps its credentials:
// VERCEL_TOKEN_STORAGE, or the credStorage of config.json, with "auto"
// meaning the auth.json file when it holds token data and the keyring
// otherwise. The default is the file.
func cliCredentialStorage(configDir string) (credentialStorage, error) {
	if override, ok := os.LookupEnv("VERCEL_TOKEN_STORAGE"); ok {
		return configuredStorage(configDir, credentialStorage(override), "VERCEL_TOKEN_STORAGE")
	}

	content, err := os.ReadFile(filepath.Join(configDir, "config.json"))
	if errors.Is(err, os.ErrNotExist) {
		return credentialStorageFile, nil
	}

	if err != nil {
		return "", fmt.Errorf("read vercel cli config: %w", err)
	}

	var config struct {
		CredStorage *string `json:"credStorage"`
	}

	if err := json.Unmarshal(trimByteOrderMark(content), &config); err != nil {
		return "", fmt.Errorf("decode vercel cli config: %w", err)
	}

	if config.CredStorage == nil {
		return credentialStorageFile, nil
	}

	return configuredStorage(configDir, credentialStorage(*config.CredStorage), "credStorage")
}

func configuredStorage(configDir string, storage credentialStorage, source string) (credentialStorage, error) {
	switch storage {
	case credentialStorageKeyring, credentialStorageFile:
		return storage, nil

	case credentialStorageAuto:
		config, ok := readCLIAuthConfig(configDir)
		if ok && config.hasTokenData() {
			return credentialStorageFile, nil
		}

		return credentialStorageKeyring, nil

	default:
		return "", fmt.Errorf("invalid value for `%s`: %q; expected one of \"auto\", \"file\", \"keyring\"", source, storage)
	}
}

// cliAccessTokenRequest describes the lookup of the CLI's access token.
type cliAccessTokenRequest struct {
	configDir  string
	buffer     time.Duration
	httpClient *http.Client
}

// cliAccessToken returns the CLI's access token from auth.json, refreshing
// it with its refresh token when it expires.
//
// Go adaptation: upstream clears auth.json when the refresh fails. This
// port leaves the CLI's configuration unchanged.
func cliAccessToken(ctx context.Context, request cliAccessTokenRequest) (string, error) {
	unlock := credential.LockRefresh(authConfigPath(request.configDir))
	defer unlock()

	config, ok := readCLIAuthConfig(request.configDir)
	if !ok || !config.hasTokenData() {
		return "", errors.New("no vercel cli access token found; log in with `vercel login`")
	}

	if config.isValid(request.buffer) {
		return config.Token, nil
	}

	if config.RefreshToken == "" {
		return "", errors.New("failed to refresh the vercel cli access token: no refresh token available")
	}

	tokens, err := refreshCLIAccessToken(ctx, request.httpClient, config.RefreshToken)
	if err != nil {
		return "", fmt.Errorf("failed to refresh the vercel cli access token: %w", err)
	}

	if err := writeCLIAuthTokens(request.configDir, tokens); err != nil {
		return "", err
	}

	return tokens.AccessToken, nil
}

// oauthTokens is a successful OAuth token response.
type oauthTokens struct {
	AccessToken  string   `json:"access_token"`
	TokenType    string   `json:"token_type"`
	ExpiresIn    *float64 `json:"expires_in"`
	RefreshToken string   `json:"refresh_token"`
}

func (t oauthTokens) validate() error {
	switch {
	case t.AccessToken == "":
		return errors.New("missing access_token in response")
	case t.TokenType != "Bearer":
		return errors.New("invalid token_type in response")
	case t.ExpiresIn == nil:
		return errors.New("missing expires_in in response")
	default:
		return nil
	}
}

// refreshCLIAccessToken exchanges a refresh token at the token endpoint of
// Vercel's OpenID configuration.
func refreshCLIAccessToken(ctx context.Context, httpClient *http.Client, refreshToken string) (oauthTokens, error) {
	server, err := discoverAuthorizationServer(ctx, httpClient)
	if err != nil {
		return oauthTokens{}, err
	}

	form := url.Values{
		"client_id": {
			vercelCLIClientID,
		},
		"grant_type": {
			"refresh_token",
		},
		"refresh_token": {
			refreshToken,
		},
	}

	request, err := newOAuthFormRequest(ctx, server.TokenEndpoint, form)
	if err != nil {
		return oauthTokens{}, fmt.Errorf("create token refresh request: %w", err)
	}

	var tokens oauthTokens

	var failure struct {
		Error string `json:"error"`
	}

	status, err := doOAuthJSON(httpClient, request, &tokens, &failure)
	if err != nil {
		return oauthTokens{}, err
	}

	if status < 200 || status >= 300 {
		if failure.Error != "" {
			return oauthTokens{}, errors.New(failure.Error)
		}

		return oauthTokens{}, errors.New("token refresh failed")
	}

	if err := tokens.validate(); err != nil {
		return oauthTokens{}, err
	}

	if tokens.RefreshToken == "" {
		tokens.RefreshToken = refreshToken
	}

	return tokens, nil
}

// authorizationServer is the part of Vercel's OpenID configuration that the
// CLI's OAuth flows use.
type authorizationServer struct {
	DeviceAuthorizationEndpoint string `json:"device_authorization_endpoint"`
	TokenEndpoint               string `json:"token_endpoint"`
}

// discoverAuthorizationServer reads Vercel's OpenID configuration, which
// must name a token endpoint.
func discoverAuthorizationServer(ctx context.Context, httpClient *http.Client) (authorizationServer, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, vercelIssuer+"/.well-known/openid-configuration", nil)
	if err != nil {
		return authorizationServer{}, fmt.Errorf("create oauth discovery request: %w", err)
	}

	request.Header.Set("User-Agent", userAgent)

	var server authorizationServer

	status, err := doOAuthJSON(httpClient, request, &server, nil)
	if err != nil {
		return authorizationServer{}, err
	}

	if status < 200 || status >= 300 {
		return authorizationServer{}, errors.New("failed to discover oauth endpoints")
	}

	if server.TokenEndpoint == "" {
		return authorizationServer{}, errors.New("invalid oauth discovery response")
	}

	return server, nil
}

// doOAuthJSON sends a request and decodes a successful response into
// success and a failed one into failure, when it is not nil.
func doOAuthJSON(httpClient *http.Client, request *http.Request, success any, failure any) (int, error) {
	response, err := httpClient.Do(request)
	if err != nil {
		return 0, err
	}

	defer func() {
		_ = response.Body.Close()
	}()

	content, err := io.ReadAll(io.LimitReader(response.Body, maxErrorBodyBytes))
	if err != nil {
		return 0, fmt.Errorf("read oauth response: %w", err)
	}

	target := success
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		target = failure
	}

	if target != nil {
		if err := json.Unmarshal(content, target); err != nil {
			return 0, fmt.Errorf("decode oauth response: %w", err)
		}
	}

	return response.StatusCode, nil
}

// writeCLIAuthTokens stores new tokens in auth.json, keeping its other
// members, with an atomic replace readable only by the user. Tokens without
// a refresh token remove the stored one. An auth.json that is not a JSON
// object is reported instead of overwritten.
func writeCLIAuthTokens(configDir string, tokens oauthTokens) error {
	path := authConfigPath(configDir)

	members, err := readAuthMembers(path)
	if err != nil {
		return err
	}

	expiresAt := time.Now().Unix() + int64(*tokens.ExpiresIn)

	members["token"] = jsonString(tokens.AccessToken)
	members["expiresAt"] = json.RawMessage(strconv.FormatInt(expiresAt, 10))

	if tokens.RefreshToken == "" {
		delete(members, "refreshToken")
	} else {
		members["refreshToken"] = jsonString(tokens.RefreshToken)
	}

	content, err := json.MarshalIndent(members, "", "  ")
	if err != nil {
		return fmt.Errorf("encode vercel cli auth config: %w", err)
	}

	if err := credential.WriteFileAtomically(path, append(content, '\n')); err != nil {
		return fmt.Errorf("write vercel cli auth config: %w", err)
	}

	return nil
}

// readAuthMembers reads the members of auth.json. A missing file has no
// members.
func readAuthMembers(path string) (map[string]json.RawMessage, error) {
	content, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return make(map[string]json.RawMessage), nil
	}

	if err != nil {
		return nil, fmt.Errorf("read vercel cli auth config: %w", err)
	}

	var members map[string]json.RawMessage
	if err := json.Unmarshal(trimByteOrderMark(content), &members); err != nil {
		return nil, fmt.Errorf("decode vercel cli auth config %s: %w", path, err)
	}

	if members == nil {
		return nil, fmt.Errorf("vercel cli auth config %s is not a json object", path)
	}

	return members, nil
}

func jsonString(value string) json.RawMessage {
	encoded, _ := json.Marshal(value)

	return encoded
}
