package grokbuild

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"math"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/acp"
	"github.com/aarock1234/ai/pkg/harness/credential"
)

// xaiClientID is the OAuth client of the Grok CLI.
const xaiClientID = "b1a00492-073a-47ea-816f-4c329264a828"

// subscriptionBaseURL is the Grok CLI chat proxy that serves subscription
// logins.
const subscriptionBaseURL = "https://cli-chat-proxy.grok.com/v1"

// scopeSeparator separates the issuer from the client ID in an auth record
// scope.
const scopeSeparator = "::"

// maxDiscoveryResponseBytes bounds an OpenID discovery response.
const maxDiscoveryResponseBytes = 1 << 20

// secondsThreshold separates expiries in seconds from expiries in
// milliseconds.
const secondsThreshold = 10_000_000_000

// subscriptionResolver resolves the host's Grok CLI login, sending the
// OAuth requests of a refresh with client.
type subscriptionResolver struct {
	// client is nil for http.DefaultClient.
	client *http.Client
}

// resolveEnvironment adds the host's Grok CLI login, routed through the
// Grok CLI chat proxy, when no xAI API key or AI Gateway credential is
// configured. It is the adapter's acp.Settings.ResolveAuthenticationEnvironment.
func (r subscriptionResolver) resolveEnvironment(ctx context.Context, event acp.AuthenticationEnvironmentEvent) (map[string]string, error) {
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

	accessToken, ok, err := r.readSubscription(ctx, authFilePath(event.Env))
	if err != nil {
		return nil, err
	}

	if !ok {
		return event.Env, nil
	}

	environment := make(map[string]string, len(event.Env)+4)
	maps.Copy(environment, event.Env)
	environment[apiKeyVariable] = accessToken
	environment[apiBaseURLVariable] = subscriptionBaseURL
	environment[modelsBaseURLVariable] = subscriptionBaseURL
	environment[chatProxyBaseURLVariable] = subscriptionBaseURL

	return environment, nil
}

// authFilePath returns the Grok CLI auth file: auth.json in GROK_HOME, or
// in .grok of the user's home directory.
func authFilePath(env map[string]string) string {
	grokHome, ok := env["GROK_HOME"]
	if !ok {
		grokHome = filepath.Join(credential.HomeDirectory(), ".grok")
	}

	absolute, err := filepath.Abs(grokHome)
	if err != nil {
		absolute = grokHome
	}

	return filepath.Join(absolute, "auth.json")
}

// oauthRecord is the selected OAuth login of an auth file.
type oauthRecord struct {
	scope        string
	issuer       string
	accessToken  string
	refreshToken string
	expiresAt    time.Time
}

// readSubscription returns the access token of the Grok CLI's OAuth login,
// refreshing and storing it when it expires soon. It reports false when no
// login is stored.
func (r subscriptionResolver) readSubscription(ctx context.Context, authPath string) (string, bool, error) {
	unlock := credential.LockRefresh(authPath)
	defer unlock()

	content, err := os.ReadFile(authPath)
	if err != nil {
		return "", false, nil
	}

	records, scopes, ok := parseAuthRecords(content)
	if !ok {
		return "", false, nil
	}

	record, ok := selectOAuthRecord(records, scopes)
	if !ok {
		return "", false, nil
	}

	if !credential.IsAccessTokenExpiringSoon(record.expiresAt, time.Now(), 0) {
		return record.accessToken, true, nil
	}

	accessToken, err := r.refreshSubscription(ctx, refreshTarget{
		authPath: authPath,
		records:  records,
		record:   record,
	})
	if err != nil {
		return "", false, err
	}

	return accessToken, true, nil
}

// parseAuthRecords decodes an auth file as a JSON object, returning its
// records and their scopes in file order.
func parseAuthRecords(content []byte) (map[string]json.RawMessage, []string, bool) {
	var records map[string]json.RawMessage
	if err := json.Unmarshal(content, &records); err != nil || records == nil {
		return nil, nil, false
	}

	scopes, err := objectKeys(content)
	if err != nil {
		return nil, nil, false
	}

	return records, scopes, true
}

// objectKeys returns the keys of a JSON object in document order.
func objectKeys(content []byte) ([]string, error) {
	decoder := json.NewDecoder(bytes.NewReader(content))

	if _, err := decoder.Token(); err != nil {
		return nil, err
	}

	var keys []string

	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return nil, err
		}

		key, ok := token.(string)
		if !ok {
			return nil, fmt.Errorf("unexpected json object key %v", token)
		}

		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			return nil, err
		}

		keys = append(keys, key)
	}

	return keys, nil
}

// authRecordJSON is the part of an auth record the adapter reads.
type authRecordJSON struct {
	Key          *string         `json:"key"`
	RefreshToken *string         `json:"refresh_token"`
	ExpiresAt    json.RawMessage `json:"expires_at"`
}

// selectOAuthRecord returns the first record of the Grok CLI's OAuth client
// with a key, a refresh token and an expiry.
func selectOAuthRecord(records map[string]json.RawMessage, scopes []string) (oauthRecord, bool) {
	for _, scope := range scopes {
		issuer, clientID, ok := cutLast(scope, scopeSeparator)
		if !ok || issuer == "" || clientID != xaiClientID {
			continue
		}

		var candidate authRecordJSON
		if err := json.Unmarshal(records[scope], &candidate); err != nil {
			continue
		}

		if candidate.Key == nil || candidate.RefreshToken == nil {
			continue
		}

		expiresAt, ok := normalizeExpiresAt(candidate.ExpiresAt)
		if !ok {
			continue
		}

		return oauthRecord{
			scope:        scope,
			issuer:       issuer,
			accessToken:  *candidate.Key,
			refreshToken: *candidate.RefreshToken,
			expiresAt:    expiresAt,
		}, true
	}

	return oauthRecord{}, false
}

// cutLast slices value around the last separator.
func cutLast(value string, separator string) (string, string, bool) {
	index := strings.LastIndex(value, separator)
	if index < 0 {
		return "", "", false
	}

	return value[:index], value[index+len(separator):], true
}

// normalizeExpiresAt decodes an expiry in seconds or milliseconds since the
// epoch, or as a date string.
func normalizeExpiresAt(raw json.RawMessage) (time.Time, bool) {
	var number float64
	if err := json.Unmarshal(raw, &number); err == nil {
		if number < secondsThreshold {
			number *= 1000
		}

		return time.UnixMilli(int64(math.Round(number))), true
	}

	var text string
	if err := json.Unmarshal(raw, &text); err != nil {
		return time.Time{}, false
	}

	parsed, err := time.Parse(time.RFC3339Nano, text)
	if err != nil {
		return time.Time{}, false
	}

	return parsed, true
}

// refreshTarget is the login to refresh and the auth file that stores it.
type refreshTarget struct {
	authPath string
	records  map[string]json.RawMessage
	record   oauthRecord
}

// refreshSubscription refreshes an access token through the token endpoint
// of the login's issuer and stores the new tokens in the auth file.
func (r subscriptionResolver) refreshSubscription(ctx context.Context, target refreshTarget) (string, error) {
	tokenURL, err := r.discoverTokenEndpoint(ctx, target.record.issuer)
	if err != nil {
		return "", err
	}

	refreshed, err := credential.RefreshOAuthAccessToken(ctx, credential.RefreshOptions{
		TokenURL:     tokenURL,
		ClientID:     xaiClientID,
		RefreshToken: target.record.refreshToken,
		Client:       r.client,
	})
	if err != nil {
		return "", fmt.Errorf("refresh grok subscription: %w", err)
	}

	refreshToken := target.record.refreshToken
	if refreshed.RefreshToken != "" {
		refreshToken = refreshed.RefreshToken
	}

	updated, err := updatedRecord(target.records[target.record.scope], storedTokens{
		accessToken:  refreshed.AccessToken,
		refreshToken: refreshToken,
		expiresAt:    refreshed.ExpiresAt,
	})
	if err != nil {
		return "", err
	}

	records := maps.Clone(target.records)
	records[target.record.scope] = updated

	if err := writeAuthFile(target.authPath, records); err != nil {
		return "", fmt.Errorf("store refreshed grok subscription: %w", err)
	}

	return refreshed.AccessToken, nil
}

// openIDConfiguration is the part of an OpenID configuration the refresh
// reads.
type openIDConfiguration struct {
	TokenEndpoint *string `json:"token_endpoint"`
}

// discoverTokenEndpoint reads the token endpoint from the issuer's OpenID
// configuration and checks that it belongs to the issuer.
func (r subscriptionResolver) discoverTokenEndpoint(ctx context.Context, issuer string) (string, error) {
	discoveryURL := strings.TrimRight(issuer, "/") + "/.well-known/openid-configuration"

	request, err := http.NewRequestWithContext(ctx, http.MethodGet, discoveryURL, nil)
	if err != nil {
		return "", fmt.Errorf("create grok oauth discovery request: %w", err)
	}

	response, err := r.httpClient().Do(request)
	if err != nil {
		return "", fmt.Errorf("grok oauth discovery: %w", err)
	}

	defer func() {
		_ = response.Body.Close()
	}()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return "", fmt.Errorf("grok oauth discovery failed with status %d", response.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(response.Body, maxDiscoveryResponseBytes))
	if err != nil {
		return "", fmt.Errorf("read grok oauth discovery response: %w", err)
	}

	var discovery openIDConfiguration
	if err := json.Unmarshal(body, &discovery); err != nil || discovery.TokenEndpoint == nil {
		return "", errors.New("grok oauth discovery returned no token endpoint")
	}

	tokenEndpoint := *discovery.TokenEndpoint

	if err := validateTokenEndpoint(issuer, tokenEndpoint); err != nil {
		return "", err
	}

	return tokenEndpoint, nil
}

// httpClient returns the configured client or http.DefaultClient.
func (r subscriptionResolver) httpClient() *http.Client {
	if r.client == nil {
		return http.DefaultClient
	}

	return r.client
}

// validateTokenEndpoint requires an HTTPS token endpoint on the same site
// as the issuer, so that a tampered discovery response cannot redirect the
// refresh token to another host.
//
// Go adaptation: without the public suffix list, the site of a host is
// approximated by its last two labels.
func validateTokenEndpoint(issuer string, tokenEndpoint string) error {
	issuerURL, err := url.Parse(issuer)
	if err != nil {
		return fmt.Errorf("parse grok oauth issuer: %w", err)
	}

	endpointURL, err := url.Parse(tokenEndpoint)
	if err != nil {
		return fmt.Errorf("parse grok oauth token endpoint: %w", err)
	}

	if endpointURL.Scheme != "https" {
		return fmt.Errorf("grok oauth token endpoint %q must use https", tokenEndpoint)
	}

	issuerSite := siteOf(issuerURL.Hostname())
	endpointSite := siteOf(endpointURL.Hostname())

	if issuerSite == "" || endpointSite != issuerSite {
		return fmt.Errorf("grok oauth token endpoint %q is not on the site of issuer %q", tokenEndpoint, issuer)
	}

	return nil
}

// siteOf returns the last two labels of a host name, lowercased.
func siteOf(host string) string {
	labels := strings.Split(strings.ToLower(strings.TrimSuffix(host, ".")), ".")
	if len(labels) < 2 {
		return strings.Join(labels, ".")
	}

	return strings.Join(labels[len(labels)-2:], ".")
}

// storedTokens are the refreshed tokens written to the auth file.
type storedTokens struct {
	accessToken  string
	refreshToken string
	expiresAt    time.Time
}

// updatedRecord returns the record with the refreshed tokens and their
// expiry in milliseconds, keeping its other fields.
func updatedRecord(original json.RawMessage, tokens storedTokens) (json.RawMessage, error) {
	var record map[string]json.RawMessage
	if err := json.Unmarshal(original, &record); err != nil || record == nil {
		record = make(map[string]json.RawMessage, 3)
	}

	key, err := json.Marshal(tokens.accessToken)
	if err != nil {
		return nil, fmt.Errorf("encode grok auth record key: %w", err)
	}

	refreshToken, err := json.Marshal(tokens.refreshToken)
	if err != nil {
		return nil, fmt.Errorf("encode grok auth record refresh_token: %w", err)
	}

	record["key"] = key
	record["refresh_token"] = refreshToken
	record["expires_at"] = json.RawMessage(strconv.FormatInt(tokens.expiresAt.UnixMilli(), 10))

	encoded, err := json.Marshal(record)
	if err != nil {
		return nil, fmt.Errorf("encode grok auth record: %w", err)
	}

	return encoded, nil
}

// writeAuthFile replaces auth.json atomically, readable only by its owner.
func writeAuthFile(authPath string, records map[string]json.RawMessage) error {
	content, err := json.MarshalIndent(records, "", "  ")
	if err != nil {
		return err
	}

	return credential.WriteFileAtomically(authPath, append(content, '\n'))
}
