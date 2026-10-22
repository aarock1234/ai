package githubcopilot

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"maps"
	"net/url"
	"slices"
	"strings"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/acp"
)

// Environment variables naming the GitHub host.
const (
	hostVariable   = "COPILOT_GH_HOST"
	ghHostVariable = "GH_HOST"
)

// defaultGitHubHost is the GitHub host without a host override.
const defaultGitHubHost = "github.com"

// copilotTokenVariable holds the token the native subscription resolves
// to.
const copilotTokenVariable = "COPILOT_GITHUB_TOKEN"

// Environment variables of a direct model provider configured for Copilot.
const (
	providerAPIKeyVariable  = "COPILOT_PROVIDER_API_KEY"
	providerBaseURLVariable = "COPILOT_PROVIDER_BASE_URL"
)

// tokenVariables hold GitHub tokens, in Copilot's order of precedence.
var tokenVariables = []string{
	copilotTokenVariable,
	"GH_TOKEN",
	"GITHUB_TOKEN",
}

// authorizationSchemes are the schemes Copilot sends GitHub tokens with.
var authorizationSchemes = []string{
	"Bearer",
	"token",
}

// GitHub OAuth tokens are "gho_" followed by a 36-character alphanumeric
// body. Sandbox placeholders mark their body with oauthPlaceholderMarker.
const (
	oauthTokenBodyLength   = 36
	oauthPlaceholderMarker = "aisdkhc"
)

// credentialTransformations inject the real GitHub tokens into requests to
// the GitHub and Copilot hosts, and the real provider API key or the
// agent's request headers into model requests.
func credentialTransformations(event acp.CredentialBrokeringEvent) ([]harness.RequestTransformation, error) {
	githubHost, err := normalizeGitHubHost(configuredGitHubHost(event.Env))
	if err != nil {
		return nil, err
	}

	var transformations []harness.RequestTransformation

	for _, host := range githubHosts(githubHost) {
		transformations = append(transformations, tokenTransformations(host, event)...)
	}

	providerTransformation, ok, err := providerCredentialTransformation(event)
	if err != nil {
		return nil, err
	}

	if ok {
		return append(transformations, providerTransformation), nil
	}

	if event.Headers == nil {
		return transformations, nil
	}

	for _, host := range copilotHosts(githubHost) {
		transformations = append(transformations, harness.RequestTransformation{
			Match: harness.RequestMatch{
				Host: host,
			},
			Transform: harness.RequestTransform{
				Headers: event.Headers,
			},
		})
	}

	return transformations, nil
}

// configuredGitHubHost returns the GitHub host the environment selects.
func configuredGitHubHost(env map[string]string) string {
	host, ok := firstVariable(env, hostVariable, ghHostVariable)
	if !ok {
		return defaultGitHubHost
	}

	return host
}

// githubHosts are the hosts GitHub tokens are sent to: the GitHub host with
// its subdomains, and Copilot's API hosts for github.com.
func githubHosts(githubHost string) []string {
	hosts := []string{
		githubHost,
		"*." + githubHost,
	}

	if githubHost != defaultGitHubHost {
		return hosts
	}

	return append(hosts, copilotHosts(githubHost)...)
}

// copilotHosts are the hosts that serve Copilot's model requests.
func copilotHosts(githubHost string) []string {
	if githubHost == defaultGitHubHost {
		return []string{
			"githubcopilot.com",
			"*.githubcopilot.com",
		}
	}

	return []string{
		githubHost,
		"*." + githubHost,
	}
}

// tokenTransformations replace each placeholder token's authorization
// headers on one host with the real token.
func tokenTransformations(host string, event acp.CredentialBrokeringEvent) []harness.RequestTransformation {
	var transformations []harness.RequestTransformation

	for _, name := range tokenVariables {
		token := event.Env[name]
		placeholder := event.SandboxEnv[name]

		if token == "" || placeholder == "" {
			continue
		}

		for _, scheme := range authorizationSchemes {
			transformations = append(transformations, authorizationTransformation(authorizationReplacement{
				host:        host,
				placeholder: scheme + " " + placeholder,
				credential:  scheme + " " + token,
			}))
		}
	}

	return transformations
}

// authorizationReplacement replaces one Authorization header on one host.
type authorizationReplacement struct {
	host        string
	placeholder string
	credential  string
}

// authorizationTransformation returns the request transformation of a
// replacement.
func authorizationTransformation(replacement authorizationReplacement) harness.RequestTransformation {
	return harness.RequestTransformation{
		Match: harness.RequestMatch{
			Host: replacement.host,
			Headers: []harness.KeyValueMatcher{
				{
					Key: &harness.Matcher{
						Exact: "Authorization",
					},
					Value: &harness.Matcher{
						Exact: replacement.placeholder,
					},
				},
			},
		},
		Transform: harness.RequestTransform{
			Headers: map[string]string{
				"Authorization": replacement.credential,
			},
		},
	}
}

// providerCredentialTransformation injects the real provider API key and
// the agent's request headers into requests to a configured provider base
// URL. It reports false without a brokered provider credential.
func providerCredentialTransformation(event acp.CredentialBrokeringEvent) (harness.RequestTransformation, bool, error) {
	credential := event.Env[providerAPIKeyVariable]
	placeholder := event.SandboxEnv[providerAPIKeyVariable]
	baseURL := event.Env[providerBaseURLVariable]

	if credential == "" || placeholder == "" || baseURL == "" {
		return harness.RequestTransformation{}, false, nil
	}

	headers := maps.Clone(event.Headers)
	if headers == nil {
		headers = make(map[string]string, 1)
	}

	headers["Authorization"] = "Bearer " + credential

	transformation, err := harness.NewCredentialRequestTransformation(harness.CredentialRequestTransformationOptions{
		MatchURL: baseURL,
		MatchHeaders: map[string]string{
			"Authorization": "Bearer " + placeholder,
		},
		TransformHeaders: headers,
	})
	if err != nil {
		return harness.RequestTransformation{}, false, err
	}

	return transformation, true, nil
}

// newCredentialForwarding returns the credential forwarding of the
// adapter: GitHub token placeholders take the GitHub OAuth token format
// before the caller's forwarding applies.
func newCredentialForwarding(forwarding harness.CredentialForwardingFunc) harness.CredentialForwardingFunc {
	return func(ctx context.Context, event harness.CredentialForwardingEvent) (string, error) {
		if isTokenPlaceholder(event) {
			event.Credential = oauthTokenPlaceholder(event.Credential)
		}

		if forwarding == nil {
			return event.Credential, nil
		}

		return forwarding(ctx, event)
	}
}

// isTokenPlaceholder reports a generated placeholder for a GitHub token.
func isTokenPlaceholder(event harness.CredentialForwardingEvent) bool {
	return slices.Contains(tokenVariables, event.EnvironmentVariableName) &&
		harness.IsSandboxCredentialPlaceholder(event.Credential)
}

// oauthTokenPlaceholder derives a placeholder in the GitHub OAuth token
// format, because Copilot CLI validates the format of environment tokens
// before it makes the requests credential brokering transforms.
func oauthTokenPlaceholder(placeholder string) string {
	digest := sha256.Sum256([]byte(placeholder))
	body := hex.EncodeToString(digest[:])[:oauthTokenBodyLength-len(oauthPlaceholderMarker)]

	return "gho_" + oauthPlaceholderMarker + body
}

// normalizeGitHubHost returns the hostname of a GitHub host or URL.
func normalizeGitHubHost(host string) (string, error) {
	parsed, err := parseGitHubHost(host)
	if err != nil {
		return "", err
	}

	return parsed.Hostname(), nil
}

// normalizeGitHubOrigin returns the origin of a GitHub host or URL, and
// false when it is not a valid URL.
func normalizeGitHubOrigin(host string) (string, bool) {
	parsed, err := parseGitHubHost(host)
	if err != nil {
		return "", false
	}

	origin := parsed.Scheme + "://" + parsed.Hostname()

	port := parsed.Port()
	if port != "" && !isDefaultPort(parsed.Scheme, port) {
		origin += ":" + port
	}

	return origin, true
}

// parseGitHubHost parses a host or URL, assuming HTTPS without a scheme.
// The scheme and host are lowercase, as in a WHATWG URL.
func parseGitHubHost(host string) (*url.URL, error) {
	address := host
	if !strings.Contains(address, "://") {
		address = "https://" + address
	}

	parsed, err := url.Parse(address)
	if err != nil {
		return nil, fmt.Errorf("parse github host %q: %w", host, err)
	}

	if parsed.Hostname() == "" {
		return nil, fmt.Errorf("parse github host %q: missing hostname", host)
	}

	parsed.Scheme = strings.ToLower(parsed.Scheme)
	parsed.Host = strings.ToLower(parsed.Host)

	return parsed, nil
}

// isDefaultPort reports the default port of an HTTP scheme.
func isDefaultPort(scheme string, port string) bool {
	return (scheme == "https" && port == "443") || (scheme == "http" && port == "80")
}
