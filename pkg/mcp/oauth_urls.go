package mcp

import (
	"net/http"
	"net/url"
	"regexp"
	"strings"

	"github.com/aarock1234/ai/internal/download"
	"github.com/aarock1234/ai/internal/origin"
)

// pathnameOf returns the URL's path as JavaScript's URL.pathname does for
// hierarchical URLs: "/" when empty.
func pathnameOf(target *url.URL) string {
	if target.EscapedPath() == "" {
		return "/"
	}

	return target.EscapedPath()
}

// normalizeURL returns raw in a canonical form, so that URLs from metadata
// and storage compare equal when they name the same endpoint.
func normalizeURL(raw string) (string, error) {
	parsed, err := parseAbsoluteURL(raw)
	if err != nil {
		return "", err
	}

	normalized := *parsed
	normalized.Scheme = strings.ToLower(parsed.Scheme)
	normalized.Host = strings.TrimPrefix(origin.Of(parsed), normalized.Scheme+"://")

	if normalized.Path == "" && normalized.Opaque == "" {
		normalized.Path = "/"
	}

	return normalized.String(), nil
}

// resolvePath returns the URL with base's origin and the absolute path.
func resolvePath(base *url.URL, path string) *url.URL {
	return &url.URL{
		Scheme: base.Scheme,
		Host:   base.Host,
		Path:   path,
	}
}

// isLoopbackHost reports loopback hosts, for which OAuth allows plain HTTP
// (RFC 8252 section 7.3, RFC 6761 section 6.3).
func isLoopbackHost(hostname string) bool {
	normalized := strings.TrimRight(strings.ToLower(hostname), ".")

	return normalized == "localhost" ||
		strings.HasSuffix(normalized, ".localhost") ||
		normalized == "127.0.0.1" ||
		normalized == "[::1]" ||
		normalized == "::1"
}

func isHTTPScheme(target *url.URL) bool {
	scheme := strings.ToLower(target.Scheme)

	return scheme == "http" || scheme == "https"
}

// trustedLoopbackOrigin returns the origin of a loopback authorization
// server, which the developer configured locally, and empty otherwise.
func trustedLoopbackOrigin(authorizationServerURL *url.URL) string {
	if !isLoopbackHost(authorizationServerURL.Hostname()) {
		return ""
	}

	return origin.Of(authorizationServerURL)
}

// assertSafeOAuthEndpoint guards a metadata-derived OAuth URL against
// server-side request forgery before it is requested. Loopback endpoints
// pass only when allowLoopback establishes that they belong to a locally
// configured server.
func assertSafeOAuthEndpoint(endpoint *url.URL, allowLoopback bool) error {
	if allowLoopback && isHTTPScheme(endpoint) && isLoopbackHost(endpoint.Hostname()) {
		return nil
	}

	if err := download.ValidateURL(endpoint.String()); err != nil {
		return &OAuthError{
			Message: "oauth endpoint url is not allowed: " + endpoint.String(),
			Cause:   err,
		}
	}

	return nil
}

// resourceURLFromServerURL returns the server URL without its fragment,
// which RFC 8707 forbids in resource indicators.
func resourceURLFromServerURL(serverURL *url.URL) *url.URL {
	resource := *serverURL
	resource.Fragment = ""
	resource.RawFragment = ""

	return &resource
}

// resourceURLString formats a resource indicator without the trailing slash
// of a pathless URL, as the MCP specification recommends.
func resourceURLString(resource *url.URL) string {
	text := resource.String()

	if pathnameOf(resource) == "/" && strings.HasSuffix(text, "/") {
		return strings.TrimSuffix(text, "/")
	}

	return text
}

// isResourceAllowed reports whether requested has the origin of configured
// and a path at or below configured's path.
func isResourceAllowed(requested *url.URL, configured *url.URL) bool {
	if !origin.SameURL(requested, configured) {
		return false
	}

	requestedPath := pathnameOf(requested)
	configuredPath := pathnameOf(configured)

	if len(requestedPath) < len(configuredPath) {
		return false
	}

	return strings.HasPrefix(withTrailingSlash(requestedPath), withTrailingSlash(configuredPath))
}

func withTrailingSlash(path string) string {
	if strings.HasSuffix(path, "/") {
		return path
	}

	return path + "/"
}

var (
	resourceMetadataParameter = regexp.MustCompile(`(?i)(?:^|[,\s])resource_metadata="([^"]*)"`)
	scopeParameter            = regexp.MustCompile(`(?i)(?:^|[,\s])scope="([^"]*)"`)
)

// wwwAuthenticateParameters are the parameters of a Bearer challenge.
type wwwAuthenticateParameters struct {
	resourceMetadataURL *url.URL
	scope               string
}

// parseWWWAuthenticate reads the protected resource metadata URL (RFC 9728)
// and the required scope from a Bearer WWW-Authenticate challenge.
func parseWWWAuthenticate(headers http.Header) wwwAuthenticateParameters {
	header := headers.Get("WWW-Authenticate")

	challengeType, rest, _ := strings.Cut(header, " ")
	scheme, _, _ := strings.Cut(rest, " ")

	if !strings.EqualFold(challengeType, "bearer") || scheme == "" {
		return wwwAuthenticateParameters{}
	}

	var parameters wwwAuthenticateParameters

	if match := scopeParameter.FindStringSubmatch(header); match != nil {
		parameters.scope = match[1]
	}

	if match := resourceMetadataParameter.FindStringSubmatch(header); match != nil {
		if metadataURL, err := parseAbsoluteURL(match[1]); err == nil {
			parameters.resourceMetadataURL = metadataURL
		}
	}

	return parameters
}
