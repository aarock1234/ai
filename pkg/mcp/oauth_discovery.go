package mcp

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"strings"

	"github.com/aarock1234/ai/internal/download"
	"github.com/aarock1234/ai/internal/origin"
)

// maxMetadataBytes bounds a metadata document.
const maxMetadataBytes = 1 << 20

// Well-known metadata documents.
const (
	wellKnownAuthorizationServer = "oauth-authorization-server"
	wellKnownProtectedResource   = "oauth-protected-resource"
	wellKnownOpenIDConfiguration = "openid-configuration"
)

// metadataResponse is a fetched metadata document. body is set only for
// successful responses.
type metadataResponse struct {
	status int
	body   []byte
}

func (r *metadataResponse) ok() bool {
	return r.status >= 200 && r.status <= 299
}

func (r *metadataResponse) isClientError() bool {
	return r.status >= 400 && r.status <= 499
}

// fetchMetadata fetches a metadata document with the download guards: the
// URL and every redirect are validated unless they are on trustedOrigin.
// It returns nil without an error when the server could not be reached, so
// that discovery tries the next candidate.
func fetchMetadata(ctx context.Context, client *http.Client, target *url.URL, protocolVersion string, trustedOrigin string) (*metadataResponse, error) {
	file, err := download.Fetch(ctx, download.Request{
		URL: target.String(),
		Headers: http.Header{
			"Accept": {
				"application/json",
			},
			"Mcp-Protocol-Version": {
				protocolVersion,
			},
		},
		CredentialedOrigin: origin.Of(target),
		TrustedOrigin:      trustedOrigin,
		Client:             client,
		MaxBytes:           maxMetadataBytes,
	})
	if err == nil {
		return &metadataResponse{
			status: http.StatusOK,
			body:   file.Data,
		}, nil
	}

	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	downloadErr, ok := errors.AsType[*download.Error](err)
	if !ok {
		return nil, err
	}

	if downloadErr.StatusCode != 0 {
		return &metadataResponse{
			status: downloadErr.StatusCode,
		}, nil
	}

	if isUnreachable(downloadErr) {
		return nil, nil
	}

	return nil, err
}

// isUnreachable reports a network failure, as opposed to a URL the download
// guards rejected.
func isUnreachable(downloadErr *download.Error) bool {
	return downloadErr.Message == "" && downloadErr.Cause != nil
}

// buildWellKnownPath inserts the well-known prefix before the path:
// /.well-known/<prefix><path>.
func buildWellKnownPath(prefix string, pathname string) string {
	return "/.well-known/" + prefix + strings.TrimSuffix(pathname, "/")
}

// discoverProtectedResourceMetadata fetches the server's protected resource
// metadata from metadataURL, or from the well-known locations derived from
// serverURL.
func discoverProtectedResourceMetadata(
	ctx context.Context,
	client *http.Client,
	serverURL *url.URL,
	metadataURL *url.URL,
) (*protectedResourceMetadata, error) {
	// The configured MCP server is a trusted request target; redirects
	// leaving its origin are still validated.
	trustedOrigin := origin.Of(serverURL)

	target := metadataURL
	if target == nil {
		target = resolvePath(serverURL, buildWellKnownPath(wellKnownProtectedResource, serverURL.EscapedPath()))
		target.RawQuery = serverURL.RawQuery
	}

	response, err := fetchMetadata(ctx, client, target, LatestProtocolVersion, trustedOrigin)
	if err != nil {
		return nil, err
	}

	if metadataURL == nil && shouldTryRootMetadata(response, serverURL) {
		root := resolvePath(serverURL, "/.well-known/"+wellKnownProtectedResource)

		response, err = fetchMetadata(ctx, client, root, LatestProtocolVersion, trustedOrigin)
		if err != nil {
			return nil, err
		}
	}

	if response == nil || response.status == http.StatusNotFound {
		return nil, errors.New("resource server does not implement oauth 2.0 protected resource metadata")
	}

	if !response.ok() {
		return nil, fmt.Errorf("http %d trying to load well-known oauth protected resource metadata", response.status)
	}

	metadata, err := decodeValidated(response.body, protectedResourceMetadata.validate)
	if err != nil {
		return nil, fmt.Errorf("decode oauth protected resource metadata: %w", err)
	}

	return &metadata, nil
}

func shouldTryRootMetadata(response *metadataResponse, serverURL *url.URL) bool {
	return response == nil || (response.isClientError() && pathnameOf(serverURL) != "/")
}

// discoveryKind distinguishes OAuth and OpenID Connect metadata documents.
type discoveryKind string

const (
	discoveryOAuth  discoveryKind = "oauth"
	discoveryOpenID discoveryKind = "oidc"
)

// discoveryCandidate is a location that may hold the authorization server's
// metadata, with the issuer the document must name.
type discoveryCandidate struct {
	url            *url.URL
	kind           discoveryKind
	expectedIssuer string
}

// buildDiscoveryCandidates lists the metadata locations in priority order:
// OAuth metadata for the path, OAuth metadata at the root, then OpenID
// Connect metadata.
func buildDiscoveryCandidates(authorizationServerURL *url.URL) []discoveryCandidate {
	rootIssuer := origin.Of(authorizationServerURL)
	pathname := pathnameOf(authorizationServerURL)

	if pathname == "/" {
		return []discoveryCandidate{
			{
				url:            resolvePath(authorizationServerURL, "/.well-known/"+wellKnownAuthorizationServer),
				kind:           discoveryOAuth,
				expectedIssuer: rootIssuer,
			},
			{
				url:            resolvePath(authorizationServerURL, "/.well-known/"+wellKnownOpenIDConfiguration),
				kind:           discoveryOpenID,
				expectedIssuer: rootIssuer,
			},
		}
	}

	pathname = strings.TrimSuffix(pathname, "/")
	pathIssuer := rootIssuer + pathname

	return []discoveryCandidate{
		{
			url:            resolvePath(authorizationServerURL, "/.well-known/"+wellKnownAuthorizationServer+pathname),
			kind:           discoveryOAuth,
			expectedIssuer: pathIssuer,
		},
		{
			url:            resolvePath(authorizationServerURL, "/.well-known/"+wellKnownAuthorizationServer),
			kind:           discoveryOAuth,
			expectedIssuer: rootIssuer,
		},
		{
			url:            resolvePath(authorizationServerURL, "/.well-known/"+wellKnownOpenIDConfiguration+pathname),
			kind:           discoveryOpenID,
			expectedIssuer: pathIssuer,
		},
		{
			url:            resolvePath(authorizationServerURL, pathname+"/.well-known/"+wellKnownOpenIDConfiguration),
			kind:           discoveryOpenID,
			expectedIssuer: pathIssuer,
		},
	}
}

// discoverAuthorizationServerMetadata fetches the authorization server's
// metadata. It returns nil when no candidate location has it.
func discoverAuthorizationServerMetadata(
	ctx context.Context,
	client *http.Client,
	authorizationServerURL *url.URL,
	trustedOrigin string,
) (*AuthorizationServerMetadata, error) {
	for _, candidate := range buildDiscoveryCandidates(authorizationServerURL) {
		response, err := fetchMetadata(ctx, client, candidate.url, LatestProtocolVersion, trustedOrigin)
		if err != nil {
			return nil, err
		}

		if response == nil || response.isClientError() {
			continue
		}

		if !response.ok() {
			return nil, fmt.Errorf("http %d trying to load %s metadata from %s", response.status, candidate.kind, candidate.url)
		}

		return decodeAuthorizationServerMetadata(response.body, candidate)
	}

	return nil, nil
}

func decodeAuthorizationServerMetadata(body []byte, candidate discoveryCandidate) (*AuthorizationServerMetadata, error) {
	validate := AuthorizationServerMetadata.validateOAuth
	if candidate.kind == discoveryOpenID {
		validate = AuthorizationServerMetadata.validateOpenID
	}

	metadata, err := decodeValidated(body, validate)
	if err != nil {
		return nil, fmt.Errorf("decode %s metadata from %s: %w", candidate.kind, candidate.url, err)
	}

	if err := assertIssuerMatches(metadata, candidate.expectedIssuer); err != nil {
		return nil, err
	}

	// The MCP specification requires OpenID providers to support S256 PKCE.
	if candidate.kind == discoveryOpenID && !slices.Contains(metadata.CodeChallengeMethodsSupported, codeChallengeMethod) {
		return nil, fmt.Errorf(
			"incompatible oidc provider at %s: does not support S256 code challenge method required by mcp specification",
			candidate.url,
		)
	}

	return &metadata, nil
}

// assertIssuerMatches requires the metadata to name the expected issuer. A
// trailing slash is tolerated for a root issuer.
func assertIssuerMatches(metadata AuthorizationServerMetadata, expectedIssuer string) error {
	if metadata.Issuer == expectedIssuer {
		return nil
	}

	expectedURL, err := url.Parse(expectedIssuer)
	isRootIssuer := err == nil && origin.Of(expectedURL) == expectedIssuer
	if isRootIssuer && metadata.Issuer == expectedIssuer+"/" {
		return nil
	}

	return &OAuthError{
		Message: fmt.Sprintf(
			"oauth authorization server metadata issuer %s does not match expected issuer %s",
			metadata.Issuer,
			expectedIssuer,
		),
	}
}
