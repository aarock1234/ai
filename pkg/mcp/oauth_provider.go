package mcp

import (
	"context"
	"net/http"
	"net/url"
)

// OAuthClientProvider stores the OAuth state of one client and performs the
// browser redirect, which the caller implements: RedirectToAuthorization
// sends the user to the authorization server, and the application's redirect
// handler calls Auth with the returned authorization code.
//
// A provider may also implement the optional interfaces OAuthClientAuthenticator,
// OAuthCredentialInvalidator, OAuthDynamicRegistrationReporter,
// OAuthClientInformationSaver, OAuthAuthorizationServerInformationLoader,
// OAuthAuthorizationServerInformationSaver,
// OAuthAuthorizationServerURLValidator, OAuthStateProvider,
// OAuthStateSaver, OAuthStoredStateProvider and OAuthResourceURLValidator.
type OAuthClientProvider interface {
	// Tokens returns the stored tokens, or nil when there are none.
	Tokens(ctx context.Context) (*OAuthTokens, error)

	SaveTokens(ctx context.Context, tokens OAuthTokens) error

	// RedirectToAuthorization sends the user to authorizationURL, for
	// example by opening a browser or answering with an HTTP redirect.
	RedirectToAuthorization(ctx context.Context, authorizationURL *url.URL) error

	// SaveCodeVerifier stores the PKCE code verifier of the authorization
	// in progress.
	SaveCodeVerifier(ctx context.Context, codeVerifier string) error

	CodeVerifier(ctx context.Context) (string, error)

	// RedirectURL is the redirect URI registered for the client.
	RedirectURL() string

	ClientMetadata() OAuthClientMetadata

	// ClientInformation returns the client's credentials, or nil when the
	// client is not registered yet. Pre-registered confidential clients must
	// specify their authorization server's issuer or provide stored
	// authorization server information.
	ClientInformation(ctx context.Context) (*OAuthClientInformation, error)
}

// ClientAuthenticationRequest is a token request that an
// OAuthClientAuthenticator authenticates by modifying Headers or Params.
type ClientAuthenticationRequest struct {
	Headers http.Header
	Params  url.Values

	// AuthorizationServerURL is the authorization server being called.
	AuthorizationServerURL string

	// Metadata is the authorization server's metadata, or nil when it could
	// not be discovered.
	Metadata *AuthorizationServerMetadata
}

// OAuthClientAuthenticator replaces the default client authentication of
// token requests, for example with a JWT client assertion.
type OAuthClientAuthenticator interface {
	AddClientAuthentication(ctx context.Context, request ClientAuthenticationRequest) error
}

// OAuthCredentialScope selects the credentials to invalidate.
type OAuthCredentialScope string

// Credential scopes.
const (
	OAuthCredentialsAll      OAuthCredentialScope = "all"
	OAuthCredentialsClient   OAuthCredentialScope = "client"
	OAuthCredentialsTokens   OAuthCredentialScope = "tokens"
	OAuthCredentialsVerifier OAuthCredentialScope = "verifier"
)

// OAuthCredentialInvalidator deletes credentials the server reported as no
// longer valid. For OAuthCredentialsTokens, tokens identifies the token
// generation being invalidated when known; providers that share storage
// should delete only that generation so that a concurrent refresh's tokens
// survive.
type OAuthCredentialInvalidator interface {
	InvalidateCredentials(ctx context.Context, scope OAuthCredentialScope, tokens *OAuthTokens) error
}

// OAuthDynamicRegistrationReporter reports whether the current client
// information came from dynamic registration. Only such credentials are
// discarded and re-registered after client authentication errors.
type OAuthDynamicRegistrationReporter interface {
	IsClientInformationDynamicallyRegistered(ctx context.Context) (bool, error)
}

// OAuthClientInformationSaver stores client information, which dynamic
// client registration requires.
type OAuthClientInformationSaver interface {
	SaveClientInformation(ctx context.Context, clientInformation OAuthClientInformation) error
}

// OAuthAuthorizationServerInformationLoader returns the stored authorization
// server binding, or nil when there is none.
type OAuthAuthorizationServerInformationLoader interface {
	AuthorizationServerInformation(ctx context.Context) (*OAuthAuthorizationServerInformation, error)
}

// OAuthAuthorizationServerInformationSaver stores the authorization server
// binding before the redirect. Without it, the binding is saved with the
// client information.
type OAuthAuthorizationServerInformationSaver interface {
	SaveAuthorizationServerInformation(ctx context.Context, information OAuthAuthorizationServerInformation) error
}

// OAuthAuthorizationServerURLValidator restricts the authorization servers
// that protected resource metadata may select, before their metadata is
// fetched. It does not bind client credentials to the server.
type OAuthAuthorizationServerURLValidator interface {
	ValidateAuthorizationServerURL(ctx context.Context, serverURL string, authorizationServerURL string) error
}

// OAuthStateProvider returns the state parameter of a new authorization.
type OAuthStateProvider interface {
	State(ctx context.Context) (string, error)
}

// OAuthStateSaver stores the state parameter of the authorization in
// progress.
type OAuthStateSaver interface {
	SaveState(ctx context.Context, state string) error
}

// OAuthStoredStateProvider returns the stored state parameter, which the
// authorization callback must echo. ok is false when none is stored.
type OAuthStoredStateProvider interface {
	StoredState(ctx context.Context) (state string, ok bool, err error)
}

// OAuthResourceURLValidator selects the resource indicator (RFC 8707) sent
// to the authorization server, replacing the default check against the
// protected resource metadata. resource is the metadata's resource, or
// empty when there is no metadata; a nil URL sends no resource.
type OAuthResourceURLValidator interface {
	ValidateResourceURL(ctx context.Context, serverURL *url.URL, resource string) (*url.URL, error)
}
