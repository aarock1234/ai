package mcp

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/aarock1234/ai/internal/origin"
)

// AuthResult is the outcome of Auth.
type AuthResult string

// Auth results.
const (
	// AuthResultAuthorized means valid tokens were saved.
	AuthResultAuthorized AuthResult = "AUTHORIZED"

	// AuthResultRedirect means the user was sent to the authorization server;
	// call Auth again with the authorization code from the redirect.
	AuthResultRedirect AuthResult = "REDIRECT"
)

// AuthOptions configure Auth.
type AuthOptions struct {
	// ServerURL is the MCP server URL. It is required.
	ServerURL string

	// AuthorizationCode is the code from the authorization redirect. Set it
	// to complete an authorization; leave it empty to refresh tokens or
	// start a new authorization.
	AuthorizationCode string

	// CallbackState is the state parameter of the authorization redirect.
	CallbackState string

	// CallbackIssuer is the iss parameter of the authorization redirect, if
	// the authorization server sent one.
	CallbackIssuer string

	// Scope is the scope a WWW-Authenticate challenge requested.
	Scope string

	// ResourceMetadataURL is the protected resource metadata URL from a
	// WWW-Authenticate challenge. It must have the server's origin.
	ResourceMetadataURL *url.URL

	// HTTPClient sends discovery, token and registration requests. Nil uses
	// http.DefaultClient for configured origins and, for every other origin,
	// a client that rejects private network addresses after DNS resolution.
	// Discovery requests to other origins always use that validating client.
	HTTPClient *http.Client
}

// Auth runs the OAuth 2.1 authorization flow for an MCP server: it discovers
// the protected resource and authorization server metadata, registers the
// client dynamically when needed, and then exchanges an authorization code,
// refreshes the stored tokens, or starts a new authorization with PKCE by
// calling provider.RedirectToAuthorization.
//
// Client credentials and tokens are bound to the authorization server that
// issued them; metadata naming a different server fails with an *OAuthError
// whose code is OAuthAuthorizationServerMismatch.
func Auth(ctx context.Context, provider OAuthClientProvider, options AuthOptions) (AuthResult, error) {
	attempt := &refreshAttempt{}

	result, err := authorize(ctx, provider, options, flowState{
		refreshAttempt: attempt,
	})
	if err == nil {
		return result, nil
	}

	switch {
	case hasOAuthErrorCode(err, OAuthInvalidClient, OAuthUnauthorizedClient):
		return reauthorizeWithNewClient(ctx, provider, options, err)
	case hasOAuthErrorCode(err, OAuthInvalidGrant) && attempt.tokens != nil:
		return reauthorizeAfterRejectedRefresh(ctx, provider, options, attempt.tokens)
	default:
		return "", err
	}
}

// reauthorizeWithNewClient discards dynamically registered credentials that
// the server rejected and starts over with a new registration.
func reauthorizeWithNewClient(ctx context.Context, provider OAuthClientProvider, options AuthOptions, cause error) (AuthResult, error) {
	if options.AuthorizationCode != "" {
		return "", cause
	}

	dynamic, err := isDynamicallyRegistered(ctx, provider)
	if err != nil {
		return "", err
	}

	if !dynamic {
		return "", cause
	}

	if err := invalidateCredentials(ctx, provider, OAuthCredentialsAll, nil); err != nil {
		return "", err
	}

	return authorize(ctx, provider, options, flowState{})
}

// reauthorizeAfterRejectedRefresh discards the tokens of a rejected refresh
// and authorizes again with the same authorization server binding.
func reauthorizeAfterRejectedRefresh(ctx context.Context, provider OAuthClientProvider, options AuthOptions, tokens *OAuthTokens) (AuthResult, error) {
	retryInformation := serverInformationFromCredentials(tokens.Issuer, tokens.AuthorizationServer, tokens.TokenEndpoint)

	if err := invalidateCredentials(ctx, provider, OAuthCredentialsTokens, tokens); err != nil {
		return "", err
	}

	return authorize(ctx, provider, options, flowState{
		retryInformation: retryInformation,
	})
}

// refreshAttempt records the tokens of a refresh, so that Auth can discard
// exactly those tokens when the refresh is rejected.
type refreshAttempt struct {
	tokens *OAuthTokens
}

// flowState carries state between Auth's attempts.
type flowState struct {
	refreshAttempt *refreshAttempt

	// retryInformation is the binding validated before a rejected refresh.
	retryInformation *OAuthAuthorizationServerInformation
}

// authFlow is one pass through the authorization flow.
type authFlow struct {
	provider OAuthClientProvider
	options  AuthOptions
	state    flowState
	client   *http.Client

	serverURL              *url.URL
	authorizationServerURL *url.URL
	resourceMetadata       *protectedResourceMetadata
	resource               *url.URL
	metadata               *AuthorizationServerMetadata
	current                OAuthAuthorizationServerInformation
	scope                  string

	clientInformation *OAuthClientInformation
	tokens            *OAuthTokens

	// callbackInformation is the binding stored before the redirect, used
	// when completing an authorization.
	callbackInformation *OAuthAuthorizationServerInformation
}

func authorize(ctx context.Context, provider OAuthClientProvider, options AuthOptions, state flowState) (AuthResult, error) {
	flow := &authFlow{
		provider: provider,
		options:  options,
		state:    state,
		client:   options.HTTPClient,
	}

	if err := flow.discover(ctx); err != nil {
		return "", err
	}

	if err := flow.loadCredentials(ctx); err != nil {
		return "", err
	}

	if flow.clientInformation == nil {
		if err := flow.registerClient(ctx); err != nil {
			return "", err
		}
	}

	if flow.isCallback() {
		return flow.completeAuthorization(ctx)
	}

	if flow.tokens != nil && flow.tokens.RefreshToken != "" {
		refreshed, err := flow.refreshTokens(ctx)
		if err != nil {
			return "", err
		}

		if refreshed {
			return AuthResultAuthorized, nil
		}
	}

	return flow.startAuthorization(ctx)
}

func (f *authFlow) isCallback() bool {
	return f.options.AuthorizationCode != ""
}

// discover selects the authorization server and the resource indicator and
// fetches the authorization server's metadata.
func (f *authFlow) discover(ctx context.Context) error {
	serverURL, err := parseAbsoluteURL(f.options.ServerURL)
	if err != nil {
		return fmt.Errorf("mcp server url: %w", err)
	}

	f.serverURL = serverURL

	metadataURL := f.options.ResourceMetadataURL
	if metadataURL != nil && !origin.SameURL(metadataURL, serverURL) {
		return &OAuthError{
			Message: fmt.Sprintf(
				"oauth protected resource metadata url %s must have the same origin as the mcp server url %s",
				metadataURL,
				origin.Of(serverURL),
			),
		}
	}

	// Servers without protected resource metadata use the legacy behavior
	// of acting as their own authorization server.
	resourceMetadata, err := discoverProtectedResourceMetadata(ctx, f.client, serverURL, metadataURL)
	if err == nil {
		f.resourceMetadata = resourceMetadata
	} else if ctx.Err() != nil {
		return ctx.Err()
	}

	if err := f.loadCallbackInformation(ctx); err != nil {
		return err
	}

	if err := f.selectAuthorizationServer(); err != nil {
		return err
	}

	resource, err := selectResourceURL(ctx, serverURL, f.provider, f.resourceMetadata)
	if err != nil {
		return err
	}

	f.resource = resource

	if validator, ok := f.provider.(OAuthAuthorizationServerURLValidator); ok {
		if err := validator.ValidateAuthorizationServerURL(ctx, serverURL.String(), f.authorizationServerURL.String()); err != nil {
			return err
		}
	}

	metadata, err := discoverAuthorizationServerMetadata(ctx, f.client, f.authorizationServerURL, f.trustedAuthorizationServerOrigin())
	if err != nil {
		return err
	}

	f.metadata = metadata

	current, err := newServerInformation(f.authorizationServerURL.String(), metadata)
	if err != nil {
		return err
	}

	f.current = current

	if f.state.retryInformation != nil {
		if err := assertServerInformationMatches(*f.state.retryInformation, current); err != nil {
			return err
		}
	}

	f.scope = selectScope(f.options.Scope, f.resourceMetadata, f.provider.ClientMetadata())

	return nil
}

// loadCallbackInformation reads the binding pinned before the redirect. A
// callback may not have the metadata URL of the original challenge, so the
// pinned server is used when rediscovery selects none.
func (f *authFlow) loadCallbackInformation(ctx context.Context) error {
	if !f.isCallback() {
		return nil
	}

	information, err := f.provider.ClientInformation(ctx)
	if err != nil {
		return err
	}

	f.clientInformation = information
	if information == nil {
		return nil
	}

	f.callbackInformation, err = storedServerInformation(ctx, f.provider, *information, nil)

	return err
}

// selectAuthorizationServer uses the first server of the protected resource
// metadata, then the pinned callback server, then the MCP server itself.
// A server outside the MCP server's origin must pass the request guards.
func (f *authFlow) selectAuthorizationServer() error {
	selected := f.serverURL.String()

	switch {
	case f.resourceMetadata != nil && len(f.resourceMetadata.AuthorizationServers) > 0:
		selected = f.resourceMetadata.AuthorizationServers[0]
	case f.callbackInformation != nil:
		selected = f.callbackInformation.AuthorizationServerURL
	}

	authorizationServerURL, err := parseAbsoluteURL(selected)
	if err != nil {
		return fmt.Errorf("authorization server url: %w", err)
	}

	f.authorizationServerURL = authorizationServerURL

	if f.trustedAuthorizationServerOrigin() != "" {
		return nil
	}

	return assertSafeOAuthEndpoint(authorizationServerURL, false)
}

// trustedAuthorizationServerOrigin returns the authorization server's origin
// when it is the developer-configured MCP server's origin, or both are
// loopback hosts, and empty otherwise.
func (f *authFlow) trustedAuthorizationServerOrigin() string {
	sameServer := origin.SameURL(f.authorizationServerURL, f.serverURL)
	bothLoopback := isLoopbackHost(f.serverURL.Hostname()) && isLoopbackHost(f.authorizationServerURL.Hostname())

	if sameServer || bothLoopback {
		return origin.Of(f.authorizationServerURL)
	}

	return ""
}

// loadCredentials loads the client information and tokens and checks that
// they are bound to the current authorization server.
func (f *authFlow) loadCredentials(ctx context.Context) error {
	if !f.isCallback() {
		information, err := f.provider.ClientInformation(ctx)
		if err != nil {
			return err
		}

		tokens, err := f.provider.Tokens(ctx)
		if err != nil {
			return err
		}

		f.clientInformation = information
		f.tokens = tokens
	}

	if f.clientInformation == nil {
		return nil
	}

	return f.checkClientBinding(ctx)
}

func (f *authFlow) checkClientBinding(ctx context.Context) error {
	information := *f.clientInformation

	// A flow binding must never override the issuer of the credentials.
	if information.Issuer != "" && information.Issuer != f.current.Issuer {
		return authorizationServerMismatch("oauth authorization server does not match the issuer of the client credentials")
	}

	clientBinding := serverInformationFromCredentials(information.Issuer, information.AuthorizationServer, information.TokenEndpoint)
	if clientBinding != nil {
		if err := assertServerInformationMatches(*clientBinding, f.current); err != nil {
			return err
		}
	}

	stored := f.callbackInformation
	if stored == nil {
		var err error

		stored, err = storedServerInformation(ctx, f.provider, information, f.tokens)
		if err != nil {
			return err
		}
	}

	if stored == nil {
		stored = f.state.retryInformation
	}

	if stored != nil {
		if err := assertServerInformationMatches(*stored, f.current); err != nil {
			return err
		}
	}

	// An MCP server's choice of authorization server must not become a
	// trust binding for an existing secret.
	_, hasAuthenticator := f.provider.(OAuthClientAuthenticator)
	isConfidential := information.ClientSecret != "" || hasAuthenticator

	if isConfidential && information.Issuer == "" && stored == nil {
		return authorizationServerMismatch(
			"oauth client credentials must be bound to an authorization server: configure their issuer or store authorization server information",
		)
	}

	return nil
}

// registerClient registers the client dynamically and saves its
// credentials bound to the current authorization server.
func (f *authFlow) registerClient(ctx context.Context) error {
	if f.isCallback() {
		return errors.New("existing oauth client information is required when exchanging an authorization code")
	}

	saver, ok := f.provider.(OAuthClientInformationSaver)
	if !ok {
		return errors.New("oauth client information must be saveable for dynamic registration")
	}

	clientMetadata := f.provider.ClientMetadata()
	clientMetadata.Scope = f.scope

	registered, err := registerClient(ctx, f.client, f.authorizationServerURL, f.metadata, clientMetadata)
	if err != nil {
		return err
	}

	information := bindClientInformation(registered.OAuthClientInformation, f.current)

	if err := saver.SaveClientInformation(ctx, information); err != nil {
		return err
	}

	f.clientInformation = &information

	return nil
}

// completeAuthorization validates the callback and exchanges the code for
// tokens.
func (f *authFlow) completeAuthorization(ctx context.Context) (AuthResult, error) {
	if err := f.checkCallbackState(ctx); err != nil {
		return "", err
	}

	stored := f.callbackInformation
	if stored == nil {
		var err error

		stored, err = storedServerInformation(ctx, f.provider, *f.clientInformation, nil)
		if err != nil {
			return "", err
		}
	}

	if stored == nil {
		return "", &OAuthError{
			Message: "stored oauth authorization server metadata is required when exchanging an authorization code",
		}
	}

	if err := f.checkCallbackIssuer(*stored); err != nil {
		return "", err
	}

	if err := assertServerInformationMatches(*stored, f.current); err != nil {
		return "", err
	}

	codeVerifier, err := f.provider.CodeVerifier(ctx)
	if err != nil {
		return "", err
	}

	params := url.Values{
		"code": {
			f.options.AuthorizationCode,
		},
		"code_verifier": {
			codeVerifier,
		},
		"redirect_uri": {
			f.provider.RedirectURL(),
		},
	}

	tokens, err := exchangeAuthorization(ctx, f.tokenRequest(params))
	if err != nil {
		return "", err
	}

	if err := f.provider.SaveTokens(ctx, bindTokens(tokens, f.current)); err != nil {
		return "", err
	}

	return AuthResultAuthorized, nil
}

// checkCallbackState rejects a callback whose state differs from the stored
// state, which indicates cross-site request forgery.
func (f *authFlow) checkCallbackState(ctx context.Context) error {
	stateProvider, ok := f.provider.(OAuthStoredStateProvider)
	if !ok {
		return nil
	}

	expected, stored, err := stateProvider.StoredState(ctx)
	if err != nil {
		return err
	}

	if stored && expected != f.options.CallbackState {
		return errors.New("oauth state parameter mismatch - possible csrf attack")
	}

	return nil
}

// checkCallbackIssuer validates the iss parameter of the authorization
// response (RFC 9207).
func (f *authFlow) checkCallbackIssuer(stored OAuthAuthorizationServerInformation) error {
	if f.options.CallbackIssuer == "" {
		return nil
	}

	expected := stored.Issuer
	if expected == "" && f.metadata != nil {
		expected = f.metadata.Issuer
	}

	if expected == "" {
		expected = f.authorizationServerURL.String()
	}

	if f.options.CallbackIssuer == expected {
		return nil
	}

	return &OAuthError{
		Message: fmt.Sprintf(
			"oauth authorization response issuer %s does not match expected issuer %s",
			f.options.CallbackIssuer,
			expected,
		),
	}
}

// refreshTokens refreshes the stored tokens when they are bound to the
// current authorization server. It reports false when the flow should
// continue with a new authorization.
func (f *authFlow) refreshTokens(ctx context.Context) (bool, error) {
	tokens := *f.tokens

	stored, err := storedServerInformation(ctx, f.provider, *f.clientInformation, &tokens)
	if err != nil {
		return false, err
	}

	if stored == nil {
		return false, invalidateCredentials(ctx, f.provider, OAuthCredentialsTokens, &tokens)
	}

	if err := assertServerInformationMatches(*stored, f.current); err != nil {
		return false, err
	}

	if f.state.refreshAttempt != nil {
		f.state.refreshAttempt.tokens = &tokens
	}

	refreshed, err := refreshAuthorization(ctx, f.tokenRequest(url.Values{
		"refresh_token": {
			tokens.RefreshToken,
		},
	}), tokens.RefreshToken)
	if err != nil {
		if isRecoverableRefreshError(err) {
			return false, nil
		}

		return false, err
	}

	if err := f.provider.SaveTokens(ctx, bindTokens(refreshed, f.current)); err != nil {
		return false, err
	}

	return true, nil
}

// isRecoverableRefreshError reports refresh failures after which a new
// authorization may succeed: server errors and failures that are not OAuth
// errors. Client, grant and binding errors are escalated.
func isRecoverableRefreshError(err error) bool {
	oauthErr, ok := errors.AsType[*OAuthError](err)
	if !ok {
		return true
	}

	switch oauthErr.Code {
	case "", OAuthInvalidClient, OAuthInvalidGrant, OAuthUnauthorizedClient, OAuthAuthorizationServerMismatch:
		return false
	default:
		return true
	}
}

// startAuthorization pins the authorization server, saves the PKCE verifier
// and redirects the user.
func (f *authFlow) startAuthorization(ctx context.Context) (AuthResult, error) {
	state, err := f.newState(ctx)
	if err != nil {
		return "", err
	}

	authorizationURL, codeVerifier, err := startAuthorization(authorizationStart{
		authorizationServerURL: f.authorizationServerURL,
		metadata:               f.metadata,
		clientInformation:      *f.clientInformation,
		redirectURL:            f.provider.RedirectURL(),
		scope:                  f.scope,
		state:                  state,
		resource:               f.resource,
	})
	if err != nil {
		return "", err
	}

	saved, err := saveServerInformation(ctx, f.provider, *f.clientInformation, f.current)
	if err != nil {
		return "", err
	}

	if !saved {
		return "", &OAuthError{
			Message: "oauth authorization server metadata must be saveable before starting authorization",
		}
	}

	if err := f.provider.SaveCodeVerifier(ctx, codeVerifier); err != nil {
		return "", err
	}

	if err := f.provider.RedirectToAuthorization(ctx, authorizationURL); err != nil {
		return "", err
	}

	return AuthResultRedirect, nil
}

func (f *authFlow) newState(ctx context.Context) (string, error) {
	stateProvider, ok := f.provider.(OAuthStateProvider)
	if !ok {
		return "", nil
	}

	state, err := stateProvider.State(ctx)
	if err != nil {
		return "", err
	}

	if saver, ok := f.provider.(OAuthStateSaver); ok && state != "" {
		if err := saver.SaveState(ctx, state); err != nil {
			return "", err
		}
	}

	return state, nil
}

func (f *authFlow) tokenRequest(params url.Values) tokenRequest {
	authenticator, _ := f.provider.(OAuthClientAuthenticator)

	return tokenRequest{
		authorizationServerURL: f.authorizationServerURL,
		metadata:               f.metadata,
		clientInformation:      *f.clientInformation,
		params:                 params,
		resource:               f.resource,
		authenticator:          authenticator,
		client:                 f.client,
	}
}

// selectScope prefers the challenged scope, then the protected resource's
// supported scopes, then the client metadata's scope.
func selectScope(scope string, resourceMetadata *protectedResourceMetadata, clientMetadata OAuthClientMetadata) string {
	if scope != "" {
		return scope
	}

	if resourceMetadata != nil && len(resourceMetadata.ScopesSupported) > 0 {
		return strings.Join(resourceMetadata.ScopesSupported, " ")
	}

	return clientMetadata.Scope
}

// selectResourceURL returns the resource indicator for the authorization
// server: the provider's choice when it validates resources, and otherwise
// the protected resource metadata's resource after checking that it covers
// the server URL.
func selectResourceURL(
	ctx context.Context,
	serverURL *url.URL,
	provider OAuthClientProvider,
	resourceMetadata *protectedResourceMetadata,
) (*url.URL, error) {
	defaultResource := resourceURLFromServerURL(serverURL)

	if validator, ok := provider.(OAuthResourceURLValidator); ok {
		resource := ""
		if resourceMetadata != nil {
			resource = resourceMetadata.Resource
		}

		return validator.ValidateResourceURL(ctx, defaultResource, resource)
	}

	if resourceMetadata == nil {
		return nil, nil
	}

	configured, err := parseAbsoluteURL(resourceMetadata.Resource)
	if err != nil {
		return nil, err
	}

	if !isResourceAllowed(defaultResource, configured) {
		return nil, fmt.Errorf(
			"protected resource %s does not match expected %s (or origin)",
			resourceMetadata.Resource,
			defaultResource,
		)
	}

	return configured, nil
}

func isDynamicallyRegistered(ctx context.Context, provider OAuthClientProvider) (bool, error) {
	reporter, ok := provider.(OAuthDynamicRegistrationReporter)
	if !ok {
		return false, nil
	}

	return reporter.IsClientInformationDynamicallyRegistered(ctx)
}

func invalidateCredentials(ctx context.Context, provider OAuthClientProvider, scope OAuthCredentialScope, tokens *OAuthTokens) error {
	invalidator, ok := provider.(OAuthCredentialInvalidator)
	if !ok {
		return nil
	}

	return invalidator.InvalidateCredentials(ctx, scope, tokens)
}
