package mcp

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"slices"
	"strings"

	"github.com/aarock1234/ai/internal/download"
	"github.com/aarock1234/ai/internal/origin"
)

// maxOAuthResponseBytes bounds a token or registration response.
const maxOAuthResponseBytes = 1 << 20

const (
	responseTypeCode    = "code"
	codeChallengeMethod = "S256"
)

// errOAuthRedirect rejects redirects of credential-carrying requests, so
// that codes, verifiers and secrets are never replayed to another target.
var errOAuthRedirect = errors.New("oauth credential requests must not be redirected")

// pkcePair is a PKCE code verifier and its S256 challenge.
type pkcePair struct {
	verifier  string
	challenge string
}

// newPKCEPair returns a 52-character verifier with 256 bits of randomness,
// drawn from the unreserved characters RFC 7636 allows.
func newPKCEPair() pkcePair {
	verifier := rand.Text() + rand.Text()
	digest := sha256.Sum256([]byte(verifier))

	return pkcePair{
		verifier:  verifier,
		challenge: base64.RawURLEncoding.EncodeToString(digest[:]),
	}
}

// authorizationStart describes a new authorization request.
type authorizationStart struct {
	authorizationServerURL *url.URL
	metadata               *AuthorizationServerMetadata
	clientInformation      OAuthClientInformation
	redirectURL            string
	scope                  string
	state                  string
	resource               *url.URL
}

// startAuthorization returns the authorization URL to send the user to and
// the PKCE code verifier to keep for the token exchange.
func startAuthorization(start authorizationStart) (*url.URL, string, error) {
	authorizationURL, err := authorizationEndpoint(start)
	if err != nil {
		return nil, "", err
	}

	pkce := newPKCEPair()

	query := authorizationURL.Query()
	query.Set("response_type", responseTypeCode)
	query.Set("client_id", start.clientInformation.ClientID)
	query.Set("code_challenge", pkce.challenge)
	query.Set("code_challenge_method", codeChallengeMethod)
	query.Set("redirect_uri", start.redirectURL)

	if start.state != "" {
		query.Set("state", start.state)
	}

	if start.scope != "" {
		query.Set("scope", start.scope)
	}

	// The OpenID offline_access scope requires the consent prompt.
	if strings.Contains(start.scope, "offline_access") {
		query.Add("prompt", "consent")
	}

	if start.resource != nil {
		query.Set("resource", resourceURLString(start.resource))
	}

	authorizationURL.RawQuery = query.Encode()

	return authorizationURL, pkce.verifier, nil
}

func authorizationEndpoint(start authorizationStart) (*url.URL, error) {
	if start.metadata == nil {
		return resolvePath(start.authorizationServerURL, "/authorize"), nil
	}

	if !slices.Contains(start.metadata.ResponseTypesSupported, responseTypeCode) {
		return nil, fmt.Errorf("incompatible auth server: does not support response type %s", responseTypeCode)
	}

	if !slices.Contains(start.metadata.CodeChallengeMethodsSupported, codeChallengeMethod) {
		return nil, fmt.Errorf("incompatible auth server: does not support code challenge method %s", codeChallengeMethod)
	}

	return parseAbsoluteURL(start.metadata.AuthorizationEndpoint)
}

// tokenRequest is a grant sent to the token endpoint.
type tokenRequest struct {
	authorizationServerURL *url.URL
	metadata               *AuthorizationServerMetadata
	clientInformation      OAuthClientInformation
	params                 url.Values
	resource               *url.URL
	authenticator          OAuthClientAuthenticator
	client                 *http.Client
}

// exchangeAuthorization exchanges an authorization code for tokens.
func exchangeAuthorization(ctx context.Context, request tokenRequest) (OAuthTokens, error) {
	return requestTokens(ctx, request, "authorization_code", "")
}

// refreshAuthorization exchanges a refresh token for new tokens, keeping the
// refresh token when the server does not issue a new one.
func refreshAuthorization(ctx context.Context, request tokenRequest, refreshToken string) (OAuthTokens, error) {
	return requestTokens(ctx, request, "refresh_token", refreshToken)
}

func requestTokens(ctx context.Context, request tokenRequest, grantType string, refreshToken string) (OAuthTokens, error) {
	tokenURL, err := tokenEndpoint(request.authorizationServerURL, request.metadata)
	if err != nil {
		return OAuthTokens{}, err
	}

	isTrusted := origin.Of(tokenURL) == trustedLoopbackOrigin(request.authorizationServerURL)
	if err := assertSafeOAuthEndpoint(tokenURL, isTrusted); err != nil {
		return OAuthTokens{}, err
	}

	if request.metadata != nil && request.metadata.GrantTypesSupported != nil &&
		!slices.Contains(request.metadata.GrantTypesSupported, grantType) {
		return OAuthTokens{}, fmt.Errorf("incompatible auth server: does not support grant type %s", grantType)
	}

	headers := http.Header{
		"Content-Type": {
			"application/x-www-form-urlencoded",
		},
		"Accept": {
			"application/json",
		},
	}

	params := request.params
	params.Set("grant_type", grantType)

	if err := authenticateClient(ctx, request, headers, params); err != nil {
		return OAuthTokens{}, err
	}

	if request.resource != nil {
		params.Set("resource", resourceURLString(request.resource))
	}

	body, err := postOAuthRequest(ctx, oauthPost{
		client:    request.client,
		target:    tokenURL,
		isTrusted: isTrusted,
		headers:   headers,
		body:      []byte(params.Encode()),
	})
	if err != nil {
		return OAuthTokens{}, err
	}

	tokens, err := decodeValidated(body, OAuthTokens.validate)
	if err != nil {
		return OAuthTokens{}, fmt.Errorf("decode oauth token response: %w", err)
	}

	if tokens.RefreshToken == "" {
		tokens.RefreshToken = refreshToken
	}

	return tokens, nil
}

func tokenEndpoint(authorizationServerURL *url.URL, metadata *AuthorizationServerMetadata) (*url.URL, error) {
	if metadata == nil || metadata.TokenEndpoint == "" {
		return resolvePath(authorizationServerURL, "/token"), nil
	}

	return parseAbsoluteURL(metadata.TokenEndpoint)
}

// clientAuthMethod is how the client authenticates to the token endpoint.
type clientAuthMethod string

const (
	clientSecretBasic clientAuthMethod = "client_secret_basic"
	clientSecretPost  clientAuthMethod = "client_secret_post"
	clientAuthNone    clientAuthMethod = "none"
)

// authenticateClient adds the client's credentials to a token request, with
// the provider's authenticator when it has one.
func authenticateClient(ctx context.Context, request tokenRequest, headers http.Header, params url.Values) error {
	if request.authenticator != nil {
		return request.authenticator.AddClientAuthentication(ctx, ClientAuthenticationRequest{
			Headers:                headers,
			Params:                 params,
			AuthorizationServerURL: request.authorizationServerURL.String(),
			Metadata:               request.metadata,
		})
	}

	var supportedMethods []string
	if request.metadata != nil {
		supportedMethods = request.metadata.TokenEndpointAuthMethodsSupported
	}

	information := request.clientInformation
	method := selectClientAuthMethod(information, supportedMethods)

	switch method {
	case clientSecretBasic:
		credentials := base64.StdEncoding.EncodeToString([]byte(information.ClientID + ":" + information.ClientSecret))
		headers.Set("Authorization", "Basic "+credentials)
	case clientSecretPost:
		params.Set("client_id", information.ClientID)
		if information.ClientSecret != "" {
			params.Set("client_secret", information.ClientSecret)
		}
	default:
		params.Set("client_id", information.ClientID)
	}

	return nil
}

// selectClientAuthMethod prefers client_secret_basic, then
// client_secret_post when the client has a secret, and none otherwise.
func selectClientAuthMethod(information OAuthClientInformation, supportedMethods []string) clientAuthMethod {
	hasClientSecret := information.ClientSecret != ""

	switch {
	case len(supportedMethods) == 0 && hasClientSecret:
		return clientSecretPost
	case len(supportedMethods) == 0:
		return clientAuthNone
	case hasClientSecret && slices.Contains(supportedMethods, string(clientSecretBasic)):
		return clientSecretBasic
	case hasClientSecret && slices.Contains(supportedMethods, string(clientSecretPost)):
		return clientSecretPost
	case slices.Contains(supportedMethods, string(clientAuthNone)):
		return clientAuthNone
	case hasClientSecret:
		return clientSecretPost
	default:
		return clientAuthNone
	}
}

// registerClient performs dynamic client registration (RFC 7591).
func registerClient(
	ctx context.Context,
	client *http.Client,
	authorizationServerURL *url.URL,
	metadata *AuthorizationServerMetadata,
	clientMetadata OAuthClientMetadata,
) (registeredClient, error) {
	registrationURL, err := registrationEndpoint(authorizationServerURL, metadata)
	if err != nil {
		return registeredClient{}, err
	}

	isTrusted := origin.Of(registrationURL) == trustedLoopbackOrigin(authorizationServerURL)
	if err := assertSafeOAuthEndpoint(registrationURL, isTrusted); err != nil {
		return registeredClient{}, err
	}

	if clientMetadata.ApplicationType == "" {
		clientMetadata.ApplicationType, err = inferApplicationType(clientMetadata.RedirectURIs)
		if err != nil {
			return registeredClient{}, err
		}
	}

	body, err := json.Marshal(clientMetadata)
	if err != nil {
		return registeredClient{}, fmt.Errorf("encode oauth client metadata: %w", err)
	}

	headers := http.Header{
		"Content-Type": {
			"application/json",
		},
	}

	response, err := postOAuthRequest(ctx, oauthPost{
		client:    client,
		target:    registrationURL,
		isTrusted: isTrusted,
		headers:   headers,
		body:      body,
	})
	if err != nil {
		return registeredClient{}, err
	}

	registered, err := decodeValidated(response, registeredClient.validate)
	if err != nil {
		return registeredClient{}, fmt.Errorf("decode oauth client registration: %w", err)
	}

	return registered, nil
}

func registrationEndpoint(authorizationServerURL *url.URL, metadata *AuthorizationServerMetadata) (*url.URL, error) {
	if metadata == nil {
		return resolvePath(authorizationServerURL, "/register"), nil
	}

	if metadata.RegistrationEndpoint == "" {
		return nil, errors.New("incompatible auth server: does not support dynamic client registration")
	}

	return parseAbsoluteURL(metadata.RegistrationEndpoint)
}

// inferApplicationType returns "native" when every redirect URI is a
// loopback HTTP URI or uses a custom scheme, and "web" otherwise.
func inferApplicationType(redirectURIs []string) (string, error) {
	for _, redirectURI := range redirectURIs {
		parsed, err := parseAbsoluteURL(redirectURI)
		if err != nil {
			return "", fmt.Errorf("redirect uri: %w", err)
		}

		if isHTTPScheme(parsed) && !isLoopbackHost(parsed.Hostname()) {
			return "web", nil
		}
	}

	return "native", nil
}

// oauthPost is a credential-carrying POST to a token or registration
// endpoint. IsTrusted marks an endpoint on the locally configured loopback
// authorization server.
type oauthPost struct {
	client    *http.Client
	target    *url.URL
	isTrusted bool
	headers   http.Header
	body      []byte
}

// postOAuthRequest posts a credential-carrying request without following
// redirects and returns the successful response body. Failed responses
// become an *OAuthError.
func postOAuthRequest(ctx context.Context, post oauthPost) ([]byte, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, post.target.String(), bytes.NewReader(post.body))
	if err != nil {
		return nil, fmt.Errorf("create oauth request: %w", err)
	}

	request.Header = post.headers

	response, err := withoutRedirects(post.httpClient()).Do(request)
	if err != nil {
		return nil, fmt.Errorf("send oauth request to %s: %w", post.target, err)
	}
	defer closeBody(response.Body)

	responseBody, err := io.ReadAll(io.LimitReader(response.Body, maxOAuthResponseBytes))
	if err != nil {
		return nil, fmt.Errorf("read oauth response from %s: %w", post.target, err)
	}

	if response.StatusCode < 200 || response.StatusCode > 299 {
		return nil, parseOAuthErrorResponse(response.StatusCode, responseBody)
	}

	return responseBody, nil
}

// httpClient returns the configured client. Without one, trusted endpoints
// use http.DefaultClient and every other endpoint uses the client that
// validates resolved addresses, as upstream pins DNS for untrusted
// endpoints, so that a public hostname cannot be rebound to a private
// address between validation and connection.
func (p oauthPost) httpClient() *http.Client {
	switch {
	case p.client != nil:
		return p.client
	case p.isTrusted:
		return http.DefaultClient
	default:
		return download.ValidatingClient()
	}
}

// withoutRedirects returns a copy of client that fails on redirects.
func withoutRedirects(client *http.Client) *http.Client {
	copied := *client
	copied.CheckRedirect = func(*http.Request, []*http.Request) error {
		return errOAuthRedirect
	}

	return &copied
}

// oauthErrorResponse is an OAuth 2.0 error response body.
type oauthErrorResponse struct {
	Error            string `json:"error"`
	ErrorDescription string `json:"error_description,omitempty"`
	ErrorURI         string `json:"error_uri,omitempty"`
}

// parseOAuthErrorResponse converts an error response into an *OAuthError.
// A body that is not an OAuth error becomes a server_error with the raw
// body.
func parseOAuthErrorResponse(status int, body []byte) *OAuthError {
	var response oauthErrorResponse

	err := json.Unmarshal(body, &response)
	if err == nil {
		err = requireFields(body, "error")
	}

	if err != nil {
		return &OAuthError{
			Code:    OAuthServerError,
			Message: fmt.Sprintf("http %d: invalid oauth error response: %v. raw body: %s", status, err, body),
		}
	}

	return &OAuthError{
		Code:    OAuthErrorCode(response.Error),
		Message: response.ErrorDescription,
		URI:     response.ErrorURI,
	}
}

func closeBody(body io.Closer) {
	// The response outcome is already decided when the body is closed.
	_ = body.Close()
}
