package gateway

import (
	"bytes"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/apikey"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	// protocolVersion is the AI Gateway protocol version this package
	// speaks.
	protocolVersion = "0.0.1"

	// apiKeyVariable holds the API key or Vercel access token.
	apiKeyVariable = "AI_GATEWAY_API_KEY"

	// oidcTokenVariable holds the Vercel OIDC token of a deployment or a
	// linked development environment.
	oidcTokenVariable = "VERCEL_OIDC_TOKEN"

	// authMethodHeader reports how the request is authenticated.
	authMethodHeader = "ai-gateway-auth-method"

	// teamHeader scopes a request to a Vercel team.
	teamHeader = "x-vercel-ai-gateway-team"
)

// authMethod is how a request is authenticated.
type authMethod string

// Authentication methods. The zero value means unknown.
const (
	authAPIKey authMethod = "api-key"
	authOIDC   authMethod = "oidc"
)

// observabilityVariables map Vercel deployment environment variables onto
// the observability headers sent with model calls.
var observabilityVariables = []struct {
	variable string
	header   string
}{
	{
		variable: "VERCEL_DEPLOYMENT_ID",
		header:   "ai-o11y-deployment-id",
	},
	{
		variable: "VERCEL_ENV",
		header:   "ai-o11y-environment",
	},
	{
		variable: "VERCEL_REGION",
		header:   "ai-o11y-region",
	},
	{
		variable: "VERCEL_PROJECT_ID",
		header:   "ai-o11y-project-id",
	},
}

// credentials are the token of a request and how it was obtained.
type credentials struct {
	token  string
	method authMethod
}

// loadCredentials prefers the configured key, then AI_GATEWAY_API_KEY, then
// the Vercel OIDC token.
func (p *Provider) loadCredentials() (credentials, error) {
	if apiKey := apikey.Lookup(p.apiKey, apiKeyVariable); apiKey != "" {
		return credentials{
			token:  apiKey,
			method: authAPIKey,
		}, nil
	}

	if token := os.Getenv(oidcTokenVariable); token != "" {
		return credentials{
			token:  token,
			method: authOIDC,
		}, nil
	}

	return credentials{}, fmt.Errorf(
		"%w: set %s, use WithAPIKey or provide a Vercel OIDC token in %s",
		model.ErrLoadAPIKey,
		apiKeyVariable,
		oidcTokenVariable,
	)
}

// providerHeaders returns the authentication, protocol and team headers
// followed by the configured headers. A missing credential is reported as an
// authentication Error.
func (p *Provider) providerHeaders() (http.Header, error) {
	loaded, err := p.loadCredentials()
	if err != nil {
		return nil, authenticationError(authenticationErrorSettings{
			statusCode: http.StatusUnauthorized,
			cause:      err,
		})
	}

	headers := http.Header{
		"Authorization": {
			"Bearer " + loaded.token,
		},
		"Ai-Gateway-Protocol-Version": {
			protocolVersion,
		},
		http.CanonicalHeaderKey(authMethodHeader): {
			string(loaded.method),
		},
	}

	if p.teamIDOrSlug != "" {
		headers.Set(teamHeader, p.teamIDOrSlug)
	}

	return apicall.MergeHeaders(headers, p.headers), nil
}

// modelRequest describes a model call to path. Later headers take
// precedence: provider, call, model configuration and observability.
func (p *Provider) modelRequest(path string, callHeaders http.Header, modelHeaders http.Header) (apicall.Request, authMethod, error) {
	headers, err := p.providerHeaders()
	if err != nil {
		return apicall.Request{}, "", err
	}

	merged := apicall.MergeHeaders(headers, callHeaders, modelHeaders, observabilityHeaders())

	return p.apiRequest(p.baseURL+path, merged), parseAuthMethod(merged), nil
}

// accountRequest describes a call to an account endpoint.
func (p *Provider) accountRequest(endpoint string) (apicall.Request, authMethod, error) {
	headers, err := p.providerHeaders()
	if err != nil {
		return apicall.Request{}, "", err
	}

	return p.apiRequest(endpoint, headers), parseAuthMethod(headers), nil
}

func (p *Provider) apiRequest(endpoint string, headers http.Header) apicall.Request {
	return apicall.Request{
		Client:     p.httpClient,
		URL:        endpoint,
		Headers:    headers,
		ParseError: parseError,
	}
}

// origin returns the scheme and host of the base URL, where the account
// endpoints live.
func (p *Provider) origin() (*url.URL, error) {
	parsed, err := url.Parse(p.baseURL)
	if err != nil {
		return nil, fmt.Errorf("%w: base URL: %w", model.ErrInvalidArgument, err)
	}

	return &url.URL{
		Scheme: parsed.Scheme,
		Host:   parsed.Host,
	}, nil
}

// observabilityHeaders reports the Vercel deployment of the caller.
func observabilityHeaders() http.Header {
	headers := make(http.Header)

	for _, mapping := range observabilityVariables {
		if value := os.Getenv(mapping.variable); value != "" {
			headers.Set(mapping.header, value)
		}
	}

	return headers
}

// parseAuthMethod reads the authentication method header. Unknown values
// are ignored.
func parseAuthMethod(headers http.Header) authMethod {
	switch method := authMethod(headers.Get(authMethodHeader)); method {
	case authAPIKey, authOIDC:
		return method
	default:
		return ""
	}
}

// parseError keeps the failed response body as the message. The gateway
// error that wraps the failure decodes the body itself.
func parseError(body []byte) string {
	return strings.TrimSpace(string(bytes.ToValidUTF8(body, nil)))
}
