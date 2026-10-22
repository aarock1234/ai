package mcp

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"strings"
)

// OAuthTokens is an OAuth 2.1 token response. Issuer, AuthorizationServer
// and TokenEndpoint bind stored tokens to the authorization server that
// issued them; Auth sets them before saving.
type OAuthTokens struct {
	AccessToken  string   `json:"access_token"`
	IDToken      string   `json:"id_token,omitempty"`
	TokenType    string   `json:"token_type"`
	ExpiresIn    *float64 `json:"expires_in,omitempty"`
	Scope        string   `json:"scope,omitempty"`
	RefreshToken string   `json:"refresh_token,omitempty"`

	Issuer              string `json:"issuer,omitempty"`
	AuthorizationServer string `json:"authorization_server,omitempty"`
	TokenEndpoint       string `json:"token_endpoint,omitempty"`
}

func (t OAuthTokens) validate(raw json.RawMessage) error {
	if err := requireFields(raw, "access_token", "token_type"); err != nil {
		return err
	}

	return validateSafeURLs(map[string]string{
		"issuer":               t.Issuer,
		"authorization_server": t.AuthorizationServer,
		"token_endpoint":       t.TokenEndpoint,
	})
}

// OAuthClientInformation holds the client's credentials. Pre-registered
// confidential clients must set Issuer or provide stored authorization server
// information; Auth binds dynamically registered clients itself.
type OAuthClientInformation struct {
	ClientID              string   `json:"client_id"`
	ClientSecret          string   `json:"client_secret,omitempty"`
	ClientIDIssuedAt      *float64 `json:"client_id_issued_at,omitempty"`
	ClientSecretExpiresAt *float64 `json:"client_secret_expires_at,omitempty"`

	Issuer              string `json:"issuer,omitempty"`
	AuthorizationServer string `json:"authorization_server,omitempty"`
	TokenEndpoint       string `json:"token_endpoint,omitempty"`
}

// OAuthClientMetadata is the client metadata sent in dynamic client
// registration (RFC 7591).
type OAuthClientMetadata struct {
	RedirectURIs []string `json:"redirect_uris"`

	// ApplicationType is "native" or "web". Empty infers it from the
	// redirect URIs: native when all of them are loopback or custom-scheme
	// URIs.
	ApplicationType string `json:"application_type,omitempty"`

	TokenEndpointAuthMethod string          `json:"token_endpoint_auth_method,omitempty"`
	GrantTypes              []string        `json:"grant_types,omitempty"`
	ResponseTypes           []string        `json:"response_types,omitempty"`
	ClientName              string          `json:"client_name,omitempty"`
	ClientURI               string          `json:"client_uri,omitempty"`
	LogoURI                 string          `json:"logo_uri,omitempty"`
	Scope                   string          `json:"scope,omitempty"`
	Contacts                []string        `json:"contacts,omitempty"`
	TOSURI                  string          `json:"tos_uri,omitempty"`
	PolicyURI               string          `json:"policy_uri,omitempty"`
	JWKSURI                 string          `json:"jwks_uri,omitempty"`
	JWKS                    json.RawMessage `json:"jwks,omitempty"`
	SoftwareID              string          `json:"software_id,omitempty"`
	SoftwareVersion         string          `json:"software_version,omitempty"`
	SoftwareStatement       string          `json:"software_statement,omitempty"`
}

// registeredClient is a dynamic client registration response.
type registeredClient struct {
	OAuthClientMetadata
	OAuthClientInformation
}

func (r registeredClient) validate(raw json.RawMessage) error {
	if err := requireFields(raw, "client_id", "redirect_uris"); err != nil {
		return err
	}

	return validateSafeURLs(map[string]string{
		"client_uri": r.ClientURI,
		"logo_uri":   r.LogoURI,
		"tos_uri":    r.TOSURI,
		"jwks_uri":   r.JWKSURI,
	})
}

// OAuthAuthorizationServerInformation identifies the authorization server
// that client credentials and tokens are bound to.
type OAuthAuthorizationServerInformation struct {
	Issuer                 string
	AuthorizationServerURL string
	TokenEndpoint          string
}

// AuthorizationServerMetadata is OAuth 2.0 authorization server metadata
// (RFC 8414) or OpenID Connect provider metadata.
type AuthorizationServerMetadata struct {
	Issuer                                     string   `json:"issuer"`
	AuthorizationEndpoint                      string   `json:"authorization_endpoint"`
	TokenEndpoint                              string   `json:"token_endpoint"`
	RegistrationEndpoint                       string   `json:"registration_endpoint,omitempty"`
	AuthorizationResponseIssParameterSupported *bool    `json:"authorization_response_iss_parameter_supported,omitempty"`
	ClientIDMetadataDocumentSupported          *bool    `json:"client_id_metadata_document_supported,omitempty"`
	ScopesSupported                            []string `json:"scopes_supported,omitempty"`
	ResponseTypesSupported                     []string `json:"response_types_supported"`
	GrantTypesSupported                        []string `json:"grant_types_supported,omitempty"`
	CodeChallengeMethodsSupported              []string `json:"code_challenge_methods_supported,omitempty"`
	TokenEndpointAuthMethodsSupported          []string `json:"token_endpoint_auth_methods_supported,omitempty"`
	TokenEndpointAuthSigningAlgValuesSupported []string `json:"token_endpoint_auth_signing_alg_values_supported,omitempty"`

	// OpenID Connect members.
	UserinfoEndpoint                 string   `json:"userinfo_endpoint,omitempty"`
	JWKSURI                          string   `json:"jwks_uri,omitempty"`
	SubjectTypesSupported            []string `json:"subject_types_supported,omitempty"`
	IDTokenSigningAlgValuesSupported []string `json:"id_token_signing_alg_values_supported,omitempty"`
	ClaimsSupported                  []string `json:"claims_supported,omitempty"`
}

func (m AuthorizationServerMetadata) validateOAuth(raw json.RawMessage) error {
	if err := requireFields(raw, "issuer", "authorization_endpoint", "token_endpoint", "response_types_supported"); err != nil {
		return err
	}

	return validateSafeURLs(map[string]string{
		"authorization_endpoint": m.AuthorizationEndpoint,
		"token_endpoint":         m.TokenEndpoint,
		"registration_endpoint":  m.RegistrationEndpoint,
	})
}

func (m AuthorizationServerMetadata) validateOpenID(raw json.RawMessage) error {
	if err := m.validateOAuth(raw); err != nil {
		return err
	}

	if err := requireFields(raw, "jwks_uri", "subject_types_supported", "id_token_signing_alg_values_supported"); err != nil {
		return err
	}

	return validateSafeURLs(map[string]string{
		"userinfo_endpoint": m.UserinfoEndpoint,
		"jwks_uri":          m.JWKSURI,
	})
}

// protectedResourceMetadata is OAuth 2.0 protected resource metadata
// (RFC 9728).
type protectedResourceMetadata struct {
	Resource             string   `json:"resource"`
	AuthorizationServers []string `json:"authorization_servers,omitempty"`
	ScopesSupported      []string `json:"scopes_supported,omitempty"`
}

func (m protectedResourceMetadata) validate(raw json.RawMessage) error {
	if err := requireFields(raw, "resource"); err != nil {
		return err
	}

	if _, err := parseAbsoluteURL(m.Resource); err != nil {
		return fmt.Errorf("resource: %w", err)
	}

	for _, server := range m.AuthorizationServers {
		if err := validateSafeURL(server); err != nil {
			return fmt.Errorf("authorization_servers: %w", err)
		}
	}

	return nil
}

// validateSafeURLs checks each nonempty URL with validateSafeURL.
func validateSafeURLs(urls map[string]string) error {
	for name, value := range urls {
		if value == "" {
			continue
		}

		if err := validateSafeURL(value); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
	}

	return nil
}

// validateSafeURL accepts absolute URLs except javascript:, data: and
// vbscript: URLs.
func validateSafeURL(value string) error {
	parsed, err := parseAbsoluteURL(value)
	if err != nil {
		return err
	}

	switch strings.ToLower(parsed.Scheme) {
	case "javascript", "data", "vbscript":
		return errors.New("url cannot use javascript:, data:, or vbscript: scheme")
	default:
		return nil
	}
}

func parseAbsoluteURL(value string) (*url.URL, error) {
	parsed, err := url.Parse(value)
	if err != nil {
		return nil, fmt.Errorf("invalid url %q: %w", value, err)
	}

	if parsed.Scheme == "" {
		return nil, fmt.Errorf("url %q is not absolute", value)
	}

	return parsed, nil
}

// decodeValidated decodes raw into a value and runs its validation.
func decodeValidated[Value any](raw []byte, validate func(value Value, raw json.RawMessage) error) (Value, error) {
	var value Value
	if err := json.Unmarshal(raw, &value); err != nil {
		return value, err
	}

	if err := validate(value, raw); err != nil {
		return value, err
	}

	return value, nil
}
