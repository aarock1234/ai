// Package bedrock holds the authentication, endpoint and stream error rules
// shared by the Amazon Bedrock providers.
package bedrock

import (
	"context"
	"net/http"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/apikey"
	"github.com/aarock1234/ai/internal/awssigv4"
)

const (
	// APIKeyVariable holds a Bedrock API key for bearer authentication.
	APIKeyVariable = "AWS_BEARER_TOKEN_BEDROCK"

	// defaultSigningService is the SigV4 service name of the Bedrock runtimes.
	defaultSigningService = "bedrock"

	// apiKeyAlternative names bearer authentication in credential errors.
	apiKeyAlternative = APIKeyVariable + " or WithAPIKey"
)

// Config is the authentication and endpoint configuration of a provider.
type Config struct {
	// Region is the configured AWS region. Empty reads AWS_REGION.
	Region string

	// APIKey enables bearer authentication. Empty reads
	// AWS_BEARER_TOKEN_BEDROCK; without a key, requests are signed with
	// SigV4.
	APIKey string

	// Credentials are the SigV4 credential sources.
	Credentials awssigv4.CredentialSettings

	// BaseURL replaces the regional endpoint.
	BaseURL string

	// HTTPClient sends requests. Nil uses http.DefaultClient.
	HTTPClient *http.Client

	// SigningService is the SigV4 service name, such as "bedrock-mantle".
	// Empty uses "bedrock".
	SigningService string
}

// Authorize returns request with the provider's authentication: a bearer
// API key header, or a client that signs requests with SigV4 for the
// Bedrock service.
func (c Config) Authorize(ctx context.Context, request apicall.Request) (apicall.Request, error) {
	request.Client = c.HTTPClient

	if apiKey := c.apiKey(); apiKey != "" {
		request.Headers = apicall.MergeHeaders(request.Headers, http.Header{
			"Authorization": {
				"Bearer " + apiKey,
			},
		})

		return request, nil
	}

	signer, err := c.signer(ctx)
	if err != nil {
		return apicall.Request{}, err
	}

	request.Client = awssigv4.Client(c.HTTPClient, signer)

	return request, nil
}

// apiKey returns the configured or environment API key, trimmed. A blank
// key selects SigV4 authentication.
func (c Config) apiKey() string {
	return strings.TrimSpace(apikey.Lookup(c.APIKey, APIKeyVariable))
}

func (c Config) signer(ctx context.Context) (awssigv4.Signer, error) {
	region, err := awssigv4.LoadRegion(c.Region)
	if err != nil {
		return awssigv4.Signer{}, err
	}

	settings := c.Credentials
	settings.APIKeyAlternative = apiKeyAlternative

	credentials, err := awssigv4.LoadCredentials(ctx, settings)
	if err != nil {
		return awssigv4.Signer{}, err
	}

	return awssigv4.Signer{
		Credentials: credentials,
		Region:      region,
		Service:     c.signingService(),
	}, nil
}

// signingService returns the configured SigV4 service name or "bedrock".
func (c Config) signingService() string {
	if c.SigningService == "" {
		return defaultSigningService
	}

	return c.SigningService
}
