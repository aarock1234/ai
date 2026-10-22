package anthropicaws

import (
	"net/http"

	"github.com/aarock1234/ai/internal/apikey"
	"github.com/aarock1234/ai/internal/awssigv4"
	"github.com/aarock1234/ai/internal/origin"
)

const (
	// signingService is the SigV4 service name of Claude Platform on AWS.
	signingService = "aws-external-anthropic"

	// apiKeyAlternative names API key authentication in credential errors.
	apiKeyAlternative = apiKeyVariable + " or WithAPIKey"
)

// authTransport authenticates each request to the provider's endpoint with
// the API key, or signs it with SigV4, resolving the key or credentials
// when the request is sent. Requests to other origins, such as batch result
// downloads, are sent without credentials.
type authTransport struct {
	settings settings
	base     http.RoundTripper
}

// RoundTrip adds the x-api-key header or signs the request when it targets
// the endpoint's origin.
func (t *authTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	isServiceRequest, err := t.isServiceRequest(request)
	if err != nil {
		closeRequestBody(request)

		return nil, err
	}

	if !isServiceRequest {
		return t.base.RoundTrip(request)
	}

	apiKey := apikey.Lookup(t.settings.apiKey, apiKeyVariable)

	if apiKey != "" {
		authorized := request.Clone(request.Context())
		authorized.Header.Set("X-Api-Key", apiKey)

		return t.base.RoundTrip(authorized)
	}

	signer, err := t.signer(request)
	if err != nil {
		closeRequestBody(request)

		return nil, err
	}

	signing := &awssigv4.Transport{
		Base:   t.base,
		Signer: signer,
	}

	return signing.RoundTrip(request)
}

// isServiceRequest reports whether request targets the origin of the
// provider's base URL.
func (t *authTransport) isServiceRequest(request *http.Request) (bool, error) {
	baseURL, err := t.settings.resolveBaseURL()
	if err != nil {
		return false, err
	}

	return origin.Same(request.URL.String(), baseURL), nil
}

func (t *authTransport) signer(request *http.Request) (awssigv4.Signer, error) {
	region, err := awssigv4.LoadRegion(t.settings.region)
	if err != nil {
		return awssigv4.Signer{}, err
	}

	credentialSettings := t.settings.credentials
	credentialSettings.APIKeyAlternative = apiKeyAlternative

	credentials, err := awssigv4.LoadCredentials(request.Context(), credentialSettings)
	if err != nil {
		return awssigv4.Signer{}, err
	}

	return awssigv4.Signer{
		Credentials: credentials,
		Region:      region,
		Service:     signingService,
	}, nil
}

// closeRequestBody closes the body of a request that is not sent, as
// RoundTrip must.
func closeRequestBody(request *http.Request) {
	if request.Body != nil {
		_ = request.Body.Close()
	}
}
