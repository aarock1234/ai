package bedrock

import (
	"net/http"

	"github.com/aarock1234/ai/internal/awssigv4"
	"github.com/aarock1234/ai/internal/origin"
)

// Transport returns a transport that authenticates each request to the
// origin of the URL baseURL returns like Authorize, resolving the API key or
// credentials when the request is sent. Requests to other origins are sent
// without credentials. It serves clients that are fixed before any request,
// such as the Anthropic and Mantle providers'.
func (c Config) Transport(base http.RoundTripper, baseURL func() (string, error)) http.RoundTripper {
	return &authTransport{
		config:  c,
		baseURL: baseURL,
		base:    base,
	}
}

type authTransport struct {
	config  Config
	baseURL func() (string, error)
	base    http.RoundTripper
}

// RoundTrip adds a bearer API key or signs the request with SigV4 when the
// request targets the base URL's origin.
func (t *authTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	isServiceRequest, err := t.isServiceRequest(request)
	if err != nil {
		closeRequestBody(request)

		return nil, err
	}

	if !isServiceRequest {
		return t.base.RoundTrip(request)
	}

	if apiKey := t.config.apiKey(); apiKey != "" {
		authorized := request.Clone(request.Context())
		authorized.Header.Set("Authorization", "Bearer "+apiKey)

		return t.base.RoundTrip(authorized)
	}

	signer, err := t.config.signer(request.Context())
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

// isServiceRequest reports whether request targets the origin of the base
// URL.
func (t *authTransport) isServiceRequest(request *http.Request) (bool, error) {
	baseURL, err := t.baseURL()
	if err != nil {
		return false, err
	}

	return origin.Same(request.URL.String(), baseURL), nil
}

// closeRequestBody closes the body of a request that is not sent, as
// RoundTrip must.
func closeRequestBody(request *http.Request) {
	if request.Body != nil {
		_ = request.Body.Close()
	}
}
