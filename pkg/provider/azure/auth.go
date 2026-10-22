package azure

import (
	"fmt"
	"net/http"
	"net/url"
	"slices"

	"github.com/aarock1234/ai/internal/apikey"
	"github.com/aarock1234/ai/internal/origin"
	"github.com/aarock1234/ai/pkg/model"
)

// api identifies the Azure service behind a request, which selects the API
// key header and the description in errors.
type api string

const (
	apiOpenAI api = "openai"
	apiSpeech api = "speech"
	apiMAI    api = "mai"
)

// keyHeader returns the header that carries the API key.
func (a api) keyHeader() string {
	if a == apiSpeech {
		return "Ocp-Apim-Subscription-Key"
	}

	return "Api-Key"
}

// description names the service in errors.
func (a api) description() string {
	switch a {
	case apiSpeech:
		return "Azure Speech"
	case apiMAI:
		return "Azure MAI"
	default:
		return "Azure OpenAI"
	}
}

// authHeaders returns the API key header of a service. Token provider
// authentication happens in the HTTP client's transport, so it adds no
// headers here.
func (s settings) authHeaders(service api) (http.Header, error) {
	if s.apiKey != "" && s.tokenSource != nil {
		return nil, fmt.Errorf(
			"%w: both an API key and a token source were provided; use only one authentication method",
			model.ErrInvalidArgument,
		)
	}

	if s.tokenSource != nil {
		return http.Header{}, nil
	}

	apiKey := apikey.Lookup(s.apiKey, apiKeyVariable)

	if apiKey == "" {
		return nil, fmt.Errorf(
			"%w: %s API key is missing; set %s or use WithAPIKey",
			model.ErrLoadAPIKey,
			service.description(),
			apiKeyVariable,
		)
	}

	return http.Header{
		service.keyHeader(): {
			apiKey,
		},
	}, nil
}

// client returns the HTTP client of every request. With a token source, it
// adds a bearer token to requests that carry no Authorization header.
func (s settings) client() *http.Client {
	if s.tokenSource == nil {
		return s.httpClient
	}

	base := s.httpClient
	if base == nil {
		base = http.DefaultClient
	}

	transport := base.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}

	return &http.Client{
		Transport: &tokenTransport{
			base:        transport,
			tokenSource: s.tokenSource,
			origins:     s.serviceOrigins,
		},
		CheckRedirect: base.CheckRedirect,
		Jar:           base.Jar,
		Timeout:       base.Timeout,
	}
}

// serviceOrigins returns the origins of the OpenAI, Azure Speech and MAI
// endpoints that can be resolved. Only these receive access tokens.
func (s settings) serviceOrigins() []string {
	prefixes := []func() (string, error){
		s.openaiURLPrefix,
		func() (string, error) {
			return s.speechURL("")
		},
		func() (string, error) {
			return s.maiURL("")
		},
	}

	var origins []string

	for _, prefix := range prefixes {
		location, err := prefix()
		if err != nil {
			continue
		}

		parsed, err := url.Parse(location)
		if err != nil {
			continue
		}

		origins = append(origins, origin.Of(parsed))
	}

	return origins
}

// tokenTransport authenticates requests to the Azure endpoints with
// Microsoft Entra ID tokens.
type tokenTransport struct {
	base        http.RoundTripper
	tokenSource TokenSource
	origins     func() []string
}

// RoundTrip adds a bearer token to requests for a configured endpoint that
// have no Authorization header; requests to other origins, such as result
// downloads, are sent unchanged. The request is cloned, as RoundTrippers
// must not modify it.
func (t *tokenTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	if request.Header.Get("Authorization") != "" || !t.isServiceRequest(request) {
		return t.base.RoundTrip(request)
	}

	token, err := t.tokenSource(request.Context())
	if err != nil {
		closeRequestBody(request)

		return nil, fmt.Errorf("azure token source: %w", err)
	}

	authenticated := request.Clone(request.Context())
	authenticated.Header.Set("Authorization", "Bearer "+token)

	return t.base.RoundTrip(authenticated)
}

func (t *tokenTransport) isServiceRequest(request *http.Request) bool {
	requestOrigin := origin.Of(request.URL)

	return slices.Contains(t.origins(), requestOrigin)
}

// closeRequestBody closes the body of a request that will not be sent, as
// RoundTrippers must.
func closeRequestBody(request *http.Request) {
	if request.Body == nil {
		return
	}

	_ = request.Body.Close()
}
