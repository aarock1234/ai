package xai

import (
	"fmt"
	"net/http"

	"github.com/aarock1234/ai/pkg/provider/googlevertex"
)

// TokenSource returns an OAuth 2.0 access token with the
// https://www.googleapis.com/auth/cloud-platform scope, such as a token from
// golang.org/x/oauth2/google. It is called for every request.
type TokenSource = googlevertex.TokenSource

// client returns the HTTP client of every request. With a token source, it
// sets a bearer token on every request, replacing configured Authorization
// headers.
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
		},
		CheckRedirect: base.CheckRedirect,
		Jar:           base.Jar,
		Timeout:       base.Timeout,
	}
}

// tokenTransport authenticates requests with OAuth access tokens.
type tokenTransport struct {
	base        http.RoundTripper
	tokenSource TokenSource
}

// RoundTrip sends a clone of the request with a bearer token, as
// RoundTrippers must not modify the request.
func (t *tokenTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	token, err := t.tokenSource(request.Context())
	if err != nil {
		if request.Body != nil {
			_ = request.Body.Close()
		}

		return nil, fmt.Errorf("google vertex xai token source: %w", err)
	}

	authenticated := request.Clone(request.Context())
	authenticated.Header.Set("Authorization", "Bearer "+token)

	return t.base.RoundTrip(authenticated)
}
