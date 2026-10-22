package openaicompatible

import (
	"fmt"
	"net/http"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/pkg/model"
)

// URLBuilder returns the full URL of an API path such as
// "/chat/completions". It returns an error when the settings needed to build
// the URL, such as a resource name or project, are missing or invalid.
type URLBuilder func(path string) (string, error)

// InfallibleURL adapts a URL function that cannot fail, such as one that
// joins a fixed base URL and the path, into a URLBuilder.
func InfallibleURL(url func(path string) string) URLBuilder {
	return func(path string) (string, error) {
		return url(path), nil
	}
}

// apiRequestSettings describes one call to a model endpoint.
type apiRequestSettings struct {
	path        string
	buildURL    URLBuilder
	headers     func() (http.Header, error)
	callHeaders http.Header
	client      *http.Client
	parseError  ErrorParser
}

// newAPIRequest builds the request URL and combines the provider headers
// with the call headers, which take precedence.
func newAPIRequest(settings apiRequestSettings) (apicall.Request, error) {
	requestURL, err := settings.requestURL()
	if err != nil {
		return apicall.Request{}, err
	}

	var providerHeaders http.Header
	if settings.headers != nil {
		headers, err := settings.headers()
		if err != nil {
			return apicall.Request{}, err
		}

		providerHeaders = headers
	}

	parseError := settings.parseError
	if parseError == nil {
		parseError = ParseError
	}

	return apicall.Request{
		Client:     settings.client,
		URL:        requestURL,
		Headers:    apicall.MergeHeaders(providerHeaders, settings.callHeaders),
		ParseError: apicall.ErrorParser(parseError),
	}, nil
}

// requestURL returns the URL of the path.
func (s apiRequestSettings) requestURL() (string, error) {
	if s.buildURL == nil {
		return "", fmt.Errorf("%w: model config sets no BuildURL", model.ErrInvalidArgument)
	}

	return s.buildURL(s.path)
}
