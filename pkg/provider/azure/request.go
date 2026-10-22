package azure

import (
	"net/http"

	"github.com/aarock1234/ai/internal/apicall"
)

// serviceRequest describes a call to an Azure Speech or MAI URL, with the
// service's authentication, the provider headers and the call headers.
func (s settings) serviceRequest(service api, requestURL string, callHeaders http.Header) (apicall.Request, error) {
	authentication, err := s.authHeaders(service)
	if err != nil {
		return apicall.Request{}, err
	}

	return apicall.Request{
		Client:     s.client(),
		URL:        requestURL,
		Headers:    apicall.MergeHeaders(authentication, s.headers, callHeaders),
		ParseError: parseError,
	}, nil
}
