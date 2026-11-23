package apicall

import (
	"fmt"
	"net/http"

	"github.com/aarock1234/ai/internal/origin"
)

// maxRedirects matches the limit of net/http's default redirect policy.
const maxRedirects = 10

// withCredentialSafeRedirects returns a copy of client that sends only the
// User-Agent header once a redirect leaves the origin of the original
// request. net/http strips Authorization and cookies on such redirects, but
// providers also carry API keys in custom headers. The client's own
// CheckRedirect, when set, still decides whether to follow.
func withCredentialSafeRedirects(client *http.Client) *http.Client {
	copied := *client
	followRedirect := client.CheckRedirect

	copied.CheckRedirect = func(request *http.Request, via []*http.Request) error {
		if !origin.SameURL(request.URL, via[0].URL) {
			request.Header = userAgentOnly(request.Header)
		}

		if followRedirect != nil {
			return followRedirect(request, via)
		}

		if len(via) >= maxRedirects {
			return fmt.Errorf("stopped after %d redirects", maxRedirects)
		}

		return nil
	}

	return &copied
}

// userAgentOnly returns the User-Agent values of headers and nothing else.
func userAgentOnly(headers http.Header) http.Header {
	reduced := make(http.Header)

	if values := headers.Values("User-Agent"); len(values) > 0 {
		reduced["User-Agent"] = values
	}

	return reduced
}
