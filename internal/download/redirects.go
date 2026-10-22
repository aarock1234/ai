package download

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/aarock1234/ai/internal/origin"
)

const (
	maxRedirects     = 10
	defaultUserAgent = "ai-sdk-go"
)

// blockedHeaders are never sent to an untrusted URL: hop-by-hop transport,
// host routing, proxy and origin spoofing, cloud metadata and cookies.
var blockedHeaders = []string{
	"Connection",
	"Keep-Alive",
	"Te",
	"Trailer",
	"Transfer-Encoding",
	"Upgrade",
	"Host",
	"Forwarded",
	"Proxy-Authorization",
	"Via",
	"X-Forwarded-For",
	"X-Forwarded-Host",
	"X-Forwarded-Proto",
	"X-Real-Ip",
	"Metadata",
	"Metadata-Flavor",
	"X-Aws-Ec2-Metadata-Token",
	"X-Metadata-Token",
	"Cookie",
	"Set-Cookie",
}

// untrustedFirstHopHeaders are the non-credential metadata headers sent to a
// URL outside the credentialed origin.
var untrustedFirstHopHeaders = map[string]bool{
	"Accept":              true,
	"Accept-Language":     true,
	"Baggage":             true,
	"Cache-Control":       true,
	"Idempotency-Key":     true,
	"If-Match":            true,
	"If-Modified-Since":   true,
	"If-None-Match":       true,
	"If-Range":            true,
	"If-Unmodified-Since": true,
	"Pragma":              true,
	"Range":               true,
	"Traceparent":         true,
	"Tracestate":          true,
	"User-Agent":          true,
	"X-Correlation-Id":    true,
	"X-Request-Id":        true,
}

// redirectStatusCodes are the statuses whose Location is followed.
var redirectStatusCodes = map[int]bool{
	http.StatusMovedPermanently:  true,
	http.StatusFound:             true,
	http.StatusSeeOther:          true,
	http.StatusTemporaryRedirect: true,
	http.StatusPermanentRedirect: true,
}

// followRedirects requests the URL and follows redirects one hop at a time,
// validating each target before it is requested. The caller closes the
// returned response body.
func followRedirects(ctx context.Context, request Request) (*http.Response, error) {
	current, err := url.Parse(request.URL)
	if err != nil {
		return nil, &Error{
			URL:     request.URL,
			Message: fmt.Sprintf("invalid URL: %s", request.URL),
			Cause:   err,
		}
	}

	headers := firstHopHeaders(request, current)

	for range maxRedirects + 1 {
		response, err := fetchHop(ctx, request, current, headers)
		if err != nil {
			return nil, err
		}

		next, isRedirect := redirectTarget(response, current)
		if !isRedirect {
			return response, nil
		}

		closeBody(response.Body)

		if err := validateRedirectScheme(next); err != nil {
			return nil, err
		}

		if !origin.SameURL(next, current) {
			headers = userAgentOnly(headers)
		}

		current = next
	}

	return nil, &Error{
		URL:     request.URL,
		Message: fmt.Sprintf("too many redirects (max %d)", maxRedirects),
	}
}

// fetchHop sends one request without following redirects. Hops outside the
// trusted origin are validated and use the address-validating client.
func fetchHop(ctx context.Context, request Request, target *url.URL, headers http.Header) (*http.Response, error) {
	client := validatingClient()

	if isTrustedOrigin(target, request.TrustedOrigin) {
		client = withoutRedirects(request.Client)
	} else if err := validateURL(target); err != nil {
		return nil, err
	}

	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodGet, target.String(), nil)
	if err != nil {
		return nil, &Error{
			URL:   target.String(),
			Cause: err,
		}
	}

	httpRequest.Header = headers.Clone()

	response, err := client.Do(httpRequest)
	if err != nil {
		return nil, transportFailure(target.String(), err)
	}

	return response, nil
}

// redirectTarget returns the resolved Location of a redirect response.
func redirectTarget(response *http.Response, current *url.URL) (*url.URL, bool) {
	location := response.Header.Get("Location")
	if !redirectStatusCodes[response.StatusCode] || location == "" {
		return nil, false
	}

	next, err := current.Parse(location)
	if err != nil {
		return nil, false
	}

	return next, true
}

// validateRedirectScheme rejects a redirect to anything but http or https,
// such as a data or file URL, even from the trusted origin.
func validateRedirectScheme(target *url.URL) error {
	scheme := strings.ToLower(target.Scheme)
	if scheme == "http" || scheme == "https" {
		return nil
	}

	return &Error{
		URL:     target.String(),
		Message: fmt.Sprintf("redirect URL scheme must be http or https, got %s:", target.Scheme),
	}
}

// transportFailure keeps a *Error raised while validating a dialed address
// and wraps any other transport failure.
func transportFailure(location string, err error) error {
	if downloadErr, ok := errors.AsType[*Error](err); ok {
		return downloadErr
	}

	return &Error{
		URL:   location,
		Cause: err,
	}
}

// firstHopHeaders sanitizes the caller's headers and, outside the
// credentialed origin, keeps only non-credential metadata.
func firstHopHeaders(request Request, target *url.URL) http.Header {
	headers := request.Headers.Clone()
	if headers == nil {
		headers = make(http.Header)
	}

	for _, name := range blockedHeaders {
		headers.Del(name)
	}

	credentialedOrigin := request.CredentialedOrigin
	if credentialedOrigin == "" {
		credentialedOrigin = request.TrustedOrigin
	}

	if !isTrustedOrigin(target, credentialedOrigin) {
		for name := range headers {
			if !untrustedFirstHopHeaders[http.CanonicalHeaderKey(name)] {
				headers.Del(name)
			}
		}
	}

	if headers.Get("User-Agent") == "" {
		headers.Set("User-Agent", defaultUserAgent)
	}

	return headers
}

func userAgentOnly(headers http.Header) http.Header {
	return http.Header{
		"User-Agent": {
			headers.Get("User-Agent"),
		},
	}
}

// isTrustedOrigin reports whether target has the origin of the configured
// reference URL. An empty or invalid reference is never trusted.
func isTrustedOrigin(target *url.URL, reference string) bool {
	if reference == "" {
		return false
	}

	parsed, err := url.Parse(reference)
	if err != nil {
		return false
	}

	return origin.SameURL(target, parsed)
}

// withoutRedirects returns a copy of client that returns redirect responses
// instead of following them.
func withoutRedirects(client *http.Client) *http.Client {
	if client == nil {
		client = http.DefaultClient
	}

	copied := *client
	copied.CheckRedirect = returnRedirect

	return &copied
}

func returnRedirect(*http.Request, []*http.Request) error {
	return http.ErrUseLastResponse
}

// ValidatingClient returns a client that checks the resolved address of
// every connection, rejecting private, loopback and other non-public
// addresses after DNS resolution, so that a public hostname cannot be
// rebound to a private address. It returns redirect responses instead of
// following them, so that callers validate each target. The clients share
// one transport, which reuses connections and ignores proxy settings that
// would bypass the address check. This is the Go counterpart of the
// upstream DNS-pinned download fetch.
func ValidatingClient() *http.Client {
	copied := *validatingClient()

	return &copied
}

// validatingClient is the shared client behind ValidatingClient.
var validatingClient = sync.OnceValue(func() *http.Client {
	dialer := &net.Dialer{
		Timeout:   30 * time.Second,
		KeepAlive: 30 * time.Second,
		Control:   validateDialAddress,
	}

	transport := &http.Transport{
		DialContext:           dialer.DialContext,
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          100,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: time.Second,
	}

	return &http.Client{
		Transport:     transport,
		CheckRedirect: returnRedirect,
	}
})
