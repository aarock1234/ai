// Package download fetches files from untrusted URLs, such as URLs in prompts
// or provider responses. It ports the upstream download-blob,
// fetch-untrusted-url and validate-download-url guards against server-side
// request forgery: only http, https and data URLs are accepted, local and
// private hosts are rejected before every hop and again after DNS
// resolution, redirects are validated before they are followed, credentials
// stay with their origin, and the downloaded size is capped. ValidatingClient
// exposes the address-validating client for other requests to untrusted
// URLs. Data URLs are decoded with internal/dataurl.
package download

import (
	"context"
	"fmt"
	"io"
	"net/http"

	"github.com/aarock1234/ai/internal/dataurl"
)

// DefaultMaxBytes is the default download size limit: 2 GiB.
const DefaultMaxBytes int64 = 2 << 30

// Error reports a failed download.
type Error struct {
	URL string

	// StatusCode is zero when no response was received.
	StatusCode int
	StatusText string

	// Message overrides the default description.
	Message string

	Cause error
}

// Error describes the failed download.
func (e *Error) Error() string {
	switch {
	case e.Message != "":
		return e.Message
	case e.Cause != nil:
		return fmt.Sprintf("failed to download %s: %v", e.URL, e.Cause)
	default:
		return fmt.Sprintf("failed to download %s: %d %s", e.URL, e.StatusCode, e.StatusText)
	}
}

// Unwrap returns the underlying cause.
func (e *Error) Unwrap() error {
	return e.Cause
}

// File is the content of a downloaded URL.
type File struct {
	Data []byte

	// MediaType is the response's Content-Type, or empty when absent.
	MediaType string
}

// Request describes one download.
type Request struct {
	// URL is the http, https or data URL to fetch.
	URL string

	// Headers are sanitized before sending. Unless URL has the same origin as
	// CredentialedOrigin, the first request carries only non-credential
	// metadata headers such as Accept and User-Agent. A redirect to another
	// origin drops every header except User-Agent.
	Headers http.Header

	// CredentialedOrigin is the configured origin, such as the provider base
	// URL, allowed to receive every header. Empty uses TrustedOrigin.
	CredentialedOrigin string

	// TrustedOrigin is a configured origin, such as the provider base URL,
	// whose requests skip URL and address validation, so that self-hosted
	// endpoints on private networks keep working. It must never come from
	// response data.
	TrustedOrigin string

	// Client sends requests to TrustedOrigin. Nil uses http.DefaultClient.
	// Every other request uses a transport that validates resolved addresses.
	Client *http.Client

	// MaxBytes limits the size of the download. Zero selects
	// DefaultMaxBytes.
	MaxBytes int64
}

// Fetch downloads request.URL. It returns a *Error when the URL is not
// allowed, a request fails, the response status is not successful, or the
// body exceeds the size limit. Data URLs are decoded without a network
// request.
func Fetch(ctx context.Context, request Request) (*File, error) {
	if dataurl.Is(request.URL) {
		return decodeDataURL(request.URL, maxBytesOf(request))
	}

	response, err := followRedirects(ctx, request)
	if err != nil {
		return nil, err
	}
	defer closeBody(response.Body)

	if response.StatusCode < 200 || response.StatusCode > 299 {
		return nil, &Error{
			URL:        request.URL,
			StatusCode: response.StatusCode,
			StatusText: http.StatusText(response.StatusCode),
		}
	}

	data, err := readWithSizeLimit(response, request.URL, maxBytesOf(request))
	if err != nil {
		return nil, err
	}

	return &File{
		Data:      data,
		MediaType: response.Header.Get("Content-Type"),
	}, nil
}

// closeBody closes a response body that has been read or abandoned. Its
// error is ignored: the outcome of the download is already decided by then.
func closeBody(body io.Closer) {
	_ = body.Close()
}

func maxBytesOf(request Request) int64 {
	if request.MaxBytes <= 0 {
		return DefaultMaxBytes
	}

	return request.MaxBytes
}

// readWithSizeLimit reads the body, failing early on a declared length above
// maxBytes and while reading once more than maxBytes arrive.
func readWithSizeLimit(response *http.Response, location string, maxBytes int64) ([]byte, error) {
	if response.ContentLength > maxBytes {
		return nil, &Error{
			URL: location,
			Message: fmt.Sprintf(
				"download of %s exceeded maximum size of %d bytes (Content-Length: %d)",
				location,
				maxBytes,
				response.ContentLength,
			),
		}
	}

	data, err := io.ReadAll(io.LimitReader(response.Body, maxBytes+1))
	if err != nil {
		return nil, &Error{
			URL:   location,
			Cause: err,
		}
	}

	if int64(len(data)) > maxBytes {
		return nil, &Error{
			URL:     location,
			Message: fmt.Sprintf("download of %s exceeded maximum size of %d bytes", location, maxBytes),
		}
	}

	return data, nil
}
