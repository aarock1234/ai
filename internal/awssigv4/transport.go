package awssigv4

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
)

// Transport signs requests before passing them to Base. By default it signs
// every request, treating a missing body as an empty payload, so that GET
// and DELETE requests such as skill listings authenticate too.
type Transport struct {
	// Base sends the signed requests. Nil uses http.DefaultTransport.
	Base http.RoundTripper

	Signer Signer

	// SignPostBodiesOnly restricts signing to POST requests with a nonempty
	// body and sends every other request unsigned, as the upstream aws4fetch
	// wrappers do.
	SignPostBodiesOnly bool
}

// RoundTrip signs a copy of request and sends it.
func (t *Transport) RoundTrip(request *http.Request) (*http.Response, error) {
	if t.SignPostBodiesOnly && request.Method != http.MethodPost {
		return t.base().RoundTrip(request)
	}

	body, err := readBody(request)
	if err != nil {
		return nil, err
	}

	prepared := request.Clone(request.Context())
	setBody(prepared, body)

	if t.SignPostBodiesOnly && len(body) == 0 {
		return t.base().RoundTrip(prepared)
	}

	t.Signer.Sign(prepared, body)

	return t.base().RoundTrip(prepared)
}

func (t *Transport) base() http.RoundTripper {
	if t.Base == nil {
		return http.DefaultTransport
	}

	return t.Base
}

// Client returns a copy of client whose transport signs requests with
// signer. A nil client is http.DefaultClient.
func Client(client *http.Client, signer Signer) *http.Client {
	return WrapClient(client, func(base http.RoundTripper) http.RoundTripper {
		return &Transport{
			Base:   base,
			Signer: signer,
		}
	})
}

// WrapClient returns a copy of client whose transport is wrap applied to the
// client's transport. A nil client is http.DefaultClient and a nil transport
// is http.DefaultTransport.
func WrapClient(client *http.Client, wrap func(base http.RoundTripper) http.RoundTripper) *http.Client {
	if client == nil {
		client = http.DefaultClient
	}

	base := client.Transport
	if base == nil {
		base = http.DefaultTransport
	}

	wrapped := *client
	wrapped.Transport = wrap(base)

	return &wrapped
}

// readBody reads and closes the body of request. A missing body is an empty
// payload.
func readBody(request *http.Request) ([]byte, error) {
	if request.Body == nil || request.Body == http.NoBody {
		return nil, nil
	}

	defer func() {
		_ = request.Body.Close()
	}()

	body, err := io.ReadAll(request.Body)
	if err != nil {
		return nil, fmt.Errorf("read request body for signing: %w", err)
	}

	return body, nil
}

// setBody replaces the consumed body of request with body. An empty body
// becomes http.NoBody so that bodiless methods stay bodiless.
func setBody(request *http.Request, body []byte) {
	if len(body) == 0 {
		request.Body = http.NoBody
		request.ContentLength = 0
		request.GetBody = func() (io.ReadCloser, error) {
			return http.NoBody, nil
		}

		return
	}

	request.Body = io.NopCloser(bytes.NewReader(body))
	request.ContentLength = int64(len(body))
	request.GetBody = func() (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(body)), nil
	}
}
