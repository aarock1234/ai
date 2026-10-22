package mcp

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"runtime"
	"strings"
	"sync"
)

// maxErrorBodyBytes bounds the body read from a failed HTTP response.
const maxErrorBodyBytes = 64 << 10

// errRedirectNotAllowed rejects redirects of transports that do not follow
// them.
var errRedirectNotAllowed = errors.New("mcp http transport does not follow redirects")

// userAgentSuffix identifies the SDK and runtime in the User-Agent header.
var userAgentSuffix = fmt.Sprintf("ai-sdk-go/mcp/%s runtime/%s", Version, runtime.Version())

// httpSettings are the options shared by the HTTP transports.
type httpSettings struct {
	headers      http.Header
	client       *http.Client
	authProvider OAuthClientProvider
}

// newHTTPClient returns client, or http.DefaultClient, configured to follow
// redirects or to fail on them.
func newHTTPClient(client *http.Client, followRedirects bool) *http.Client {
	if client == nil {
		client = http.DefaultClient
	}

	copied := *client
	if !followRedirects {
		copied.CheckRedirect = func(*http.Request, []*http.Request) error {
			return errRedirectNotAllowed
		}
	}

	return &copied
}

// requestHeaders returns the configured headers with base, the protocol
// version, the bearer token and the User-Agent suffix.
func requestHeaders(ctx context.Context, settings httpSettings, base http.Header, protocolVersion string) (http.Header, error) {
	headers := settings.headers.Clone()
	if headers == nil {
		headers = make(http.Header)
	}

	for name, values := range base {
		headers[http.CanonicalHeaderKey(name)] = values
	}

	headers.Set("Mcp-Protocol-Version", protocolVersion)

	if settings.authProvider != nil {
		tokens, err := settings.authProvider.Tokens(ctx)
		if err != nil {
			return nil, fmt.Errorf("load oauth tokens: %w", err)
		}

		if tokens != nil && tokens.AccessToken != "" {
			headers.Set("Authorization", "Bearer "+tokens.AccessToken)
		}
	}

	headers.Set("User-Agent", strings.TrimSpace(headers.Get("User-Agent")+" "+userAgentSuffix))

	return headers, nil
}

// authorizer runs one OAuth recovery flow for concurrent 401 responses of a
// transport.
type authorizer struct {
	provider  OAuthClientProvider
	serverURL string
	client    *http.Client

	mu       sync.Mutex
	inFlight *authorization
}

// authorization is a running Auth call that other callers wait for.
type authorization struct {
	done   chan struct{}
	result AuthResult
	err    error
}

// authorize runs Auth or waits for the call already running. ctx bounds the
// wait; the first caller's ctx bounds the shared flow.
func (a *authorizer) authorize(ctx context.Context, challenge wwwAuthenticateParameters) (AuthResult, error) {
	a.mu.Lock()
	if running := a.inFlight; running != nil {
		a.mu.Unlock()

		select {
		case <-running.done:
			return running.result, running.err
		case <-ctx.Done():
			return "", ctx.Err()
		}
	}

	running := &authorization{
		done: make(chan struct{}),
	}
	a.inFlight = running
	a.mu.Unlock()

	running.result, running.err = Auth(ctx, a.provider, AuthOptions{
		ServerURL:           a.serverURL,
		Scope:               challenge.scope,
		ResourceMetadataURL: challenge.resourceMetadataURL,
		HTTPClient:          a.client,
	})

	a.mu.Lock()
	a.inFlight = nil
	a.mu.Unlock()

	close(running.done)

	return running.result, running.err
}

// recoverUnauthorized runs the OAuth flow after a 401 response and reports
// an error unless it produced new tokens.
func (a *authorizer) recoverUnauthorized(ctx context.Context, response *http.Response) error {
	result, err := a.authorize(ctx, parseWWWAuthenticate(response.Header))
	if err != nil {
		return err
	}

	if result != AuthResultAuthorized {
		return ErrUnauthorized
	}

	return nil
}

// readErrorBody returns the beginning of a failed response's body, or
// empty when it cannot be read.
func readErrorBody(response *http.Response) (string, bool) {
	body, err := io.ReadAll(io.LimitReader(response.Body, maxErrorBodyBytes))
	if err != nil {
		return "", false
	}

	return string(body), true
}

// mergeContexts returns a context that ends when either parent ends. It
// keeps the values of primary.
func mergeContexts(primary context.Context, secondary context.Context) (context.Context, context.CancelFunc) {
	merged, cancel := context.WithCancelCause(primary)
	stop := context.AfterFunc(secondary, func() {
		cancel(context.Cause(secondary))
	})

	return merged, func() {
		stop()
		cancel(context.Canceled)
	}
}
