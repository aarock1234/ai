package livetranscription

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/aarock1234/ai/internal/apicall"
	"github.com/aarock1234/ai/internal/websocket"
	"github.com/aarock1234/ai/pkg/model"
)

// DialOptions configure Dial.
type DialOptions struct {
	// URL is the ws or wss URL of the connection.
	URL string

	// DisplayURL replaces URL in errors when URL carries a credential, such
	// as an API key query parameter.
	DisplayURL string

	// Client sends the handshake. Nil uses http.DefaultClient.
	Client *http.Client

	Header    http.Header
	Protocols []string

	// ParseError reads the provider's message from a rejected handshake.
	ParseError apicall.ErrorParser
}

// Dial opens the WebSocket connection of a live transcription. A rejected
// handshake is reported as a *model.APICallError, like a failed HTTP call;
// cancellation reports the context's error.
func Dial(ctx context.Context, options DialOptions) (*websocket.Conn, error) {
	conn, err := websocket.Dial(ctx, options.URL, websocket.DialOptions{
		Client:    options.Client,
		Header:    options.Header,
		Protocols: options.Protocols,
	})
	if err != nil {
		return nil, dialError(ctx, options, err)
	}

	return conn, nil
}

// WebSocketURL converts an http or https URL to ws or wss. Other schemes
// are kept.
func WebSocketURL(rawURL string) (*url.URL, error) {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return nil, fmt.Errorf("%w: parse websocket url: %w", model.ErrInvalidArgument, err)
	}

	switch parsed.Scheme {
	case "http":
		parsed.Scheme = "ws"
	case "https":
		parsed.Scheme = "wss"
	}

	return parsed, nil
}

func dialError(ctx context.Context, options DialOptions, err error) error {
	if ctx.Err() != nil {
		return context.Cause(ctx)
	}

	displayURL := options.DisplayURL
	if displayURL == "" {
		displayURL = options.URL
	}

	redactURL(err, displayURL)

	handshake, rejected := errors.AsType[*websocket.HandshakeError](err)
	if !rejected {
		return &model.APICallError{
			Message: "provider request failed",
			URL:     displayURL,
			Cause:   err,
		}
	}

	return &model.APICallError{
		Message:         handshakeMessage(handshake, options.ParseError),
		URL:             displayURL,
		StatusCode:      handshake.StatusCode,
		ResponseHeaders: handshake.Header,
		ResponseBody:    string(handshake.Body),
		IsRetryable:     model.RetryableStatus(handshake.StatusCode),
		Cause:           err,
	}
}

// redactURL replaces the URL that net/http reports in a transport error,
// which may carry a credential, with the display URL. The error was created
// by this dial, so it is changed in place.
func redactURL(err error, displayURL string) {
	transportErr, ok := errors.AsType[*url.Error](err)
	if !ok {
		return
	}

	transportErr.URL = displayURL
}

// handshakeMessage returns the provider's error message, the status text,
// or the handshake failure.
func handshakeMessage(handshake *websocket.HandshakeError, parse apicall.ErrorParser) string {
	if parse != nil {
		if message := parse(handshake.Body); message != "" {
			return message
		}
	}

	if handshake.StatusCode != http.StatusSwitchingProtocols {
		if text := http.StatusText(handshake.StatusCode); text != "" {
			return strings.ToLower(text)
		}
	}

	return handshake.Message
}
