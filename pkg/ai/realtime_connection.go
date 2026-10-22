package ai

import (
	"context"
	"fmt"
	"net/http"
	"slices"

	"github.com/aarock1234/ai/internal/websocket"
	"github.com/aarock1234/ai/pkg/model"
)

// RealtimeRelay is an application-owned WebSocket endpoint that relays the
// provider's realtime protocol, such as a proxy that adds credentials.
type RealtimeRelay struct {
	URL       string
	Protocols []string
	Headers   http.Header
}

// realtimeDialTarget is where and how a session connects.
type realtimeDialTarget struct {
	url       string
	header    http.Header
	protocols []string
}

// dialRealtime opens the session's WebSocket: through the relay when one is
// configured, with the model's server configuration when it has one, and
// otherwise with a client secret.
func dialRealtime(ctx context.Context, options RealtimeSessionOptions, config model.RealtimeSessionConfig) (*websocket.Conn, error) {
	target, err := resolveRealtimeDialTarget(ctx, options, config)
	if err != nil {
		return nil, err
	}

	conn, err := websocket.Dial(ctx, target.url, websocket.DialOptions{
		Client:    options.HTTPClient,
		Header:    target.header,
		Protocols: target.protocols,
	})
	if err != nil {
		if ctx.Err() != nil {
			return nil, context.Cause(ctx)
		}

		return nil, fmt.Errorf("connect to realtime model: %w", err)
	}

	return conn, nil
}

func resolveRealtimeDialTarget(ctx context.Context, options RealtimeSessionOptions, config model.RealtimeSessionConfig) (realtimeDialTarget, error) {
	realtimeModel := options.Model
	capabilities := realtimeModel.Capabilities()

	if relay := options.Relay; relay != nil {
		if err := checkRealtimeConnection(capabilities, model.RealtimeServerWebSocket); err != nil {
			return realtimeDialTarget{}, err
		}

		return realtimeDialTarget{
			url:       relay.URL,
			header:    relay.Headers,
			protocols: relay.Protocols,
		}, nil
	}

	configurer, hasServerConfig := realtimeModel.(model.RealtimeServerWebSocketConfigurer)
	supportsServerWebSocket := supportsRealtimeConnection(capabilities, model.RealtimeServerWebSocket)

	if hasServerConfig && supportsServerWebSocket {
		return serverWebSocketTarget(ctx, configurer)
	}

	if err := checkRealtimeConnection(capabilities, model.RealtimeClientSecretWebSocket); err != nil {
		return realtimeDialTarget{}, err
	}

	return clientSecretTarget(ctx, realtimeModel, config)
}

func serverWebSocketTarget(ctx context.Context, configurer model.RealtimeServerWebSocketConfigurer) (realtimeDialTarget, error) {
	config, err := configurer.ServerWebSocketConfig(ctx)
	if err != nil {
		return realtimeDialTarget{}, err
	}

	return realtimeDialTarget{
		url:    config.URL,
		header: config.Headers,
	}, nil
}

// clientSecretTarget creates a client secret and connects with it, as a
// browser client would.
func clientSecretTarget(ctx context.Context, realtimeModel model.RealtimeModel, config model.RealtimeSessionConfig) (realtimeDialTarget, error) {
	creator, isCreator := realtimeModel.(model.RealtimeClientSecretCreator)
	configurer, isConfigurer := realtimeModel.(model.RealtimeWebSocketConfigurer)

	if !isCreator || !isConfigurer {
		return realtimeDialTarget{}, fmt.Errorf(
			"%w: the %s realtime model %q supports no WebSocket connection",
			model.ErrUnsupported,
			realtimeModel.Provider(),
			realtimeModel.ModelID(),
		)
	}

	secret, err := creator.CreateClientSecret(ctx, model.RealtimeClientSecretOptions{
		SessionConfig: &config,
	})
	if err != nil {
		return realtimeDialTarget{}, err
	}

	if secret == nil {
		return realtimeDialTarget{}, fmt.Errorf(
			"%w: the %s realtime model %q returned no client secret",
			model.ErrInvalidResponse,
			realtimeModel.Provider(),
			realtimeModel.ModelID(),
		)
	}

	webSocket := configurer.WebSocketConfig(*secret)

	return realtimeDialTarget{
		url:       webSocket.URL,
		protocols: webSocket.Protocols,
	}, nil
}

// checkRealtimeConnection rejects connections the model does not declare.
func checkRealtimeConnection(capabilities model.RealtimeCapabilities, connection model.RealtimeConnection) error {
	if supportsRealtimeConnection(capabilities, connection) {
		return nil
	}

	return fmt.Errorf("%w: realtime model does not support %s", model.ErrUnsupported, connection)
}

// supportsRealtimeConnection reports whether a WebSocket connection of the
// kind is declared. Models that declare no connections support the client
// secret connection.
func supportsRealtimeConnection(capabilities model.RealtimeCapabilities, connection model.RealtimeConnection) bool {
	connections := capabilities.Connections
	if connections == nil {
		connections = []model.RealtimeConnection{
			model.RealtimeClientSecretWebSocket,
		}
	}

	if !slices.Contains(connections, connection) {
		return false
	}

	return capabilities.Transports == nil || slices.Contains(capabilities.Transports, model.RealtimeWebSocketTransport)
}
