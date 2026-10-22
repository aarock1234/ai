package bridgeadapter

import (
	"context"
	"fmt"

	"github.com/aarock1234/ai/pkg/harness"
	"github.com/aarock1234/ai/pkg/harness/bridge"
	"github.com/aarock1234/ai/pkg/sandbox"
)

// adapterIdentity names an adapter in its errors.
type adapterIdentity struct {
	harnessID string

	// displayName is the adapter's name in messages, such as "Claude Code".
	displayName string
}

// unsupported returns a capability error of the adapter.
func (a adapterIdentity) unsupported(message string) error {
	return &harness.CapabilityUnsupportedError{
		Message:   message,
		HarnessID: a.harnessID,
	}
}

// validateBasicSandbox requires an explicit port and endpoint for sandbox
// sessions that cannot resolve ports themselves.
func (a adapterIdentity) validateBasicSandbox(session sandbox.Session, settings bridge.ConnectionSettings) error {
	if _, ok := session.(harness.NetworkSession); ok {
		return nil
	}

	if settings.Port == 0 {
		return a.unsupported(fmt.Sprintf(
			"The %s harness requires an explicit port when using a basic sandbox session.",
			a.displayName,
		))
	}

	if settings.PortEndpoint == nil {
		return a.unsupported(fmt.Sprintf(
			"The %s harness requires an explicit port endpoint when using a basic sandbox session.",
			a.displayName,
		))
	}

	return nil
}

// resolvePort returns the port the bridge binds: the configured port, or the
// first port the sandbox exposes.
func (a adapterIdentity) resolvePort(session sandbox.Session, settings bridge.ConnectionSettings) (int, error) {
	if settings.Port != 0 {
		return settings.Port, nil
	}

	if network, ok := session.(harness.NetworkSession); ok && len(network.Ports()) > 0 {
		return network.Ports()[0], nil
	}

	return 0, a.unsupported(fmt.Sprintf(
		"The %s harness needs a TCP port exposed by the sandbox. Create the sandbox with ports or set the adapter's Port setting.",
		a.harnessID,
	))
}

// resolveEndpoint returns the WebSocket endpoint of the bridge port: the
// configured endpoint, or the sandbox's endpoint of the port.
func (a adapterIdentity) resolveEndpoint(ctx context.Context, session sandbox.Session, settings bridge.ConnectionSettings, port int) (harness.PortEndpoint, error) {
	if settings.PortEndpoint != nil {
		return *settings.PortEndpoint, nil
	}

	network, ok := session.(harness.NetworkSession)
	if !ok {
		return harness.PortEndpoint{}, a.unsupported(fmt.Sprintf(
			"The %s harness requires an explicit port endpoint when using a basic sandbox session.",
			a.displayName,
		))
	}

	endpoint, err := network.PortEndpoint(ctx, harness.PortEndpointOptions{
		Port:     port,
		Protocol: harness.PortWebSocket,
	})
	if err != nil {
		return harness.PortEndpoint{}, fmt.Errorf("resolve %s bridge endpoint: %w", a.harnessID, err)
	}

	return endpoint, nil
}

// sandboxID returns the ID of a network session, or empty.
func sandboxID(session sandbox.Session) string {
	if network, ok := session.(harness.NetworkSession); ok {
		return network.ID()
	}

	return ""
}
