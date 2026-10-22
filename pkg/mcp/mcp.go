// Package mcp is a Model Context Protocol client. It connects to one MCP
// server over a Transport, negotiates the protocol version and capabilities,
// and exposes the server's tools as ai.Tool values together with its
// resources, prompts and completions.
//
// The package ports the upstream @ai-sdk/mcp package. Go adaptations:
//
//   - Cancellation and deadlines come from context.Context. They replace the
//     upstream abort signals and the timeout and maxTotalTimeout request
//     options. Canceling a request sends notifications/cancelled to the
//     server.
//   - The transport configuration union is replaced by constructors:
//     NewHTTPTransport (Streamable HTTP), NewSSETransport (legacy HTTP with
//     Server-Sent Events) and NewStdioTransport (a subprocess). Custom
//     transports implement Transport.
//   - Transports deliver incoming messages through Transport.Receive. The
//     client owns one goroutine that reads them, correlates responses with
//     requests by ID and dispatches server requests; Close waits for it.
//   - The OAuth browser redirect is left to OAuthClientProvider, which the
//     caller implements; Auth completes the flow after the redirect.
//   - Experimental events persist subscriptions in an EventStore that the
//     caller implements, and EventWebhook is an http.Handler. Managed
//     adapters return EventOperations synchronously, as upstream requires.
package mcp

import "slices"

// Protocol versions.
const (
	// LatestProtocolVersion is the stateless protocol version, negotiated with
	// server/discover by transports that support it.
	LatestProtocolVersion = "2026-07-28"

	// LatestLegacyProtocolVersion is the newest protocol version negotiated
	// with initialize.
	LatestLegacyProtocolVersion = "2025-11-25"
)

// SupportedProtocolVersions lists every protocol version the client accepts,
// newest first.
var SupportedProtocolVersions = []string{
	LatestProtocolVersion,
	LatestLegacyProtocolVersion,
	"2025-06-18",
	"2025-03-26",
	"2024-11-05",
}

// Client identity defaults.
const (
	// DefaultClientName is the client name sent when ClientOptions.ClientName
	// is empty.
	DefaultClientName = "ai-sdk-mcp-client"

	// DefaultClientVersion is the client version sent when
	// ClientOptions.Version is empty.
	DefaultClientVersion = "1.0.0"
)

// Version is the version of this package, reported in the User-Agent header
// of HTTP transports.
const Version = "1.0.0"

func isSupportedProtocolVersion(version string) bool {
	return slices.Contains(SupportedProtocolVersions, version)
}
