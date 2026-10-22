package mcp

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"sync/atomic"
	"time"
)

// protocolDiscoveryTimeout bounds the server/discover probe; servers that do
// not answer in time are initialized with the legacy handshake.
const protocolDiscoveryTimeout = time.Second

// modernProtocolErrorCodes are server/discover errors that come from a
// stateless-protocol server and must not trigger the legacy fallback.
var modernProtocolErrorCodes = []int{
	-32020,
	-32021,
	-32022,
}

// ClientOptions configure NewClient.
type ClientOptions struct {
	// Transport connects to the server. It is required; the client starts and
	// closes it.
	Transport Transport

	// DisableProtocolVersionDiscovery skips the server/discover probe on
	// transports that support it. Set it for legacy servers that require
	// initialize to be the first request.
	DisableProtocolVersionDiscovery bool

	// InitialInitializeResult resumes a previous session: the client starts
	// the transport and reuses this result without a new initialize
	// handshake.
	InitialInitializeResult *InitializeResult

	// ClientName is the name sent to the server. Empty uses
	// DefaultClientName.
	ClientName string

	// Version is the client version sent to the server. Empty uses
	// DefaultClientVersion.
	Version string

	// Capabilities are advertised during initialization. The application
	// handles the corresponding server requests, such as elicitation through
	// Client.HandleElicitation.
	Capabilities ClientCapabilities

	// MaxRetries is the number of retries of tools/call requests that fail
	// transiently, such as with HTTP 429 or a reset connection. JSON-RPC
	// errors from the server are not retried. Zero disables retries.
	MaxRetries int

	// OnUncaughtError receives errors that no caller can receive, such as an
	// unparsable message or a failed response to a server request.
	OnUncaughtError func(ctx context.Context, err error)

	// OnNotification receives notifications from the server, in order, on the
	// goroutine that reads messages; it must return promptly. Without it,
	// notifications are reported to OnUncaughtError as unsupported, as
	// upstream does.
	OnNotification func(ctx context.Context, notification Notification)

	// ExperimentalEvents configures event subscriptions, upstream's
	// experimental_events: a private durable store for direct webhook
	// subscriptions (Client.Events) or a managed adapter
	// (Client.ManagedEvents).
	ExperimentalEvents EventsOptions
}

// protocolEra distinguishes the initialize handshake from the stateless
// protocol negotiated with server/discover.
type protocolEra string

const (
	protocolEraLegacy protocolEra = "legacy"
	protocolEraModern protocolEra = "modern"
)

// Client is a connection to one MCP server. It is safe for concurrent use.
// Create a client per server to combine tools from several servers.
type Client struct {
	transport       Transport
	clientInfo      Implementation
	capabilities    ClientCapabilities
	maxRetries      int
	onUncaughtError func(ctx context.Context, err error)
	onNotification  func(ctx context.Context, notification Notification)
	events          *Events
	managedEvents   *ManagedEvents

	// lifetime carries the creation context's values to goroutines the
	// client starts; it is canceled by Close.
	lifetime       context.Context
	cancelLifetime context.CancelFunc
	workers        sync.WaitGroup

	nextRequestID atomic.Int64

	mu                 sync.Mutex
	stopping           bool
	closed             bool
	pending            map[int64]chan responseOutcome
	serverRequests     map[string]context.CancelFunc
	elicitationHandler ElicitationHandler
	initializeResult   InitializeResult
	protocolEra        protocolEra
	protocolVersion    string
	toolHeaderBindings map[string][]toolHeaderBinding
}

// responseOutcome is the response to a pending request, or the reason none
// will arrive.
type responseOutcome struct {
	message Message
	err     error
}

// NewClient starts options.Transport and negotiates the protocol with the
// server: it probes with server/discover when the transport supports it and
// falls back to the initialize handshake. ctx bounds connection and
// initialization; the client then lives until Close.
func NewClient(ctx context.Context, options ClientOptions) (*Client, error) {
	if options.Transport == nil {
		return nil, &ClientError{
			Message: "a transport is required",
		}
	}

	if options.MaxRetries < 0 {
		return nil, &ClientError{
			Message: "maxRetries must be >= 0",
		}
	}

	eventOperations, err := newEventOperations(options.ExperimentalEvents, options.Transport)
	if err != nil {
		return nil, err
	}

	client := newClient(ctx, options)
	client.attachEvents(options.ExperimentalEvents, eventOperations)

	if err := client.connect(ctx, options); err != nil {
		// The initialization failure takes precedence over a failed cleanup.
		_ = client.shutdown(ctx)

		if ctx.Err() != nil {
			return nil, &ClientError{
				Message: "mcp client initialization was aborted",
				Cause:   context.Cause(ctx),
			}
		}

		return nil, err
	}

	return client, nil
}

func newClient(ctx context.Context, options ClientOptions) *Client {
	lifetime, cancelLifetime := context.WithCancel(context.WithoutCancel(ctx))

	return &Client{
		transport:       options.Transport,
		clientInfo:      clientInfoOf(options),
		capabilities:    options.Capabilities,
		maxRetries:      options.MaxRetries,
		onUncaughtError: options.OnUncaughtError,
		onNotification:  options.OnNotification,
		lifetime:        lifetime,
		cancelLifetime:  cancelLifetime,
		pending:         make(map[int64]chan responseOutcome),
		serverRequests:  make(map[string]context.CancelFunc),
		initializeResult: InitializeResult{
			ProtocolVersion: LatestLegacyProtocolVersion,
		},
		protocolEra:        protocolEraLegacy,
		protocolVersion:    LatestLegacyProtocolVersion,
		toolHeaderBindings: make(map[string][]toolHeaderBinding),
	}
}

// attachEvents creates the direct event operations and, with managed
// operations, the managed events that share their discovery.
func (c *Client) attachEvents(options EventsOptions, operations EventOperations) {
	c.events = &Events{
		client:            c,
		store:             options.Store,
		validateArguments: options.ValidateArguments,
	}

	if operations == nil {
		return
	}

	c.managedEvents = &ManagedEvents{
		EventOperations: operations,
		events:          c.events,
	}
}

func clientInfoOf(options ClientOptions) Implementation {
	info := Implementation{
		Name:    options.ClientName,
		Version: options.Version,
	}

	if info.Name == "" {
		info.Name = DefaultClientName
	}

	if info.Version == "" {
		info.Version = DefaultClientVersion
	}

	return info
}

// ServerInfo returns the server's name and version as reported during
// initialization.
func (c *Client) ServerInfo() Implementation {
	c.mu.Lock()
	defer c.mu.Unlock()

	return c.initializeResult.ServerInfo
}

// InitializeResult returns the initialization result used by the client,
// from the server or from ClientOptions.InitialInitializeResult.
func (c *Client) InitializeResult() InitializeResult {
	c.mu.Lock()
	defer c.mu.Unlock()

	return c.initializeResult
}

// Instructions returns the server's usage instructions from initialization,
// for example to include in a system prompt. It is empty when the server
// sent none.
func (c *Client) Instructions() string {
	c.mu.Lock()
	defer c.mu.Unlock()

	return c.initializeResult.Instructions
}

func (c *Client) connect(ctx context.Context, options ClientOptions) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	if err := c.transport.Start(ctx); err != nil {
		return err
	}

	c.spawn(c.readMessages)

	if options.InitialInitializeResult != nil {
		return c.applyInitializeResult(*options.InitialInitializeResult)
	}

	if !options.DisableProtocolVersionDiscovery && supportsProtocolDiscovery(c.transport) {
		discovered, err := c.discoverProtocol(ctx)
		if err != nil || discovered {
			return err
		}
	}

	return c.initialize(ctx)
}

func supportsProtocolDiscovery(transport Transport) bool {
	discoverer, ok := transport.(ProtocolDiscoveryTransport)

	return ok && discoverer.SupportsProtocolVersionDiscovery()
}

// initialize performs the legacy handshake: initialize followed by
// notifications/initialized.
func (c *Client) initialize(ctx context.Context) error {
	c.setProtocol(protocolEraLegacy, LatestLegacyProtocolVersion)

	var result InitializeResult

	err := c.request(ctx, rpcCall{
		method: "initialize",
		params: initializeParams{
			ProtocolVersion: LatestLegacyProtocolVersion,
			Capabilities:    c.capabilities,
			ClientInfo:      c.clientInfo,
		},
		result: &result,
	})
	if err != nil {
		return err
	}

	if err := c.applyInitializeResult(result); err != nil {
		return err
	}

	return c.notify(ctx, "notifications/initialized", nil)
}

// discoverProtocol probes for a stateless-protocol server. It reports false
// when the server does not answer the probe successfully, so that the
// caller falls back to initialize.
func (c *Client) discoverProtocol(ctx context.Context) (bool, error) {
	c.setProtocol(protocolEraModern, LatestProtocolVersion)

	probeCtx, cancel := context.WithTimeout(ctx, protocolDiscoveryTimeout)
	defer cancel()

	var result discoverResult

	err := c.request(probeCtx, rpcCall{
		method: "server/discover",
		result: &result,
	})
	if err == nil {
		return true, c.applyDiscoverResult(result)
	}

	if ctx.Err() != nil {
		return false, ctx.Err()
	}

	if isModernProtocolError(err) {
		return false, err
	}

	return false, nil
}

func isModernProtocolError(err error) bool {
	clientErr, ok := errors.AsType[*ClientError](err)

	return ok && clientErr.Code != nil && slices.Contains(modernProtocolErrorCodes, *clientErr.Code)
}

func (c *Client) applyDiscoverResult(result discoverResult) error {
	if !slices.Contains(result.SupportedVersions, LatestProtocolVersion) {
		return &ClientError{
			Message: "server does not support the requested protocol version: " + LatestProtocolVersion,
		}
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	serverInfo := c.initializeResult.ServerInfo
	if discovered, ok := discoveredServerInfo(result.Meta); ok {
		serverInfo = discovered
	}

	c.initializeResult = InitializeResult{
		ProtocolVersion: LatestProtocolVersion,
		Capabilities:    result.Capabilities,
		ServerInfo:      serverInfo,
		Instructions:    result.Instructions,
	}

	return nil
}

func (c *Client) applyInitializeResult(result InitializeResult) error {
	if !isSupportedProtocolVersion(result.ProtocolVersion) {
		return &ClientError{
			Message: "server's protocol version is not supported: " + result.ProtocolVersion,
		}
	}

	c.setProtocol(protocolEraLegacy, result.ProtocolVersion)

	c.mu.Lock()
	defer c.mu.Unlock()

	c.initializeResult = result

	return nil
}

// setProtocol records the protocol in use and tells the transport.
func (c *Client) setProtocol(era protocolEra, version string) {
	c.mu.Lock()
	c.protocolEra = era
	c.protocolVersion = version
	c.mu.Unlock()

	if setter, ok := c.transport.(ProtocolVersionTransport); ok {
		setter.SetProtocolVersion(version)
	}
}

func (c *Client) protocol() (protocolEra, string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	return c.protocolEra, c.protocolVersion
}

func (c *Client) serverCapabilities() ServerCapabilities {
	c.mu.Lock()
	defer c.mu.Unlock()

	return c.initializeResult.Capabilities
}

// Close disconnects from the server, fails pending requests and waits for
// the client's goroutines. ctx bounds graceful transport shutdown. Calling
// Close again has no effect.
func (c *Client) Close(ctx context.Context) error {
	c.mu.Lock()
	if c.stopping {
		c.mu.Unlock()

		return nil
	}
	c.mu.Unlock()

	return c.shutdown(ctx)
}

// shutdown stops the client and its transport. It is used by Close and when
// initialization fails.
func (c *Client) shutdown(ctx context.Context) error {
	c.mu.Lock()
	c.stopping = true
	c.mu.Unlock()

	c.cancelLifetime()

	err := c.transport.Close(ctx)

	c.workers.Wait()
	c.connectionClosed()

	if err != nil {
		return fmt.Errorf("close mcp transport: %w", err)
	}

	return nil
}

// spawn runs work on a goroutine that Close waits for, unless the client is
// stopping.
func (c *Client) spawn(work func()) bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.stopping {
		return false
	}

	c.workers.Go(work)

	return true
}

// connectionClosed fails every pending request. The client accepts no
// further requests.
func (c *Client) connectionClosed() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.closed = true

	for id, outcome := range c.pending {
		outcome <- responseOutcome{
			err: &ClientError{
				Message: "connection closed",
				Cause:   ErrConnectionClosed,
			},
		}

		delete(c.pending, id)
	}
}

func (c *Client) reportError(err error) {
	if c.onUncaughtError != nil {
		c.onUncaughtError(c.lifetime, err)
	}
}
