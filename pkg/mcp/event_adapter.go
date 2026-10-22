package mcp

import (
	"context"
	"encoding/json"
	"time"
)

// ManagedSubscriptionStatus is the state of a managed subscription.
type ManagedSubscriptionStatus string

// Managed subscription statuses.
const (
	ManagedSubscriptionPending   ManagedSubscriptionStatus = "pending"
	ManagedSubscriptionActive    ManagedSubscriptionStatus = "active"
	ManagedSubscriptionNeedsAuth ManagedSubscriptionStatus = "needs_auth"
	ManagedSubscriptionStopped   ManagedSubscriptionStatus = "stopped"
	ManagedSubscriptionExpired   ManagedSubscriptionStatus = "expired"
	ManagedSubscriptionFailed    ManagedSubscriptionStatus = "failed"
)

// ManagedCleanupStatus is the progress of a managed backend's cleanup of an
// upstream subscription.
type ManagedCleanupStatus string

// Managed cleanup statuses.
const (
	ManagedCleanupPending   ManagedCleanupStatus = "pending"
	ManagedCleanupComplete  ManagedCleanupStatus = "complete"
	ManagedCleanupExhausted ManagedCleanupStatus = "exhausted"
)

// ManagedSubscription is a durable watch owned by a managed backend, not an
// MCP lease.
type ManagedSubscription struct {
	// ID is the backend subscription ID used by the managed operations.
	ID        string                     `json:"id"`
	Name      string                     `json:"name"`
	Arguments map[string]json.RawMessage `json:"arguments"`

	Status ManagedSubscriptionStatus `json:"status"`

	// ExpiresAt is the application's monitoring deadline, or nil until the
	// subscription is stopped.
	ExpiresAt *time.Time `json:"expiresAt"`

	// CleanupStatus is empty when the backend does not report it.
	CleanupStatus ManagedCleanupStatus `json:"cleanupStatus,omitempty"`
}

// ManagedSubscribeInput is the input of EventOperations.Subscribe.
type ManagedSubscribeInput struct {
	Name      string
	Arguments map[string]json.RawMessage

	// Context is application metadata, not an authorization grant.
	Context map[string]json.RawMessage

	// IdempotencyKey is reused to recover an uncertain create result.
	IdempotencyKey string

	// TTL is the application lifetime. Zero monitors until the subscription
	// is stopped.
	TTL time.Duration
}

// ListManagedSubscriptionsParams select a page of managed subscriptions.
type ListManagedSubscriptionsParams struct {
	Cursor string

	// Limit is zero for the backend's default.
	Limit int

	// Status filters by status when it is not empty.
	Status ManagedSubscriptionStatus
}

// ListManagedSubscriptionsResult is one page of managed subscriptions.
type ListManagedSubscriptionsResult struct {
	Subscriptions []ManagedSubscription

	// NextCursor is empty on the last page.
	NextCursor string
}

// EventOperations are subscription operations bound by the application to
// an authorized account and destination. The backend owns callback
// provisioning, verification, persistence, renewal and cleanup.
// Implementations must honor ctx and preserve actionable errors. Canceling
// a request does not cancel a remote subscription that was already
// accepted.
type EventOperations interface {
	Subscribe(ctx context.Context, input ManagedSubscribeInput) (*ManagedSubscription, error)
	GetSubscription(ctx context.Context, id string) (*ManagedSubscription, error)
	ListSubscriptions(ctx context.Context, params ListManagedSubscriptionsParams) (*ListManagedSubscriptionsResult, error)
	Unsubscribe(ctx context.Context, id string) (*ManagedSubscription, error)
}

// EventTransportType names the kind of transport passed to an
// EventAdapter.
type EventTransportType string

// Event transport types.
const (
	EventTransportHTTP   EventTransportType = "http"
	EventTransportSSE    EventTransportType = "sse"
	EventTransportCustom EventTransportType = "custom"
)

// EventTransportInfo describes the client's transport, without credentials
// or a live transport. URL is the configured endpoint of HTTP and SSE
// transports, not a resolved redirect target, and empty for custom
// transports, including stdio.
type EventTransportInfo struct {
	Type EventTransportType
	URL  string
}

// EventAdapter creates the managed operations once per client, before the
// transport starts.
type EventAdapter interface {
	CreateAdapter(transport EventTransportInfo) (EventOperations, error)
}

// EventAdapterFunc adapts a function to EventAdapter, for example to bind
// operations that do not depend on the transport.
type EventAdapterFunc func(transport EventTransportInfo) (EventOperations, error)

// CreateAdapter calls f.
func (f EventAdapterFunc) CreateAdapter(transport EventTransportInfo) (EventOperations, error) {
	return f(transport)
}

// ManagedEvents discovers events over MCP and manages subscriptions through
// the adapter's operations.
type ManagedEvents struct {
	EventOperations

	events *Events
}

// ManagedEvents returns the managed event operations, or nil when
// EventsOptions.Adapter is not configured.
func (c *Client) ManagedEvents() *ManagedEvents {
	return c.managedEvents
}

// List requests one page of the server's events.
func (m *ManagedEvents) List(ctx context.Context, params PaginatedParams) (*ListEventsResult, error) {
	return m.events.List(ctx, params)
}

// newEventOperations validates the events options and creates the managed
// operations, if an adapter is configured.
func newEventOperations(options EventsOptions, transport Transport) (EventOperations, error) {
	if options.Adapter == nil {
		return nil, nil
	}

	if options.Store != nil || options.ValidateArguments != nil {
		return nil, &ClientError{
			Message: "configure experimental events with either an adapter or a store, not both; managed adapters own argument validation",
		}
	}

	operations, err := options.Adapter.CreateAdapter(eventTransportInfo(transport))
	if err != nil {
		return nil, err
	}

	if operations == nil {
		return nil, &ClientError{
			Message: "experimental events adapter returned no operations",
		}
	}

	return operations, nil
}

func eventTransportInfo(transport Transport) EventTransportInfo {
	switch typed := transport.(type) {
	case *HTTPTransport:
		return EventTransportInfo{
			Type: EventTransportHTTP,
			URL:  typed.url.String(),
		}

	case *SSETransport:
		return EventTransportInfo{
			Type: EventTransportSSE,
			URL:  typed.url.String(),
		}

	default:
		return EventTransportInfo{
			Type: EventTransportCustom,
		}
	}
}
