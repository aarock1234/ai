package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"time"
)

// EventDeliveryMode is a way a server can deliver events.
type EventDeliveryMode string

// Event delivery modes. The client subscribes with webhook delivery only.
const (
	EventDeliveryWebhook EventDeliveryMode = "webhook"
	EventDeliveryPoll    EventDeliveryMode = "poll"
	EventDeliveryPush    EventDeliveryMode = "push"
)

var eventDeliveryModes = []EventDeliveryMode{
	EventDeliveryWebhook,
	EventDeliveryPoll,
	EventDeliveryPush,
}

// EventDefinition is an event as listed by the server. The event API is
// experimental and may change.
type EventDefinition struct {
	Name        string              `json:"name"`
	Description string              `json:"description,omitempty"`
	Delivery    []EventDeliveryMode `json:"delivery"`

	// InputSchema is the JSON Schema of the subscription arguments, an
	// object or a boolean.
	InputSchema json.RawMessage `json:"inputSchema"`

	// PayloadSchema is the JSON Schema of the event data, an object or a
	// boolean.
	PayloadSchema json.RawMessage `json:"payloadSchema"`

	Meta map[string]json.RawMessage `json:"_meta,omitempty"`
}

func (d EventDefinition) validate() error {
	if d.Name == "" {
		return errors.New("event definition must have a name")
	}

	if len(d.Delivery) == 0 {
		return fmt.Errorf("event %q must list at least one delivery mode", d.Name)
	}

	for _, mode := range d.Delivery {
		if !slices.Contains(eventDeliveryModes, mode) {
			return fmt.Errorf("event %q has an unknown delivery mode %q", d.Name, mode)
		}
	}

	if !isJSONSchema(d.InputSchema) || !isJSONSchema(d.PayloadSchema) {
		return fmt.Errorf("event %q must have inputSchema and payloadSchema objects or booleans", d.Name)
	}

	return nil
}

// isJSONSchema reports whether raw is a JSON Schema: an object or a
// boolean.
func isJSONSchema(raw json.RawMessage) bool {
	var value bool
	if json.Unmarshal(raw, &value) == nil {
		return true
	}

	return isJSONObject(raw)
}

// supportsWebhooks reports whether the event can be delivered by webhook.
func (d EventDefinition) supportsWebhooks() bool {
	return slices.Contains(d.Delivery, EventDeliveryWebhook)
}

// ListEventsResult is one page of the server's events.
type ListEventsResult struct {
	Events []EventDefinition `json:"events"`

	// NextCursor requests the following page; it is empty on the last page.
	NextCursor string                     `json:"nextCursor,omitempty"`
	Meta       map[string]json.RawMessage `json:"_meta,omitempty"`
}

func (r *ListEventsResult) validate(raw json.RawMessage) error {
	if err := requireFields(raw, "events"); err != nil {
		return err
	}

	for _, definition := range r.Events {
		if err := definition.validate(); err != nil {
			return err
		}
	}

	return nil
}

// SubscribeEventResult is the server's response to events/subscribe.
type SubscribeEventResult struct {
	// ID is the server-derived subscription ID.
	ID string `json:"id"`

	// RefreshBefore is when the subscription must be refreshed, or nil when
	// it does not expire.
	RefreshBefore *time.Time `json:"refreshBefore"`

	// Cursor is the replay position, or nil.
	Cursor *string `json:"cursor"`

	// Truncated reports that events were lost before the cursor.
	Truncated bool `json:"truncated"`

	Meta map[string]json.RawMessage `json:"_meta,omitempty"`
}

func (r *SubscribeEventResult) validate(raw json.RawMessage) error {
	if err := requireFields(raw, "id", "truncated"); err != nil {
		return err
	}

	if err := requireMembers(raw, "refreshBefore"); err != nil {
		return err
	}

	return nil
}

// requireMembers reports a missing member of the object raw. Unlike
// requireFields, it accepts null values.
func requireMembers(raw json.RawMessage, names ...string) error {
	var members map[string]json.RawMessage
	if err := json.Unmarshal(raw, &members); err != nil {
		return err
	}

	for _, name := range names {
		if _, ok := members[name]; !ok {
			return fmt.Errorf("missing required member %q", name)
		}
	}

	return nil
}

// Event is one event delivered to a webhook.
type Event struct {
	EventID   string                     `json:"eventId"`
	Name      string                     `json:"name"`
	Timestamp time.Time                  `json:"timestamp"`
	Data      map[string]json.RawMessage `json:"data"`

	// Cursor is the replay position after the event, or nil.
	Cursor *string `json:"cursor,omitempty"`

	// Raw is the event as received, including members this package does
	// not interpret.
	Raw json.RawMessage `json:"-"`
}

// EventGap reports that some events are no longer available. The
// subscription continues from Cursor and is marked truncated.
type EventGap struct {
	Cursor string `json:"cursor"`

	// Raw is the control message as received.
	Raw json.RawMessage `json:"-"`
}

// EventTermination reports that the server ended the subscription.
type EventTermination struct {
	Error EventTerminationError `json:"error"`

	// Raw is the control message as received.
	Raw json.RawMessage `json:"-"`
}

// EventTerminationError is the reason a server ended a subscription.
type EventTerminationError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

// EventDelivery is how a subscription delivers events: a webhook URL and the
// secret that signs its deliveries.
type EventDelivery struct {
	Mode   EventDeliveryMode `json:"mode"`
	URL    string            `json:"url"`
	Secret string            `json:"secret,omitempty"`
}

// EventSubscriptionStatus is whether a subscription was confirmed by the
// server.
type EventSubscriptionStatus string

// Event subscription statuses.
const (
	// EventSubscriptionPending is a subscription whose subscribe or refresh
	// response is not persisted yet. Deliveries are retried until it is.
	EventSubscriptionPending EventSubscriptionStatus = "pending"

	EventSubscriptionActive EventSubscriptionStatus = "active"
)

// EventTTL is a requested subscription lifetime.
type EventTTL struct {
	// Duration is the lifetime, sent in whole milliseconds. Zero requests a
	// subscription without expiration.
	Duration time.Duration `json:"duration"`
}

// EventSubscription is a subscription as the application persists it,
// including the signing secret. Store it privately and never expose it to a
// browser.
type EventSubscription struct {
	// Key routes webhook deliveries to the subscription before the server
	// assigns an ID.
	Key string `json:"key"`

	// ID is the server-derived subscription ID, empty while pending the
	// first subscribe response.
	ID string `json:"id,omitempty"`

	Name       string                     `json:"name"`
	Arguments  map[string]json.RawMessage `json:"arguments"`
	Definition EventDefinition            `json:"definition"`
	Delivery   EventDelivery              `json:"delivery"`

	RefreshBefore *time.Time `json:"refreshBefore,omitempty"`
	Cursor        *string    `json:"cursor"`
	Truncated     bool       `json:"truncated"`

	TTL    *EventTTL      `json:"ttl,omitempty"`
	MaxAge *time.Duration `json:"maxAge,omitempty"`

	Status EventSubscriptionStatus `json:"status"`
}

// info returns the subscription without its secret.
func (s EventSubscription) info() EventSubscriptionInfo {
	return EventSubscriptionInfo{
		ID:        s.ID,
		Name:      s.Name,
		Arguments: s.Arguments,
		Delivery: EventDelivery{
			Mode: EventDeliveryWebhook,
			URL:  s.Delivery.URL,
		},
		RefreshBefore: s.RefreshBefore,
	}
}

// EventSubscriptionInfo is the subscription passed to webhook handlers. It
// excludes the signing secret.
type EventSubscriptionInfo struct {
	ID            string
	Name          string
	Arguments     map[string]json.RawMessage
	Delivery      EventDelivery
	RefreshBefore *time.Time
}

// EventSubscriptionPatch lists the fields to change in a stored
// subscription. Nil and false members are left unchanged.
type EventSubscriptionPatch struct {
	ID *string

	// SetRefreshBefore applies RefreshBefore, which may be nil to clear it.
	SetRefreshBefore bool
	RefreshBefore    *time.Time

	// SetCursor applies Cursor, which may be nil to clear it.
	SetCursor bool
	Cursor    *string

	Truncated *bool

	// Status is applied when it is not empty.
	Status EventSubscriptionStatus
}

// Apply merges the patch into subscription.
func (p EventSubscriptionPatch) Apply(subscription *EventSubscription) {
	if p.ID != nil {
		subscription.ID = *p.ID
	}

	if p.SetRefreshBefore {
		subscription.RefreshBefore = p.RefreshBefore
	}

	if p.SetCursor {
		subscription.Cursor = p.Cursor
	}

	if p.Truncated != nil {
		subscription.Truncated = *p.Truncated
	}

	if p.Status != "" {
		subscription.Status = p.Status
	}
}

// EventStore persists subscriptions. Use durable, private storage scoped to
// one MCP server and authenticated principal. Get and GetByID return nil
// without an error for unknown subscriptions. Update must atomically merge
// only the patched fields, for example with EventSubscriptionPatch.Apply.
type EventStore interface {
	Get(ctx context.Context, key string) (*EventSubscription, error)
	GetByID(ctx context.Context, id string) (*EventSubscription, error)
	Set(ctx context.Context, subscription EventSubscription) error
	Update(ctx context.Context, key string, patch EventSubscriptionPatch) error
	Delete(ctx context.Context, key string) error
}

// EventArgumentsValidation is passed to EventsOptions.ValidateArguments.
type EventArgumentsValidation struct {
	Definition EventDefinition
	Arguments  map[string]json.RawMessage
}

// EventsOptions configure experimental events, upstream's
// experimental_events: either a Store for direct webhook subscriptions or a
// managed Adapter, not both.
type EventsOptions struct {
	// Store persists direct subscriptions.
	Store EventStore

	// ValidateArguments optionally validates subscription arguments, for
	// example against Definition.InputSchema. An error rejects the
	// subscription before it is persisted or sent.
	ValidateArguments func(ctx context.Context, validation EventArgumentsValidation) error

	// Adapter delegates subscriptions to a managed backend. Managed adapters
	// own argument validation.
	Adapter EventAdapter
}
