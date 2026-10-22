package mcp

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"net/url"
	"strconv"
	"time"
)

// eventCallbackKey is the query parameter of a webhook URL that routes
// deliveries to the stored subscription.
const eventCallbackKey = "mcp_event_subscription"

// Events manages direct webhook subscriptions to the server's events,
// persisted in EventsOptions.Store. The event API is experimental and may
// change.
type Events struct {
	client            *Client
	store             EventStore
	validateArguments func(ctx context.Context, validation EventArgumentsValidation) error
}

// WebhookDelivery configures the webhook of a subscription.
type WebhookDelivery struct {
	// URL is the HTTPS callback served by an EventWebhook, mounted before
	// subscribing.
	URL string

	// AllowInsecureLocalhost allows HTTP callbacks to the literal loopback
	// addresses 127.0.0.1 and [::1]. Use it in development only.
	AllowInsecureLocalhost bool

	// Secret signs deliveries. Empty generates one with NewEventSecret. It
	// is persisted before the subscribe request.
	Secret string
}

// SubscribeEventOptions configure Events.Subscribe.
type SubscribeEventOptions struct {
	Name string

	// Arguments are the subscription arguments. Nil sends an empty object.
	Arguments map[string]json.RawMessage

	Delivery WebhookDelivery

	// Cursor resumes from a replay position. Nil starts from the present.
	Cursor *string

	// TTL requests a lifetime. Nil leaves it to the server.
	TTL *EventTTL

	// MaxAge bounds the age of replayed events. Nil leaves it to the server.
	MaxAge *time.Duration
}

// EventSubscriptionRef identifies a stored subscription by its server ID or
// its routing key. Set exactly one.
type EventSubscriptionRef struct {
	ID  string
	Key string
}

// subscribeEventParams are the parameters of events/subscribe.
type subscribeEventParams struct {
	Name      string                     `json:"name"`
	Arguments map[string]json.RawMessage `json:"arguments"`
	Delivery  EventDelivery              `json:"delivery"`
	Cursor    *string                    `json:"cursor"`
	TTLMs     json.RawMessage            `json:"ttlMs,omitempty"`
	MaxAgeMs  *int64                     `json:"maxAgeMs,omitempty"`
}

// unsubscribeEventParams are the parameters of events/unsubscribe.
type unsubscribeEventParams struct {
	Name      string                     `json:"name"`
	Arguments map[string]json.RawMessage `json:"arguments"`
	Delivery  EventDelivery              `json:"delivery"`
}

// emptyResult is a result without members the client reads.
type emptyResult struct {
	Meta map[string]json.RawMessage `json:"_meta,omitempty"`
}

// Events returns the direct event operations. Without EventsOptions.Store,
// List works and the subscription methods fail.
func (c *Client) Events() *Events {
	return c.events
}

// List requests one page of the server's events.
func (e *Events) List(ctx context.Context, params PaginatedParams) (*ListEventsResult, error) {
	var result ListEventsResult

	err := e.client.request(ctx, rpcCall{
		method: "events/list",
		params: params,
		result: &result,
	})
	if err != nil {
		return nil, err
	}

	return &result, nil
}

// Subscribe validates the callback, finds the event, persists a pending
// subscription with its signing secret and subscribes. On failure the
// pending subscription is kept, because the server may have registered the
// callback; refresh or unsubscribe it with its key.
func (e *Events) Subscribe(ctx context.Context, options SubscribeEventOptions) (*SubscribeEventResult, error) {
	store, err := e.requireStore()
	if err != nil {
		return nil, err
	}

	callback, err := validateCallbackURL(options.Delivery)
	if err != nil {
		return nil, err
	}

	if err := validateSubscriptionLifetime(options.TTL, options.MaxAge); err != nil {
		return nil, err
	}

	definition, err := e.findDefinition(ctx, options.Name)
	if err != nil {
		return nil, err
	}

	arguments := cloneArguments(options.Arguments)

	if e.validateArguments != nil {
		validation := EventArgumentsValidation{
			Definition: definition,
			Arguments:  arguments,
		}

		if err := e.validateArguments(ctx, validation); err != nil {
			return nil, err
		}
	}

	subscription, err := newPendingSubscription(pendingSubscription{
		options:    options,
		arguments:  arguments,
		definition: definition,
		callback:   callback,
	})
	if err != nil {
		return nil, err
	}

	if err := store.Set(ctx, subscription); err != nil {
		return nil, err
	}

	return e.subscribe(ctx, subscription)
}

// pendingSubscription holds the validated inputs of a new subscription.
type pendingSubscription struct {
	options    SubscribeEventOptions
	arguments  map[string]json.RawMessage
	definition EventDefinition
	callback   *url.URL
}

// newPendingSubscription adds a routing key to the callback URL and
// prepares the subscription to persist before the server is asked to
// verify the callback.
func newPendingSubscription(pending pendingSubscription) (EventSubscription, error) {
	secret := pending.options.Delivery.Secret
	if secret == "" {
		secret = NewEventSecret()
	}

	if _, err := decodeEventSecret(secret); err != nil {
		return EventSubscription{}, err
	}

	key := rand.Text()

	query := pending.callback.Query()
	query.Set(eventCallbackKey, key)
	pending.callback.RawQuery = query.Encode()

	return EventSubscription{
		Key:        key,
		Name:       pending.options.Name,
		Arguments:  pending.arguments,
		Definition: pending.definition,
		Delivery: EventDelivery{
			Mode:   EventDeliveryWebhook,
			URL:    pending.callback.String(),
			Secret: secret,
		},
		Cursor: pending.options.Cursor,
		TTL:    pending.options.TTL,
		MaxAge: pending.options.MaxAge,
		Status: EventSubscriptionPending,
	}, nil
}

// Refresh renews a stored subscription. The subscription is marked pending
// first, so that deliveries are retried until the renewal is persisted; it
// stays pending on failure, because the server may have renewed it.
func (e *Events) Refresh(ctx context.Context, ref EventSubscriptionRef) (*SubscribeEventResult, error) {
	subscription, err := e.storedSubscription(ctx, ref)
	if err != nil {
		return nil, err
	}

	patch := EventSubscriptionPatch{
		Status: EventSubscriptionPending,
	}

	if err := e.store.Update(ctx, subscription.Key, patch); err != nil {
		return nil, err
	}

	return e.subscribe(ctx, *subscription)
}

// Unsubscribe ends a stored subscription on the server and deletes it.
func (e *Events) Unsubscribe(ctx context.Context, ref EventSubscriptionRef) error {
	subscription, err := e.storedSubscription(ctx, ref)
	if err != nil {
		return err
	}

	var result emptyResult

	err = e.client.request(ctx, rpcCall{
		method: "events/unsubscribe",
		params: unsubscribeEventParams{
			Name:      subscription.Name,
			Arguments: subscription.Arguments,
			Delivery: EventDelivery{
				Mode: EventDeliveryWebhook,
				URL:  subscription.Delivery.URL,
			},
		},
		result: &result,
	})
	if err != nil {
		return err
	}

	return e.store.Delete(ctx, subscription.Key)
}

// subscribe sends events/subscribe for a stored subscription and persists
// the server's response as active.
func (e *Events) subscribe(ctx context.Context, subscription EventSubscription) (*SubscribeEventResult, error) {
	var result SubscribeEventResult

	err := e.client.request(ctx, rpcCall{
		method: "events/subscribe",
		params: subscribeParams(subscription),
		result: &result,
	})
	if err != nil {
		return nil, err
	}

	patch := EventSubscriptionPatch{
		ID:               &result.ID,
		SetRefreshBefore: true,
		RefreshBefore:    result.RefreshBefore,
		SetCursor:        true,
		Cursor:           result.Cursor,
		Truncated:        &result.Truncated,
		Status:           EventSubscriptionActive,
	}

	if err := e.store.Update(ctx, subscription.Key, patch); err != nil {
		return nil, err
	}

	return &result, nil
}

func subscribeParams(subscription EventSubscription) subscribeEventParams {
	params := subscribeEventParams{
		Name:      subscription.Name,
		Arguments: subscription.Arguments,
		Delivery:  subscription.Delivery,
		Cursor:    subscription.Cursor,
	}

	if subscription.TTL != nil {
		params.TTLMs = ttlMilliseconds(*subscription.TTL)
	}

	if subscription.MaxAge != nil {
		maxAgeMs := subscription.MaxAge.Milliseconds()
		params.MaxAgeMs = &maxAgeMs
	}

	return params
}

// ttlMilliseconds encodes a lifetime in milliseconds, or null for a
// subscription without expiration.
func ttlMilliseconds(ttl EventTTL) json.RawMessage {
	if ttl.Duration == 0 {
		return json.RawMessage("null")
	}

	return json.RawMessage(strconv.FormatInt(ttl.Duration.Milliseconds(), 10))
}

func (e *Events) requireStore() (EventStore, error) {
	if e.store == nil {
		return nil, &ClientError{
			Message: "configure experimental events store to manage direct event subscriptions",
		}
	}

	return e.store, nil
}

func (e *Events) storedSubscription(ctx context.Context, ref EventSubscriptionRef) (*EventSubscription, error) {
	store, err := e.requireStore()
	if err != nil {
		return nil, err
	}

	if (ref.ID == "") == (ref.Key == "") {
		return nil, &ClientError{
			Message: "identify the mcp event subscription by exactly one of id or key",
		}
	}

	var subscription *EventSubscription
	if ref.ID != "" {
		subscription, err = store.GetByID(ctx, ref.ID)
	} else {
		subscription, err = store.Get(ctx, ref.Key)
	}

	if err != nil {
		return nil, err
	}

	if subscription == nil {
		return nil, &ClientError{
			Message: "unknown mcp event subscription: " + ref.ID + ref.Key,
		}
	}

	return subscription, nil
}

// findDefinition pages through the server's events until it finds name,
// which must support webhook delivery.
func (e *Events) findDefinition(ctx context.Context, name string) (EventDefinition, error) {
	visited := make(map[string]bool)
	cursor := ""

	for {
		page, err := e.List(ctx, PaginatedParams{
			Cursor: cursor,
		})
		if err != nil {
			return EventDefinition{}, err
		}

		for _, definition := range page.Events {
			if definition.Name == name {
				return webhookDefinition(definition)
			}
		}

		cursor = page.NextCursor
		if cursor == "" {
			return EventDefinition{}, &ClientError{
				Message: "unknown mcp event: " + name,
			}
		}

		if visited[cursor] {
			return EventDefinition{}, &ClientError{
				Message: "repeated mcp events pagination cursor",
			}
		}

		visited[cursor] = true
	}
}

func webhookDefinition(definition EventDefinition) (EventDefinition, error) {
	if !definition.supportsWebhooks() {
		return EventDefinition{}, &ClientError{
			Message: "mcp event " + definition.Name + " does not support webhook delivery",
		}
	}

	return definition, nil
}

// validateCallbackURL requires an HTTPS callback without credentials or a
// fragment, or HTTP to a literal loopback address when allowed.
func validateCallbackURL(delivery WebhookDelivery) (*url.URL, error) {
	invalid := &ClientError{
		Message: "mcp event callback urls must use https without credentials or fragments",
	}

	callback, err := url.Parse(delivery.URL)
	if err != nil || !callback.IsAbs() || callback.Host == "" {
		return nil, invalid
	}

	if callback.User != nil || callback.Fragment != "" || callback.RawFragment != "" {
		return nil, invalid
	}

	if callback.Scheme == "https" || isLocalHTTPCallback(callback, delivery.AllowInsecureLocalhost) {
		return callback, nil
	}

	return nil, invalid
}

func isLocalHTTPCallback(callback *url.URL, allowed bool) bool {
	if !allowed || callback.Scheme != "http" {
		return false
	}

	hostname := callback.Hostname()

	return hostname == "127.0.0.1" || hostname == "::1"
}

func validateSubscriptionLifetime(ttl *EventTTL, maxAge *time.Duration) error {
	if isValidTTL(ttl) && isValidMaxAge(maxAge) {
		return nil
	}

	return &ClientError{
		Message: "invalid mcp event subscription lifetime or replay age",
	}
}

// isValidTTL accepts no lifetime, zero for no expiration, and lifetimes of
// at least one millisecond.
func isValidTTL(ttl *EventTTL) bool {
	if ttl == nil || ttl.Duration == 0 {
		return true
	}

	return ttl.Duration >= time.Millisecond
}

func isValidMaxAge(maxAge *time.Duration) bool {
	return maxAge == nil || *maxAge >= 0
}

// cloneArguments copies the arguments so that the stored subscription does
// not share them with the caller. Nil becomes an empty object.
func cloneArguments(arguments map[string]json.RawMessage) map[string]json.RawMessage {
	cloned := make(map[string]json.RawMessage, len(arguments))

	for name, value := range arguments {
		cloned[name] = bytes.Clone(value)
	}

	return cloned
}
