package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"time"
)

// maxEventBodyBytes bounds the size of a webhook delivery.
const maxEventBodyBytes = 256 << 10

// EventPayloadValidation is passed to EventWebhookOptions.ValidatePayload.
type EventPayloadValidation struct {
	Definition EventDefinition
	Data       map[string]json.RawMessage
}

// EventNotification is a verified event passed to
// EventWebhookOptions.OnEvent.
type EventNotification struct {
	Subscription EventSubscriptionInfo
	Event        Event
}

// EventGapNotification is a verified replay gap passed to
// EventWebhookOptions.OnGap.
type EventGapNotification struct {
	Subscription EventSubscriptionInfo

	// MessageID is the webhook-id of the delivery.
	MessageID string

	Gap EventGap
}

// EventTerminationNotification is a verified termination passed to
// EventWebhookOptions.OnTerminated.
type EventTerminationNotification struct {
	Subscription EventSubscriptionInfo

	// MessageID is the webhook-id of the delivery.
	MessageID string

	Termination EventTermination
}

// EventWebhookOptions configure NewEventWebhook.
type EventWebhookOptions struct {
	// Store holds the subscriptions created by Events.Subscribe. It is
	// required.
	Store EventStore

	// ValidatePayload optionally validates event data, for example against
	// Definition.PayloadSchema. An error rejects the delivery with 400
	// before OnEvent runs. The envelope and signature are always checked.
	ValidatePayload func(ctx context.Context, validation EventPayloadValidation) error

	// OnEvent receives each verified event. It is required. Return only
	// after durable acceptance or successful processing; an error responds
	// 503 so that the server retries. Deliveries may be duplicated or
	// reordered: deduplicate by subscription ID and event ID. The webhook
	// does not run agents.
	OnEvent func(ctx context.Context, notification EventNotification) error

	// OnGap handles a signed replay gap: some events are no longer
	// available. Reconcile application state, then return after durable
	// acceptance. The webhook then saves the new cursor and the truncated
	// flag. An error responds 503 without changing the stored subscription.
	// Deduplicate by subscription ID and message ID.
	OnGap func(ctx context.Context, notification EventGapNotification) error

	// OnTerminated handles signed termination: the server ended the
	// subscription. The webhook deletes the stored subscription after it
	// returns. An error responds 503 without changing the stored
	// subscription. Deduplicate by subscription ID and message ID.
	OnTerminated func(ctx context.Context, notification EventTerminationNotification) error

	// OnError receives errors of the store and the callbacks.
	OnError func(ctx context.Context, err error)
}

// EventWebhook is an http.Handler for webhook deliveries of events, upstream's
// createMCPEventWebhook. Mount it at the callback URL before subscribing.
// It answers the server's verification challenge with the pending secret
// saved by Events.Subscribe.
type EventWebhook struct {
	options EventWebhookOptions
}

var _ http.Handler = (*EventWebhook)(nil)

// NewEventWebhook returns a webhook handler.
func NewEventWebhook(options EventWebhookOptions) (*EventWebhook, error) {
	if options.Store == nil {
		return nil, &ClientError{
			Message: "an event store is required",
		}
	}

	if options.OnEvent == nil {
		return nil, &ClientError{
			Message: "an event handler is required",
		}
	}

	return &EventWebhook{
		options: options,
	}, nil
}

// webhookStatusError ends a delivery with an HTTP status.
type webhookStatusError struct {
	status int
}

func (e *webhookStatusError) Error() string {
	return fmt.Sprintf("mcp event webhook responded %d", e.status)
}

func respondWith(status int) error {
	return &webhookStatusError{
		status: status,
	}
}

// ServeHTTP verifies and handles one delivery.
func (w *EventWebhook) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodPost {
		response.Header().Set("Allow", http.MethodPost)
		response.WriteHeader(http.StatusMethodNotAllowed)

		return
	}

	if !isJSONContentType(request.Header.Get("Content-Type")) {
		response.WriteHeader(http.StatusUnsupportedMediaType)

		return
	}

	if request.ContentLength > maxEventBodyBytes {
		response.WriteHeader(http.StatusRequestEntityTooLarge)

		return
	}

	challenge, err := w.handle(request)
	if err != nil {
		w.respondError(request.Context(), response, err)

		return
	}

	if challenge != nil {
		writeChallenge(response, *challenge)

		return
	}

	response.WriteHeader(http.StatusNoContent)
}

func isJSONContentType(contentType string) bool {
	mediaType, _, err := mime.ParseMediaType(contentType)

	return err == nil && mediaType == "application/json"
}

// respondError writes the status of a rejected delivery, or 503 after
// reporting an unexpected error so that the server retries.
func (w *EventWebhook) respondError(ctx context.Context, response http.ResponseWriter, err error) {
	if statusErr, ok := errors.AsType[*webhookStatusError](err); ok {
		response.WriteHeader(statusErr.status)

		return
	}

	w.reportError(ctx, err)
	response.WriteHeader(http.StatusServiceUnavailable)
}

func (w *EventWebhook) reportError(ctx context.Context, err error) {
	if w.options.OnError != nil {
		w.options.OnError(ctx, err)
	}
}

// verificationChallenge is the server's callback verification request and
// its answer.
type verificationChallenge struct {
	Challenge string `json:"challenge"`
}

func writeChallenge(response http.ResponseWriter, challenge verificationChallenge) {
	response.Header().Set("Content-Type", "application/json")
	response.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(response).Encode(challenge)
}

// delivery is an authenticated webhook request.
type delivery struct {
	subscription EventSubscription
	messageID    string
	body         []byte
}

// handle authenticates the delivery and dispatches it. It returns the
// answer to a verification challenge, if the delivery is one.
func (w *EventWebhook) handle(request *http.Request) (*verificationChallenge, error) {
	ctx := request.Context()

	verified, err := w.authenticate(request)
	if err != nil {
		return nil, err
	}

	if !hasRawMember(verified.body, "type") {
		return nil, w.handleEvent(ctx, verified)
	}

	var envelope struct {
		Type string `json:"type"`
	}

	if err := json.Unmarshal(verified.body, &envelope); err != nil {
		return nil, respondWith(http.StatusBadRequest)
	}

	if envelope.Type == "verification" {
		return verificationAnswer(verified.body)
	}

	return nil, w.handleControl(ctx, verified)
}

// authenticate finds the subscription the delivery is routed to, reads the
// bounded body and checks its signature.
func (w *EventWebhook) authenticate(request *http.Request) (delivery, error) {
	ctx := request.Context()

	key := request.URL.Query().Get(eventCallbackKey)
	id := request.Header.Get("X-MCP-Subscription-Id")

	if key == "" || id == "" {
		return delivery{}, respondWith(http.StatusUnauthorized)
	}

	subscription, err := w.options.Store.Get(ctx, key)
	if err != nil {
		return delivery{}, err
	}

	if subscription == nil || (subscription.ID != "" && subscription.ID != id) {
		return delivery{}, respondWith(http.StatusUnauthorized)
	}

	body, err := readBoundedBody(request.Body)
	if err != nil {
		return delivery{}, err
	}

	valid, err := verifyEventSignature(request.Header, body, subscription.Delivery.Secret)
	if err != nil {
		return delivery{}, err
	}

	if !valid {
		return delivery{}, respondWith(http.StatusUnauthorized)
	}

	return delivery{
		subscription: *subscription,
		messageID:    request.Header.Get("webhook-id"),
		body:         body,
	}, nil
}

// readBoundedBody reads the body, rejecting bodies above the limit
// whatever their declared length.
func readBoundedBody(body io.Reader) ([]byte, error) {
	if body == nil || body == http.NoBody {
		return nil, respondWith(http.StatusBadRequest)
	}

	content, err := io.ReadAll(io.LimitReader(body, maxEventBodyBytes+1))
	if err != nil {
		return nil, fmt.Errorf("read mcp event delivery: %w", err)
	}

	if len(content) > maxEventBodyBytes {
		return nil, respondWith(http.StatusRequestEntityTooLarge)
	}

	return content, nil
}

func verificationAnswer(body []byte) (*verificationChallenge, error) {
	var challenge verificationChallenge
	if err := json.Unmarshal(body, &challenge); err != nil || challenge.Challenge == "" {
		return nil, respondWith(http.StatusBadRequest)
	}

	return &challenge, nil
}

// checkConfirmed requires a subscription whose subscribe or refresh
// response is persisted. Servers can deliver before it is, so the delivery
// is retried. A verified termination may end a pending renewal, because
// access may have been revoked.
func checkConfirmed(subscription EventSubscription, isTermination bool) error {
	if subscription.ID == "" {
		return respondWith(http.StatusServiceUnavailable)
	}

	if subscription.Status != EventSubscriptionActive && !isTermination {
		return respondWith(http.StatusServiceUnavailable)
	}

	return nil
}

// eventControl is a gap or termination control message.
type eventControl struct {
	Type   string                 `json:"type"`
	Cursor *string                `json:"cursor"`
	Error  *EventTerminationError `json:"error"`
}

func (w *EventWebhook) handleControl(ctx context.Context, verified delivery) error {
	var control eventControl
	if err := json.Unmarshal(verified.body, &control); err != nil {
		return respondWith(http.StatusBadRequest)
	}

	switch {
	case control.Type == "gap" && control.Cursor != nil:
		if err := checkConfirmed(verified.subscription, false); err != nil {
			return err
		}

		return w.handleGap(ctx, verified, EventGap{
			Cursor: *control.Cursor,
			Raw:    verified.body,
		})

	case control.Type == "terminated" && isValidTermination(verified.body, control.Error):
		if err := checkConfirmed(verified.subscription, true); err != nil {
			return err
		}

		return w.handleTermination(ctx, verified, EventTermination{
			Error: *control.Error,
			Raw:   verified.body,
		})

	default:
		return respondWith(http.StatusBadRequest)
	}
}

// isValidTermination requires an error with a code and a message.
func isValidTermination(body []byte, terminationError *EventTerminationError) bool {
	if terminationError == nil {
		return false
	}

	var members struct {
		Error json.RawMessage `json:"error"`
	}

	if err := json.Unmarshal(body, &members); err != nil {
		return false
	}

	return requireFields(members.Error, "code", "message") == nil
}

func (w *EventWebhook) handleGap(ctx context.Context, verified delivery, gap EventGap) error {
	if w.options.OnGap != nil {
		notification := EventGapNotification{
			Subscription: verified.subscription.info(),
			MessageID:    verified.messageID,
			Gap:          gap,
		}

		if err := w.options.OnGap(ctx, notification); err != nil {
			return err
		}
	}

	truncated := true

	patch := EventSubscriptionPatch{
		SetCursor: true,
		Cursor:    &gap.Cursor,
		Truncated: &truncated,
	}

	return w.options.Store.Update(ctx, verified.subscription.Key, patch)
}

func (w *EventWebhook) handleTermination(ctx context.Context, verified delivery, termination EventTermination) error {
	if w.options.OnTerminated != nil {
		notification := EventTerminationNotification{
			Subscription: verified.subscription.info(),
			MessageID:    verified.messageID,
			Termination:  termination,
		}

		if err := w.options.OnTerminated(ctx, notification); err != nil {
			return err
		}
	}

	return w.options.Store.Delete(ctx, verified.subscription.Key)
}

func (w *EventWebhook) handleEvent(ctx context.Context, verified delivery) error {
	subscription := verified.subscription

	if err := checkConfirmed(subscription, false); err != nil {
		return err
	}

	if isExpired(subscription.RefreshBefore) {
		return respondWith(http.StatusGone)
	}

	event, hasCursor, err := parseEvent(verified.body)
	if err != nil || event.Name != subscription.Name || event.EventID != verified.messageID {
		return respondWith(http.StatusBadRequest)
	}

	if err := w.validatePayload(ctx, subscription.Definition, event.Data); err != nil {
		return err
	}

	notification := EventNotification{
		Subscription: subscription.info(),
		Event:        event,
	}

	if err := w.options.OnEvent(ctx, notification); err != nil {
		return err
	}

	if !hasCursor {
		return nil
	}

	patch := EventSubscriptionPatch{
		SetCursor: true,
		Cursor:    event.Cursor,
	}

	return w.options.Store.Update(ctx, subscription.Key, patch)
}

// validatePayload runs the application's payload validation. A rejected
// payload is reported and answered with 400.
func (w *EventWebhook) validatePayload(ctx context.Context, definition EventDefinition, data map[string]json.RawMessage) error {
	if w.options.ValidatePayload == nil {
		return nil
	}

	validation := EventPayloadValidation{
		Definition: definition,
		Data:       data,
	}

	if err := w.options.ValidatePayload(ctx, validation); err != nil {
		w.reportError(ctx, err)

		return respondWith(http.StatusBadRequest)
	}

	return nil
}

func isExpired(refreshBefore *time.Time) bool {
	return refreshBefore != nil && !refreshBefore.After(time.Now())
}

// parseEvent decodes and checks an event envelope. It reports whether the
// event has a cursor member, which may be null.
func parseEvent(body []byte) (Event, bool, error) {
	var event Event
	if err := json.Unmarshal(body, &event); err != nil {
		return Event{}, false, err
	}

	if err := requireFields(body, "eventId", "name", "timestamp", "data"); err != nil {
		return Event{}, false, err
	}

	if event.EventID == "" || event.Name == "" || event.Data == nil {
		return Event{}, false, errors.New("invalid mcp event envelope")
	}

	event.Raw = body

	return event, hasRawMember(body, "cursor"), nil
}

func hasRawMember(raw json.RawMessage, name string) bool {
	return requireMembers(raw, name) == nil
}
