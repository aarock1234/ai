package devtools

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/aarock1234/ai/internal/id"
)

// heartbeatInterval is the interval of heartbeat events, which keep idle
// event streams open.
const heartbeatInterval = 30 * time.Second

// clientEventBuffer is the number of events buffered for a slow client.
// Further events are dropped until it catches up; every event only tells
// the viewer to reload.
const clientEventBuffer = 16

// serverEvent is one server-sent event.
type serverEvent struct {
	name string
	data json.RawMessage
}

// eventBroker sends events to the connected viewers.
type eventBroker struct {
	mu      sync.Mutex
	clients map[chan serverEvent]struct{}
}

func newEventBroker() *eventBroker {
	return &eventBroker{
		clients: map[chan serverEvent]struct{}{},
	}
}

func (b *eventBroker) subscribe() chan serverEvent {
	events := make(chan serverEvent, clientEventBuffer)

	b.mu.Lock()
	defer b.mu.Unlock()

	b.clients[events] = struct{}{}

	return events
}

func (b *eventBroker) unsubscribe(events chan serverEvent) {
	b.mu.Lock()
	defer b.mu.Unlock()

	delete(b.clients, events)
}

// broadcast sends an event to every client without waiting for slow
// clients.
func (b *eventBroker) broadcast(event serverEvent) {
	b.mu.Lock()
	defer b.mu.Unlock()

	for events := range b.clients {
		select {
		case events <- event:
		default:
		}
	}
}

// handleEvents streams server-sent events until the client disconnects or
// the server shuts down.
func (s *Server) handleEvents(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming is not supported", http.StatusInternalServerError)

		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	events := s.events.subscribe()
	defer s.events.unsubscribe(events)

	connected, err := json.Marshal(struct {
		ClientID string `json:"clientId"`
	}{
		ClientID: id.New(""),
	})
	if err != nil {
		return
	}

	if !writeEvent(w, flusher, serverEvent{
		name: "connected",
		data: connected,
	}) {
		return
	}

	heartbeat := time.NewTicker(heartbeatInterval)
	defer heartbeat.Stop()

	for {
		select {
		case <-r.Context().Done():
			return
		case event := <-events:
			if !writeEvent(w, flusher, event) {
				return
			}
		case now := <-heartbeat.C:
			if !writeEvent(w, flusher, heartbeatEvent(now)) {
				return
			}
		}
	}
}

func heartbeatEvent(now time.Time) serverEvent {
	return serverEvent{
		name: "heartbeat",
		data: fmt.Appendf(nil, `{"time":%d}`, now.UnixMilli()),
	}
}

// writeEvent writes one event and reports whether the client is still
// connected.
func writeEvent(w http.ResponseWriter, flusher http.Flusher, event serverEvent) bool {
	if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event.name, event.data); err != nil {
		return false
	}

	flusher.Flush()

	return true
}
