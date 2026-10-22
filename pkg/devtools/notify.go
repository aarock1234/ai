package devtools

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"time"
)

// notifyTimeout bounds a viewer notification, so that recording does not
// stall when nothing listens on the viewer port.
const notifyTimeout = 500 * time.Millisecond

// changeEvent names a change of the database.
type changeEvent string

const (
	eventRun        changeEvent = "run"
	eventStep       changeEvent = "step"
	eventStepUpdate changeEvent = "step-update"
	eventClear      changeEvent = "clear"
)

// notification is the body of POST /api/notify.
type notification struct {
	Event     changeEvent `json:"event"`
	Timestamp int64       `json:"timestamp"`
	DBPath    string      `json:"dbPath"`
}

// notifier tells a running viewer that the database changed.
type notifier struct {
	client       *http.Client
	url          string
	databasePath string
}

func newNotifier(port int, databasePath string) notifier {
	return notifier{
		client: &http.Client{
			Timeout: notifyTimeout,
		},
		url:          "http://localhost:" + strconv.Itoa(port) + "/api/notify",
		databasePath: databasePath,
	}
}

// notify posts the change to the viewer and waits for the response. It
// ignores every failure, because the viewer may not be running. The
// notification is sent even when ctx is cancelled, so that aborted steps
// still reach the viewer.
func (n notifier) notify(ctx context.Context, event changeEvent) {
	body, err := json.Marshal(notification{
		Event:     event,
		Timestamp: time.Now().UnixMilli(),
		DBPath:    n.databasePath,
	})
	if err != nil {
		return
	}

	request, err := http.NewRequestWithContext(context.WithoutCancel(ctx), http.MethodPost, n.url, bytes.NewReader(body))
	if err != nil {
		return
	}

	request.Header.Set("Content-Type", "application/json")

	response, err := n.client.Do(request)
	if err != nil {
		return
	}

	_ = response.Body.Close()
}
