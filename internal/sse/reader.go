// Package sse provides a generic Server-Sent Events reader that parses
// the SSE wire format from an io.Reader into an iterator of events.
package sse

import (
	"bufio"
	"io"
	"iter"
	"strings"
)

// Event is a single SSE event parsed from the wire format.
type Event struct {
	// Type is the event type from the "event:" field. Empty string
	// means the default message type.
	Type string

	// Data is the event payload from the "data:" field. Multiple data
	// lines within the same event are joined with newlines.
	Data string

	// ID is the event ID from the "id:" field.
	ID string
}

// Read parses SSE events from r and yields them lazily. It stops when
// r returns io.EOF or when the caller breaks out of the range loop.
// The [DONE] sentinel (used by OpenAI-compatible APIs) is recognized
// and terminates the iterator without yielding an event.
func Read(r io.Reader) iter.Seq[Event] {
	return func(yield func(Event) bool) {
		scanner := bufio.NewScanner(r)
		var current Event
		hasData := false

		for scanner.Scan() {
			line := scanner.Text()

			// Blank line = event boundary. Dispatch if we accumulated data.
			if line == "" {
				if hasData {
					current.Data = strings.TrimSuffix(current.Data, "\n")
					if !yield(current) {
						return
					}
				}
				current = Event{}
				hasData = false

				continue
			}

			// Skip SSE comments.
			if strings.HasPrefix(line, ":") {
				continue
			}

			field, value, _ := strings.Cut(line, ":")
			value = strings.TrimPrefix(value, " ")

			switch field {
			case "data":
				// The [DONE] sentinel terminates the stream.
				if value == "[DONE]" {
					return
				}
				current.Data += value + "\n"
				hasData = true
			case "event":
				current.Type = value
			case "id":
				current.ID = value
			}
		}

		// Flush any trailing event without a final blank line.
		if hasData {
			current.Data = strings.TrimSuffix(current.Data, "\n")
			yield(current)
		}
	}
}
