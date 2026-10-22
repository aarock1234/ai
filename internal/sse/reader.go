// Package sse reads Server-Sent Events streams.
package sse

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"iter"
	"strings"
)

const (
	maxLineBytes  = 4 << 20
	maxEventBytes = 8 << 20
)

// ErrEventTooLarge reports a line or event above the buffering limits.
var ErrEventTooLarge = errors.New("sse event exceeds size limit")

// Event is one dispatched event.
type Event struct {
	// Type is the "event" field; empty means the default "message" type.
	Type string

	// Data joins the event's "data" lines with newlines.
	Data string

	// ID is the most recent "id" field, which persists across events.
	ID string
}

// Read parses events from r. It accepts LF, CRLF and CR line endings and
// ignores comments and unknown fields. As the specification requires, an
// event without a terminating blank line at EOF is discarded; providers detect
// truncation through their own terminal events. Read does not close r.
func Read(r io.Reader) iter.Seq2[Event, error] {
	return func(yield func(Event, error) bool) {
		scanner := bufio.NewScanner(r)
		scanner.Buffer(make([]byte, 0, 64<<10), maxLineBytes)
		scanner.Split(splitLines)

		var parser parser
		for scanner.Scan() {
			event, ready, err := parser.line(scanner.Text())
			if err != nil {
				yield(Event{}, err)

				return
			}

			if ready && !yield(event, nil) {
				return
			}
		}

		if err := scanner.Err(); err != nil {
			if errors.Is(err, bufio.ErrTooLong) {
				err = ErrEventTooLarge
			}
			yield(Event{}, fmt.Errorf("read sse stream: %w", err))
		}
	}
}

// parser accumulates the fields of one event.
type parser struct {
	eventType string
	data      strings.Builder
	hasData   bool
	lastID    string
	size      int
	started   bool
}

// line consumes one line and reports a completed event when the line ends it.
func (p *parser) line(line string) (Event, bool, error) {
	if !p.started {
		line = strings.TrimPrefix(line, "\uFEFF")
		p.started = true
	}

	if line == "" {
		event, ready := p.dispatch()

		return event, ready, nil
	}

	p.size += len(line) + 1
	if p.size > maxEventBytes {
		return Event{}, false, ErrEventTooLarge
	}

	if strings.HasPrefix(line, ":") {
		return Event{}, false, nil
	}

	field, value, _ := strings.Cut(line, ":")
	value = strings.TrimPrefix(value, " ")

	switch field {
	case "data":
		p.data.WriteString(value)
		p.data.WriteByte('\n')
		p.hasData = true

	case "event":
		p.eventType = value

	case "id":
		if !strings.ContainsRune(value, 0) {
			p.lastID = value
		}
	}

	return Event{}, false, nil
}

// dispatch returns the accumulated event, if it has data, and resets the
// per-event fields.
func (p *parser) dispatch() (Event, bool) {
	event := Event{
		Type: p.eventType,
		Data: strings.TrimSuffix(p.data.String(), "\n"),
		ID:   p.lastID,
	}
	ready := p.hasData

	p.eventType = ""
	p.data.Reset()
	p.hasData = false
	p.size = 0

	return event, ready
}

// splitLines splits on LF, CRLF or a lone CR.
func splitLines(data []byte, atEOF bool) (advance int, token []byte, err error) {
	for index, value := range data {
		if value == '\n' {
			return index + 1, data[:index], nil
		}

		if value != '\r' {
			continue
		}

		next := index + 1
		if next == len(data) && !atEOF {
			return 0, nil, nil
		}

		if next < len(data) && data[next] == '\n' {
			return next + 1, data[:index], nil
		}

		return next, data[:index], nil
	}

	if atEOF && len(data) > 0 {
		return len(data), data, nil
	}

	return 0, nil, nil
}
