package sse

import (
	"errors"
	"io"
	"strings"
	"testing"
)

// collect drains the iterator, failing the test on any yielded error.
func collect(t *testing.T, input string) []Event {
	t.Helper()

	var events []Event
	for e, err := range Read(strings.NewReader(input)) {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		events = append(events, e)
	}

	return events
}

func TestRead_singleEvent(t *testing.T) {
	events := collect(t, "data: hello world\n\n")

	if len(events) != 1 {
		t.Fatalf("want 1 event, got %d", len(events))
	}
	if events[0].Data != "hello world" {
		t.Errorf("want data %q, got %q", "hello world", events[0].Data)
	}
}

func TestRead_multipleEvents(t *testing.T) {
	events := collect(t, "data: first\n\ndata: second\n\n")

	if len(events) != 2 {
		t.Fatalf("want 2 events, got %d", len(events))
	}
	if events[0].Data != "first" {
		t.Errorf("event 0: want %q, got %q", "first", events[0].Data)
	}
	if events[1].Data != "second" {
		t.Errorf("event 1: want %q, got %q", "second", events[1].Data)
	}
}

func TestRead_multiLineData(t *testing.T) {
	events := collect(t, "data: line one\ndata: line two\ndata: line three\n\n")

	if len(events) != 1 {
		t.Fatalf("want 1 event, got %d", len(events))
	}

	want := "line one\nline two\nline three"
	if events[0].Data != want {
		t.Errorf("want data %q, got %q", want, events[0].Data)
	}
}

func TestRead_eventTypeAndID(t *testing.T) {
	events := collect(t, "event: message\nid: 42\ndata: payload\n\n")

	if len(events) != 1 {
		t.Fatalf("want 1 event, got %d", len(events))
	}
	if events[0].Type != "message" {
		t.Errorf("want type %q, got %q", "message", events[0].Type)
	}
	if events[0].ID != "42" {
		t.Errorf("want id %q, got %q", "42", events[0].ID)
	}
	if events[0].Data != "payload" {
		t.Errorf("want data %q, got %q", "payload", events[0].Data)
	}
}

func TestRead_doneTermination(t *testing.T) {
	events := collect(t, "data: first\n\ndata: [DONE]\n\n")

	if len(events) != 1 {
		t.Fatalf("want 1 event before [DONE], got %d", len(events))
	}
	if events[0].Data != "first" {
		t.Errorf("want data %q, got %q", "first", events[0].Data)
	}
}

func TestRead_comments(t *testing.T) {
	events := collect(t, ": this is a comment\ndata: actual data\n\n")

	if len(events) != 1 {
		t.Fatalf("want 1 event, got %d", len(events))
	}
	if events[0].Data != "actual data" {
		t.Errorf("want data %q, got %q", "actual data", events[0].Data)
	}
}

func TestRead_emptyDataField(t *testing.T) {
	events := collect(t, "data:\n\n")

	if len(events) != 1 {
		t.Fatalf("want 1 event, got %d", len(events))
	}
	if events[0].Data != "" {
		t.Errorf("want empty data, got %q", events[0].Data)
	}
}

func TestRead_trailingEventWithoutBlankLine(t *testing.T) {
	events := collect(t, "data: trailing")

	if len(events) != 1 {
		t.Fatalf("want 1 event (flushed), got %d", len(events))
	}
	if events[0].Data != "trailing" {
		t.Errorf("want data %q, got %q", "trailing", events[0].Data)
	}
}

func TestRead_breakEarly(t *testing.T) {
	input := "data: one\n\ndata: two\n\ndata: three\n\n"
	count := 0
	for _, err := range Read(strings.NewReader(input)) {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		count++
		if count == 1 {
			break
		}
	}

	if count != 1 {
		t.Errorf("want 1 event consumed, got %d", count)
	}
}

func TestRead_jsonData(t *testing.T) {
	events := collect(t, `data: {"choices":[{"delta":{"content":"Hi"}}]}`+"\n\n")

	if len(events) != 1 {
		t.Fatalf("want 1 event, got %d", len(events))
	}

	want := `{"choices":[{"delta":{"content":"Hi"}}]}`
	if events[0].Data != want {
		t.Errorf("want data %q, got %q", want, events[0].Data)
	}
}

func TestRead_crlfLineEndings(t *testing.T) {
	events := collect(t, "event: message\r\ndata: payload\r\n\r\n")

	if len(events) != 1 {
		t.Fatalf("want 1 event, got %d", len(events))
	}
	if events[0].Type != "message" {
		t.Errorf("want type %q, got %q", "message", events[0].Type)
	}
	if events[0].Data != "payload" {
		t.Errorf("want data %q, got %q", "payload", events[0].Data)
	}
}

func TestRead_largeLine(t *testing.T) {
	// A data line well past the 64 KB bufio.Scanner default must
	// survive intact (base64 image chunks routinely exceed it).
	payload := strings.Repeat("x", 256<<10)
	events := collect(t, "data: "+payload+"\n\n")

	if len(events) != 1 {
		t.Fatalf("want 1 event, got %d", len(events))
	}
	if events[0].Data != payload {
		t.Errorf("payload corrupted: want %d bytes, got %d", len(payload), len(events[0].Data))
	}
}

func TestRead_lineExceedingLimit(t *testing.T) {
	payload := strings.Repeat("x", maxLineSize+1)
	var gotErr error
	for _, err := range Read(strings.NewReader("data: " + payload + "\n\n")) {
		if err != nil {
			gotErr = err
		}
	}

	if gotErr == nil {
		t.Fatal("want error for oversized line, got none")
	}
}

// failingReader returns some data, then an error.
type failingReader struct {
	data string
	read bool
	err  error
}

func (r *failingReader) Read(p []byte) (int, error) {
	if !r.read {
		r.read = true

		return copy(p, r.data), nil
	}

	return 0, r.err
}

func TestRead_readerError(t *testing.T) {
	cause := errors.New("connection reset")
	reader := &failingReader{
		data: "data: partial\n\n",
		err:  cause,
	}

	var events []Event
	var gotErr error
	for e, err := range Read(reader) {
		if err != nil {
			gotErr = err

			continue
		}
		events = append(events, e)
	}

	if len(events) != 1 {
		t.Fatalf("want 1 event before failure, got %d", len(events))
	}
	if gotErr == nil {
		t.Fatal("want error from failing reader, got none")
	}
	if !errors.Is(gotErr, cause) {
		t.Errorf("want wrapped cause, got %v", gotErr)
	}
}

func TestRead_eofIsClean(t *testing.T) {
	// io.EOF from the reader is a clean end, not an error.
	for _, err := range Read(io.LimitReader(strings.NewReader("data: x\n\n"), 9)) {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}
}
