package sse

import (
	"strings"
	"testing"
)

func TestRead_singleEvent(t *testing.T) {
	input := "data: hello world\n\n"
	var events []Event
	for e := range Read(strings.NewReader(input)) {
		events = append(events, e)
	}

	if len(events) != 1 {
		t.Fatalf("want 1 event, got %d", len(events))
	}
	if events[0].Data != "hello world" {
		t.Errorf("want data %q, got %q", "hello world", events[0].Data)
	}
}

func TestRead_multipleEvents(t *testing.T) {
	input := "data: first\n\ndata: second\n\n"
	var events []Event
	for e := range Read(strings.NewReader(input)) {
		events = append(events, e)
	}

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
	input := "data: line one\ndata: line two\ndata: line three\n\n"
	var events []Event
	for e := range Read(strings.NewReader(input)) {
		events = append(events, e)
	}

	if len(events) != 1 {
		t.Fatalf("want 1 event, got %d", len(events))
	}

	want := "line one\nline two\nline three"
	if events[0].Data != want {
		t.Errorf("want data %q, got %q", want, events[0].Data)
	}
}

func TestRead_eventTypeAndID(t *testing.T) {
	input := "event: message\nid: 42\ndata: payload\n\n"
	var events []Event
	for e := range Read(strings.NewReader(input)) {
		events = append(events, e)
	}

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
	input := "data: first\n\ndata: [DONE]\n\n"
	var events []Event
	for e := range Read(strings.NewReader(input)) {
		events = append(events, e)
	}

	if len(events) != 1 {
		t.Fatalf("want 1 event before [DONE], got %d", len(events))
	}
	if events[0].Data != "first" {
		t.Errorf("want data %q, got %q", "first", events[0].Data)
	}
}

func TestRead_comments(t *testing.T) {
	input := ": this is a comment\ndata: actual data\n\n"
	var events []Event
	for e := range Read(strings.NewReader(input)) {
		events = append(events, e)
	}

	if len(events) != 1 {
		t.Fatalf("want 1 event, got %d", len(events))
	}
	if events[0].Data != "actual data" {
		t.Errorf("want data %q, got %q", "actual data", events[0].Data)
	}
}

func TestRead_emptyDataField(t *testing.T) {
	input := "data:\n\n"
	var events []Event
	for e := range Read(strings.NewReader(input)) {
		events = append(events, e)
	}

	if len(events) != 1 {
		t.Fatalf("want 1 event, got %d", len(events))
	}
	if events[0].Data != "" {
		t.Errorf("want empty data, got %q", events[0].Data)
	}
}

func TestRead_trailingEventWithoutBlankLine(t *testing.T) {
	input := "data: trailing"
	var events []Event
	for e := range Read(strings.NewReader(input)) {
		events = append(events, e)
	}

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
	for range Read(strings.NewReader(input)) {
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
	input := `data: {"choices":[{"delta":{"content":"Hi"}}]}` + "\n\n"
	var events []Event
	for e := range Read(strings.NewReader(input)) {
		events = append(events, e)
	}

	if len(events) != 1 {
		t.Fatalf("want 1 event, got %d", len(events))
	}

	want := `{"choices":[{"delta":{"content":"Hi"}}]}`
	if events[0].Data != want {
		t.Errorf("want data %q, got %q", want, events[0].Data)
	}
}
