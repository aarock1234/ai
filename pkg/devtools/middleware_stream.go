package devtools

import (
	"context"
	"encoding/json"
	"sync"
	"time"

	"github.com/aarock1234/ai/internal/wirejson"
	"github.com/aarock1234/ai/pkg/model"
)

// streamRecording records the parts of one streaming call as they pass to
// the consumer. The step is recorded once: when the stream ends, when the
// consumer stops early or closes it, or when the store aborts active steps.
type streamRecording struct {
	recorder  *middlewareRecorder
	ctx       context.Context
	stepID    string
	startedAt time.Time
	source    *model.StreamResult

	// passRawChunks tells whether the caller requested raw chunks itself.
	passRawChunks bool

	mu               sync.Mutex
	currentText      map[string]string
	currentReasoning map[string]string
	fullStreamChunks []model.StreamPart
	rawChunks        []json.RawMessage
	collectedOutput  streamOutput

	finishedRecording sync.Once
}

// streamOutput collects the output of a streaming call.
type streamOutput struct {
	TextParts      []streamedBlock
	ReasoningParts []streamedBlock
	ToolCalls      []model.StreamPart
	Content        []model.StreamPart
	FinishReason   *model.FinishReason
	Usage          *model.Usage
}

// streamedBlock is the complete text of a text or reasoning block.
type streamedBlock struct {
	ID   string `json:"id"`
	Text string `json:"text"`
}

// streamOutputRecord is the recorded output of a streaming call. Content
// is omitted until a file or tool result arrives.
type streamOutputRecord struct {
	TextParts      []streamedBlock        `json:"textParts"`
	ReasoningParts []streamedBlock        `json:"reasoningParts"`
	ToolCalls      []wirejson.StreamPart  `json:"toolCalls"`
	Content        []wirejson.StreamPart  `json:"content,omitempty"`
	FinishReason   *wirejson.FinishReason `json:"finishReason,omitempty"`
	Usage          *wirejson.Usage        `json:"usage,omitempty"`
}

func newStreamOutputRecord(output streamOutput) streamOutputRecord {
	record := streamOutputRecord{
		TextParts:      nonNil(output.TextParts),
		ReasoningParts: nonNil(output.ReasoningParts),
		ToolCalls:      recordJSON.StreamParts(output.ToolCalls),
		Content:        recordJSON.StreamParts(output.Content),
	}

	if output.FinishReason != nil {
		record.FinishReason = new(wirejson.EncodeFinishReason(*output.FinishReason))
	}

	if output.Usage != nil {
		record.Usage = new(wirejson.EncodeUsage(*output.Usage))
	}

	return record
}

// nonNil returns blocks, or an empty slice that encodes as an array.
func nonNil(blocks []streamedBlock) []streamedBlock {
	if blocks == nil {
		return []streamedBlock{}
	}

	return blocks
}

// result returns the stream that the consumer reads.
func (s *streamRecording) result() *model.StreamResult {
	recorded := model.NewStreamResult(s.parts, s.release)
	recorded.Request = s.source.Request
	recorded.Response = s.source.Response

	return recorded
}

// parts passes the source parts to the consumer and records them. Raw
// parts are recorded, and passed on only when the caller requested them.
func (s *streamRecording) parts(yield func(model.StreamPart) bool) {
	for part := range s.source.Parts() {
		if raw, ok := part.(model.Raw); ok {
			s.recordRaw(raw)

			if s.passRawChunks && !yield(part) {
				s.finish(new(abortedError))

				return
			}

			continue
		}

		s.record(part)

		if !yield(part) {
			s.finish(new(abortedError))

			return
		}
	}

	s.finish(nil)
}

// release closes the source. Closing a stream before it ended records the
// step as aborted.
func (s *streamRecording) release() error {
	err := s.source.Close()
	s.finish(new(abortedError))

	return err
}

func (s *streamRecording) recordRaw(raw model.Raw) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.rawChunks = append(s.rawChunks, raw.Value)
}

// record collects the output of one part.
func (s *streamRecording) record(part model.StreamPart) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.fullStreamChunks = append(s.fullStreamChunks, part)
	output := &s.collectedOutput

	switch typed := part.(type) {
	case model.TextStart:
		s.currentText[typed.ID] = ""
	case model.TextDelta:
		s.currentText[typed.ID] += typed.Delta
	case model.TextEnd:
		output.TextParts = append(output.TextParts, streamedBlock{
			ID:   typed.ID,
			Text: s.currentText[typed.ID],
		})
	case model.ReasoningStart:
		s.currentReasoning[typed.ID] = ""
	case model.ReasoningDelta:
		s.currentReasoning[typed.ID] += typed.Delta
	case model.ReasoningEnd:
		output.ReasoningParts = append(output.ReasoningParts, streamedBlock{
			ID:   typed.ID,
			Text: s.currentReasoning[typed.ID],
		})
	case model.ToolCall:
		output.ToolCalls = append(output.ToolCalls, typed)
	case model.File, model.ReasoningFile, model.ToolResult:
		output.Content = append(output.Content, typed)
	case model.Finish:
		output.FinishReason = &typed.FinishReason
		output.Usage = &typed.Usage
	}
}

// finish records the step once. A nil errorMessage records a completed
// stream.
func (s *streamRecording) finish(errorMessage *string) {
	s.finishedRecording.Do(func() {
		result := s.snapshot()
		result.Error = errorMessage

		s.recorder.recordResult(s.ctx, s.stepID, result)
	})
}

// abortedResult returns the result to record when the store aborts the
// step before the stream ends.
func (s *streamRecording) abortedResult() stepResult {
	result := s.snapshot()
	result.Error = new(abortedError)

	return result
}

// snapshot returns the step result with the output collected so far.
func (s *streamRecording) snapshot() stepResult {
	s.mu.Lock()
	defer s.mu.Unlock()

	output := newStreamOutputRecord(s.collectedOutput)

	result := stepResult{
		Duration:    time.Since(s.startedAt),
		Output:      s.serializeRecorded("step output", output),
		RawRequest:  rawJSON(s.source.Request.Body),
		RawResponse: s.serializeRecorded("stream chunks", recordJSON.StreamParts(s.fullStreamChunks)),
		RawChunks:   s.serializeRecorded("raw chunks", rawChunkValues(s.rawChunks)),
	}

	if output.Usage != nil {
		result.Usage = s.serializeRecorded("step usage", output.Usage)
	}

	return result
}

// serializeRecorded encodes a record, logging a failure and returning nil
// instead.
func (s *streamRecording) serializeRecorded(name string, record any) *string {
	encoded, err := serialize(record)
	if err != nil {
		s.recorder.logFailure(s.ctx, "serializing "+name, err)

		return nil
	}

	return &encoded
}

// rawChunkValues returns the raw chunks as JSON values.
func rawChunkValues(chunks []json.RawMessage) []json.RawMessage {
	values := make([]json.RawMessage, 0, len(chunks))

	for _, chunk := range chunks {
		values = append(values, wirejson.RawValue(chunk))
	}

	return values
}
