package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"

	"github.com/aarock1234/ai/pkg/model"
)

// enrichedPart is a stream part with the partial output parsed from the text
// so far, when it changed.
type enrichedPart struct {
	part       StreamPart
	partial    any
	hasPartial bool
}

// partialOutputEnrichment describes how partial outputs are derived.
type partialOutputEnrichment struct {
	parts  iter.Seq[StreamPart]
	output Output

	// enabled turns parsing on. When it is off, parts pass through
	// unchanged.
	enabled bool
}

// enrichWithPartialOutput parses the first text block of each step as it
// streams. Text deltas are held back until the partial output changes, so
// that each emitted delta carries the partial output it produced.
func enrichWithPartialOutput(enrichment partialOutputEnrichment) iter.Seq[enrichedPart] {
	return func(yield func(enrichedPart) bool) {
		tracker := &partialOutputTracker{
			output: enrichment.output,
		}

		for part := range enrichment.parts {
			if !enrichment.enabled {
				if !yield(plainPart(part)) {
					return
				}

				continue
			}

			for _, enriched := range tracker.process(part) {
				if !yield(enriched) {
					return
				}
			}
		}
	}
}

// partialOutputTracker accumulates the text of one step.
type partialOutputTracker struct {
	output Output

	textID           string
	text             string
	pendingText      string
	pendingMetadata  model.ProviderMetadata
	lastPublishedKey string
	hasPublished     bool
}

func (t *partialOutputTracker) reset() {
	*t = partialOutputTracker{
		output: t.output,
	}
}

// process returns the parts to emit for one input part.
func (t *partialOutputTracker) process(part StreamPart) []enrichedPart {
	switch typed := part.(type) {
	case StartStepPart, streamRetryBoundaryPart:
		t.reset()

	case FinishStepPart:
		if t.pendingText != "" {
			return []enrichedPart{
				t.publishPendingText(nil, false),
				plainPart(part),
			}
		}

	case TextStartPart:
		t.claimTextID(typed.ID)

	case TextEndPart:
		if t.claimTextID(typed.ID) && t.pendingText != "" {
			return []enrichedPart{
				t.publishPendingText(nil, false),
				plainPart(part),
			}
		}

	case TextDeltaPart:
		if t.claimTextID(typed.ID) {
			return t.processTextDelta(typed)
		}
	}

	return []enrichedPart{
		plainPart(part),
	}
}

// claimTextID reports whether a text part belongs to the tracked text
// block, which is the first block of the step.
func (t *partialOutputTracker) claimTextID(textID string) bool {
	if t.textID == "" {
		t.textID = textID
	}

	return t.textID == textID
}

func (t *partialOutputTracker) processTextDelta(delta TextDeltaPart) []enrichedPart {
	t.text += delta.Text
	t.pendingText += delta.Text

	if delta.ProviderMetadata != nil {
		t.pendingMetadata = delta.ProviderMetadata
	}

	if delta.Text == "" && delta.ProviderMetadata != nil {
		return []enrichedPart{
			plainPart(delta),
		}
	}

	partial, ok := t.output.ParsePartialOutput(t.text)
	if !ok {
		return nil
	}

	key := partialOutputKey(partial)
	if t.hasPublished && key == t.lastPublishedKey {
		return nil
	}

	t.lastPublishedKey = key
	t.hasPublished = true

	return []enrichedPart{
		t.publishPendingText(partial, true),
	}
}

// publishPendingText emits the held-back text as one delta.
func (t *partialOutputTracker) publishPendingText(partial any, hasPartial bool) enrichedPart {
	published := enrichedPart{
		part: TextDeltaPart{
			ID:               t.textID,
			Text:             t.pendingText,
			ProviderMetadata: t.pendingMetadata,
		},
		partial:    partial,
		hasPartial: hasPartial,
	}

	t.pendingText = ""

	return published
}

func plainPart(part StreamPart) enrichedPart {
	return enrichedPart{
		part: part,
	}
}

// partialOutputKey identifies a partial output for change detection.
func partialOutputKey(partial any) string {
	if text, ok := partial.(string); ok {
		return text
	}

	encoded, err := json.Marshal(partial)
	if err != nil {
		return fmt.Sprint(partial)
	}

	return string(encoded)
}

// streamRecorder records the content and steps of a stream as the consumer
// receives it, and invokes the stream callbacks.
type streamRecorder struct {
	run *streamTextRun

	steps                   []StepResult
	initialResponseMessages []model.Message

	// stepModel is the model of the current step, set by the producer.
	stepModel StepModel

	content         []ContentPart
	activeText      map[string]int
	activeReasoning map[string]int
	request         StepRequest
	warnings        []model.Warning

	hasFinish       bool
	finishReason    model.UnifiedFinishReason
	rawFinishReason string
	totalUsage      LanguageModelUsage

	// lastErr is the latest error part; noOutputErr is set when a step
	// produced nothing.
	lastErr     error
	noOutputErr error

	// stepErr reports a step that could not be recorded.
	stepErr error

	aborted bool
}

func newStreamRecorder(run *streamTextRun) *streamRecorder {
	return &streamRecorder{
		run:             run,
		activeText:      make(map[string]int),
		activeReasoning: make(map[string]int),
	}
}

// record processes one part and returns error parts for parts that are
// invalid in their position.
func (r *streamRecorder) record(ctx context.Context, part StreamPart) []StreamPart {
	errorPart, isError := part.(ErrorPart)
	callbacksHandled := isError && errorPart.callbacksHandled

	if r.run.options.OnChunk != nil && !callbacksHandled {
		r.run.options.OnChunk(ctx, StreamTextChunkEvent{
			Chunk: part,
		})
	}

	switch typed := part.(type) {
	case ErrorPart:
		r.recordError(ctx, typed)

	case CustomContent:
		r.addContent(typed)

	case SourceContent:
		r.addContent(typed)

	case FileContent:
		r.addContent(typed)

	case ReasoningFileContent:
		r.addContent(typed)

	case ToolCall:
		r.addContent(typed)

	case ToolError:
		r.addContent(typed)

	case ToolApprovalRequest:
		r.addContent(typed)

	case ToolApprovalResponse:
		r.addContent(typed)

	case ToolResult:
		if !typed.Preliminary {
			r.addContent(typed)
		}

	case TextStartPart:
		r.activeText[typed.ID] = len(r.content)
		r.addContent(TextContent{
			ProviderMetadata: typed.ProviderMetadata,
		})

	case TextDeltaPart:
		return r.updateText(typed.ID, func(text *TextContent) {
			text.Text += typed.Text
			text.ProviderMetadata = metadataOrDefault(typed.ProviderMetadata, text.ProviderMetadata)
		})

	case TextEndPart:
		invalid := r.updateText(typed.ID, func(text *TextContent) {
			text.ProviderMetadata = metadataOrDefault(typed.ProviderMetadata, text.ProviderMetadata)

			if typed.Citations != nil {
				text.Citations = typed.Citations
			}
		})
		delete(r.activeText, typed.ID)

		return invalid

	case ReasoningStartPart:
		r.activeReasoning[typed.ID] = len(r.content)
		r.addContent(ReasoningContent{
			ProviderMetadata: typed.ProviderMetadata,
		})

	case ReasoningDeltaPart:
		return r.updateReasoning(typed.ID, func(reasoning *ReasoningContent) {
			reasoning.Text += typed.Text
			reasoning.ProviderMetadata = metadataOrDefault(typed.ProviderMetadata, reasoning.ProviderMetadata)
		})

	case ReasoningEndPart:
		invalid := r.updateReasoning(typed.ID, func(reasoning *ReasoningContent) {
			reasoning.ProviderMetadata = metadataOrDefault(typed.ProviderMetadata, reasoning.ProviderMetadata)
		})
		delete(r.activeReasoning, typed.ID)

		return invalid

	case StartStepPart:
		r.restartContent(typed.Request, typed.Warnings)

	case FinishStepPart:
		return r.recordStep(ctx, typed)

	case FinishPart:
		r.hasFinish = true
		r.finishReason = typed.FinishReason
		r.rawFinishReason = typed.RawFinishReason
		r.totalUsage = typed.TotalUsage
	}

	return nil
}

// addContent records a content part unless the stream keeps no content.
func (r *streamRecorder) addContent(part ContentPart) {
	if r.run.retention.collectContent {
		r.content = append(r.content, part)
	}
}

// restartContent starts recording a new step or a retried attempt.
func (r *streamRecorder) restartContent(request StepRequest, warnings []model.Warning) {
	r.content = nil
	r.activeText = make(map[string]int)
	r.activeReasoning = make(map[string]int)
	r.request = request
	r.warnings = warnings
}

// resetAttempt drops the content of a failed model call attempt.
func (r *streamRecorder) resetAttempt(boundary streamRetryBoundaryPart) {
	r.restartContent(boundary.request, boundary.warnings)
}

// recordError records an error part and calls OnError, unless the stream
// retry decision already did.
func (r *streamRecorder) recordError(ctx context.Context, part ErrorPart) {
	r.lastErr = part.Err

	var noOutput *NoOutputGeneratedError
	if errors.As(part.Err, &noOutput) {
		r.noOutputErr = part.Err
	}

	if r.run.options.OnError != nil && !part.callbacksHandled {
		r.run.options.OnError(ctx, StreamTextErrorEvent{
			Error: part.Err,
		})
	}
}

func (r *streamRecorder) updateText(partID string, update func(text *TextContent)) []StreamPart {
	index, active := r.activeText[partID]
	if active && !r.run.retention.collectContent {
		return nil
	}

	text, isText := contentAt[TextContent](r.content, index)

	if !active || !isText {
		return []StreamPart{
			ErrorPart{
				Err: &InvalidStreamPartError{
					Message: fmt.Sprintf("text part %s not found", partID),
				},
			},
		}
	}

	update(&text)
	r.content[index] = text

	return nil
}

func (r *streamRecorder) updateReasoning(partID string, update func(reasoning *ReasoningContent)) []StreamPart {
	index, active := r.activeReasoning[partID]
	if active && !r.run.retention.collectContent {
		return nil
	}

	reasoning, isReasoning := contentAt[ReasoningContent](r.content, index)

	if !active || !isReasoning {
		return []StreamPart{
			ErrorPart{
				Err: &InvalidStreamPartError{
					Message: fmt.Sprintf("reasoning part %s not found", partID),
				},
			},
		}
	}

	update(&reasoning)
	r.content[index] = reasoning

	return nil
}

// recordStep completes the current step and calls OnStepEnd.
func (r *streamRecorder) recordStep(ctx context.Context, finish FinishStepPart) []StreamPart {
	messages, err := toResponseMessages(ctx, r.content, r.run.tools)
	if err != nil {
		r.lastErr = err
		r.stepErr = err

		return []StreamPart{
			ErrorPart{
				Err: err,
			},
		}
	}

	step := StepResult{
		CallID:           r.run.callID,
		StepNumber:       len(r.steps),
		Model:            r.stepModel,
		Content:          r.content,
		FinishReason:     finish.FinishReason,
		RawFinishReason:  finish.RawFinishReason,
		Usage:            finish.Usage,
		Performance:      finish.Performance,
		Warnings:         r.warnings,
		Request:          r.request,
		ProviderMetadata: finish.ProviderMetadata,
		Response: StepResponse{
			ResponseInfo: finish.Response,
			Messages:     messages,
		},
	}

	r.run.notifyStepEnd(ctx, step, r.run.options.OnStepEnd)

	r.steps = append(r.steps, step)

	return nil
}

// finish computes the outcome of a stream that ran to its end and calls
// OnEnd unless the stream was aborted.
func (r *streamRecorder) finish(ctx context.Context) streamOutcome {
	if len(r.steps) == 0 || r.noOutputErr != nil {
		return streamOutcome{
			err: r.failure(ctx),
		}
	}

	finalStep := r.steps[len(r.steps)-1]

	output, outputErr := r.run.output.ParseCompleteOutput(ctx, finalStep.Text(), OutputContext{
		Response:     finalStep.Response.ResponseInfo,
		Usage:        finalStep.Usage,
		FinishReason: finalStep.FinishReason,
	})
	if outputErr != nil {
		output = nil
	}

	result := newGenerateTextResult(generateTextResultInput{
		steps:                   r.steps,
		initialResponseMessages: r.initialResponseMessages,
		totalUsage:              r.totalUsage,
		output:                  output,
	})

	result.FinishReason = model.FinishOther
	if r.hasFinish {
		result.FinishReason = r.finishReason
		result.RawFinishReason = r.rawFinishReason
	}

	outcome := streamOutcome{
		result: result,
		summary: StreamTextSummary{
			TotalUsage:      result.TotalUsage,
			FinishReason:    result.FinishReason,
			RawFinishReason: result.RawFinishReason,
		},
		output:    output,
		outputErr: outputErr,
	}

	if r.aborted {
		outcome.err = context.Cause(ctx)
	} else {
		r.run.notifyEnd(ctx, GenerateTextEndEvent{
			CallID:             r.run.callID,
			GenerateTextResult: *result,
		}, r.run.options.OnEnd)
	}

	if !r.run.retention.replay {
		outcome.result = nil
		r.steps = nil
	}

	return outcome
}

// failure explains why a stream produced no result.
func (r *streamRecorder) failure(ctx context.Context) error {
	if r.aborted {
		return context.Cause(ctx)
	}

	if r.noOutputErr != nil {
		return r.noOutputErr
	}

	return &NoOutputGeneratedError{
		Cause: r.lastErr,
	}
}

// contentAt returns the content part at index as a Part.
func contentAt[Part ContentPart](content []ContentPart, index int) (Part, bool) {
	var zero Part
	if index < 0 || index >= len(content) {
		return zero, false
	}

	typed, ok := content[index].(Part)

	return typed, ok
}

func metadataOrDefault(metadata, fallback model.ProviderMetadata) model.ProviderMetadata {
	if metadata != nil {
		return metadata
	}

	return fallback
}
