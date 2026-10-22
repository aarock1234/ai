package harness

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/aarock1234/ai/pkg/ai"
	"github.com/aarock1234/ai/pkg/model"
)

// errUnclosedStep reports a terminal finish with step content that no step
// boundary captured, which adapters must not produce.
var errUnclosedStep = errors.New("harness agent: received terminal finish with unclosed step content; harness adapters must emit a finish-step part before finish")

// turnResult accumulates the parts of one turn into steps and the final
// result. It is the writer side of the upstream HarnessStreamTextResult: the
// turn driver pushes translated parts, step boundaries and the final outcome;
// readers see them through StreamResult. It is safe for concurrent use.
type turnResult struct {
	mu sync.Mutex

	log  *partLog
	done chan struct{}

	tools    []ai.Tool
	output   ai.Output
	provider string
	callID   string
	modelID  string

	// modelOutputs are the model-facing tool outputs submitted to the
	// runtime, keyed by tool call ID.
	modelOutputs map[string]model.ToolResultOutput

	partial partialOutputTracker

	steps        []ai.StepResult
	content      []ai.ContentPart
	openText     map[string]int
	openThoughts map[string]int
	stepWarnings []model.Warning
	stepNumber   int
	stepStarted  bool

	totalUsage           ai.LanguageModelUsage
	finalFinishReason    model.UnifiedFinishReason
	finalRawFinishReason string
	finalMetadata        model.ProviderMetadata

	settled bool
	outcome turnOutcome
}

// turnOutcome is the final state of a turn's result.
type turnOutcome struct {
	result *ai.GenerateTextResult
	err    error

	// outputErr is the failure to parse the configured output.
	outputErr error
}

// turnResultSetup configures a turn result.
type turnResultSetup struct {
	tools     []ai.Tool
	output    ai.Output
	harnessID string
	callID    string
	modelID   string
}

// newTurnResult returns a result whose stream starts with an ai.StartPart,
// as StreamText's does; UI message streams rely on it.
func newTurnResult(setup turnResultSetup) *turnResult {
	result := &turnResult{
		log:               newPartLog(),
		done:              make(chan struct{}),
		tools:             setup.tools,
		output:            setup.output,
		provider:          "harness:" + setup.harnessID,
		callID:            setup.callID,
		modelID:           setup.modelID,
		modelOutputs:      make(map[string]model.ToolResultOutput),
		openText:          make(map[string]int),
		openThoughts:      make(map[string]int),
		finalFinishReason: model.FinishOther,
	}

	result.log.append(loggedPart{
		part: ai.StartPart{},
	})

	return result
}

// enqueue forwards a part and records it in the current step.
func (r *turnResult) enqueue(part ai.StreamPart) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.startStep()
	r.write(part)
	r.record(part)
}

// enqueueContinuation forwards a part that arrived between model calls, such
// as an approval response or a client tool result, without attributing it
// to a step.
func (r *turnResult) enqueueContinuation(part ai.StreamPart) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.write(part)
}

// enqueueRaw forwards adapter-specific data outside any step.
func (r *turnResult) enqueueRaw(value json.RawMessage) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.write(ai.RawPart{
		Value: value,
	})
}

// setModelID records the runtime's model, unless a step already completed.
func (r *turnResult) setModelID(modelID string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if len(r.steps) == 0 {
		r.modelID = modelID
	}
}

// addWarnings records warnings of the current step.
func (r *turnResult) addWarnings(warnings []model.Warning) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.stepWarnings = append(r.stepWarnings, warnings...)
}

// setToolModelOutput records the model-facing output submitted for a call.
func (r *turnResult) setToolModelOutput(toolCallID string, output model.ToolResultOutput) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.modelOutputs[toolCallID] = output
}

// discardCurrentStepContent drops content replayed while a resumed step
// closes.
func (r *turnResult) discardCurrentStepContent() {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.resetStep()
}

// stepBoundary is the end of a model step.
type stepBoundary struct {
	finishReason model.FinishReason
	usage        model.Usage
	metadata     model.ProviderMetadata
}

// finishStep closes the current step, records its result and forwards a
// FinishStepPart.
func (r *turnResult) finishStep(ctx context.Context, boundary stepBoundary) (ai.StepResult, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.startStep()

	usage := languageModelUsage(boundary.usage)
	response := r.stepResponse()

	messages, err := responseMessages(ctx, responseMessageInput{
		content:      r.content,
		tools:        r.tools,
		modelOutputs: r.modelOutputs,
	})
	if err != nil {
		return ai.StepResult{}, err
	}

	step := ai.StepResult{
		CallID:     r.callID,
		StepNumber: r.stepNumber,
		Model: ai.StepModel{
			Provider: r.provider,
			ModelID:  r.modelID,
		},
		Content:         r.content,
		FinishReason:    boundary.finishReason.Unified,
		RawFinishReason: boundary.finishReason.Raw,
		Usage:           usage,
		Warnings:        r.stepWarnings,
		Response: ai.StepResponse{
			ResponseInfo: response,
			Messages:     messages,
		},
		ProviderMetadata: boundary.metadata,
	}

	r.steps = append(r.steps, step)

	r.write(ai.FinishStepPart{
		Response:         response,
		Usage:            usage,
		FinishReason:     step.FinishReason,
		RawFinishReason:  step.RawFinishReason,
		ProviderMetadata: step.ProviderMetadata,
	})

	r.totalUsage = addUsage(r.totalUsage, usage)
	r.finalFinishReason = step.FinishReason
	r.finalRawFinishReason = step.RawFinishReason
	r.finalMetadata = step.ProviderMetadata

	r.stepNumber++
	r.resetStep()

	return step, nil
}

// turnFinish is the terminal finish of a turn, when the adapter reported one.
type turnFinish struct {
	finishReason model.FinishReason
	totalUsage   model.Usage
	metadata     model.ProviderMetadata
}

// finish completes the result: it builds the aggregated result, parses the
// output, forwards a FinishPart and closes the stream. Without a terminal
// finish the result keeps the values of the last step. It is idempotent.
func (r *turnResult) finish(ctx context.Context, final *turnFinish) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.settled {
		return
	}

	if len(r.content) > 0 {
		r.settle(ai.ErrorPart{
			Err: errUnclosedStep,
		}, errUnclosedStep)

		return
	}

	if final != nil {
		r.finalFinishReason = final.finishReason.Unified
		r.finalRawFinishReason = final.finishReason.Raw
		r.finalMetadata = final.metadata
		r.totalUsage = languageModelUsage(final.totalUsage)
	}

	result := r.aggregate()
	output, outputErr := r.parseOutput(ctx, result.FinalStep)
	result.Output = output

	r.write(ai.FinishPart{
		FinishReason:    r.finalFinishReason,
		RawFinishReason: r.finalRawFinishReason,
		TotalUsage:      r.totalUsage,
	})

	r.settled = true
	r.outcome = turnOutcome{
		result:    result,
		outputErr: outputErr,
	}
	r.log.close()
	close(r.done)
}

// abort settles the result as cancelled: it forwards an ai.AbortPart, as
// StreamText does, instead of an error part. It is idempotent.
func (r *turnResult) abort(err error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.settled {
		return
	}

	r.settle(ai.AbortPart{
		Reason: err.Error(),
	}, err)
}

// fail settles the result with a fatal error part. It is idempotent.
func (r *turnResult) fail(err error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.settled {
		return
	}

	r.settle(ai.ErrorPart{
		Err: err,
	}, err)
}

// settle forwards a terminal part and ends the result with err. The caller
// holds mu.
func (r *turnResult) settle(part ai.StreamPart, err error) {
	r.write(part)
	r.settled = true
	r.outcome = turnOutcome{
		err: err,
	}
	r.log.close()
	close(r.done)
}

// aggregate builds the result of the whole turn. The caller holds mu.
func (r *turnResult) aggregate() *ai.GenerateTextResult {
	result := summarizeSteps(r.steps, r.finalStep(), r.totalUsage)
	result.FinishReason = r.finalFinishReason
	result.RawFinishReason = r.finalRawFinishReason
	result.ProviderMetadata = r.finalMetadata

	return result
}

// finalStep returns the last step, or an empty step when none completed.
// The caller holds mu.
func (r *turnResult) finalStep() ai.StepResult {
	if len(r.steps) > 0 {
		return r.steps[len(r.steps)-1]
	}

	return ai.StepResult{
		CallID: r.callID,
		Model: ai.StepModel{
			Provider: r.provider,
			ModelID:  r.modelID,
		},
		FinishReason:    r.finalFinishReason,
		RawFinishReason: r.finalRawFinishReason,
		Response: ai.StepResponse{
			ResponseInfo: r.stepResponse(),
		},
	}
}

// parseOutput parses the final step's text with the configured output. The
// text itself is the output when none is configured. The caller holds mu.
func (r *turnResult) parseOutput(ctx context.Context, finalStep ai.StepResult) (any, error) {
	if r.output == nil {
		return finalStep.Text(), nil
	}

	output, err := r.output.ParseCompleteOutput(ctx, finalStep.Text(), ai.OutputContext{
		Response:     finalStep.Response.ResponseInfo,
		Usage:        finalStep.Usage,
		FinishReason: finalStep.FinishReason,
	})
	if err != nil {
		return nil, fmt.Errorf("parse harness output: %w", err)
	}

	return output, nil
}

// stepResponse describes the response of the current step. The caller holds
// mu.
func (r *turnResult) stepResponse() model.ResponseInfo {
	return model.ResponseInfo{
		ResponseMetadata: model.ResponseMetadata{
			ID:        stepResponseID(r.callID, r.stepNumber),
			Timestamp: time.Now(),
			ModelID:   r.modelID,
		},
	}
}

// startStep forwards a StartStepPart before the first part of a step. The
// caller holds mu.
func (r *turnResult) startStep() {
	if r.stepStarted {
		return
	}

	r.stepStarted = true
	r.write(ai.StartStepPart{
		Warnings: r.stepWarnings,
	})
}

// resetStep clears the content of the current step. The caller holds mu.
func (r *turnResult) resetStep() {
	r.content = nil
	clear(r.openText)
	clear(r.openThoughts)
	r.stepWarnings = nil
	r.stepStarted = false
}

// write appends a part to the stream, applying the partial output
// projection. The caller holds mu.
func (r *turnResult) write(part ai.StreamPart) {
	if r.output == nil {
		r.log.append(loggedPart{
			part: part,
		})

		return
	}

	r.log.append(r.partial.project(r.output, part)...)
}

// record adds a part to the current step's content. The caller holds mu.
func (r *turnResult) record(part ai.StreamPart) {
	switch typed := part.(type) {
	case ai.TextStartPart:
		r.openText[typed.ID] = r.appendContent(ai.TextContent{
			ProviderMetadata: typed.ProviderMetadata,
		})
	case ai.TextDeltaPart:
		r.recordTextDelta(typed)
	case ai.TextEndPart:
		r.recordTextEnd(typed)
	case ai.ReasoningStartPart:
		r.openThoughts[typed.ID] = r.appendContent(ai.ReasoningContent{
			ProviderMetadata: typed.ProviderMetadata,
		})
	case ai.ReasoningDeltaPart:
		r.recordReasoningDelta(typed)
	case ai.ReasoningEndPart:
		r.recordReasoningEnd(typed)
	case ai.ToolCall, ai.ToolApprovalRequest, ai.ToolApprovalResponse, ai.ToolResult, ai.ToolError:
		r.appendContent(part.(ai.ContentPart))
	}
}

// appendContent adds content and returns its index. The caller holds mu.
func (r *turnResult) appendContent(content ai.ContentPart) int {
	r.content = append(r.content, content)

	return len(r.content) - 1
}

// recordTextDelta appends to an open text block, opening one when the
// adapter skipped the start part. The caller holds mu.
func (r *turnResult) recordTextDelta(part ai.TextDeltaPart) {
	index, open := r.openText[part.ID]
	if !open {
		index = r.appendContent(ai.TextContent{
			ProviderMetadata: part.ProviderMetadata,
		})
		r.openText[part.ID] = index
	}

	text := r.content[index].(ai.TextContent)
	text.Text += part.Text

	if part.ProviderMetadata != nil {
		text.ProviderMetadata = part.ProviderMetadata
	}

	r.content[index] = text
}

// recordTextEnd closes a text block. The caller holds mu.
func (r *turnResult) recordTextEnd(part ai.TextEndPart) {
	index, open := r.openText[part.ID]
	if !open {
		return
	}

	text := r.content[index].(ai.TextContent)
	if part.ProviderMetadata != nil {
		text.ProviderMetadata = part.ProviderMetadata
	}

	if part.Citations != nil {
		text.Citations = part.Citations
	}

	r.content[index] = text
	delete(r.openText, part.ID)
}

// recordReasoningDelta appends to an open reasoning block. The caller holds
// mu.
func (r *turnResult) recordReasoningDelta(part ai.ReasoningDeltaPart) {
	index, open := r.openThoughts[part.ID]
	if !open {
		index = r.appendContent(ai.ReasoningContent{
			ProviderMetadata: part.ProviderMetadata,
		})
		r.openThoughts[part.ID] = index
	}

	reasoning := r.content[index].(ai.ReasoningContent)
	reasoning.Text += part.Text

	if part.ProviderMetadata != nil {
		reasoning.ProviderMetadata = part.ProviderMetadata
	}

	r.content[index] = reasoning
}

// recordReasoningEnd closes a reasoning block. The caller holds mu.
func (r *turnResult) recordReasoningEnd(part ai.ReasoningEndPart) {
	index, open := r.openThoughts[part.ID]
	if !open {
		return
	}

	reasoning := r.content[index].(ai.ReasoningContent)
	if part.ProviderMetadata != nil {
		reasoning.ProviderMetadata = part.ProviderMetadata
	}

	r.content[index] = reasoning
	delete(r.openThoughts, part.ID)
}
