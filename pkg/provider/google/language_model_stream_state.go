package google

import (
	"encoding/json"
	"slices"
	"strconv"

	"github.com/aarock1234/ai/internal/toolnames"
	"github.com/aarock1234/ai/pkg/model"
)

// streamState converts stream chunks into stream parts. Consecutive text and
// reasoning parts share one block; function calls may arrive whole or as
// partial arguments over several chunks.
type streamState struct {
	toolNames  toolnames.Mapping
	generateID func() string

	// metadataNamespaces key the part and finish metadata.
	metadataNamespaces []string

	finishReason model.FinishReason

	// usage is the usage of the current request; continuedUsage sums the
	// completed requests of an automatically continued response.
	usage                   *UsageMetadata
	continuedUsage          *UsageMetadata
	initialPromptTokenCount *int

	promptFeedback      *PromptFeedback
	blockReason         string
	grounding           *GroundingMetadata
	urlContext          *URLContextMetadata
	safetyRatings       []SafetyRating
	finishMessage       *string
	continuationToken   string
	hasToolCalls        bool
	hasResponseMetadata bool

	textBlockID      string
	reasoningBlockID string
	blockCounter     int

	emittedSourceURLs       map[string]bool
	lastCodeExecutionCallID string
	lastServerToolCallID    string
	streamingCalls          []streamingCall

	pending []model.StreamPart
}

// streamingCall is a function call whose arguments are still streaming.
type streamingCall struct {
	toolCallID  string
	toolName    string
	accumulator *argumentAccumulator
	metadata    model.ProviderMetadata
}

func newStreamState(call preparedCall) *streamState {
	return &streamState{
		toolNames:          call.toolNames,
		generateID:         call.generateID,
		metadataNamespaces: call.metadataNamespaces,
		finishReason: model.FinishReason{
			Unified: model.FinishOther,
		},
		emittedSourceURLs: make(map[string]bool),
	}
}

func (s *streamState) emit(part model.StreamPart) {
	s.pending = append(s.pending, part)
}

// fail marks the response as failed.
func (s *streamState) fail() {
	s.finishReason = model.FinishReason{
		Unified: model.FinishError,
	}
}

// process returns the parts of one chunk.
func (s *streamState) process(chunk generateContentResponse) []model.StreamPart {
	s.pending = nil

	if chunk.Error != nil {
		s.fail()
		s.emit(model.StreamError{
			Err: chunk.Error,
		})

		return s.pending
	}

	if !s.hasResponseMetadata && chunk.ResponseID != "" {
		s.hasResponseMetadata = true
		s.emit(model.ResponseMetadata{
			ID: chunk.ResponseID,
		})
	}

	if chunk.UsageMetadata != nil {
		s.usage = chunk.UsageMetadata
	}

	s.trackPromptFeedback(chunk.PromptFeedback)

	candidate := chunk.firstCandidate()
	if candidate == nil {
		return s.pending
	}

	s.trackCandidateMetadata(candidate)

	// A blocked prompt produces no content, but later chunks still report
	// usage and metadata.
	if s.blockReason != "" {
		return s.pending
	}

	s.emitSources(candidate.GroundingMetadata)

	parts := candidate.parts()
	for _, chunkPart := range parts {
		s.processPart(chunkPart)
	}

	// Function calls are handled after the other parts so that the finish
	// reason below sees them.
	for _, chunkPart := range parts {
		if chunkPart.FunctionCall != nil {
			s.processFunctionCall(*chunkPart.FunctionCall, signatureMetadata(s.metadataNamespaces, chunkPart.ThoughtSignature))
		}
	}

	s.trackFinish(candidate.FinishReason)

	return s.pending
}

func (s *streamState) trackPromptFeedback(feedback *PromptFeedback) {
	if feedback == nil || s.blockReason != "" {
		return
	}

	s.promptFeedback = feedback

	if !isConfirmedBlockReason(feedback.BlockReason) {
		return
	}

	s.blockReason = *feedback.BlockReason
	s.finishReason = model.FinishReason{
		Unified: model.FinishContentFilter,
		Raw:     s.blockReason,
	}
}

func (s *streamState) trackCandidateMetadata(candidate *candidate) {
	if candidate.GroundingMetadata != nil {
		s.grounding = candidate.GroundingMetadata
	}

	if candidate.URLContextMetadata != nil {
		s.urlContext = candidate.URLContextMetadata
	}

	if candidate.SafetyRatings != nil {
		s.safetyRatings = candidate.SafetyRatings
	}

	if candidate.FinishMessage != nil {
		s.finishMessage = candidate.FinishMessage
	}

	if candidate.ContinuationToken != "" {
		s.continuationToken = candidate.ContinuationToken
	}
}

// emitSources emits URL sources once each. Document sources are reported in
// the grounding metadata only.
func (s *streamState) emitSources(grounding *GroundingMetadata) {
	for _, source := range extractSources(grounding, s.generateID) {
		if source.SourceType != model.SourceURL || s.emittedSourceURLs[source.URL] {
			continue
		}

		s.emittedSourceURLs[source.URL] = true
		s.emit(source)
	}
}

// trackFinish records the finish reason. The usage of a request that is
// continued is banked, because the next request reports its own usage.
func (s *streamState) trackFinish(raw string) {
	if raw == finishReasonContinuation {
		if s.continuedUsage == nil {
			s.initialPromptTokenCount = promptTokenCount(s.usage)
		}

		s.continuedUsage = addUsageMetadata(s.continuedUsage, s.usage)
		s.usage = nil
	}

	if raw == "" {
		return
	}

	s.finishReason = model.FinishReason{
		Unified: mapFinishReason(raw, s.hasToolCalls),
		Raw:     raw,
	}
}

func (s *streamState) processPart(chunkPart part) {
	switch {
	case chunkPart.ExecutableCode != nil && chunkPart.ExecutableCode.Code != "":
		s.processCodeExecutionCall(*chunkPart.ExecutableCode)

	case chunkPart.CodeExecutionResult != nil:
		s.processCodeExecutionResult(*chunkPart.CodeExecutionResult)

	case chunkPart.Text != nil:
		s.processText(chunkPart)

	case chunkPart.InlineData != nil:
		s.processInlineData(chunkPart)

	case chunkPart.ToolCall != nil:
		s.processServerToolCall(*chunkPart.ToolCall, chunkPart.ThoughtSignature)

	case chunkPart.ToolResponse != nil:
		s.processServerToolResponse(*chunkPart.ToolResponse, chunkPart.ThoughtSignature)
	}
}

func (s *streamState) processCodeExecutionCall(input CodeExecutionInput) {
	callID := s.generateID()
	s.lastCodeExecutionCallID = callID

	s.emit(model.ToolCall{
		ToolCallID:       callID,
		ToolName:         s.toolNames.ToApplication(codeExecutionToolName),
		Input:            encodeCompact(input),
		ProviderExecuted: true,
	})
}

func (s *streamState) processCodeExecutionResult(output CodeExecutionOutput) {
	if s.lastCodeExecutionCallID == "" {
		return
	}

	s.emit(model.ToolResult{
		ToolCallID: s.lastCodeExecutionCallID,
		ToolName:   s.toolNames.ToApplication(codeExecutionToolName),
		Result:     json.RawMessage(encodeCompact(output)),
	})
}

// processText continues the open text or reasoning block, or closes it and
// opens one of the other kind. An empty text part carries a thought
// signature for the open text block.
func (s *streamState) processText(chunkPart part) {
	metadata := signatureMetadata(s.metadataNamespaces, chunkPart.ThoughtSignature)
	text := *chunkPart.Text

	if text == "" {
		if metadata != nil && s.textBlockID != "" {
			s.emit(model.TextDelta{
				ID:               s.textBlockID,
				ProviderMetadata: metadata,
			})
		}

		return
	}

	if chunkPart.Thought {
		s.endTextBlock()

		if s.reasoningBlockID == "" {
			s.reasoningBlockID = s.nextBlockID()
			s.emit(model.ReasoningStart{
				ID:               s.reasoningBlockID,
				ProviderMetadata: metadata,
			})
		}

		s.emit(model.ReasoningDelta{
			ID:               s.reasoningBlockID,
			Delta:            text,
			ProviderMetadata: metadata,
		})

		return
	}

	s.endReasoningBlock()

	if s.textBlockID == "" {
		s.textBlockID = s.nextBlockID()
		s.emit(model.TextStart{
			ID:               s.textBlockID,
			ProviderMetadata: metadata,
		})
	}

	s.emit(model.TextDelta{
		ID:               s.textBlockID,
		Delta:            text,
		ProviderMetadata: metadata,
	})
}

func (s *streamState) processInlineData(chunkPart part) {
	s.endTextBlock()
	s.endReasoningBlock()

	file, err := inlineFile(chunkPart, signatureMetadata(s.metadataNamespaces, chunkPart.ThoughtSignature))
	if err != nil {
		s.emit(model.StreamError{
			Err: err,
		})

		return
	}

	switch file := file.(type) {
	case model.File:
		s.emit(file)
	case model.ReasoningFile:
		s.emit(file)
	}
}

func (s *streamState) processServerToolCall(call serverToolCall, signature string) {
	callID := call.ID
	if callID == "" {
		callID = s.generateID()
	}

	s.lastServerToolCallID = callID

	s.emit(model.ToolCall{
		ToolCallID:       callID,
		ToolName:         serverToolNamePrefix + call.ToolType,
		Input:            compactArguments(call.Args),
		ProviderExecuted: true,
		Dynamic:          true,
		ProviderMetadata: serverToolMetadata(s.metadataNamespaces, signature, callID, call.ToolType),
	})
}

func (s *streamState) processServerToolResponse(response serverToolResponse, signature string) {
	callID := firstNonEmpty(s.lastServerToolCallID, response.ID)
	if callID == "" {
		callID = s.generateID()
	}

	s.lastServerToolCallID = ""

	s.emit(model.ToolResult{
		ToolCallID:       callID,
		ToolName:         serverToolNamePrefix + response.ToolType,
		Result:           json.RawMessage(compactArguments(response.Response)),
		ProviderMetadata: serverToolMetadata(s.metadataNamespaces, signature, callID, response.ToolType),
	})
}

func (s *streamState) nextBlockID() string {
	blockID := strconv.Itoa(s.blockCounter)
	s.blockCounter++

	return blockID
}

func (s *streamState) endTextBlock() {
	if s.textBlockID == "" {
		return
	}

	s.emit(model.TextEnd{
		ID: s.textBlockID,
	})
	s.textBlockID = ""
}

func (s *streamState) endReasoningBlock() {
	if s.reasoningBlockID == "" {
		return
	}

	s.emit(model.ReasoningEnd{
		ID: s.reasoningBlockID,
	})
	s.reasoningBlockID = ""
}

// finish closes open blocks and returns the final parts with the combined
// usage and the response metadata.
func (s *streamState) finish(groundingPerRequest []*GroundingMetadata, requestGrounding *GroundingMetadata) []model.StreamPart {
	s.pending = nil
	usage := addUsageMetadata(s.continuedUsage, s.usage)

	s.endTextBlock()
	s.endReasoningBlock()

	metadata := ChatMetadata{
		PromptFeedback:          s.promptFeedback,
		GroundingMetadata:       s.grounding,
		URLContextMetadata:      s.urlContext,
		SafetyRatings:           s.safetyRatings,
		UsageMetadata:           usage,
		FinishMessage:           s.finishMessage,
		InitialPromptTokenCount: s.initialPromptTokenCount,
	}

	if usage != nil {
		metadata.ServiceTier = usage.ServiceTier
	}

	if len(groundingPerRequest) > 0 {
		metadata.GroundingMetadataPerRequest = append(slices.Clone(groundingPerRequest), requestGrounding)
	}

	// The token is surfaced only when the response stopped early, not for
	// the checkpoint tokens of a stream that finished normally.
	stoppedEarly := s.finishReason.Raw == finishReasonMaxTokens || s.finishReason.Raw == finishReasonContinuation
	if stoppedEarly {
		metadata.ContinuationToken = s.continuationToken
	}

	s.emit(model.Finish{
		Usage:            convertUsage(usage),
		FinishReason:     s.finishReason,
		ProviderMetadata: wrapMetadata(s.metadataNamespaces, metadata),
	})

	return s.pending
}
