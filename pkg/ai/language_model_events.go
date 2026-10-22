package ai

import (
	"time"

	"github.com/aarock1234/ai/pkg/model"
)

// LanguageModelCallStartEvent describes a language model call immediately
// before it starts. Unlike the step start event, it covers only the model
// call, not the client-side tool execution that may follow.
type LanguageModelCallStartEvent struct {
	CallID   string
	Provider string
	ModelID  string

	System   string
	Messages []model.Message

	// Tools are the tool declarations sent to the model.
	Tools []model.Tool

	CallSettings CallSettings
}

// LanguageModelCallEndEvent describes a language model call after its
// response was normalized and parsed, before any client-side tool runs.
type LanguageModelCallEndEvent struct {
	CallID   string
	Provider string

	// ModelID is the model that the response reports.
	ModelID string

	FinishReason model.UnifiedFinishReason
	Usage        LanguageModelUsage

	// Content is the content the model produced, without tool outputs of
	// client-side tools.
	Content []ContentPart

	ResponseID       string
	ProviderMetadata model.ProviderMetadata

	Performance LanguageModelCallPerformance
}

// LanguageModelCallPerformance measures one language model call. Durations
// replace the upstream millisecond values.
type LanguageModelCallPerformance struct {
	// ResponseTime is the time spent waiting for the model response.
	ResponseTime time.Duration

	// EffectiveOutputTokensPerSecond is the output tokens divided by the
	// response time.
	EffectiveOutputTokensPerSecond float64

	// OutputTokensPerSecond is the output tokens divided by the time after
	// the first output chunk. It is only set for streaming calls.
	OutputTokensPerSecond *float64

	// InputTokensPerSecond is the input tokens divided by the time to the
	// first output chunk. It is only set for streaming calls.
	InputTokensPerSecond *float64

	// EffectiveTotalTokensPerSecond is the input and output tokens divided by
	// the response time.
	EffectiveTotalTokensPerSecond float64

	// TimeToFirstOutput is the time until the first output chunk. It is only
	// set for streaming calls.
	TimeToFirstOutput *time.Duration

	// TimeBetweenOutputChunks summarizes the gaps between output chunks. It
	// is only set for streaming calls with at least two output chunks.
	TimeBetweenOutputChunks *OutputChunkTimingStats
}

// generatedCallPerformance measures a non-streaming model call.
func generatedCallPerformance(usage LanguageModelUsage, responseTime time.Duration) LanguageModelCallPerformance {
	return LanguageModelCallPerformance{
		ResponseTime:                   responseTime,
		EffectiveOutputTokensPerSecond: tokensPerSecond(usage.OutputTokens, responseTime.Seconds()),
		EffectiveTotalTokensPerSecond:  tokensPerSecond(addTokenCounts(usage.InputTokens, usage.OutputTokens), responseTime.Seconds()),
	}
}
