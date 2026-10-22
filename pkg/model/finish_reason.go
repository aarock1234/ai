package model

// UnifiedFinishReason is a provider-independent reason for ending generation.
type UnifiedFinishReason string

// Unified finish reasons.
const (
	FinishStop          UnifiedFinishReason = "stop"
	FinishLength        UnifiedFinishReason = "length"
	FinishContentFilter UnifiedFinishReason = "content-filter"
	FinishToolCalls     UnifiedFinishReason = "tool-calls"
	FinishError         UnifiedFinishReason = "error"
	FinishOther         UnifiedFinishReason = "other"
)

// FinishReason explains why generation ended.
type FinishReason struct {
	Unified UnifiedFinishReason

	// Raw is the provider's native reason, if any.
	Raw string
}
