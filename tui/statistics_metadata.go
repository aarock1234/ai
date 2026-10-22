package tui

import (
	"encoding/json"
)

// metadataStatistics are the statistics that a remote agent reports in the
// metadata of its assistant messages.
type metadataStatistics struct {
	Usage       *metadataUsage       `json:"usage"`
	Performance *metadataPerformance `json:"performance"`
}

type metadataPerformance struct {
	OutputTokensPerSecond *float64 `json:"outputTokensPerSecond"`
}

// metadataUsage accepts the usage shapes of current and earlier AI SDK
// versions.
type metadataUsage struct {
	TotalTokens      tokenCount `json:"totalTokens"`
	InputTokens      tokenCount `json:"inputTokens"`
	OutputTokens     tokenCount `json:"outputTokens"`
	PromptTokens     *int       `json:"promptTokens"`
	CompletionTokens *int       `json:"completionTokens"`
}

// tokenCount is a token count given as a number or as an object with a
// total.
type tokenCount struct {
	value *int
}

// UnmarshalJSON reads a number or an object with a numeric total, and
// ignores other values.
func (c *tokenCount) UnmarshalJSON(data []byte) error {
	var number int
	if json.Unmarshal(data, &number) == nil {
		c.value = &number

		return nil
	}

	var detailed struct {
		Total *int `json:"total"`
	}

	if json.Unmarshal(data, &detailed) == nil {
		c.value = detailed.Total
	}

	return nil
}

func (u *metadataUsage) inputTokens() *int {
	if u.InputTokens.value != nil {
		return u.InputTokens.value
	}

	return u.PromptTokens
}

func (u *metadataUsage) outputTokens() *int {
	if u.OutputTokens.value != nil {
		return u.OutputTokens.value
	}

	return u.CompletionTokens
}

func (u *metadataUsage) totalTokens() *int {
	if u.TotalTokens.value != nil {
		return u.TotalTokens.value
	}

	input := u.inputTokens()
	output := u.outputTokens()

	if input == nil || output == nil {
		return nil
	}

	return new(*input + *output)
}

// recordMetadata publishes the statistics in a remote response's message
// metadata, keeping earlier values that it does not report.
func (s *responseStatistics) recordMetadata(response int, metadata json.RawMessage) {
	var reported metadataStatistics
	if len(metadata) == 0 || json.Unmarshal(metadata, &reported) != nil {
		return
	}

	statistics := s.byResponse[response]

	if reported.Usage != nil {
		statistics.totalTokens = firstKnown(reported.Usage.totalTokens(), statistics.totalTokens)
		statistics.outputTokens = firstKnown(reported.Usage.outputTokens(), statistics.outputTokens)
	}

	if reported.Performance != nil && reported.Performance.OutputTokensPerSecond != nil {
		statistics.outputTokensPerSecond = reported.Performance.OutputTokensPerSecond
	}

	s.totalTokens = firstKnown(statistics.totalTokens, s.totalTokens)
	s.byResponse[response] = statistics
}

func firstKnown[Value any](values ...*Value) *Value {
	for _, value := range values {
		if value != nil {
			return value
		}
	}

	return nil
}
