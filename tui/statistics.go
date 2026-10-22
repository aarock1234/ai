package tui

import (
	"math"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/pkg/ai"
)

// stepStatistics are the statistics of the latest step of a turn.
type stepStatistics struct {
	totalTokens           *int
	outputTokens          *int
	outputTokensPerSecond *float64
}

// responseStatistics tracks the token statistics that the UI shows: the
// total tokens of the latest turn in the header, and the output statistic
// of each response in its assistant sections. Like upstream, they come
// from the latest step of a turn and appear when the turn finishes.
type responseStatistics struct {
	totalTokens *int
	latestStep  stepStatistics
	byResponse  map[int]stepStatistics
}

// reset clears the statistics of a response when a turn starts.
func (s *responseStatistics) reset(response int) {
	s.totalTokens = nil
	s.latestStep = stepStatistics{}
	delete(s.byResponse, response)
}

// recordStep remembers the usage of a finished step.
func (s *responseStatistics) recordStep(step ai.FinishStepPart) {
	s.latestStep = stepStatistics{
		totalTokens:           totalTokens(step.Usage),
		outputTokens:          step.Usage.OutputTokens,
		outputTokensPerSecond: step.Performance.OutputTokensPerSecond,
	}
}

// recordFinish publishes the statistics of a finished turn, falling back
// to the total usage when no step reported usage.
func (s *responseStatistics) recordFinish(response int, finish ai.FinishPart) {
	latest := s.latestStep

	if latest.totalTokens == nil && latest.outputTokens == nil {
		latest.totalTokens = totalTokens(finish.TotalUsage)
		latest.outputTokens = finish.TotalUsage.OutputTokens
	}

	s.totalTokens = latest.totalTokens
	s.byResponse[response] = latest
}

// format returns the statistic of a response in the selected mode, or the
// empty string when it is unknown.
func (s *responseStatistics) format(response int, mode ResponseStatisticsMode) string {
	statistics := s.byResponse[response]

	if mode == StatisticsOutputTokenCount {
		return formatTokenCount(statistics.outputTokens, 0)
	}

	if statistics.outputTokensPerSecond == nil {
		return ""
	}

	return formatNumber(*statistics.outputTokensPerSecond) + " tok/s"
}

// totalTokens returns the total tokens of a usage, or the sum of its input
// and output tokens.
func totalTokens(usage ai.LanguageModelUsage) *int {
	if usage.TotalTokens != nil {
		return usage.TotalTokens
	}

	if usage.InputTokens == nil || usage.OutputTokens == nil {
		return nil
	}

	return new(*usage.InputTokens + *usage.OutputTokens)
}

// formatTokenCount formats a token count, with its share of the context
// window when contextSize is positive. It returns the empty string when
// the count is unknown.
func formatTokenCount(tokens *int, contextSize int) string {
	if tokens == nil {
		return ""
	}

	unit := "tokens"
	if *tokens == 1 {
		unit = "token"
	}

	formatted := groupThousands(strconv.Itoa(*tokens)) + " " + unit
	if contextSize <= 0 {
		return formatted
	}

	percentage := math.Round(float64(*tokens) / float64(contextSize) * 100)

	return formatted + " " + groupThousands(strconv.Itoa(int(percentage))) + "%"
}

// formatNumber formats an integer with thousands separators and other
// numbers with at most one fraction digit.
func formatNumber(value float64) string {
	if value == math.Trunc(value) {
		return groupThousands(strconv.FormatInt(int64(value), 10))
	}

	formatted := strconv.FormatFloat(value, 'f', 1, 64)
	whole, fraction, _ := strings.Cut(formatted, ".")
	if fraction == "0" {
		return groupThousands(whole)
	}

	return groupThousands(whole) + "." + fraction
}

// groupThousands inserts commas between groups of three digits.
func groupThousands(digits string) string {
	sign := ""
	if unsigned, ok := strings.CutPrefix(digits, "-"); ok {
		sign = "-"
		digits = unsigned
	}

	var grouped strings.Builder

	for index, digit := range digits {
		if index > 0 && (len(digits)-index)%3 == 0 {
			grouped.WriteByte(',')
		}

		grouped.WriteRune(digit)
	}

	return sign + grouped.String()
}
