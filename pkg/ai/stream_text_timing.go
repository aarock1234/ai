package ai

import (
	"math"
	"slices"
	"time"
)

// outputTiming measures when output chunks arrive.
type outputTiming struct {
	firstOutput  *time.Duration
	previous     time.Time
	betweenParts []time.Duration
}

func (t *outputTiming) observe(now, started time.Time) {
	if t.firstOutput == nil {
		elapsed := now.Sub(started)
		t.firstOutput = &elapsed
	} else {
		t.betweenParts = append(t.betweenParts, now.Sub(t.previous))
	}

	t.previous = now
}

// performance computes the model call's performance from its usage and
// response time.
func (t *outputTiming) performance(usage LanguageModelUsage, responseTime time.Duration) LanguageModelCallPerformance {
	performance := generatedCallPerformance(usage, responseTime)
	performance.TimeToFirstOutput = t.firstOutput

	if t.firstOutput != nil {
		outputRate := tokensPerSecond(usage.OutputTokens, (responseTime - *t.firstOutput).Seconds())
		inputRate := tokensPerSecond(usage.InputTokens, t.firstOutput.Seconds())

		performance.OutputTokensPerSecond = &outputRate
		performance.InputTokensPerSecond = &inputRate
	}

	if len(t.betweenParts) > 0 {
		stats := chunkTimingStats(t.betweenParts)
		performance.TimeBetweenOutputChunks = &stats
	}

	return performance
}

func chunkTimingStats(gaps []time.Duration) OutputChunkTimingStats {
	sorted := slices.Clone(gaps)
	slices.Sort(sorted)

	var sum time.Duration
	for _, gap := range gaps {
		sum += gap
	}

	return OutputChunkTimingStats{
		Min:    sorted[0],
		P10:    nearestRankPercentile(sorted, 0.1),
		Median: nearestRankPercentile(sorted, 0.5),
		Avg:    sum / time.Duration(len(gaps)),
		P90:    nearestRankPercentile(sorted, 0.9),
		Max:    sorted[len(sorted)-1],
	}
}

func nearestRankPercentile(sorted []time.Duration, percentile float64) time.Duration {
	rank := int(math.Ceil(percentile*float64(len(sorted)))) - 1

	return sorted[max(rank, 0)]
}
