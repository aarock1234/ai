// Package reasoning maps the provider-independent reasoning level of a call
// onto provider efforts and thinking token budgets, reporting the same
// warnings as the upstream provider utilities.
package reasoning

import (
	"fmt"
	"math"

	"github.com/aarock1234/ai/pkg/model"
)

// DefaultMinBudget is upstream's default minimum thinking budget. Callers
// pass it as BudgetLimits.MinBudget unless the model accepts smaller budgets.
const DefaultMinBudget = 1024

// defaultBudgetShares are the shares of the maximum output tokens used as
// the thinking budget of each level.
var defaultBudgetShares = map[model.ReasoningEffort]float64{
	model.ReasoningMinimal: 0.02,
	model.ReasoningLow:     0.1,
	model.ReasoningMedium:  0.3,
	model.ReasoningHigh:    0.6,
	model.ReasoningXHigh:   0.9,
	model.ReasoningMax:     0.95,
}

// IsCustom reports whether the call selected a reasoning level, including
// "none", rather than leaving it unset or at the provider default.
func IsCustom(level model.ReasoningEffort) bool {
	return level != "" && level != model.ReasoningProviderDefault
}

// Effort maps a reasoning level to a provider effort through efforts. A level
// that maps to a differently named effort produces a compatibility warning. A
// level missing from efforts produces an unsupported warning and reports
// false.
func Effort[T ~string](level model.ReasoningEffort, efforts map[model.ReasoningEffort]T) (T, []model.Warning, bool) {
	effort, ok := efforts[level]
	if !ok {
		return "", []model.Warning{
			unsupportedWarning(level),
		}, false
	}

	if string(effort) == string(level) {
		return effort, nil, true
	}

	details := fmt.Sprintf("reasoning %q is not directly supported by this model. mapped to effort %q.", level, effort)

	return effort, []model.Warning{
		model.CompatibilityWarning("reasoning", details),
	}, true
}

// BudgetLimits describes how a model sizes thinking budgets.
type BudgetLimits struct {
	// MaxOutputTokens is the model's output limit that the level's share is
	// taken of.
	MaxOutputTokens int

	// MinBudget and MaxBudget clamp the budget; both are required. Use
	// DefaultMinBudget unless the model accepts smaller budgets.
	MinBudget int
	MaxBudget int

	// Shares overrides the share of MaxOutputTokens used for each level. Nil
	// uses the upstream defaults, from 2% for minimal to 95% for max.
	Shares map[model.ReasoningEffort]float64
}

// Budget maps a reasoning level to a thinking token budget: the level's share
// of the maximum output tokens, rounded and clamped to the limits. A level
// without a share produces an unsupported warning and reports false.
func Budget(level model.ReasoningEffort, limits BudgetLimits) (int, []model.Warning, bool) {
	shares := limits.Shares
	if shares == nil {
		shares = defaultBudgetShares
	}

	share, ok := shares[level]
	if !ok {
		return 0, []model.Warning{
			unsupportedWarning(level),
		}, false
	}

	budget := int(math.Round(float64(limits.MaxOutputTokens) * share))
	clamped := min(limits.MaxBudget, max(limits.MinBudget, budget))

	return clamped, nil, true
}

func unsupportedWarning(level model.ReasoningEffort) model.Warning {
	details := fmt.Sprintf("reasoning %q is not supported by this model.", level)

	return model.UnsupportedWarning("reasoning", details)
}
