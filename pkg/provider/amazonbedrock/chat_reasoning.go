package amazonbedrock

import (
	"github.com/aarock1234/ai/internal/reasoning"
	"github.com/aarock1234/ai/pkg/model"
	"github.com/aarock1234/ai/pkg/provider/anthropic"
)

// portableReasoningUnsupported explains why a call's reasoning level is
// ignored.
const portableReasoningUnsupported = "Portable reasoning is not supported for this model and will be ignored. If the model supports a provider-specific reasoning configuration, use providerOptions.amazonBedrock.reasoningConfig."

// reasoningEfforts map reasoning levels to Bedrock efforts.
var reasoningEfforts = map[model.ReasoningEffort]ReasoningEffort{
	model.ReasoningMinimal: ReasoningEffortLow,
	model.ReasoningLow:     ReasoningEffortLow,
	model.ReasoningMedium:  ReasoningEffortMedium,
	model.ReasoningHigh:    ReasoningEffortHigh,
	model.ReasoningXHigh:   ReasoningEffortMax,
	model.ReasoningMax:     ReasoningEffortMax,
}

// novaReasoningEfforts map reasoning levels to the efforts of Nova 2
// models, which stop at high.
var novaReasoningEfforts = map[model.ReasoningEffort]ReasoningEffort{
	model.ReasoningMinimal: ReasoningEffortLow,
	model.ReasoningLow:     ReasoningEffortLow,
	model.ReasoningMedium:  ReasoningEffortMedium,
	model.ReasoningHigh:    ReasoningEffortHigh,
	model.ReasoningXHigh:   ReasoningEffortHigh,
	model.ReasoningMax:     ReasoningEffortHigh,
}

// reasoningTarget describes the model a reasoning level is resolved for.
type reasoningTarget struct {
	modelID        string
	anthropicModel bool
	openAIModel    bool
}

// resolveReasoningConfig applies the call's reasoning level to the
// reasoning configuration. Explicit configuration values take precedence;
// a disabled configuration stays disabled.
func resolveReasoningConfig(level model.ReasoningEffort, configured *ReasoningConfig, target reasoningTarget, warnings *[]model.Warning) *ReasoningConfig {
	if !reasoning.IsCustom(level) {
		return configured
	}

	resolved := &ReasoningConfig{}
	if configured != nil {
		copied := *configured
		resolved = &copied
	}

	if isReasoningType(configured, ReasoningDisabled) {
		resolved.MaxReasoningEffort = nil
		resolved.BudgetTokens = nil

		return resolved
	}

	if target.anthropicModel {
		return resolveClaudeReasoning(level, configured, resolved, target.modelID, warnings)
	}

	if level == model.ReasoningNone {
		return configured
	}

	nova := isNova2ReasoningModel(target.modelID)
	if !target.openAIModel && !nova && configured == nil {
		*warnings = append(*warnings, model.UnsupportedWarning("reasoning", portableReasoningUnsupported))

		return configured
	}

	if resolved.MaxReasoningEffort == nil {
		efforts := reasoningEfforts
		if nova {
			efforts = novaReasoningEfforts
		}

		resolved.MaxReasoningEffort = mappedEffort(level, efforts, warnings)
	}

	if nova && resolved.Type == nil {
		resolved.Type = new(ReasoningEnabled)
	}

	return resolved
}

// resolveClaudeReasoning disables thinking for "none", sets an effort for
// adaptive models and a thinking budget for others.
func resolveClaudeReasoning(
	level model.ReasoningEffort,
	configured *ReasoningConfig,
	resolved *ReasoningConfig,
	modelID string,
	warnings *[]model.Warning,
) *ReasoningConfig {
	if level == model.ReasoningNone {
		return &ReasoningConfig{
			Type: new(ReasoningDisabled),
		}
	}

	capabilities := anthropic.CapabilitiesOf(modelID)

	if capabilities.SupportsAdaptiveThinking {
		if resolved.MaxReasoningEffort == nil {
			resolved.MaxReasoningEffort = mappedEffort(level, reasoningEfforts, warnings)
		}

		if resolved.Type == nil {
			resolved.Type = new(ReasoningAdaptive)
		}

		return resolved
	}

	if resolved.BudgetTokens == nil {
		budget, budgetWarnings, ok := reasoning.Budget(level, reasoning.BudgetLimits{
			MaxOutputTokens: capabilities.MaxOutputTokens,
			MinBudget:       reasoning.DefaultMinBudget,
			MaxBudget:       capabilities.MaxOutputTokens,
		})

		*warnings = append(*warnings, budgetWarnings...)

		if !ok {
			return configured
		}

		resolved.BudgetTokens = &budget
	}

	if resolved.Type == nil {
		resolved.Type = new(ReasoningEnabled)
	}

	return resolved
}

// mappedEffort maps a reasoning level to an effort, or nil with a warning
// when the level has no effort.
func mappedEffort(level model.ReasoningEffort, efforts map[model.ReasoningEffort]ReasoningEffort, warnings *[]model.Warning) *ReasoningEffort {
	effort, effortWarnings, ok := reasoning.Effort(level, efforts)
	*warnings = append(*warnings, effortWarnings...)

	if !ok {
		return nil
	}

	return &effort
}

func isReasoningType(config *ReasoningConfig, reasoningType ReasoningType) bool {
	return config != nil && config.Type != nil && *config.Type == reasoningType
}
