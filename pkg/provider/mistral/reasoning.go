package mistral

import (
	"github.com/aarock1234/ai/internal/reasoning"
	"github.com/aarock1234/ai/pkg/model"
)

// chatModelsWithoutReasoningEffort are chat models known not to accept a
// reasoning effort. Unknown models are assumed to accept one.
var chatModelsWithoutReasoningEffort = map[string]bool{
	"codestral-2508":          true,
	"codestral-latest":        true,
	"ministral-14b-2512":      true,
	"ministral-14b-latest":    true,
	"ministral-3b-2512":       true,
	"ministral-3b-latest":     true,
	"ministral-8b-2512":       true,
	"ministral-8b-latest":     true,
	"mistral-code-fim-latest": true,
	"mistral-code-latest":     true,
	"mistral-large-latest":    true,
	"mistral-large-2512":      true,
	"voxtral-small-2507":      true,
	"voxtral-small-latest":    true,
}

// conversationModelsWithReasoningEffort are the conversation models known to
// accept a reasoning effort. Other models are assumed not to.
var conversationModelsWithReasoningEffort = map[string]bool{
	"glm-5-2":                     true,
	"labs-leanstral-1-5":          true,
	"labs-leanstral-1-5-1":        true,
	"magistral-medium-latest":     true,
	"magistral-small-latest":      true,
	"mistral-large-4":             true,
	"mistral-large-4-0":           true,
	"mistral-medium":              true,
	"mistral-medium-2604":         true,
	"mistral-medium-3":            true,
	"mistral-medium-3-5":          true,
	"mistral-medium-3.5":          true,
	"mistral-medium-latest":       true,
	"mistral-small-2603":          true,
	"mistral-small-latest":        true,
	"mistral-vibe-cli-fast":       true,
	"mistral-vibe-cli-latest":     true,
	"mistral-vibe-cli-with-tools": true,
	"zai-glm-5-2":                 true,
}

// reasoningEfforts maps reasoning levels to Mistral's effort. Mistral only
// distinguishes on and off, so every level enables reasoning.
var reasoningEfforts = map[model.ReasoningEffort]ReasoningEffort{
	model.ReasoningMinimal: ReasoningEffortHigh,
	model.ReasoningLow:     ReasoningEffortHigh,
	model.ReasoningMedium:  ReasoningEffortHigh,
	model.ReasoningHigh:    ReasoningEffortHigh,
	model.ReasoningXHigh:   ReasoningEffortHigh,
	model.ReasoningMax:     ReasoningEffortHigh,
}

// reasoningEffort resolves the wire reasoning effort. The provider option
// wins; otherwise the reasoning level selects the effort.
func reasoningEffort(supported bool, option *ReasoningEffort, level model.ReasoningEffort) (*string, []model.Warning) {
	if !supported {
		if !reasoning.IsCustom(level) {
			return nil, nil
		}

		return nil, []model.Warning{
			model.UnsupportedWarning("reasoning", "This model does not support reasoning configuration."),
		}
	}

	if option != nil {
		return new(string(*option)), nil
	}

	if !reasoning.IsCustom(level) {
		return nil, nil
	}

	if level == model.ReasoningNone {
		return new(string(ReasoningEffortNone)), nil
	}

	effort, warnings, ok := reasoning.Effort(level, reasoningEfforts)
	if !ok {
		return nil, warnings
	}

	return new(string(effort)), warnings
}
