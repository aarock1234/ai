package moonshotai

import (
	"fmt"

	"github.com/aarock1234/ai/internal/reasoning"
	"github.com/aarock1234/ai/pkg/model"
)

// effortMap maps reasoning levels to the efforts of Kimi K3 and custom
// models.
var effortMap = map[model.ReasoningEffort]ReasoningEffort{
	model.ReasoningMinimal: ReasoningEffortLow,
	model.ReasoningLow:     ReasoningEffortLow,
	model.ReasoningMedium:  ReasoningEffortHigh,
	model.ReasoningHigh:    ReasoningEffortHigh,
	model.ReasoningXHigh:   ReasoningEffortMax,
	model.ReasoningMax:     ReasoningEffortMax,
}

// reasoningSettings are the thinking and effort members of a request.
type reasoningSettings struct {
	thinking *thinkingRequest
	effort   *ReasoningEffort
	warnings []model.Warning
}

// reasoningInputs are the call settings that control reasoning.
type reasoningInputs struct {
	modelID string
	options ChatOptions
	level   model.ReasoningEffort
}

// resolveReasoning applies the thinking and effort rules of the model's
// family.
func resolveReasoning(inputs reasoningInputs) reasoningSettings {
	var settings reasoningSettings

	if inputs.options.Thinking != nil && inputs.options.Thinking.BudgetTokens != nil {
		settings.warnings = append(settings.warnings, model.Warning{
			Type:    model.WarningDeprecated,
			Setting: "providerOptions.moonshotai.thinking.budgetTokens",
			Message: "Moonshot Chat Completions does not support budget_tokens. Remove budgetTokens; the option has been omitted.",
		})
	}

	switch ModelFamily(inputs.modelID) {
	case FamilyKimiK3:
		settings.kimiK3(inputs)
	case FamilyKimiK27:
		settings.kimiK27(inputs)
	case FamilyKimiK26:
		settings.kimiK26(inputs)
	case FamilyKimiK25:
		settings.kimiK25(inputs)
	case FamilyMoonshotV:
		settings.moonshotV1(inputs)
	default:
		settings.custom(inputs)
	}

	return settings
}

// kimiK3 always reasons and accepts only an effort.
func (s *reasoningSettings) kimiK3(inputs reasoningInputs) {
	if inputs.options.Thinking != nil {
		s.warnings = append(s.warnings, model.UnsupportedWarning(
			"thinking",
			"Kimi K3 always reasons and does not accept the thinking field. The option has been omitted.",
		))
	}

	if inputs.level == model.ReasoningNone {
		s.warnings = append(s.warnings, model.UnsupportedWarning(
			`reasoning "none"`,
			"Kimi K3 reasoning cannot be disabled.",
		))
	}

	s.effort = s.mappedEffort(inputs)
}

// kimiK27 always thinks; thinking can only be enabled explicitly.
func (s *reasoningSettings) kimiK27(inputs reasoningInputs) {
	s.warnUnsupportedEffort(inputs)

	thinkingType := requestedThinkingType(inputs.options)

	switch {
	case thinkingType == ThinkingDisabled:
		s.warnings = append(s.warnings, model.UnsupportedWarning(
			`thinking.type "disabled"`,
			"Kimi K2.7 thinking cannot be disabled.",
		))

	case inputs.level == model.ReasoningNone:
		s.warnings = append(s.warnings, model.UnsupportedWarning(
			`reasoning "none"`,
			"Kimi K2.7 thinking cannot be disabled.",
		))

	case thinkingType == ThinkingEnabled:
		s.thinking = &thinkingRequest{
			Type: ThinkingEnabled,
		}
	}
}

// kimiK26 toggles thinking and can preserve all earlier reasoning.
func (s *reasoningSettings) kimiK26(inputs reasoningInputs) {
	s.warnUnsupportedEffort(inputs)

	thinkingType := thinkingTypeOf(inputs)
	preserve := preservesReasoning(inputs.options)

	if thinkingType == "" && !preserve {
		return
	}

	if thinkingType == "" {
		thinkingType = ThinkingEnabled
	}

	s.thinking = &thinkingRequest{
		Type: thinkingType,
	}

	if preserve {
		s.thinking.Keep = thinkingKeepAll
	}
}

// kimiK25 toggles thinking.
func (s *reasoningSettings) kimiK25(inputs reasoningInputs) {
	s.warnUnsupportedEffort(inputs)

	if thinkingType := thinkingTypeOf(inputs); thinkingType != "" {
		s.thinking = &thinkingRequest{
			Type: thinkingType,
		}
	}

	s.warnPreservedHistory(inputs)
}

// moonshotV1 does not reason.
func (s *reasoningSettings) moonshotV1(inputs reasoningInputs) {
	s.warnUnsupportedEffort(inputs)

	if inputs.options.Thinking != nil {
		s.warnings = append(s.warnings, model.UnsupportedWarning(
			"thinking",
			fmt.Sprintf("thinking is not supported by model \"%s\" and has been omitted.", inputs.modelID),
		))
	}

	if reasoning.IsCustom(inputs.level) && inputs.level != model.ReasoningNone {
		s.warnings = append(s.warnings, model.UnsupportedWarning(
			"reasoning",
			fmt.Sprintf("reasoning is not supported by model \"%s\".", inputs.modelID),
		))
	}

	s.warnPreservedHistory(inputs)
}

// custom forwards an effort and an explicit thinking type.
func (s *reasoningSettings) custom(inputs reasoningInputs) {
	if inputs.level == model.ReasoningNone {
		s.warnings = append(s.warnings, model.UnsupportedWarning(
			`reasoning "none"`,
			"Use providerOptions.moonshotai.thinking to control thinking on custom models.",
		))
	}

	s.effort = s.mappedEffort(inputs)

	if thinkingType := requestedThinkingType(inputs.options); thinkingType != "" {
		s.thinking = &thinkingRequest{
			Type: thinkingType,
		}
	}

	s.warnPreservedHistory(inputs)
}

// mappedEffort prefers the requested effort and otherwise maps the
// reasoning level.
func (s *reasoningSettings) mappedEffort(inputs reasoningInputs) *ReasoningEffort {
	if inputs.options.ReasoningEffort != nil {
		return inputs.options.ReasoningEffort
	}

	if !reasoning.IsCustom(inputs.level) || inputs.level == model.ReasoningNone {
		return nil
	}

	effort, warnings, ok := reasoning.Effort(inputs.level, effortMap)
	s.warnings = append(s.warnings, warnings...)

	if !ok {
		return nil
	}

	return &effort
}

func (s *reasoningSettings) warnUnsupportedEffort(inputs reasoningInputs) {
	if inputs.options.ReasoningEffort == nil {
		return
	}

	s.warnings = append(s.warnings, model.UnsupportedWarning(
		"reasoningEffort",
		fmt.Sprintf("reasoningEffort is only supported by Kimi K3 and has been omitted for model \"%s\".", inputs.modelID),
	))
}

func (s *reasoningSettings) warnPreservedHistory(inputs reasoningInputs) {
	if !preservesReasoning(inputs.options) {
		return
	}

	s.warnings = append(s.warnings, model.UnsupportedWarning(
		fmt.Sprintf("reasoningHistory 'preserved' is not supported by model \"%s\"", inputs.modelID),
		"",
	))
}

// thinkingTypeOf prefers the requested thinking type and otherwise derives
// it from a custom reasoning level.
func thinkingTypeOf(inputs reasoningInputs) ThinkingType {
	if thinkingType := requestedThinkingType(inputs.options); thinkingType != "" {
		return thinkingType
	}

	if !reasoning.IsCustom(inputs.level) {
		return ""
	}

	if inputs.level == model.ReasoningNone {
		return ThinkingDisabled
	}

	return ThinkingEnabled
}

func requestedThinkingType(options ChatOptions) ThinkingType {
	if options.Thinking == nil || options.Thinking.Type == nil {
		return ""
	}

	return *options.Thinking.Type
}

func preservesReasoning(options ChatOptions) bool {
	return options.ReasoningHistory != nil && *options.ReasoningHistory == ReasoningHistoryPreserved
}
