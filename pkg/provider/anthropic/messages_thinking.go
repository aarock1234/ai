package anthropic

import (
	"fmt"

	"github.com/aarock1234/ai/internal/reasoning"
	"github.com/aarock1234/ai/pkg/model"
)

const defaultThinkingBudget = 1024

// thinkingPolicy holds what the thinking adjustments need to know about the
// model.
type thinkingPolicy struct {
	modelID      string
	capabilities modelCapabilities
	warnings     *callWarnings
}

// applyReasoning maps the provider-independent reasoning level to thinking
// and effort. Provider options take precedence: nothing changes when they
// set an effort.
func (p thinkingPolicy) applyReasoning(options *MessagesOptions, level model.ReasoningEffort) {
	if !reasoning.IsCustom(level) || options.Effort != "" {
		return
	}

	thinking, effort, ok := p.reasoningConfig(level)
	if !ok {
		return
	}

	if options.Thinking == nil {
		options.Thinking = thinking
	}

	if effort != "" && thinkingTypeOf(options.Thinking) != ThinkingDisabled {
		options.Effort = effort
	}
}

func (p thinkingPolicy) reasoningConfig(level model.ReasoningEffort) (*ThinkingOptions, Effort, bool) {
	if level == model.ReasoningNone {
		return p.noReasoningConfig()
	}

	if p.capabilities.supportsAdaptiveThinking {
		thinking := &ThinkingOptions{
			Type:    ThinkingAdaptive,
			Display: ThinkingDisplaySummarized,
		}

		return thinking, p.reasoningEffort(level), true
	}

	budget, warnings, ok := reasoning.Budget(level, reasoning.BudgetLimits{
		MaxOutputTokens: p.capabilities.maxOutputTokens,
		MinBudget:       reasoning.DefaultMinBudget,
		MaxBudget:       p.capabilities.maxOutputTokens,
	})
	p.warnings.add(warnings...)

	if !ok {
		return nil, "", false
	}

	thinking := &ThinkingOptions{
		Type:         ThinkingEnabled,
		BudgetTokens: new(budget),
	}

	return thinking, "", true
}

// noReasoningConfig turns thinking off, or keeps it as small as the model
// allows when it cannot be disabled.
func (p thinkingPolicy) noReasoningConfig() (*ThinkingOptions, Effort, bool) {
	if p.capabilities.supportsBetweenToolsThinking {
		thinking := &ThinkingOptions{
			Type: ThinkingBetweenTools,
		}

		return thinking, "", true
	}

	if p.capabilities.rejectsThinkingDisabled {
		p.warnings.compatibility(
			"reasoning",
			fmt.Sprintf(
				"reasoning 'none' is not supported by %s; it always uses adaptive thinking. Using effort 'low' to minimize thinking instead.",
				p.modelID,
			),
		)

		return nil, EffortLow, true
	}

	thinking := &ThinkingOptions{
		Type: ThinkingDisabled,
	}

	return thinking, "", true
}

// reasoningEffort maps a reasoning level to an effort of an adaptive
// thinking model.
func (p thinkingPolicy) reasoningEffort(level model.ReasoningEffort) Effort {
	xhigh := EffortMax
	if p.capabilities.supportsXHighEffort {
		xhigh = EffortXHigh
	}

	effort, warnings, _ := reasoning.Effort(level, map[model.ReasoningEffort]Effort{
		model.ReasoningMinimal: EffortLow,
		model.ReasoningLow:     EffortLow,
		model.ReasoningMedium:  EffortMedium,
		model.ReasoningHigh:    EffortHigh,
		model.ReasoningXHigh:   xhigh,
		model.ReasoningMax:     EffortMax,
	})
	p.warnings.add(warnings...)

	return effort
}

// normalize replaces thinking and effort settings that the model would
// reject with the closest accepted ones, warning about each change.
func (p thinkingPolicy) normalize(options *MessagesOptions) {
	p.replaceDisabledThinking(options)
	p.replaceBudgetThinking(options)
	p.lowerEffortForDisabledThinking(options)
	p.lowerEffortForBetweenTools(options)
}

// replaceDisabledThinking handles models that always think adaptively: they
// use between_tools thinking when available and otherwise drop the setting.
func (p thinkingPolicy) replaceDisabledThinking(options *MessagesOptions) {
	if !p.capabilities.rejectsThinkingDisabled || thinkingTypeOf(options.Thinking) != ThinkingDisabled {
		return
	}

	if p.capabilities.supportsBetweenToolsThinking {
		p.warnings.unsupported(
			"providerOptions.anthropic.thinking",
			fmt.Sprintf(
				"thinking cannot be disabled for %s. Using 'between_tools' thinking, the lowest thinking setting, instead.",
				p.modelID,
			),
		)

		options.Thinking = &ThinkingOptions{
			Type: ThinkingBetweenTools,
		}

		return
	}

	p.warnings.unsupported(
		"providerOptions.anthropic.thinking",
		fmt.Sprintf(
			"thinking cannot be disabled for %s; it always uses adaptive thinking. The thinking setting has been removed. Lower 'effort' to reduce thinking.",
			p.modelID,
		),
	)

	options.Thinking = nil
}

// replaceBudgetThinking switches budget thinking to adaptive thinking on
// models that reject budgets.
func (p thinkingPolicy) replaceBudgetThinking(options *MessagesOptions) {
	if !p.capabilities.rejectsBudgetThinking || thinkingTypeOf(options.Thinking) != ThinkingEnabled {
		return
	}

	details := fmt.Sprintf(
		"budget-based thinking is not supported by %s. Using adaptive thinking instead. Use 'effort' to control how much the model thinks.",
		p.modelID,
	)
	if p.capabilities.rejectsThinkingDisabled {
		details = fmt.Sprintf(
			"budget-based thinking is not supported by %s; it always uses adaptive thinking. Using adaptive thinking instead. Use 'effort' to control how much the model thinks.",
			p.modelID,
		)
	}

	p.warnings.unsupported("providerOptions.anthropic.thinking", details)

	options.Thinking = &ThinkingOptions{
		Type: ThinkingAdaptive,
	}
}

// lowerEffortForDisabledThinking keeps an explicit request to run without
// thinking on models that only allow it up to high effort.
func (p thinkingPolicy) lowerEffortForDisabledThinking(options *MessagesOptions) {
	if !p.capabilities.rejectsThinkingDisabledAboveHighEffort {
		return
	}

	if thinkingTypeOf(options.Thinking) != ThinkingDisabled || !isAboveHighEffort(options.Effort) {
		return
	}

	p.warnings.unsupported(
		"providerOptions.anthropic.effort",
		fmt.Sprintf(
			"effort '%s' is not supported by %s when thinking is disabled. The effort has been lowered to 'high'.",
			options.Effort,
			p.modelID,
		),
	)

	options.Effort = EffortHigh
}

// lowerEffortForBetweenTools keeps between_tools thinking, which accepts at
// most high effort.
func (p thinkingPolicy) lowerEffortForBetweenTools(options *MessagesOptions) {
	if thinkingTypeOf(options.Thinking) != ThinkingBetweenTools || !isAboveHighEffort(options.Effort) {
		return
	}

	p.warnings.unsupported(
		"providerOptions.anthropic.effort",
		fmt.Sprintf(
			"effort '%s' is not supported with 'between_tools' thinking. The effort has been lowered to 'high'.",
			options.Effort,
		),
	)

	options.Effort = EffortHigh
}

func isAboveHighEffort(effort Effort) bool {
	return effort == EffortXHigh || effort == EffortMax
}

func thinkingTypeOf(thinking *ThinkingOptions) ThinkingType {
	if thinking == nil {
		return ""
	}

	return thinking.Type
}

// thinkingRequest is the resolved thinking configuration of a request.
type thinkingRequest struct {
	config     *thinkingConfig
	isThinking bool
	budget     int
	display    ThinkingDisplay
	hasBinding bool
}

// resolveThinking builds the thinking object of the request. Disabled
// thinking is sent explicitly, because some models think by default, and a
// binding-only recovery request sends a thinking object without a type.
func resolveThinking(thinking *ThinkingOptions, warnings *callWarnings) thinkingRequest {
	if thinking == nil {
		return thinkingRequest{}
	}

	thinkingType := thinking.Type
	resolved := thinkingRequest{
		isThinking: isThinkingType(thinkingType),
		hasBinding: hasBlockBinding(thinking),
	}

	sendThinking := resolved.isThinking || thinkingType == ThinkingDisabled || resolved.hasBinding
	if !sendThinking {
		return resolved
	}

	config := &thinkingConfig{
		Type: thinkingType,
	}

	if thinkingType == ThinkingEnabled {
		budget := thinking.BudgetTokens
		if budget == nil {
			warnings.compatibility(
				"extended thinking",
				"thinking budget is required when thinking is enabled. using default budget of 1024 tokens.",
			)

			budget = new(defaultThinkingBudget)
		}

		config.BudgetTokens = budget
		resolved.budget = *budget
	}

	if thinkingType == ThinkingAdaptive {
		config.Display = thinking.Display
		resolved.display = thinking.Display
	}

	if resolved.hasBinding {
		config.BlockBinding = &blockBindingWire{
			PrefixMismatchBehavior: thinking.BlockBinding.PrefixMismatchBehavior,
		}
	}

	resolved.config = config

	return resolved
}

func isThinkingType(thinkingType ThinkingType) bool {
	switch thinkingType {
	case ThinkingEnabled, ThinkingAdaptive, ThinkingBetweenTools:
		return true
	default:
		return false
	}
}

// hasBlockBinding reports whether binding controls apply: they are part of
// adaptive thinking and of type-less recovery requests only.
func hasBlockBinding(thinking *ThinkingOptions) bool {
	if thinking.BlockBinding == nil {
		return false
	}

	return thinking.Type == ThinkingAdaptive || thinking.Type == ""
}
