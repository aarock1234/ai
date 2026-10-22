package anthropic

import (
	"errors"
	"fmt"
	"slices"

	"github.com/aarock1234/ai/pkg/model"
)

var (
	efforts = []Effort{
		EffortLow,
		EffortMedium,
		EffortHigh,
		EffortXHigh,
		EffortMax,
	}

	speeds = []Speed{
		SpeedFast,
		SpeedStandard,
	}
)

// validate checks the enums and ranges that the upstream options schema
// enforces.
func (o MessagesOptions) validate() error {
	checks := []error{
		checkEnum("structuredOutputMode", o.StructuredOutputMode, StructuredOutputFormat, StructuredOutputJSONTool, StructuredOutputAuto),
		checkEnum("effort", o.Effort, efforts...),
		checkEnum("speed", o.Speed, speeds...),
		checkEnum("serviceTier", o.ServiceTier, ServiceTierAuto, ServiceTierStandardOnly),
		checkEnum("inferenceGeo", o.InferenceGeo, InferenceGeoUS, InferenceGeoGlobal),
		validateThinking(o.Thinking),
		validateCacheControl("cacheControl", o.CacheControl),
		validateMCPServers(o.MCPServers),
		validateContainer(o.Container),
		validateTaskBudget(o.TaskBudget),
		validateFallbacks(o.Fallbacks),
		validateSafeguards(o.Safeguards),
		validateCompaction(o.Compaction),
		validateContextManagement(o.ContextManagement),
		validateContextExclusivity(o),
	}

	for _, err := range checks {
		if err != nil {
			return fmt.Errorf("%w: anthropic provider options: %w", model.ErrInvalidArgument, err)
		}
	}

	return nil
}

func validateThinking(thinking *ThinkingOptions) error {
	if thinking == nil {
		return nil
	}

	if thinking.Type == "" && thinking.BlockBinding == nil {
		return errors.New("thinking requires a type or blockBinding")
	}

	err := checkEnum("thinking.type", thinking.Type, ThinkingAdaptive, ThinkingEnabled, ThinkingDisabled, ThinkingBetweenTools)
	if err != nil {
		return err
	}

	err = checkEnum("thinking.display", thinking.Display, ThinkingDisplayOmitted, ThinkingDisplaySummarized, ThinkingDisplayUpdates)
	if err != nil {
		return err
	}

	if thinking.BlockBinding == nil {
		return nil
	}

	return checkRequiredEnum(
		"thinking.blockBinding.prefixMismatchBehavior",
		thinking.BlockBinding.PrefixMismatchBehavior,
		PrefixMismatchError,
		PrefixMismatchDropBlock,
	)
}

func validateCacheControl(field string, cacheControl *CacheControl) error {
	if cacheControl == nil {
		return nil
	}

	if err := checkRequiredEnum(field+".type", cacheControl.Type, CacheEphemeral); err != nil {
		return err
	}

	return checkEnum(field+".ttl", cacheControl.TTL, CacheTTL5Minutes, CacheTTL1Hour)
}

func validateMCPServers(servers []MCPServer) error {
	for _, server := range servers {
		if server.Type != "url" {
			return fmt.Errorf("mcpServers type must be \"url\", got %q", server.Type)
		}
	}

	return nil
}

func validateContainer(container *ContainerOptions) error {
	if container == nil {
		return nil
	}

	for _, skill := range container.Skills {
		if err := checkRequiredEnum("container.skills.type", skill.Type, SkillAnthropic, SkillCustom); err != nil {
			return err
		}
	}

	return nil
}

func validateTaskBudget(budget *TaskBudget) error {
	if budget == nil {
		return nil
	}

	if budget.Type != "tokens" {
		return fmt.Errorf("taskBudget type must be \"tokens\", got %q", budget.Type)
	}

	if budget.Total < 20000 {
		return fmt.Errorf("taskBudget total must be at least 20000, got %d", budget.Total)
	}

	if budget.Remaining != nil && *budget.Remaining < 0 {
		return fmt.Errorf("taskBudget remaining must not be negative, got %d", *budget.Remaining)
	}

	return nil
}

func validateFallbacks(fallbacks *Fallbacks) error {
	if fallbacks == nil {
		return nil
	}

	for _, fallback := range fallbacks.Chain {
		if err := checkEnum("fallbacks.speed", fallback.Speed, speeds...); err != nil {
			return err
		}
	}

	return nil
}

func validateSafeguards(safeguards []Safeguard) error {
	for _, safeguard := range safeguards {
		if safeguard.Type != "dangerous_tool_use" {
			return fmt.Errorf("safeguards type must be \"dangerous_tool_use\", got %q", safeguard.Type)
		}
	}

	return nil
}

func validateCompaction(compaction *Compaction) error {
	if compaction == nil {
		return nil
	}

	if compaction.Type != "summarize" {
		return fmt.Errorf("compaction type must be \"summarize\", got %q", compaction.Type)
	}

	return nil
}

func validateContextManagement(management *ContextManagement) error {
	if management == nil {
		return nil
	}

	for _, edit := range management.Edits {
		err := checkRequiredEnum("contextManagement.edits.type", edit.Type, ContextEditClearToolUses, ContextEditClearThinking, ContextEditCompact)
		if err != nil {
			return err
		}
	}

	return nil
}

// validateContextExclusivity rejects compaction combined with context
// management; the API accepts only one of them.
func validateContextExclusivity(options MessagesOptions) error {
	if options.Compaction != nil && options.ContextManagement != nil {
		return errors.New("compaction and contextManagement cannot be used together")
	}

	return nil
}

func (o SystemMessageOptions) validate() error {
	if err := checkEnum("clearAt", o.ClearAt, ClearAtNextUserMessage); err != nil {
		return err
	}

	if err := checkEnum("effort", o.Effort, efforts...); err != nil {
		return err
	}

	for _, change := range o.ToolChanges {
		if err := checkRequiredEnum("toolChanges.type", change.Type, ToolAddition, ToolRemoval); err != nil {
			return err
		}
	}

	return nil
}

// checkEnum accepts the empty value or one of allowed.
func checkEnum[T ~string](field string, value T, allowed ...T) error {
	if value == "" {
		return nil
	}

	return checkRequiredEnum(field, value, allowed...)
}

// checkRequiredEnum accepts only one of allowed.
func checkRequiredEnum[T ~string](field string, value T, allowed ...T) error {
	if slices.Contains(allowed, value) {
		return nil
	}

	return fmt.Errorf("%s has unsupported value %q", field, value)
}
