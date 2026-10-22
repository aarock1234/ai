package google

import (
	"regexp"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/internal/reasoning"
	"github.com/aarock1234/ai/pkg/model"
)

const (
	gemini25MaxOutputTokens      = 65536
	gemini25ProMaxThinkingBudget = 32768
	gemini25MaxThinkingBudget    = 24576
)

var (
	geminiModelNamePattern  = regexp.MustCompile(`(?i)(?:^|[/.])(gemini-[^/]*)$`)
	geminiProVersionPattern = regexp.MustCompile(`^gemini-(\d+)(?:\.\d+)?-pro(?:$|-)`)
	geminiFlashPattern      = regexp.MustCompile(`^gemini-(\d+)\.(\d+)-flash(?:$|-)`)
	geminiFlashLitePattern  = regexp.MustCompile(`^gemini-\d+\.\d+-flash-lite(?:-|$)`)
)

// thinkingConfig combines the provider-independent reasoning level with the
// thinking config from the provider options. An explicit budget or level in
// the options replaces the reasoning level; other option fields override
// the resolved values.
func thinkingConfig(level model.ReasoningEffort, configured *ThinkingConfig, modelID string) (*ThinkingConfig, []model.Warning) {
	explicit := configured != nil && (configured.ThinkingBudget != nil || configured.ThinkingLevel != "")

	var resolved *ThinkingConfig
	var warnings []model.Warning
	if !explicit {
		resolved, warnings = resolveThinkingConfig(level, modelID)
	}

	if resolved == nil {
		return configured, warnings
	}

	if configured == nil {
		return resolved, warnings
	}

	merged := *resolved
	merged.IncludeThoughts = configured.IncludeThoughts

	return &merged, warnings
}

func resolveThinkingConfig(level model.ReasoningEffort, modelID string) (*ThinkingConfig, []model.Warning) {
	if !reasoning.IsCustom(level) {
		return nil, nil
	}

	usesThinkingLevels := capabilitiesOf(modelID).usesGemini3Features && !strings.Contains(modelID, "gemini-3-pro-image")
	if usesThinkingLevels {
		return resolveThinkingLevel(level, modelID)
	}

	return resolveThinkingBudget(level, modelID)
}

// resolveThinkingLevel maps reasoning levels to Gemini 3 thinking levels.
// Gemini 3 thinking cannot be disabled, so "none" selects the lowest level.
func resolveThinkingLevel(level model.ReasoningEffort, modelID string) (*ThinkingConfig, []model.Warning) {
	minimumLevel := minimumThinkingLevel(modelID)

	if level == model.ReasoningNone {
		return &ThinkingConfig{
			ThinkingLevel: minimumLevel,
		}, nil
	}

	thinkingLevel, warnings, ok := reasoning.Effort(level, map[model.ReasoningEffort]ThinkingLevel{
		model.ReasoningMinimal: minimumLevel,
		model.ReasoningLow:     ThinkingLevelLow,
		model.ReasoningMedium:  ThinkingLevelMedium,
		model.ReasoningHigh:    ThinkingLevelHigh,
		model.ReasoningXHigh:   ThinkingLevelHigh,
		model.ReasoningMax:     ThinkingLevelHigh,
	})
	if !ok {
		return nil, warnings
	}

	return &ThinkingConfig{
		ThinkingLevel: thinkingLevel,
	}, warnings
}

// minimumThinkingLevel returns the lowest level a Gemini 3 model accepts.
// Pro models, Gemini Flash Latest and Flash models from 3.7 on start at low.
func minimumThinkingLevel(modelID string) ThinkingLevel {
	match := geminiModelNamePattern.FindStringSubmatch(modelID)
	if match == nil {
		return ThinkingLevelMinimal
	}

	name := strings.ToLower(match[1])
	if name == "gemini-flash-latest" {
		return ThinkingLevelLow
	}

	if pro := geminiProVersionPattern.FindStringSubmatch(name); pro != nil {
		if major, err := strconv.Atoi(pro[1]); err == nil && major >= 3 {
			return ThinkingLevelLow
		}
	}

	flash := geminiFlashPattern.FindStringSubmatch(name)
	if flash == nil || geminiFlashLitePattern.MatchString(name) {
		return ThinkingLevelMinimal
	}

	major, majorErr := strconv.Atoi(flash[1])
	minor, minorErr := strconv.Atoi(flash[2])
	if majorErr != nil || minorErr != nil {
		return ThinkingLevelMinimal
	}

	if major > 3 || (major == 3 && minor >= 7) {
		return ThinkingLevelLow
	}

	return ThinkingLevelMinimal
}

// resolveThinkingBudget maps reasoning levels to Gemini 2.5 thinking budgets.
func resolveThinkingBudget(level model.ReasoningEffort, modelID string) (*ThinkingConfig, []model.Warning) {
	if level == model.ReasoningNone {
		return &ThinkingConfig{
			ThinkingBudget: new(0),
		}, nil
	}

	budget, warnings, ok := reasoning.Budget(level, reasoning.BudgetLimits{
		MaxOutputTokens: gemini25MaxOutputTokens,
		MinBudget:       0,
		MaxBudget:       maxThinkingBudget(modelID),
	})
	if !ok {
		return nil, warnings
	}

	return &ThinkingConfig{
		ThinkingBudget: new(budget),
	}, nil
}

func maxThinkingBudget(modelID string) int {
	id := strings.ToLower(modelID)
	if strings.Contains(id, "2.5-pro") || strings.Contains(id, "gemini-3-pro-image") {
		return gemini25ProMaxThinkingBudget
	}

	return gemini25MaxThinkingBudget
}
