package anthropic

import (
	"regexp"
	"strings"
)

// modelCapabilities are the features of a Claude model that drive defaults
// and request adjustments.
type modelCapabilities struct {
	maxOutputTokens           int
	supportsStructuredOutput  bool
	supportsAdaptiveThinking  bool
	rejectsSamplingParameters bool
	supportsXHighEffort       bool

	// rejectsThinkingDisabledAboveHighEffort means disabled thinking is
	// rejected at xhigh and max effort.
	rejectsThinkingDisabledAboveHighEffort bool

	// rejectsThinkingDisabled means thinking is always adaptive: disabled
	// and enabled thinking are rejected.
	rejectsThinkingDisabled bool

	// rejectsBudgetThinking means budget-based thinking is rejected.
	rejectsBudgetThinking bool

	// rejectsForcedToolUse means tool_choice any or a named tool is
	// rejected.
	rejectsForcedToolUse bool

	// supportsBetweenToolsThinking means between_tools is the lowest
	// thinking setting.
	supportsBetweenToolsThinking bool

	// supportsStrictTools means function tools accept strict. It follows
	// supportsStructuredOutput unless a host restricts it.
	supportsStrictTools bool

	isKnownModel bool
}

// capabilityRule assigns capabilities to the model IDs it matches. Rules are
// checked in order.
type capabilityRule struct {
	matches      func(modelID string) bool
	capabilities modelCapabilities
}

var (
	sonnet4Pattern = regexp.MustCompile(`claude-sonnet-4(?:-|@)`)
	opus4Pattern   = regexp.MustCompile(`claude-opus-4(?:-|@)`)
	legacyPattern  = regexp.MustCompile(`claude-(?:instant(?:-|$)|v?2(?:$|[-.:])|3(?:$|[-.]))`)
)

// adaptiveOnlyModel always thinks adaptively and rejects sampling
// parameters.
var adaptiveOnlyModel = modelCapabilities{
	maxOutputTokens:                        128000,
	supportsStructuredOutput:               true,
	supportsAdaptiveThinking:               true,
	rejectsSamplingParameters:              true,
	supportsXHighEffort:                    true,
	rejectsThinkingDisabledAboveHighEffort: true,
	rejectsThinkingDisabled:                true,
	rejectsBudgetThinking:                  true,
	rejectsForcedToolUse:                   true,
	isKnownModel:                           true,
}

// modernModel supports adaptive and budget thinking and rejects sampling
// parameters.
var modernModel = modelCapabilities{
	maxOutputTokens:           128000,
	supportsStructuredOutput:  true,
	supportsAdaptiveThinking:  true,
	rejectsSamplingParameters: true,
	supportsXHighEffort:       true,
	isKnownModel:              true,
}

var capabilityRules = []capabilityRule{
	{
		matches: containing("claude-sonnet-5-5"),
		capabilities: with(adaptiveOnlyModel, func(c *modelCapabilities) {
			c.supportsBetweenToolsThinking = true
		}),
	},
	{
		matches:      containing("claude-opus-5-5"),
		capabilities: adaptiveOnlyModel,
	},
	{
		matches: containing("claude-haiku-5-5"),
		capabilities: with(modernModel, func(c *modelCapabilities) {
			c.rejectsThinkingDisabledAboveHighEffort = true
			c.rejectsBudgetThinking = true
		}),
	},
	{
		matches: containing("claude-opus-5"),
		capabilities: with(modernModel, func(c *modelCapabilities) {
			c.rejectsThinkingDisabledAboveHighEffort = true
		}),
	},
	{
		matches: containing("claude-fable-5-1"),
		capabilities: with(adaptiveOnlyModel, func(c *modelCapabilities) {
			c.rejectsThinkingDisabledAboveHighEffort = false
		}),
	},
	{
		matches: containing("claude-fable-5"),
		capabilities: with(adaptiveOnlyModel, func(c *modelCapabilities) {
			c.rejectsThinkingDisabledAboveHighEffort = false
			c.rejectsForcedToolUse = false
		}),
	},
	{
		matches:      containing("claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-5"),
		capabilities: modernModel,
	},
	{
		matches: containing("claude-sonnet-4-6", "claude-opus-4-6"),
		capabilities: modelCapabilities{
			maxOutputTokens:          128000,
			supportsStructuredOutput: true,
			supportsAdaptiveThinking: true,
			isKnownModel:             true,
		},
	},
	{
		matches: containing("claude-sonnet-4-5", "claude-opus-4-5", "claude-haiku-4-5"),
		capabilities: modelCapabilities{
			maxOutputTokens:          64000,
			supportsStructuredOutput: true,
			isKnownModel:             true,
		},
	},
	{
		matches: containing("claude-opus-4-1"),
		capabilities: modelCapabilities{
			maxOutputTokens:          32000,
			supportsStructuredOutput: true,
			isKnownModel:             true,
		},
	},
	{
		matches: sonnet4Pattern.MatchString,
		capabilities: modelCapabilities{
			maxOutputTokens: 64000,
			isKnownModel:    true,
		},
	},
	{
		matches: opus4Pattern.MatchString,
		capabilities: modelCapabilities{
			maxOutputTokens: 32000,
			isKnownModel:    true,
		},
	},
	{
		matches: containing("claude-3-haiku"),
		capabilities: modelCapabilities{
			maxOutputTokens: 4096,
			isKnownModel:    true,
		},
	},
	{
		matches: legacyPattern.MatchString,
		capabilities: modelCapabilities{
			maxOutputTokens: 4096,
		},
	},
	{
		// Claude IDs not matched above are assumed to be newer than the known
		// list. They stay unknown so callers still see the max output tokens
		// warning.
		matches: containing("claude-"),
		capabilities: modelCapabilities{
			maxOutputTokens:                        128000,
			supportsStructuredOutput:               true,
			supportsAdaptiveThinking:               true,
			rejectsSamplingParameters:              true,
			supportsXHighEffort:                    true,
			rejectsThinkingDisabledAboveHighEffort: true,
		},
	},
}

// nonClaudeModel keeps conservative defaults for models served through
// Anthropic-compatible APIs.
var nonClaudeModel = modelCapabilities{
	maxOutputTokens: 4096,
}

// capabilitiesOf returns the capabilities of a model ID.
func capabilitiesOf(modelID string) modelCapabilities {
	capabilities := nonClaudeModel

	for _, rule := range capabilityRules {
		if rule.matches(modelID) {
			capabilities = rule.capabilities

			break
		}
	}

	capabilities.supportsStrictTools = capabilities.supportsStructuredOutput

	return capabilities
}

func containing(fragments ...string) func(string) bool {
	return func(modelID string) bool {
		for _, fragment := range fragments {
			if strings.Contains(modelID, fragment) {
				return true
			}
		}

		return false
	}
}

func with(base modelCapabilities, adjust func(*modelCapabilities)) modelCapabilities {
	adjust(&base)

	return base
}
