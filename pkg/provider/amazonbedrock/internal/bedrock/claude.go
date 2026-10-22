package bedrock

import (
	"regexp"
	"slices"
	"strings"
)

var (
	// legacyClaudePattern matches Claude models before Claude 4.
	legacyClaudePattern = regexp.MustCompile(`claude-(?:instant|v?2|3)(?:[-.:]|$)`)

	// strictToolPatterns match the Claude models that accept strict tools
	// on Bedrock.
	strictToolPatterns = []*regexp.Regexp{
		// Original Claude 4 IDs include a release date instead of a minor
		// version.
		regexp.MustCompile(`claude-(?:opus|sonnet)-4(?:-\d{8})?(?:-v\d+)?(?:[:.]|$)`),
		regexp.MustCompile(`claude-opus-4-(?:1|5|6)(?:[-.:]|$)`),
		regexp.MustCompile(`claude-sonnet-4-(?:5|6)(?:[-.:]|$)`),
		regexp.MustCompile(`claude-haiku-4-5(?:[-.:]|$)`),
	}

	// noNativeOutputPattern matches Claude models that accept strict tools
	// but reject native structured output on Bedrock.
	noNativeOutputPattern = regexp.MustCompile(`claude-(?:sonnet-4-6|haiku-4-5)(?:[-.:]|$)`)
)

// SupportsStrictTools reports whether Bedrock accepts strict tools for a
// model. Bedrock validates its own Messages schema and rejects strict tools
// for new Claude generations until it supports them; non-Claude models and
// opaque inference profile IDs keep their behavior.
func SupportsStrictTools(modelID string) bool {
	if !strings.Contains(modelID, "claude-") || legacyClaudePattern.MatchString(modelID) {
		return true
	}

	return slices.ContainsFunc(strictToolPatterns, func(pattern *regexp.Regexp) bool {
		return pattern.MatchString(modelID)
	})
}

// SupportsNativeStructuredOutput reports whether Bedrock accepts
// output_config.format for a model.
func SupportsNativeStructuredOutput(modelID string) bool {
	return SupportsStrictTools(modelID) && !noNativeOutputPattern.MatchString(modelID)
}
