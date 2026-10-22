package harness

import (
	"encoding/json"
	"fmt"
	"slices"
)

// ToolSpec describes an application tool the runtime should offer. The
// adapter translates it into the runtime's tool format; it does not execute
// the tool but emits a ToolCallPart and waits for the agent's result.
type ToolSpec struct {
	// Name is the tool name the runtime sees and reports on tool calls.
	Name string `json:"name"`

	Description string `json:"description,omitempty"`

	// InputSchema is the JSON Schema of the input. It is empty for tools
	// without a schema.
	InputSchema json.RawMessage `json:"inputSchema,omitempty"`
}

// Skill is a self-contained instruction bundle the runtime can load into its
// context. Adapters decide how to surface skills; see WriteSkills.
type Skill struct {
	// Name is the stable kebab-case identifier of the skill.
	Name string `json:"name"`

	// Description is what the runtime sees to decide whether the skill is
	// relevant.
	Description string `json:"description"`

	// Content is the full skill content loaded when the skill is active.
	Content string `json:"content"`

	// Files are additional files of the skill. Adapters with native skill
	// directories write them next to SKILL.md; others include them with the
	// content.
	Files []SkillFile `json:"files,omitempty"`
}

// SkillFile is one additional file of a skill.
type SkillFile struct {
	// Path is a skill-relative POSIX path such as "references/codes.md".
	// Absolute paths and ".." segments are rejected.
	Path string `json:"path"`

	// Content is the UTF-8 text of the file.
	Content string `json:"content"`
}

// BuiltinToolFilteringMode selects how BuiltinToolFiltering lists tools.
type BuiltinToolFilteringMode string

// Built-in tool filtering modes.
const (
	// FilterAllow keeps only the listed tools.
	FilterAllow BuiltinToolFilteringMode = "allow"

	// FilterDeny removes the listed tools.
	FilterDeny BuiltinToolFilteringMode = "deny"
)

// BuiltinToolFiltering selects the built-in tools available in a session.
type BuiltinToolFiltering struct {
	Mode      BuiltinToolFilteringMode `json:"mode"`
	ToolNames []string                 `json:"toolNames"`
}

// IsBuiltinToolIncluded reports whether filtering keeps the named tool. Nil
// filtering keeps every tool.
func IsBuiltinToolIncluded(toolName string, filtering *BuiltinToolFiltering) bool {
	if filtering == nil {
		return true
	}

	listed := slices.Contains(filtering.ToolNames, toolName)
	if filtering.Mode == FilterAllow {
		return listed
	}

	return !listed
}

// BuiltinToolFilteringDenialReason is the reason reported to the runtime
// when the agent denies a call of an inactive built-in tool.
func BuiltinToolFilteringDenialReason(toolName string) string {
	return fmt.Sprintf("Tool '%s' is inactive due to the HarnessAgent tool filtering policy.", toolName)
}
