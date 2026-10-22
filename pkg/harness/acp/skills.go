package acp

import (
	"fmt"
	"path"
	"regexp"
	"slices"
	"strings"

	"github.com/aarock1234/ai/pkg/harness"
)

// defaultSkillsDirectory is the native skills directory relative to the
// implementation's home directory.
const defaultSkillsDirectory = ".agents/skills"

// skillNamePattern accepts kebab-case skill names.
var skillNamePattern = regexp.MustCompile(`^[a-z0-9]+(?:-[a-z0-9]+)*$`)

// resolveSkillsDirectory validates the relative skills directory and
// returns it with its absolute form below homeDir.
func resolveSkillsDirectory(homeDir string, skillsDirectory string) (string, string, error) {
	if skillsDirectory == "" {
		skillsDirectory = defaultSkillsDirectory
	}

	normalized := path.Clean(skillsDirectory)
	if !isRelativeWithoutTraversal(skillsDirectory, normalized) {
		return "", "", fmt.Errorf("acp skillsDirectory %q must be a relative POSIX path without traversal", skillsDirectory)
	}

	return skillsDirectory, path.Join(homeDir, normalized), nil
}

// validateSkills checks the skill names and attached file paths.
func validateSkills(skills []harness.Skill) error {
	var names []string

	for _, skill := range skills {
		if !skillNamePattern.MatchString(skill.Name) {
			return fmt.Errorf("invalid acp skill name %q: expected a kebab-case slug", skill.Name)
		}

		if slices.Contains(names, skill.Name) {
			return fmt.Errorf("duplicate acp skill name %q", skill.Name)
		}

		names = append(names, skill.Name)

		if err := validateSkillFiles(skill); err != nil {
			return err
		}
	}

	return nil
}

// validateSkillFiles checks the attached files of one skill.
func validateSkillFiles(skill harness.Skill) error {
	var filePaths []string

	for _, file := range skill.Files {
		normalized := path.Clean(file.Path)

		if strings.HasSuffix(file.Path, "/") || !isRelativeWithoutTraversal(file.Path, normalized) {
			return fmt.Errorf(
				"invalid acp skill file path %q for skill %q: expected a relative POSIX path without traversal",
				file.Path,
				skill.Name,
			)
		}

		if normalized == "SKILL.md" {
			return fmt.Errorf(
				"invalid acp skill file path %q for skill %q: SKILL.md is reserved for the skill definition",
				file.Path,
				skill.Name,
			)
		}

		if slices.Contains(filePaths, normalized) {
			return fmt.Errorf("duplicate acp skill file path %q for skill %q", file.Path, skill.Name)
		}

		filePaths = append(filePaths, normalized)
	}

	return nil
}

// isRelativeWithoutTraversal reports a non-empty relative POSIX path
// without backslashes, drive letters or parent references.
func isRelativeWithoutTraversal(raw string, normalized string) bool {
	switch {
	case raw == "",
		strings.Contains(raw, `\`),
		path.IsAbs(raw),
		isWindowsAbsolute(raw),
		hasParentSegment(raw),
		normalized == ".",
		normalized == "..",
		strings.HasPrefix(normalized, "../"):
		return false
	default:
		return true
	}
}

// hasParentSegment reports a ".." segment.
func hasParentSegment(value string) bool {
	return slices.Contains(strings.Split(value, "/"), "..")
}

// isWindowsAbsolute reports an absolute Windows drive path.
func isWindowsAbsolute(value string) bool {
	return len(value) >= 3 && value[1] == ':' && (value[2] == '/' || value[2] == '\\')
}
