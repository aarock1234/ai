package harness

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"path"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/internal/shellquote"
	"github.com/aarock1234/ai/pkg/sandbox"
)

// Skills manifest constants.
const (
	skillsManifestFilename = ".ai-sdk-harness-skills.json"
	skillsManifestVersion  = 1
)

// Manifest states. A pending manifest lists directories that may hold only
// part of their files.
const (
	manifestComplete = "complete"
	manifestPending  = "pending"
)

var (
	// defaultSkillNamePattern accepts the default skill names.
	defaultSkillNamePattern = regexp.MustCompile(`^[A-Za-z0-9._-]+$`)

	// skillHashPattern matches a manifest skill hash.
	skillHashPattern = regexp.MustCompile(`^[a-f0-9]{64}$`)
)

// SkillFilePathMode selects how skill file paths are validated.
type SkillFilePathMode string

// Skill file path modes. The empty mode is SkillPathsRelative.
const (
	// SkillPathsRelative requires clean relative paths.
	SkillPathsRelative SkillFilePathMode = "relative"

	// SkillPathsStripLeadingSlashes strips leading slashes, so absolute paths
	// become skill-relative.
	SkillPathsStripLeadingSlashes SkillFilePathMode = "strip-leading-slashes"
)

// WriteSkillsOptions configure WriteSkills.
type WriteSkillsOptions struct {
	Sandbox sandbox.Session

	// HomePath is the absolute directory the skills directory is relative to.
	HomePath string

	// SkillsDir is the relative directory of the runtime's skills, such as
	// ".claude/skills".
	SkillsDir string

	Skills []Skill

	// SkillNamePattern validates skill names. Nil accepts letters, digits,
	// dots, underscores and hyphens.
	SkillNamePattern *regexp.Regexp

	FilePathMode SkillFilePathMode

	// TrailingNewline ends each SKILL.md with a newline.
	TrailingNewline bool
}

// WriteSkillsResult reports the skill names WriteSkills changed.
type WriteSkillsResult struct {
	Changed   bool
	Written   []string
	Removed   []string
	Unchanged []string
}

// projectedSkill is a skill with the files to write and their hash.
type projectedSkill struct {
	name  string
	hash  string
	files []SkillFile
}

// skillsManifest records the skills the harness owns in a skills directory.
type skillsManifest struct {
	Version int                   `json:"version"`
	State   string                `json:"state"`
	Skills  []skillsManifestEntry `json:"skills"`
}

// skillsManifestEntry is one owned skill.
type skillsManifestEntry struct {
	Name string `json:"name"`
	Hash string `json:"hash"`
}

// WriteSkills materializes skills as SKILL.md directories in a runtime's
// skills directory. A manifest records the skills the harness owns, so that
// later calls rewrite only changed skills, remove skills that disappeared,
// and never overwrite directories the harness does not own. An interrupted
// write is recovered on the next call.
func WriteSkills(ctx context.Context, options WriteSkillsOptions) (WriteSkillsResult, error) {
	rootDir, err := resolveHomeSubpath(options.HomePath, options.SkillsDir, "skillsDir")
	if err != nil {
		return WriteSkillsResult{}, err
	}

	projected, err := projectSkills(options)
	if err != nil {
		return WriteSkillsResult{}, err
	}

	manifestPath := path.Join(rootDir, skillsManifestFilename)

	existing, err := readSkillsManifest(ctx, options.Sandbox, manifestPath)
	if err != nil {
		return WriteSkillsResult{}, err
	}

	writer := skillsWriter{
		session:      options.Sandbox,
		rootDir:      rootDir,
		manifestPath: manifestPath,
		skills:       projected,
	}

	if existing != nil && existing.State == manifestPending {
		return writer.recover(ctx, *existing)
	}

	return writer.update(ctx, existing)
}

// skillsWriter writes projected skills into one skills directory.
type skillsWriter struct {
	session      sandbox.Session
	rootDir      string
	manifestPath string
	skills       []projectedSkill
}

// recover rewrites every skill after an interrupted write, since each listed
// directory may hold only part of its files.
func (w skillsWriter) recover(ctx context.Context, pending skillsManifest) (WriteSkillsResult, error) {
	if err := w.ensureRoot(ctx); err != nil {
		return WriteSkillsResult{}, err
	}

	recoveryNames := manifestNames(pending.Skills)
	if err := w.removeDirectories(ctx, recoveryNames); err != nil {
		return WriteSkillsResult{}, err
	}

	if err := w.writeSkills(ctx, w.skills); err != nil {
		return WriteSkillsResult{}, err
	}

	if err := w.writeManifest(ctx, manifestComplete, w.entries()); err != nil {
		return WriteSkillsResult{}, err
	}

	var removed []string
	for _, name := range recoveryNames {
		if !w.hasSkill(name) {
			removed = append(removed, name)
		}
	}

	slices.Sort(removed)

	return WriteSkillsResult{
		Changed: len(recoveryNames) > 0 || len(w.skills) > 0,
		Written: w.names(),
		Removed: removed,
	}, nil
}

// update writes changed skills and removes skills that disappeared.
func (w skillsWriter) update(ctx context.Context, existing *skillsManifest) (WriteSkillsResult, error) {
	var previous []skillsManifestEntry
	if existing != nil {
		previous = existing.Skills
	}

	result := diffSkills(previous, w.skills)
	if !result.Changed && existing != nil {
		return result, nil
	}

	owned := manifestNames(previous)

	for _, name := range result.Written {
		if slices.Contains(owned, name) {
			continue
		}

		if err := w.assertAvailable(ctx, name); err != nil {
			return WriteSkillsResult{}, err
		}
	}

	if err := w.ensureRoot(ctx); err != nil {
		return WriteSkillsResult{}, err
	}

	if err := w.writeManifest(ctx, manifestPending, pendingEntries(previous, w.entries())); err != nil {
		return WriteSkillsResult{}, err
	}

	replaced := slices.Clone(result.Removed)
	for _, name := range result.Written {
		if slices.Contains(owned, name) {
			replaced = append(replaced, name)
		}
	}

	if err := w.removeDirectories(ctx, replaced); err != nil {
		return WriteSkillsResult{}, err
	}

	var written []projectedSkill
	for _, skill := range w.skills {
		if slices.Contains(result.Written, skill.name) {
			written = append(written, skill)
		}
	}

	if err := w.writeSkills(ctx, written); err != nil {
		return WriteSkillsResult{}, err
	}

	if err := w.writeManifest(ctx, manifestComplete, w.entries()); err != nil {
		return WriteSkillsResult{}, err
	}

	return result, nil
}

// diffSkills compares the owned skills with the projected ones.
func diffSkills(previous []skillsManifestEntry, skills []projectedSkill) WriteSkillsResult {
	previousHashes := make(map[string]string, len(previous))
	for _, entry := range previous {
		previousHashes[entry.Name] = entry.Hash
	}

	var result WriteSkillsResult

	for _, entry := range previous {
		if !slices.ContainsFunc(skills, func(skill projectedSkill) bool {
			return skill.name == entry.Name
		}) {
			result.Removed = append(result.Removed, entry.Name)
		}
	}

	for _, skill := range skills {
		if hash, owned := previousHashes[skill.name]; owned && hash == skill.hash {
			result.Unchanged = append(result.Unchanged, skill.name)
		} else {
			result.Written = append(result.Written, skill.name)
		}
	}

	slices.Sort(result.Removed)
	slices.Sort(result.Written)
	slices.Sort(result.Unchanged)

	result.Changed = len(result.Removed) > 0 || len(result.Written) > 0

	return result
}

// pendingEntries lists every skill a pending write may touch.
func pendingEntries(previous []skillsManifestEntry, next []skillsManifestEntry) []skillsManifestEntry {
	hashes := make(map[string]string)
	for _, entry := range previous {
		hashes[entry.Name] = entry.Hash
	}

	for _, entry := range next {
		hashes[entry.Name] = entry.Hash
	}

	names := make([]string, 0, len(hashes))
	for name := range hashes {
		names = append(names, name)
	}

	slices.Sort(names)

	entries := make([]skillsManifestEntry, 0, len(names))
	for _, name := range names {
		entries = append(entries, skillsManifestEntry{
			Name: name,
			Hash: hashes[name],
		})
	}

	return entries
}

// entries returns the manifest entries of the projected skills.
func (w skillsWriter) entries() []skillsManifestEntry {
	entries := make([]skillsManifestEntry, 0, len(w.skills))
	for _, skill := range w.skills {
		entries = append(entries, skillsManifestEntry{
			Name: skill.name,
			Hash: skill.hash,
		})
	}

	return entries
}

// names returns the projected skill names in order.
func (w skillsWriter) names() []string {
	names := make([]string, 0, len(w.skills))
	for _, skill := range w.skills {
		names = append(names, skill.name)
	}

	return names
}

// hasSkill reports whether a skill is projected.
func (w skillsWriter) hasSkill(name string) bool {
	return slices.Contains(w.names(), name)
}

// ensureRoot creates the skills directory.
func (w skillsWriter) ensureRoot(ctx context.Context) error {
	return runSandboxCommand(ctx, w.session, "mkdir -p "+shellquote.Quote(w.rootDir), "create skills directory "+w.rootDir)
}

// assertAvailable fails when a directory the harness does not own exists.
func (w skillsWriter) assertAvailable(ctx context.Context, name string) error {
	skillDir := path.Join(w.rootDir, name)

	result, err := w.session.Run(ctx, sandbox.ProcessOptions{
		Command: "test ! -e " + shellquote.Quote(skillDir),
	})
	if err != nil {
		return fmt.Errorf("check skill directory %s: %w", skillDir, err)
	}

	if result.ExitCode != 0 {
		return fmt.Errorf("cannot write harness skill %q: %s already exists and is not owned by the ai sdk harness", name, skillDir)
	}

	return nil
}

// removeDirectories removes skill directories.
func (w skillsWriter) removeDirectories(ctx context.Context, names []string) error {
	if len(names) == 0 {
		return nil
	}

	unique := slices.Clone(names)
	slices.Sort(unique)
	unique = slices.Compact(unique)

	quoted := make([]string, 0, len(unique))
	for _, name := range unique {
		quoted = append(quoted, shellquote.Quote(path.Join(w.rootDir, name)))
	}

	return runSandboxCommand(ctx, w.session, "rm -rf -- "+strings.Join(quoted, " "), "replace harness skills in "+w.rootDir)
}

// writeSkills writes the files of skills.
func (w skillsWriter) writeSkills(ctx context.Context, skills []projectedSkill) error {
	for _, skill := range skills {
		skillDir := path.Join(w.rootDir, skill.name)

		for _, file := range skill.files {
			err := w.session.WriteTextFile(ctx, sandbox.WriteTextFileOptions{
				Path:    path.Join(skillDir, file.Path),
				Content: file.Content,
			})
			if err != nil {
				return fmt.Errorf("write skill file %s of %q: %w", file.Path, skill.name, err)
			}
		}
	}

	return nil
}

// writeManifest replaces the manifest atomically.
func (w skillsWriter) writeManifest(ctx context.Context, state string, entries []skillsManifestEntry) error {
	return writeJSONAtomically(ctx, w.session, w.manifestPath, skillsManifest{
		Version: skillsManifestVersion,
		State:   state,
		Skills:  entries,
	})
}

// projectSkills validates the skills and renders their files, sorted by
// name.
func projectSkills(options WriteSkillsOptions) ([]projectedSkill, error) {
	pattern := options.SkillNamePattern
	if pattern == nil {
		pattern = defaultSkillNamePattern
	}

	projected := make([]projectedSkill, 0, len(options.Skills))

	for _, skill := range options.Skills {
		skillProjection, err := projectSkill(skill, pattern, options)
		if err != nil {
			return nil, err
		}

		projected = append(projected, skillProjection)
	}

	slices.SortFunc(projected, func(first, second projectedSkill) int {
		return strings.Compare(first.name, second.name)
	})

	for index := 1; index < len(projected); index++ {
		if projected[index-1].name == projected[index].name {
			return nil, fmt.Errorf("duplicate skill name: %s", projected[index].name)
		}
	}

	return projected, nil
}

// projectSkill renders SKILL.md and the additional files of one skill and
// hashes them.
func projectSkill(skill Skill, pattern *regexp.Regexp, options WriteSkillsOptions) (projectedSkill, error) {
	if !pattern.MatchString(skill.Name) || skill.Name == "." || skill.Name == ".." {
		return projectedSkill{}, fmt.Errorf("invalid skill name: %s", skill.Name)
	}

	files := map[string]string{
		"SKILL.md": renderSkillFile(skill, options.TrailingNewline),
	}

	for _, file := range skill.Files {
		filePath, err := normalizeSkillFilePath(file.Path, options.FilePathMode)
		if err != nil {
			return projectedSkill{}, fmt.Errorf("invalid skill file path for %s: %s", skill.Name, file.Path)
		}

		files[filePath] = file.Content
	}

	paths := make([]string, 0, len(files))
	for filePath := range files {
		paths = append(paths, filePath)
	}

	slices.Sort(paths)

	digest := sha256.New()
	projectedFiles := make([]SkillFile, 0, len(paths))

	for _, filePath := range paths {
		content := files[filePath]
		digest.Write([]byte(strconv.Itoa(len(filePath)) + ":" + filePath))
		digest.Write([]byte(strconv.Itoa(len(content)) + ":" + content))

		projectedFiles = append(projectedFiles, SkillFile{
			Path:    filePath,
			Content: content,
		})
	}

	return projectedSkill{
		name:  skill.Name,
		hash:  hex.EncodeToString(digest.Sum(nil)),
		files: projectedFiles,
	}, nil
}

// renderSkillFile renders SKILL.md with its front matter.
func renderSkillFile(skill Skill, trailingNewline bool) string {
	content := fmt.Sprintf("---\nname: %s\ndescription: %s\n---\n\n%s", skill.Name, skill.Description, skill.Content)
	if trailingNewline {
		content += "\n"
	}

	return content
}

// normalizeSkillFilePath validates a skill-relative file path.
func normalizeSkillFilePath(filePath string, mode SkillFilePathMode) (string, error) {
	normalized := path.Clean(filePath)
	if mode == SkillPathsStripLeadingSlashes {
		normalized = strings.TrimLeft(filePath, "/")
	}

	invalid := normalized == "" ||
		(mode != SkillPathsStripLeadingSlashes && (normalized == "." || path.IsAbs(normalized))) ||
		normalized == ".." ||
		strings.HasPrefix(normalized, "../") ||
		strings.Contains(normalized, "/../") ||
		strings.HasSuffix(normalized, "/..")

	if invalid {
		return "", errors.New("invalid skill file path")
	}

	return normalized, nil
}

// readSkillsManifest reads the manifest, returning nil when it is missing.
func readSkillsManifest(ctx context.Context, session sandbox.Session, manifestPath string) (*skillsManifest, error) {
	content, err := session.ReadTextFile(ctx, sandbox.ReadTextFileOptions{
		Path: manifestPath,
	})
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}

	if err != nil {
		return nil, fmt.Errorf("read skills manifest %s: %w", manifestPath, err)
	}

	var manifest skillsManifest
	if err := json.Unmarshal([]byte(content), &manifest); err != nil || !validSkillsManifest(manifest) {
		return nil, fmt.Errorf("invalid ai sdk harness skills manifest: %s", manifestPath)
	}

	return &manifest, nil
}

// validSkillsManifest checks a decoded manifest.
func validSkillsManifest(manifest skillsManifest) bool {
	if manifest.Version != skillsManifestVersion {
		return false
	}

	if manifest.State != manifestComplete && manifest.State != manifestPending {
		return false
	}

	seen := make(map[string]bool, len(manifest.Skills))

	for _, entry := range manifest.Skills {
		safeName := defaultSkillNamePattern.MatchString(entry.Name) && entry.Name != "." && entry.Name != ".."

		if !safeName || !skillHashPattern.MatchString(entry.Hash) || seen[entry.Name] {
			return false
		}

		seen[entry.Name] = true
	}

	return true
}

// manifestNames returns the names of manifest entries.
func manifestNames(entries []skillsManifestEntry) []string {
	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		names = append(names, entry.Name)
	}

	return names
}

// writeJSONAtomically writes a JSON file through a temporary file and a
// rename.
func writeJSONAtomically(ctx context.Context, session sandbox.Session, filePath string, value any) error {
	encoded, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Errorf("encode %s: %w", filePath, err)
	}

	temporaryPath := filePath + ".tmp"

	err = session.WriteTextFile(ctx, sandbox.WriteTextFileOptions{
		Path:    temporaryPath,
		Content: string(encoded) + "\n",
	})
	if err != nil {
		return fmt.Errorf("write %s: %w", temporaryPath, err)
	}

	command := "mv -f " + shellquote.Quote(temporaryPath) + " " + shellquote.Quote(filePath)

	return runSandboxCommand(ctx, session, command, "update "+filePath)
}

// resolveHomeSubpath validates a relative POSIX path without traversal and
// joins it under an absolute home path.
func resolveHomeSubpath(homePath string, relativePath string, field string) (string, error) {
	if strings.TrimSpace(homePath) == "" || !path.IsAbs(homePath) {
		return "", fmt.Errorf("invalid homePath %q: expected an absolute POSIX path", homePath)
	}

	trimmed := strings.TrimSpace(relativePath)
	hasTraversal := slices.Contains(strings.FieldsFunc(relativePath, func(character rune) bool {
		return character == '/' || character == '\\'
	}), "..")

	cleaned := strings.TrimRight(path.Clean(trimmed), "/")
	invalid := trimmed == "" ||
		strings.Contains(relativePath, `\`) ||
		path.IsAbs(relativePath) ||
		isWindowsAbsolute(relativePath) ||
		hasTraversal ||
		cleaned == "" ||
		cleaned == "." ||
		strings.HasPrefix(cleaned, "../")

	if invalid {
		return "", fmt.Errorf("invalid %s %q: expected a relative POSIX path without traversal", field, relativePath)
	}

	return path.Join(homePath, cleaned), nil
}

// isWindowsAbsolute reports a Windows drive or UNC path.
func isWindowsAbsolute(value string) bool {
	if strings.HasPrefix(value, `\\`) {
		return true
	}

	return len(value) >= 2 && value[1] == ':'
}
