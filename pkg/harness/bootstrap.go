package harness

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"path"
	"slices"
	"strconv"
	"strings"

	"github.com/aarock1234/ai/pkg/sandbox"
)

// Identity versions. Bumping one invalidates every snapshot and marker keyed
// by identities of that kind.
const (
	bootstrapSchemaVersion         = 1
	preparedSandboxIdentityVersion = 1
)

// HashBootstrap returns the deterministic 16-character hexadecimal identity
// of a bootstrap recipe, derived from the harness state directory (with a
// symbolic $HOME, since the concrete one is unknown before creation), the
// harness ID, the bootstrap directory, the files, the commands and the
// recipe schema version. Equivalent recipes share an identity and any
// content change produces a new one.
//
// Go adaptation: files are sorted by byte order, where upstream uses
// localeCompare; identities differ from upstream only for recipes whose
// file paths sort differently under the two orders.
func HashBootstrap(recipe Bootstrap) (string, error) {
	recipeCommands := recipe.Commands
	if recipeCommands == nil {
		recipeCommands = []BootstrapCommand{}
	}

	commands, err := marshalWithoutHTMLEscaping(recipeCommands)
	if err != nil {
		return "", fmt.Errorf("hash bootstrap commands: %w", err)
	}

	files := slices.Clone(recipe.Files)
	slices.SortStableFunc(files, func(first, second BootstrapFile) int {
		return strings.Compare(first.Path, second.Path)
	})

	hash := newIdentityHash()
	hash.add(StateDirectoryPath("$HOME"))
	hash.add(recipe.HarnessID)
	hash.add(recipe.BootstrapDir)

	for _, file := range files {
		hash.add(file.Path)
		hash.add(file.Content)
	}

	hash.add(string(commands))
	hash.add(strconv.Itoa(bootstrapSchemaVersion))

	return hash.identity(), nil
}

// ApplyBootstrapOptions configure ApplyBootstrap.
type ApplyBootstrapOptions struct {
	Session sandbox.Session
	Recipe  Bootstrap

	// Identity is the recipe's HashBootstrap identity.
	Identity string

	// StateDirectory is the harness state directory; see
	// StateDirectoryPath.
	StateDirectory string
}

// ApplyBootstrap applies a bootstrap recipe idempotently. When the marker of
// the identity exists it returns immediately; otherwise it creates the
// bootstrap directory, writes the files, runs the commands in order and
// writes the marker.
func ApplyBootstrap(ctx context.Context, options ApplyBootstrapOptions) error {
	bootstrapDir := resolveBootstrapPath(options.Recipe.BootstrapDir, options.StateDirectory)
	markerPath := path.Join(bootstrapDir, ".bootstrap-"+options.Identity+".ok")

	applied, err := sandboxFileExists(ctx, options.Session, markerPath)
	if err != nil {
		return fmt.Errorf("check bootstrap marker of harness %q: %w", options.Recipe.HarnessID, err)
	}

	if applied {
		return nil
	}

	if err := ensureSandboxDirectory(ctx, options.Session, bootstrapDir); err != nil {
		return fmt.Errorf("create bootstrap directory of harness %q: %w", options.Recipe.HarnessID, err)
	}

	for _, file := range options.Recipe.Files {
		err := options.Session.WriteTextFile(ctx, sandbox.WriteTextFileOptions{
			Path:    resolveBootstrapPath(file.Path, options.StateDirectory),
			Content: file.Content,
		})
		if err != nil {
			return fmt.Errorf("write bootstrap file %s of harness %q: %w", file.Path, options.Recipe.HarnessID, err)
		}
	}

	for _, command := range options.Recipe.Commands {
		if err := runBootstrapCommand(ctx, options, bootstrapDir, command); err != nil {
			return err
		}
	}

	err = options.Session.WriteTextFile(ctx, sandbox.WriteTextFileOptions{
		Path:    markerPath,
		Content: "",
	})
	if err != nil {
		return fmt.Errorf("write bootstrap marker of harness %q: %w", options.Recipe.HarnessID, err)
	}

	return nil
}

// runBootstrapCommand runs one recipe command from the bootstrap directory.
func runBootstrapCommand(ctx context.Context, options ApplyBootstrapOptions, bootstrapDir string, command BootstrapCommand) error {
	result, err := options.Session.Run(ctx, sandbox.ProcessOptions{
		Command:          command.Command,
		WorkingDirectory: bootstrapDir,
	})
	if err != nil {
		return fmt.Errorf("run bootstrap command of harness %q: %w", options.Recipe.HarnessID, err)
	}

	if result.ExitCode != 0 {
		return fmt.Errorf(
			"bootstrap command failed for harness %q (exit %d): %s\n%s",
			options.Recipe.HarnessID,
			result.ExitCode,
			command.Command,
			commandOutput(result),
		)
	}

	return nil
}

// CreateSandboxTemplateOptions configure CreateSandboxTemplate.
type CreateSandboxTemplateOptions struct {
	// Harnesses are prepared in harness ID order; for a repeated ID the last
	// harness is used. At least one is required.
	Harnesses []Harness

	// SandboxConfig adds the agent's bootstrap hook and working directory.
	// OnSession is not used.
	SandboxConfig SandboxConfig
}

// CreateSandboxTemplate returns a template that applies the bootstrap
// recipes of the harnesses and the SandboxConfig.OnBootstrap hook. It returns
// nil when nothing needs to be prepared. Pass the template to a sandbox
// session creator, such as the sandbox/vercel package, to reuse snapshots.
func CreateSandboxTemplate(ctx context.Context, options CreateSandboxTemplateOptions) (*SandboxTemplate, error) {
	if len(options.Harnesses) == 0 {
		return nil, errors.New("create sandbox template: at least one harness is required")
	}

	if err := validateSandboxConfig(options.SandboxConfig); err != nil {
		return nil, err
	}

	workDir, err := optionalWorkDir(options.SandboxConfig.WorkDir)
	if err != nil {
		return nil, err
	}

	recipes, err := resolveTemplateRecipes(ctx, uniqueHarnesses(options.Harnesses))
	if err != nil {
		return nil, err
	}

	recipeIdentities := make(map[string]string, len(recipes))
	for _, recipe := range recipes {
		recipeIdentities[recipe.recipe.HarnessID] = recipe.identity
	}

	identity, ok := preparedSandboxIdentity(preparedIdentityInput{
		recipeIdentities: recipeIdentities,
		bootstrapHash:    options.SandboxConfig.BootstrapHash,
		workDir:          workDir,
	})
	if !ok {
		return nil, nil
	}

	config := options.SandboxConfig

	return &SandboxTemplate{
		Identity: identity,
		Prepare: func(ctx context.Context, session sandbox.Session) error {
			for _, recipe := range recipes {
				err := runSandboxBootstrap(ctx, sandboxBootstrapInput{
					session:        session,
					recipe:         &recipe.recipe,
					recipeIdentity: recipe.identity,
				})
				if err != nil {
					return err
				}
			}

			return runSandboxBootstrap(ctx, sandboxBootstrapInput{
				session:                 session,
				workDir:                 workDir,
				onBootstrap:             config.OnBootstrap,
				bootstrapHash:           config.BootstrapHash,
				skipOnBootstrapIfMarked: true,
			})
		},
	}, nil
}

// identifiedRecipe is a bootstrap recipe with its identity.
type identifiedRecipe struct {
	recipe   Bootstrap
	identity string
}

// resolveTemplateRecipes returns the recipes of the harnesses that have one.
func resolveTemplateRecipes(ctx context.Context, harnesses []Harness) ([]identifiedRecipe, error) {
	var recipes []identifiedRecipe

	for _, harness := range harnesses {
		recipe, err := harnessBootstrap(ctx, harness)
		if err != nil {
			return nil, err
		}

		if recipe == nil {
			continue
		}

		identity, err := HashBootstrap(*recipe)
		if err != nil {
			return nil, err
		}

		recipes = append(recipes, identifiedRecipe{
			recipe:   *recipe,
			identity: identity,
		})
	}

	return recipes, nil
}

// uniqueHarnesses keeps the last harness of each ID, sorted by ID.
func uniqueHarnesses(harnesses []Harness) []Harness {
	byID := make(map[string]Harness, len(harnesses))
	for _, harness := range harnesses {
		byID[harness.HarnessID()] = harness
	}

	unique := make([]Harness, 0, len(byID))
	for _, harness := range byID {
		unique = append(unique, harness)
	}

	slices.SortFunc(unique, func(first, second Harness) int {
		return strings.Compare(first.HarnessID(), second.HarnessID())
	})

	return unique
}

// harnessBootstrap returns the harness's recipe, or nil when it has none.
func harnessBootstrap(ctx context.Context, harness Harness) (*Bootstrap, error) {
	bootstrapper, ok := harness.(Bootstrapper)
	if !ok {
		return nil, nil
	}

	recipe, err := bootstrapper.Bootstrap(ctx)
	if err != nil {
		return nil, fmt.Errorf("get bootstrap recipe of harness %q: %w", harness.HarnessID(), err)
	}

	return recipe, nil
}

// sandboxBootstrapInput describes one bootstrap run.
type sandboxBootstrapInput struct {
	session sandbox.Session

	// recipe is nil when only the application hook runs.
	recipe         *Bootstrap
	recipeIdentity string

	workDir                 string
	onBootstrap             func(ctx context.Context, event SandboxBootstrapEvent) error
	bootstrapHash           string
	skipOnBootstrapIfMarked bool
}

// runSandboxBootstrap applies a harness recipe and then the application's
// bootstrap hook, writing a marker for the hook's hash.
func runSandboxBootstrap(ctx context.Context, input sandboxBootstrapInput) error {
	if input.recipe != nil {
		stateDirectory, err := sandboxStateDirectory(ctx, input.session)
		if err != nil {
			return err
		}

		err = ApplyBootstrap(ctx, ApplyBootstrapOptions{
			Session:        input.session,
			Recipe:         *input.recipe,
			Identity:       input.recipeIdentity,
			StateDirectory: stateDirectory,
		})
		if err != nil {
			return err
		}
	}

	if input.onBootstrap == nil {
		return nil
	}

	if input.skipOnBootstrapIfMarked && input.bootstrapHash != "" {
		marked, err := hasOnBootstrapMarker(ctx, input.session, input.bootstrapHash)
		if err != nil {
			return err
		}

		if marked {
			return nil
		}
	}

	return runOnBootstrap(ctx, input)
}

// runOnBootstrap runs the application's bootstrap hook in its working
// directory and records its marker.
func runOnBootstrap(ctx context.Context, input sandboxBootstrapInput) error {
	defaultWorkingDirectory, err := ResolveSandboxDefaultWorkingDirectory(ctx, input.session)
	if err != nil {
		return err
	}

	workDir := defaultWorkingDirectory
	if input.workDir != "" {
		workDir = path.Join(defaultWorkingDirectory, input.workDir)
	}

	if err := ensureSandboxDirectory(ctx, input.session, workDir); err != nil {
		return err
	}

	err = input.onBootstrap(ctx, SandboxBootstrapEvent{
		Session: input.session,
		WorkDir: workDir,
	})
	if err != nil {
		return fmt.Errorf("run sandbox bootstrap hook: %w", err)
	}

	if input.bootstrapHash == "" {
		return nil
	}

	return writeOnBootstrapMarker(ctx, input.session, input.bootstrapHash)
}

// applySessionBootstrap ensures the harness recipe on a sandbox session.
func applySessionBootstrap(ctx context.Context, session sandbox.Session, recipe Bootstrap) error {
	identity, err := HashBootstrap(recipe)
	if err != nil {
		return err
	}

	stateDirectory, err := sandboxStateDirectory(ctx, session)
	if err != nil {
		return err
	}

	return ApplyBootstrap(ctx, ApplyBootstrapOptions{
		Session:        session,
		Recipe:         recipe,
		Identity:       identity,
		StateDirectory: stateDirectory,
	})
}

// sandboxStateDirectory returns the harness state directory of a sandbox.
func sandboxStateDirectory(ctx context.Context, session sandbox.Session) (string, error) {
	homeDir, err := ResolveSandboxHomeDir(ctx, session)
	if err != nil {
		return "", err
	}

	return StateDirectoryPath(homeDir), nil
}

// onBootstrapMarkerPath returns the marker recording that the bootstrap hook
// with bootstrapHash ran.
func onBootstrapMarkerPath(ctx context.Context, session sandbox.Session, bootstrapHash string) (string, error) {
	stateDirectory, err := sandboxStateDirectory(ctx, session)
	if err != nil {
		return "", err
	}

	digest := sha256.Sum256([]byte(bootstrapHash))

	return path.Join(stateDirectory, ".on-bootstrap", hex.EncodeToString(digest[:])+".ok"), nil
}

// hasOnBootstrapMarker reports whether the bootstrap hook already ran.
func hasOnBootstrapMarker(ctx context.Context, session sandbox.Session, bootstrapHash string) (bool, error) {
	markerPath, err := onBootstrapMarkerPath(ctx, session, bootstrapHash)
	if err != nil {
		return false, err
	}

	return sandboxFileExists(ctx, session, markerPath)
}

// writeOnBootstrapMarker records that the bootstrap hook ran.
func writeOnBootstrapMarker(ctx context.Context, session sandbox.Session, bootstrapHash string) error {
	markerPath, err := onBootstrapMarkerPath(ctx, session, bootstrapHash)
	if err != nil {
		return err
	}

	if err := ensureSandboxDirectory(ctx, session, path.Dir(markerPath)); err != nil {
		return fmt.Errorf("create bootstrap hook marker directory: %w", err)
	}

	err = session.WriteTextFile(ctx, sandbox.WriteTextFileOptions{
		Path:    markerPath,
		Content: "",
	})
	if err != nil {
		return fmt.Errorf("write bootstrap hook marker: %w", err)
	}

	return nil
}

// preparedIdentityInput is the input of a prepared sandbox identity.
type preparedIdentityInput struct {
	recipeIdentities map[string]string
	bootstrapHash    string
	workDir          string
}

// preparedSandboxIdentity combines the recipe identities of several
// harnesses with the application's bootstrap. It reports false when there
// is nothing to prepare.
func preparedSandboxIdentity(input preparedIdentityInput) (string, bool) {
	if len(input.recipeIdentities) == 0 && input.bootstrapHash == "" {
		return "", false
	}

	harnessIDs := make([]string, 0, len(input.recipeIdentities))
	for harnessID := range input.recipeIdentities {
		harnessIDs = append(harnessIDs, harnessID)
	}

	slices.Sort(harnessIDs)

	hash := newIdentityHash()
	hash.add(strconv.Itoa(preparedSandboxIdentityVersion))
	hash.add(input.workDir)
	hash.add(input.bootstrapHash)

	for _, harnessID := range harnessIDs {
		hash.add(harnessID)
		hash.add(input.recipeIdentities[harnessID])
	}

	return hash.identity(), true
}

// validateSandboxConfig checks the bootstrap settings of a sandbox config.
func validateSandboxConfig(config SandboxConfig) error {
	if (config.OnBootstrap == nil) != (config.BootstrapHash == "") {
		return errors.New("harness agent: sandbox config OnBootstrap and BootstrapHash must be set together")
	}

	_, err := optionalWorkDir(config.WorkDir)

	return err
}

// optionalWorkDir normalizes a configured working directory, keeping empty
// as unset.
func optionalWorkDir(workDir string) (string, error) {
	if workDir == "" {
		return "", nil
	}

	return normalizeSandboxWorkDir(workDir)
}

// normalizeSandboxWorkDir validates and cleans a relative working directory.
func normalizeSandboxWorkDir(workDir string) (string, error) {
	switch {
	case strings.ContainsRune(workDir, 0):
		return "", errors.New("harness agent: sandbox work directory must not contain NUL")
	case strings.Contains(workDir, `\`):
		return "", errors.New("harness agent: sandbox work directory must use POSIX path separators")
	case path.IsAbs(workDir):
		return "", errors.New("harness agent: sandbox work directory must be relative")
	}

	normalized := path.Clean(workDir)
	escapes := normalized == ".." || strings.HasPrefix(normalized, "../")
	collapsedToDefault := normalized == "." && workDir != "."

	if escapes || collapsedToDefault {
		return "", errors.New("harness agent: sandbox work directory must stay inside the default working directory")
	}

	return normalized, nil
}

// sessionWorkDirInput describes the working directory of a session.
type sessionWorkDirInput struct {
	defaultWorkingDirectory string
	harnessID               string
	sessionID               string

	// workDir is the configured relative directory; empty selects the
	// per-session directory.
	workDir string
}

// resolveSessionWorkDir returns the working directory of a session.
func resolveSessionWorkDir(input sessionWorkDirInput) string {
	workDir := input.workDir
	if workDir == "" {
		workDir = EncodePathSegment(input.harnessID) + "-" + EncodePathSegment(input.sessionID)
	}

	return path.Join(input.defaultWorkingDirectory, workDir)
}

// resolveBootstrapPath resolves a recipe path against the state directory.
func resolveBootstrapPath(recipePath string, stateDirectory string) string {
	if path.IsAbs(recipePath) {
		return path.Clean(recipePath)
	}

	return path.Join(stateDirectory, recipePath)
}

// sandboxFileExists reports whether a file exists in the sandbox.
func sandboxFileExists(ctx context.Context, session sandbox.Session, filePath string) (bool, error) {
	_, err := session.ReadTextFile(ctx, sandbox.ReadTextFileOptions{
		Path: filePath,
	})
	if errors.Is(err, fs.ErrNotExist) {
		return false, nil
	}

	if err != nil {
		return false, err
	}

	return true, nil
}

// marshalWithoutHTMLEscaping encodes value as JavaScript's JSON.stringify
// does, without escaping <, > and &.
func marshalWithoutHTMLEscaping(value any) ([]byte, error) {
	var buffer bytes.Buffer

	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)

	if err := encoder.Encode(value); err != nil {
		return nil, err
	}

	return bytes.TrimSuffix(buffer.Bytes(), []byte("\n")), nil
}

// identityHash accumulates NUL-terminated strings into a SHA-256 identity.
type identityHash struct {
	buffer bytes.Buffer
}

// newIdentityHash returns an empty identity hash.
func newIdentityHash() *identityHash {
	return &identityHash{}
}

// add appends one NUL-terminated string.
func (h *identityHash) add(value string) {
	h.buffer.WriteString(value)
	h.buffer.WriteByte(0)
}

// identity returns the first eight digest bytes as hexadecimal.
func (h *identityHash) identity() string {
	digest := sha256.Sum256(h.buffer.Bytes())

	return hex.EncodeToString(digest[:8])
}
