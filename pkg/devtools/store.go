package devtools

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/aarock1234/ai/pkg/harness/credential"
)

const (
	databaseDirectoryName = ".devtools"
	databaseFileName      = "generations.json"

	// maxDatabaseBytes caps the size of a database file named by a viewer
	// notification, so that an enormous file cannot exhaust memory.
	maxDatabaseBytes = 100 * 1024 * 1024
)

// abortedError is the error recorded for steps that did not finish because
// the request was aborted or the process stopped.
const abortedError = "Request aborted"

// StoreOptions configure OpenStore.
type StoreOptions struct {
	// ProjectDirectory is the directory that contains the .devtools
	// directory. Empty uses the working directory.
	ProjectDirectory string

	// Port is the viewer port to notify after each change. Zero uses the
	// AI_SDK_DEVTOOLS_PORT environment variable or DefaultPort.
	Port int
}

// Store records runs and steps in .devtools/generations.json. Its methods
// are safe for concurrent use.
//
// The store keeps the database in memory after the first read and rewrites
// the whole file after every change, as upstream does. Each write replaces
// the file atomically, so readers never see a partial database, and the
// file is readable only by its owner because it records prompts and
// outputs. The file is a rebuildable debugging log: concurrent writers in
// different processes overwrite each other's changes. Within a process,
// OpenStore returns one shared Store per file.
type Store struct {
	projectDirectory string
	path             string
	notifier         notifier

	mu       sync.Mutex
	database *database

	// activeSteps holds, for every step that has not finished, the result
	// to record when it is aborted.
	activeSteps map[string]func() stepResult
}

var (
	openStoresMu sync.Mutex
	openStores   = map[string]*Store{}
)

// OpenStore returns the store of the project directory. Every call for the
// same database file returns the same Store, configured by the first call.
func OpenStore(options StoreOptions) (*Store, error) {
	projectDirectory, err := resolveProjectDirectory(options.ProjectDirectory)
	if err != nil {
		return nil, err
	}

	path := databasePath(projectDirectory)

	openStoresMu.Lock()
	defer openStoresMu.Unlock()

	if store, ok := openStores[path]; ok {
		return store, nil
	}

	store := &Store{
		projectDirectory: projectDirectory,
		path:             path,
		notifier:         newNotifier(resolvePort(options.Port), path),
		activeSteps:      map[string]func() stepResult{},
	}
	openStores[path] = store

	return store, nil
}

// Path returns the absolute path of the database file.
func (s *Store) Path() string {
	return s.path
}

// Clear deletes every run and step.
func (s *Store) Clear(ctx context.Context) error {
	s.mu.Lock()
	err := s.save(emptyDatabase())
	s.mu.Unlock()

	if err != nil {
		return err
	}

	s.notifier.notify(ctx, eventClear)

	return nil
}

// AbortActiveSteps records every step that has not finished as aborted,
// with the output collected so far. Call it before the process exits, for
// example from a signal handler; it replaces upstream's SIGINT and SIGTERM
// handlers, which a library should not install.
func (s *Store) AbortActiveSteps(ctx context.Context) error {
	s.mu.Lock()
	aborted := make(map[string]stepResult, len(s.activeSteps))
	for stepID, abortResult := range s.activeSteps {
		aborted[stepID] = abortResult()
	}
	clear(s.activeSteps)
	s.mu.Unlock()

	if len(aborted) == 0 {
		return nil
	}

	var errs []error
	for stepID, result := range aborted {
		errs = append(errs, s.updateStepResult(ctx, stepID, result))
	}

	return errors.Join(errs...)
}

// trackActiveStep registers the result to record when the step is
// aborted before it finishes.
func (s *Store) trackActiveStep(stepID string, abortResult func() stepResult) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.activeSteps[stepID] = abortResult
}

// untrackActiveStep removes a finished step from the active steps.
func (s *Store) untrackActiveStep(stepID string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	delete(s.activeSteps, stepID)
}

// createRun records a run unless a run with the same ID exists, and
// returns the stored run.
func (s *Store) createRun(ctx context.Context, newRun run) (run, error) {
	s.mu.Lock()

	db := s.load()
	if index := slices.IndexFunc(db.Runs, func(existing run) bool {
		return existing.ID == newRun.ID
	}); index >= 0 {
		existing := db.Runs[index]
		s.mu.Unlock()

		return existing, nil
	}

	db.Runs = append(db.Runs, newRun)
	err := s.save(db)
	s.mu.Unlock()

	if err != nil {
		return run{}, err
	}

	s.notifier.notify(ctx, eventRun)

	return newRun, nil
}

// createStep records a step that has started and not finished.
func (s *Store) createStep(ctx context.Context, newStep step) error {
	s.mu.Lock()
	db := s.load()
	db.Steps = append(db.Steps, newStep)
	err := s.save(db)
	s.mu.Unlock()

	if err != nil {
		return err
	}

	s.notifier.notify(ctx, eventStep)

	return nil
}

// updateStepResult records the result of a step. Unknown steps are
// ignored.
func (s *Store) updateStepResult(ctx context.Context, stepID string, result stepResult) error {
	s.mu.Lock()

	db := s.load()
	index := slices.IndexFunc(db.Steps, func(candidate step) bool {
		return candidate.ID == stepID
	})
	if index < 0 {
		s.mu.Unlock()

		return nil
	}

	db.Steps[index].applyResult(result)
	err := s.save(db)
	s.mu.Unlock()

	if err != nil {
		return err
	}

	s.notifier.notify(ctx, eventStepUpdate)

	return nil
}

// stepCount returns the number of steps of a run.
func (s *Store) stepCount(runID string) int {
	s.mu.Lock()
	defer s.mu.Unlock()

	return len(s.load().stepsForRun(runID))
}

// load returns the cached database, reading it on first use. The caller
// holds s.mu.
func (s *Store) load() *database {
	if s.database == nil {
		s.database = readDatabase(s.path)
	}

	return s.database
}

// save caches db and writes it to the file. Creating the .devtools
// directory also adds it to the project's .gitignore. The caller holds
// s.mu.
func (s *Store) save(db *database) error {
	s.database = db

	directory := filepath.Dir(s.path)

	if _, err := os.Stat(directory); errors.Is(err, fs.ErrNotExist) {
		if err := os.MkdirAll(directory, 0o700); err != nil {
			return fmt.Errorf("create devtools directory: %w", err)
		}

		if err := ensureGitignore(s.projectDirectory); err != nil {
			return err
		}
	}

	encoded, err := json.MarshalIndent(db, "", "  ")
	if err != nil {
		return fmt.Errorf("encode devtools database: %w", err)
	}

	if err := credential.WriteFileAtomically(s.path, encoded); err != nil {
		return fmt.Errorf("write devtools database: %w", err)
	}

	return nil
}

// database is the content of generations.json.
type database struct {
	Runs  []run  `json:"runs"`
	Steps []step `json:"steps"`
}

func emptyDatabase() *database {
	return &database{
		Runs:  []run{},
		Steps: []step{},
	}
}

// run is a multi-step AI interaction. Child runs are nested calls made by
// a tool of their parent step.
type run struct {
	ID           string  `json:"id"`
	StartedAt    string  `json:"started_at"`
	ParentRunID  *string `json:"parent_run_id"`
	ParentStepID *string `json:"parent_step_id"`
	FunctionID   *string `json:"function_id"`
}

// stepType tells whether a step was a streaming call.
type stepType string

const (
	stepGenerate stepType = "generate"
	stepStream   stepType = "stream"
)

// step is one language model call of a run. DurationMS is nil until the
// step finishes.
type step struct {
	ID              string   `json:"id"`
	RunID           string   `json:"run_id"`
	StepNumber      int      `json:"step_number"`
	Type            stepType `json:"type"`
	ModelID         string   `json:"model_id"`
	Provider        *string  `json:"provider"`
	StartedAt       string   `json:"started_at"`
	DurationMS      *int64   `json:"duration_ms"`
	Input           string   `json:"input"`
	Output          *string  `json:"output"`
	Usage           *string  `json:"usage"`
	Error           *string  `json:"error"`
	RawRequest      *string  `json:"raw_request"`
	RawResponse     *string  `json:"raw_response"`
	RawChunks       *string  `json:"raw_chunks"`
	ProviderOptions *string  `json:"provider_options"`
}

// stepResult is the outcome of a step.
type stepResult struct {
	Duration    time.Duration
	Output      *string
	Usage       *string
	Error       *string
	RawRequest  *string
	RawResponse *string
	RawChunks   *string
}

func (s *step) applyResult(result stepResult) {
	durationMS := result.Duration.Milliseconds()

	s.DurationMS = &durationMS
	s.Output = result.Output
	s.Usage = result.Usage
	s.Error = result.Error
	s.RawRequest = result.RawRequest
	s.RawResponse = result.RawResponse
	s.RawChunks = result.RawChunks
}

// isInProgress reports whether the step has neither finished nor failed.
func (s step) isInProgress() bool {
	return s.DurationMS == nil && s.Error == nil
}

// runsByStartDescending returns the runs, the most recent first.
func (db *database) runsByStartDescending() []run {
	runs := slices.Clone(db.Runs)

	slices.SortStableFunc(runs, func(a, b run) int {
		return parseTimestamp(b.StartedAt).Compare(parseTimestamp(a.StartedAt))
	})

	return runs
}

// stepsForRun returns the steps of a run in step order.
func (db *database) stepsForRun(runID string) []step {
	var steps []step

	for _, candidate := range db.Steps {
		if candidate.RunID == runID {
			steps = append(steps, candidate)
		}
	}

	slices.SortStableFunc(steps, func(a, b step) int {
		return cmp.Compare(a.StepNumber, b.StepNumber)
	})

	return steps
}

// findRun returns the run with the ID.
func (db *database) findRun(runID string) (run, bool) {
	index := slices.IndexFunc(db.Runs, func(candidate run) bool {
		return candidate.ID == runID
	})
	if index < 0 {
		return run{}, false
	}

	return db.Runs[index], true
}

// readDatabase reads a database file. A missing or corrupted file reads as
// an empty database, as upstream starts fresh in both cases.
func readDatabase(path string) *database {
	content, err := os.ReadFile(path)
	if err != nil {
		return emptyDatabase()
	}

	db := emptyDatabase()
	if err := json.Unmarshal(content, db); err != nil {
		return emptyDatabase()
	}

	return db
}

// validateRemoteDatabasePath returns the resolved path of a database file
// named by a viewer notification. It accepts only existing regular files
// named .devtools/generations.json that are at most maxDatabaseBytes, also
// after resolving symbolic links.
func validateRemoteDatabasePath(path string) (string, bool) {
	normalized, ok := normalizeDatabasePath(path)
	if !ok {
		return "", false
	}

	info, err := os.Stat(normalized)
	if err != nil || !info.Mode().IsRegular() || info.Size() > maxDatabaseBytes {
		return "", false
	}

	resolved, err := filepath.EvalSymlinks(normalized)
	if err != nil {
		return "", false
	}

	return normalizeDatabasePath(resolved)
}

func normalizeDatabasePath(path string) (string, bool) {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return "", false
	}

	isDatabaseFile := filepath.Base(absolute) == databaseFileName
	isInDevtoolsDirectory := filepath.Base(filepath.Dir(absolute)) == databaseDirectoryName

	if !isDatabaseFile || !isInDevtoolsDirectory {
		return "", false
	}

	return absolute, true
}

// databasePath returns the path of the database file of a project.
func databasePath(projectDirectory string) string {
	return filepath.Join(projectDirectory, databaseDirectoryName, databaseFileName)
}

// projectDirectoryOf returns the project directory of a database file.
func projectDirectoryOf(path string) string {
	return filepath.Dir(filepath.Dir(path))
}

func resolveProjectDirectory(directory string) (string, error) {
	if directory == "" {
		workingDirectory, err := os.Getwd()
		if err != nil {
			return "", fmt.Errorf("resolve devtools project directory: %w", err)
		}

		return workingDirectory, nil
	}

	absolute, err := filepath.Abs(directory)
	if err != nil {
		return "", fmt.Errorf("resolve devtools project directory: %w", err)
	}

	return absolute, nil
}

func parseTimestamp(value string) time.Time {
	parsed, err := time.Parse(time.RFC3339Nano, value)
	if err != nil {
		return time.Time{}
	}

	return parsed
}

// ensureGitignore adds the .devtools directory to the project's
// .gitignore. It does nothing when the project has no .gitignore or already
// ignores the directory.
func ensureGitignore(projectDirectory string) error {
	path := filepath.Join(projectDirectory, ".gitignore")

	content, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("read .gitignore: %w", err)
	}

	if ignoresDevtoolsDirectory(string(content)) {
		return nil
	}

	entry := databaseDirectoryName + "\n"
	if len(content) > 0 && !strings.HasSuffix(string(content), "\n") {
		entry = "\n" + entry
	}

	updated := string(content) + entry
	if err := os.WriteFile(path, []byte(updated), 0o644); err != nil {
		return fmt.Errorf("update .gitignore: %w", err)
	}

	return nil
}

func ignoresDevtoolsDirectory(gitignore string) bool {
	for line := range strings.SplitSeq(gitignore, "\n") {
		trimmed := strings.TrimSpace(line)

		if trimmed == databaseDirectoryName || trimmed == databaseDirectoryName+"/" {
			return true
		}
	}

	return false
}
